/**
 * @file shield.ts
 * @description Hard Action-Masking Shield that intercepts and sanitizes every AI
 * decision so the bot can never violate Classic (Non-OP) PvP rules at runtime.
 */

import { bot, Monster } from "turu";
import {
    BLUDRUT_SAFEZONE_CELLS,
    checkTrajectoryDoorCollision,
    NPC_MANA_TAP_SAFE_HP_MIN,
    NPC_MANA_TAP_SAFE_HP_RATIO,
} from "../config/constants";
import type { MatchRulesConfig } from "../config/match_config";
import { isProtectedNpcName } from "../state/extractor";
import type {
    AtomicCombatAction,
    AtomicMovementDecision,
    CombatDecision,
    CombatObservation,
} from "../state/types";

/**
 * Finds a safe NPC (Brawler/Restorer) in the current cell whose HP is high enough
 * to be tapped for mana regeneration without any risk of killing it.
 */
export function findSafeManaTapNpc(obs: CombatObservation): Monster | null {
    for (const m of obs.cellMonsters) {
        if (m.state <= 0 || m.hp <= 0) continue;
        if (!isProtectedNpcName(m.name)) continue;
        const hpRatio = m.hp / Math.max(1, m.maxHp);
        if (m.hp >= NPC_MANA_TAP_SAFE_HP_MIN && hpRatio >= NPC_MANA_TAP_SAFE_HP_RATIO) {
            return m;
        }
    }
    return null;
}

/**
 * Checks if the bot is currently targeting a protected NPC whose HP is approaching
 * the danger zone, and immediately aborts the attack if so.
 */
export function enforceNpcProtectionGuard(obs: CombatObservation): boolean {
    if (!obs.hasTarget || !obs.targetIsNpc) return false;

    const targetMonster = bot.player.target;
    if (!targetMonster) return false;

    const hpRatio = targetMonster.hp / Math.max(1, targetMonster.maxHp);
    if (
        targetMonster.hp < NPC_MANA_TAP_SAFE_HP_MIN ||
        hpRatio < NPC_MANA_TAP_SAFE_HP_RATIO
    ) {
        bot.combat.cancelAutoAttack();
        bot.combat.cancelTarget();
        bot.log(
            `[RuleShield] Stopped mana-tapping NPC "${targetMonster.name}" at HP=${targetMonster.hp} (${Math.round(hpRatio * 100)}%) to prevent illegal NPC kill.`,
        );
        return true;
    }
    return false;
}

/**
 * Filters a proposed CombatDecision through the Non-OP ruleset constraints.
 */
export function applyRuleShield(
    decision: CombatDecision,
    obs: CombatObservation,
    _config: MatchRulesConfig,
): CombatDecision {
    let sanitizedCombat =
        decision.combatAction ??
        (decision.action === "USE_SKILL_1"
            ? "USE_SKILL_1"
            : decision.action === "USE_SKILL_2"
              ? "USE_SKILL_2"
              : decision.action === "USE_SKILL_3"
                ? "USE_SKILL_3"
                : decision.action === "USE_SKILL_4"
                  ? "USE_SKILL_4"
                  : decision.action === "MANA_TAP_NPC"
                    ? "MANA_TAP_NPC"
                    : decision.action === "TARGET_OPPONENT"
                      ? "AUTO_ATTACK"
                      : "NONE");

    let sanitizedMovement: AtomicMovementDecision = decision.movement
        ? { ...decision.movement }
        : {
              type:
                  decision.action === "SPRINT_TO_POINT" || decision.action === "SPRINT_TO_DOOR"
                      ? "SPRINT"
                      : decision.action === "WALK_TO_POINT" || decision.action === "WALK_TO_DOOR"
                        ? "WALK"
                        : "STAND",
              targetX: decision.targetX ?? Math.round(obs.selfX),
              targetY: decision.targetY ?? Math.round(obs.selfY),
              targetCell: decision.targetCell,
              targetPad: decision.targetPad,
          };

    let sanitizedReason = decision.reason;

    // 1. Hard Rule: Never use offensive skills on a protected NPC (mana-tap uses basic attacks only)
    if (obs.targetIsNpc) {
        if (
            sanitizedCombat === "USE_SKILL_1" ||
            sanitizedCombat === "USE_SKILL_2" ||
            sanitizedCombat === "USE_SKILL_3" ||
            sanitizedCombat === "USE_SKILL_4"
        ) {
            sanitizedCombat = "NONE";
            sanitizedReason += " [RuleShield: blocked offensive skill on NPC]";
        }
    }

    // 2. Hard Rule: Mana-Tap NPC is only allowed if a safe high-HP NPC exists
    if (sanitizedCombat === "MANA_TAP_NPC") {
        const safeNpc = findSafeManaTapNpc(obs);
        if (!safeNpc) {
            sanitizedCombat = "CANCEL_AUTO_ATTACK";
            sanitizedReason += " [RuleShield: blocked NPC mana tap - HP too low]";
        }
    }

    // 3. Stun Guard: Hard-lock all movement while stunned, off-balance, or immobilized
    if (obs.selfStunned) {
        if (sanitizedMovement.type !== "STAND") {
            sanitizedMovement.type = "STAND";
            sanitizedReason += " [RuleShield: frozen movement while stunned]";
        }
    }

    // 4. Safezone transition lock mid-wave
    if (
        sanitizedMovement.targetCell &&
        BLUDRUT_SAFEZONE_CELLS.has(sanitizedMovement.targetCell) &&
        !obs.inSafezone &&
        !obs.wave.canReturnToSafezone
    ) {
        sanitizedMovement.type = "STAND";
        sanitizedReason += " [RuleShield: blocked illegal mid-wave safezone jump]";
    }

    // 5. Stage Bounds Guard (Prevents "Roofing" & Door Trapping)
    if (sanitizedMovement.type !== "STAND") {
        let safeX = sanitizedMovement.targetX;
        let safeY = sanitizedMovement.targetY;

        if (obs.currentCell.toLowerCase() === "crosslower") {
            safeX = Math.min(640, Math.max(320, safeX));
            safeY = Math.min(435, Math.max(375, safeY));
        } else {
            safeY = Math.min(445, Math.max(345, safeY));
        }

        sanitizedMovement.targetX = safeX;
        sanitizedMovement.targetY = safeY;

        // Trajectory collision check when not intentionally taking a door
        if (!sanitizedMovement.targetCell) {
            const collision = checkTrajectoryDoorCollision(
                obs.currentCell,
                obs.selfX,
                obs.selfY,
                safeX,
                safeY,
            );
            if (collision) {
                const safeT = Math.max(0, collision.hitT - 0.15);
                sanitizedMovement.targetX = Math.round(obs.selfX + (safeX - obs.selfX) * safeT);
                sanitizedMovement.targetY = Math.round(obs.selfY + (safeY - obs.selfY) * safeT);
                sanitizedReason += ` [RuleShield: clamped trajectory before ${collision.door.targetAlias}]`;
            }
        }
    }

    return {
        ...decision,
        combatAction: sanitizedCombat,
        movement: sanitizedMovement,
        reason: sanitizedReason,
    };
}
