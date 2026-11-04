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
import type { CombatDecision, CombatObservation } from "../state/types";

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
    // 1. Hard Rule: Never use offensive skills (1..4) on a protected NPC (only basic auto-attack for mana tap)
    if (obs.targetIsNpc) {
        if (
            decision.action === "USE_SKILL_1" ||
            decision.action === "USE_SKILL_2" ||
            decision.action === "USE_SKILL_3" ||
            decision.action === "USE_SKILL_4"
        ) {
            return {
                action: "IDLE",
                reason: "[RuleShield] Blocked offensive skill on protected NPC (mana-tap uses basic attacks only).",
            };
        }
    }

    // 2. Hard Rule: Returning to Safezone (SZ) is prohibited mid-wave unless allowed by wave death/kill state
    if (decision.action === "RETREAT_TO_SAFEZONE" && !obs.inSafezone) {
        if (!obs.wave.canReturnToSafezone) {
            return {
                action: "IDLE",
                reason: "[RuleShield] Blocked illegal mid-wave retreat to Safezone.",
            };
        }
    }

    if (
        (decision.action === "WALK_TO_DOOR" || decision.action === "SPRINT_TO_DOOR") &&
        decision.targetCell &&
        BLUDRUT_SAFEZONE_CELLS.has(decision.targetCell) &&
        !obs.inSafezone &&
        !obs.wave.canReturnToSafezone
    ) {
        return {
            action: "IDLE",
            reason: `[RuleShield] Blocked illegal mid-wave door transition into Safezone (${decision.targetCell}).`,
        };
    }

    // 3. Stun Guard: Hard-lock all movement while stunned, off-balance, or immobilized
    // In Flash AQW, server rejects move packets when stunned. Moving on client causes massive desync!
    if (obs.selfStunned) {
        if (
            decision.action === "WALK_TO_POINT" ||
            decision.action === "SPRINT_TO_POINT" ||
            decision.action === "WALK_TO_DOOR" ||
            decision.action === "SPRINT_TO_DOOR"
        ) {
            return {
                action: "IDLE",
                reason: "[RuleShield] Blocked movement while stunned/immobilized (preventing client-server position desync).",
            };
        }
    }

    // 4. Ground-Level & Stage Bounds Guard (Prevents "Roofing" & Door Trapping):
    // In BludrutBrawl combat rooms (e.g., Crosslower), ground is strictly Y in [375, 440].
    // Any target above Y=350 causes the client avatar to clip into the ceiling ("roofing").
    // We also keep X in [300, 660] to prevent trapping against left (X<=230) and right (X>=745) doors.
    if (
        (decision.action === "WALK_TO_POINT" || decision.action === "SPRINT_TO_POINT") &&
        typeof decision.targetX === "number" &&
        typeof decision.targetY === "number"
    ) {
        let safeX = decision.targetX;
        let safeY = decision.targetY;

        if (obs.currentCell.toLowerCase() === "crosslower") {
            safeX = Math.min(640, Math.max(320, safeX));
            safeY = Math.min(435, Math.max(375, safeY));
        } else {
            // General safe floor bounds for all Bludrut screens
            safeY = Math.min(445, Math.max(345, safeY));
        }

        decision = {
            ...decision,
            targetX: safeX,
            targetY: safeY,
        };
    }

    // 5. Trajectory Door-Collision Guard:
    // When kiting inside the current screen (WALK_TO_POINT / SPRINT_TO_POINT), check if the linear
    // path from (selfX, selfY) to (targetX, targetY) crosses any 2D door activation zone.
    // If it does, clamp the destination just before the door boundary so the bot doesn't
    // accidentally trigger a room transition!
    if (
        (decision.action === "WALK_TO_POINT" || decision.action === "SPRINT_TO_POINT") &&
        !decision.targetName &&
        typeof decision.targetX === "number" &&
        typeof decision.targetY === "number"
    ) {
        const collision = checkTrajectoryDoorCollision(
            obs.currentCell,
            obs.selfX,
            obs.selfY,
            decision.targetX,
            decision.targetY,
        );
        if (collision) {
            const safeT = Math.max(0, collision.hitT - 0.15);
            const clampedX = Math.round(obs.selfX + (decision.targetX - obs.selfX) * safeT);
            const clampedY = Math.round(obs.selfY + (decision.targetY - obs.selfY) * safeT);
            return {
                ...decision,
                targetX: clampedX,
                targetY: clampedY,
                reason: `${decision.reason} [RuleShield: clamped trajectory before ${collision.door.targetAlias} door trigger area]`,
            };
        }
    }

    // 6. Hard Rule: Mana-Tap NPC is only allowed if a safe high-HP NPC exists in the cell
    if (decision.action === "MANA_TAP_NPC") {
        const safeNpc = findSafeManaTapNpc(obs);
        if (!safeNpc) {
            return {
                action: "CANCEL_AUTO_ATTACK",
                reason: "[RuleShield] Blocked NPC mana-tap: no NPC with safe HP margin available.",
            };
        }
        return {
            ...decision,
            npcMapId: safeNpc.mapId,
        };
    }

    return decision;
}
