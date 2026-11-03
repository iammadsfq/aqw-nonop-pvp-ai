/**
 * @file classic_warrior.ts
 * @description Pilot 1v1 combat policy for Classic Warrior under Non-OP rules.
 *
 * Classic Warrior Skill Kit:
 * - Skill 1: Decisive Strike (Fast physical strike, 2s CD, 15 MP)
 * - Skill 2: Imbalancing Strike (Physical strike + 4s Stun, 10s CD, 25 MP)
 * - Skill 3: Prepared Strike (Buffs next attack to guaranteed Critical Hit, 12s CD, 20 MP)
 * - Skill 4: On Guard (Reduces incoming damage by 50% for 8s, 20s CD, 30 MP)
 */

import { findSafeManaTapNpc } from "../rules/shield";
import type { CombatDecision, CombatObservation } from "../state/types";

/** Checks if the opponent has an active high-mitigation or dodge/reflect aura. */
function opponentHasDefensiveWall(obs: CombatObservation): boolean {
    return obs.targetAuras.some(
        (a) =>
            /shieldwall|bulwark|shadowblade|blurred|parry|riposte|reflect/i.test(a.name) &&
            a.remainingTime > 0.3,
    );
}

/** Checks if the player already has Prepared Strike active. */
function hasPreparedStrikeBuff(obs: CombatObservation): boolean {
    return obs.selfAuras.some(
        (a) => /prepared strike/i.test(a.name) && a.remainingTime > 0.2,
    );
}

/** Checks if the opponent is currently stunned. */
function opponentIsStunned(obs: CombatObservation): boolean {
    return obs.targetAuras.some(
        (a) => /stun|imbalancing/i.test(a.name) && a.remainingTime > 0.2,
    );
}

export function decideClassicWarrior1v1(obs: CombatObservation): CombatDecision {
    // 1. If in Safezone and healthy/mana-ready, advance to the combat cell
    if (obs.inSafezone) {
        if (obs.selfHpRatio >= 0.95 && obs.selfMpRatio >= 0.85) {
            return {
                action: "ADVANCE_TO_COMBAT_CELL",
                reason: "HP and Mana full in Safezone; advancing to combat cell.",
            };
        }
        return {
            action: "IDLE",
            reason: "Resting in Safezone before next wave.",
        };
    }

    // 2. If wave kill/death condition allows returning to Safezone and HP/MP is low, retreat to SZ
    if (obs.wave.canReturnToSafezone && (obs.selfHpRatio < 0.75 || obs.selfMpRatio < 0.5)) {
        return {
            action: "RETREAT_TO_SAFEZONE",
            reason: "Wave cleared (kill secured); returning to Safezone to reset.",
        };
    }

    // 3. Find living opponent in current cell
    const opponentInCell = obs.opponents.find(
        (p) => p.cell === obs.currentCell && p.alive,
    );

    // 4. If Mana is critically low (< 20%) and no opponent is in melee burst range, tap NPC safely
    if (obs.selfMpRatio < 0.2 && !opponentInCell) {
        const safeNpc = findSafeManaTapNpc(obs);
        if (safeNpc) {
            return {
                action: "MANA_TAP_NPC",
                npcMapId: safeNpc.mapId,
                reason: `Mana low (${Math.round(obs.selfMpRatio * 100)}%); safely tapping ${safeNpc.name} for mana.`,
            };
        }
    }

    if (!opponentInCell) {
        return {
            action: "IDLE",
            reason: "Waiting for opponent in combat cell.",
        };
    }

    // 5. Ensure we are targeting the opponent player (and not an NPC)
    if (!obs.hasTarget || obs.targetIsNpc) {
        return {
            action: "TARGET_OPPONENT",
            targetName: opponentInCell.username,
            reason: `Acquiring opponent target: ${opponentInCell.username}`,
        };
    }

    const [s1Ready, s2Ready, s3Ready, s4Ready] = obs.skillsReady;
    const enemyHasWall = opponentHasDefensiveWall(obs);

    // 6. Defensive Pop: Use Skill 4 (On Guard - 50% damage reduction) when trading in combat
    if (s4Ready && obs.selfHpRatio < 0.85 && obs.selfMp >= 30) {
        return {
            action: "USE_SKILL_4",
            reason: "Activating On Guard (50% damage reduction) during active trade.",
        };
    }

    // 7. Combo Execution: If Prepared Strike (Guaranteed Crit) is active, immediately fire Skill 1 (Decisive Strike)
    if (hasPreparedStrikeBuff(obs) && s1Ready) {
        return {
            action: "USE_SKILL_1",
            reason: "Consuming Prepared Strike buff with guaranteed-crit Decisive Strike.",
        };
    }

    // 8. Hold burst skills if opponent has an active defensive wall / dodge aura
    if (enemyHasWall) {
        return {
            action: "IDLE",
            reason: "Holding cooldowns while opponent defensive/dodge aura is active.",
        };
    }

    // 9. Stun Lock: Use Skill 2 (Imbalancing Strike) to stun opponent before burst
    if (s2Ready && !opponentIsStunned(obs) && obs.selfMp >= 25) {
        return {
            action: "USE_SKILL_2",
            reason: "Landing Imbalancing Strike (4s Stun) on unprotected opponent.",
        };
    }

    // 10. Burst Setup: Use Skill 3 (Prepared Strike) when Skill 1 is ready or nearly ready
    if (s3Ready && s1Ready && obs.selfMp >= 35) {
        return {
            action: "USE_SKILL_3",
            reason: "Priming Prepared Strike (100% Crit) for Decisive Strike combo.",
        };
    }

    // 11. Sustained DPS: Use Skill 1 (Decisive Strike) when Mana is healthy
    if (s1Ready && obs.selfMp >= 20) {
        return {
            action: "USE_SKILL_1",
            reason: "Using Decisive Strike for sustained pressure.",
        };
    }

    return {
        action: "IDLE",
        reason: "Auto-attacking and waiting for cooldowns/mana.",
    };
}
