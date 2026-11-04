/**
 * @file nn_inference.ts
 * @description Sub-millisecond Pure TypeScript Neural Network Inference Engine for
 * executing the Gymnasium-trained PPO `Battle Warrior` policy (`trained_policy_weights.ts`)
 * directly inside Turu Headless.
 *
 * Maps `CombatObservation` -> 26-dim feature vector (matching `sim/bludrut_env.py`)
 * -> `26 -> 64 (tanh) -> 64 (tanh) -> 14` logits -> Action Masking -> `CombatDecision`.
 */

import { BLUDRUT_16_SCREENS, findNextDoorTowardCell } from "../config/constants";
import { findSafeManaTapNpc } from "../rules/shield";
import type { CombatDecision, CombatObservation } from "../state/types";
import { BATTLE_WARRIOR_1V1_POLICY_WEIGHTS, type TrainedMlpPolicySpec } from "./trained_policy_weights";

const MELEE_RANGE = 301.0;

/**
 * Encodes the live Turu `CombatObservation` into the exact 26-dimensional normalized
 * observation vector expected by the Gymnasium PPO neural network (`sim/bludrut_env.py`).
 */
export function buildObservationVector26(obs: CombatObservation): number[] {
    const opp =
        obs.opponents.find((p) => p.cell === obs.currentCell && p.alive) ??
        obs.opponents.find((p) => p.alive) ??
        obs.opponents[0];

    const sameCell = opp && opp.cell === obs.currentCell && opp.alive ? 1.0 : 0.0;
    const dist =
        sameCell && opp
            ? Math.hypot(obs.selfX - opp.x, obs.selfY - opp.y)
            : 1000.0;
    const inMelee = sameCell && dist <= MELEE_RANGE ? 1.0 : 0.0;

    const screenIdx = Math.max(
        0,
        BLUDRUT_16_SCREENS.findIndex(
            (s) => s.cell.toLowerCase() === obs.currentCell.toLowerCase(),
        ),
    );

    const selfStunRem =
        obs.selfAuras.find((a) => /off balance|stun/i.test(a.name))?.remainingTime ?? 0.0;
    const prepAura = obs.selfAuras.find((a) => /prepared strike/i.test(a.name));
    const selfPrepRem = prepAura?.remainingTime ?? 0.0;
    const selfPrepStacks = prepAura ? Math.max(1, prepAura.stacks) : 0;
    const selfOnGuardRem =
        obs.selfAuras.find((a) => /on guard/i.test(a.name))?.remainingTime ?? 0.0;

    const oppStunRem =
        obs.targetAuras.find((a) => /off balance|stun/i.test(a.name))?.remainingTime ?? 0.0;
    const oppOnGuardRem =
        obs.targetAuras.find((a) => /on guard|shieldwall|bulwark/i.test(a.name))
            ?.remainingTime ?? 0.0;

    const safeNpc = findSafeManaTapNpc(obs);

    const vec = [
        obs.selfHpRatio,                                     // 0: self HP ratio
        obs.selfMpRatio,                                     // 1: self MP ratio
        obs.selfSpRatio,                                     // 2: self Stamina (intSP) ratio
        obs.canSprint ? 1.0 : 0.0,                           // 3: can sprint (SP >= 50)
        obs.selfInCombat ? 1.0 : 0.0,                        // 4: self in combat
        obs.selfInCombat ? 0.0 : 0.8,                        // 5: OOC timer progress
        0.5,                                                 // 6: server 6s regen phase
        Math.min(1.0, Math.max(0.0, obs.selfX / 960.0)),     // 7: self X normalized
        Math.min(1.0, Math.max(0.0, obs.selfY / 540.0)),     // 8: self Y normalized
        screenIdx / 15.0,                                    // 9: self cell index (0..1)
        obs.skillsReady[0] ? 0.0 : 0.5,                      // 10: a1 CD ratio
        obs.skillsReady[1] ? 0.0 : 0.5,                      // 11: a2 CD ratio
        obs.skillsReady[2] ? 0.0 : 0.5,                      // 12: a3 CD ratio
        obs.skillsReady[3] ? 0.0 : 0.5,                      // 13: a4 CD ratio
        Math.min(1.0, selfStunRem / 4.0),                    // 14: self stun remaining
        Math.min(1.0, selfPrepRem / 5.0),                    // 15: self Prepared Strike remaining
        Math.min(1.0, selfPrepStacks / 2.0),                 // 16: self Prepared Strike stacks
        Math.min(1.0, selfOnGuardRem / 10.0),                // 17: self On Guard remaining
        sameCell,                                            // 18: opponent in same screen
        inMelee,                                             // 19: opponent within melee range (301px)
        Math.min(1.0, dist / 960.0),                         // 20: 2D distance to opponent
        opp ? opp.hpRatio : 1.0,                             // 21: opponent HP ratio
        opp ? opp.mpRatio : 1.0,                             // 22: opponent MP ratio
        Math.min(1.0, oppStunRem / 4.0),                     // 23: opponent stun remaining
        Math.min(1.0, oppOnGuardRem / 10.0),                 // 24: opponent On Guard remaining
        safeNpc ? 1.0 : 0.0,                                 // 25: safe NPC available in cell
    ];

    return vec.map((v) => Math.max(-1.0, Math.min(1.0, v)));
}

function denseTanh(
    input: readonly number[],
    weights: readonly (readonly number[])[],
    bias: readonly number[],
): number[] {
    const out: number[] = new Array(bias.length);
    for (let i = 0; i < bias.length; i++) {
        let sum = bias[i]!;
        const row = weights[i]!;
        for (let j = 0; j < input.length; j++) {
            sum += row[j]! * input[j]!;
        }
        out[i] = Math.tanh(sum);
    }
    return out;
}

function denseLinear(
    input: readonly number[],
    weights: readonly (readonly number[])[],
    bias: readonly number[],
): number[] {
    const out: number[] = new Array(bias.length);
    for (let i = 0; i < bias.length; i++) {
        let sum = bias[i]!;
        const row = weights[i]!;
        for (let j = 0; j < input.length; j++) {
            sum += row[j]! * input[j]!;
        }
        out[i] = sum;
    }
    return out;
}

/**
 * Runs forward inference on the trained PPO MLP policy (`26 -> 64 -> 64 -> 14`)
 * with legal action masking and translates the winning action into a `CombatDecision`.
 */
export function inferBattleWarriorDecision(
    obs: CombatObservation,
    policy: TrainedMlpPolicySpec = BATTLE_WARRIOR_1V1_POLICY_WEIGHTS,
): CombatDecision {
    if (obs.currentCell === "Wait") {
        return {
            action: "IDLE",
            reason: "[PPO-NN] Waiting in respawn room (Wait)...",
        };
    }

    if (obs.inSafezone) {
        if (obs.selfHpRatio >= 0.95 && obs.selfMpRatio >= 0.85) {
            const exitDoor = findNextDoorTowardCell(obs.currentCell, "Crosslower");
            if (exitDoor) {
                return {
                    action: obs.canSprint ? "SPRINT_TO_DOOR" : "WALK_TO_DOOR",
                    targetCell: exitDoor.targetCell,
                    targetPad: exitDoor.targetPad,
                    targetX: exitDoor.aimX,
                    targetY: exitDoor.aimY,
                    reason: `[PPO-NN] Ready in Safezone (${obs.currentCell}); exiting door to ${exitDoor.targetAlias} (${exitDoor.targetCell}).`,
                };
            }
        }
        return {
            action: "REST_OUT_OF_COMBAT",
            reason: "[PPO-NN] Recovering in Safezone before next wave.",
        };
    }

    const oppInCell = obs.opponents.find(
        (p) => p.cell === obs.currentCell && p.alive,
    );
    const anyOpp = obs.opponents.find(
        (p) => p.alive && p.cell !== "Enter0" && p.cell !== "Enter1" && p.cell !== "Wait",
    );

    // If no opponent is in our current screen yet, navigate through doors toward the opponent (or Mid `Crosslower`)
    if (!oppInCell) {
        if (obs.selfHpRatio < 0.75 && !obs.selfInCombat) {
            return {
                action: "IDLE",
                reason: `[PPO-NN] Holding in ${obs.currentCell} to catch 6.0s Out-of-Combat server regen tick (HP=${Math.round(obs.selfHpRatio * 100)}%).`,
            };
        }
        const goalCell = anyOpp ? anyOpp.cell : "Crosslower";
        if (obs.currentCell.toLowerCase() !== goalCell.toLowerCase()) {
            const nextDoor = findNextDoorTowardCell(obs.currentCell, goalCell);
            if (nextDoor) {
                return {
                    action: obs.canSprint ? "SPRINT_TO_DOOR" : "WALK_TO_DOOR",
                    targetCell: nextDoor.targetCell,
                    targetPad: nextDoor.targetPad,
                    targetX: nextDoor.aimX,
                    targetY: nextDoor.aimY,
                    reason: `[PPO-NN] Navigating ${obs.currentCell} -> ${nextDoor.targetAlias} (${nextDoor.targetCell}) toward ${goalCell}.`,
                };
            }
        }
        return {
            action: "IDLE",
            reason: `[PPO-NN] Holding center stage at ${obs.currentCell} waiting for opponent.`,
        };
    }

    const x = buildObservationVector26(obs);
    const h1 = denseTanh(x, policy.w1, policy.b1);
    const h2 = denseTanh(h1, policy.w2, policy.b2);
    const logits = denseLinear(h2, policy.wOut, policy.bOut);

    const dist = Math.hypot(obs.selfX - oppInCell.x, obs.selfY - oppInCell.y);
    const inMelee = dist <= MELEE_RANGE;
    const safeNpc = findSafeManaTapNpc(obs);
    const prepAura = obs.selfAuras.find((a) => /prepared strike/i.test(a.name));

    // Mask out illegal/unavailable actions (`-1e9` logit penalty)
    const masked = [...logits];
    if (!obs.canSprint) {
        masked[2] = -1e9;  // SPRINT_TO_OPPONENT
        masked[8] = -1e9;  // SPRINT_KITE_AWAY
        masked[13] = -1e9; // SPRINT_DOOR_BACKWARD
    }
    // Already well inside melee range (<= 220px): do not waste actions or Stamina "approaching"
    if (dist <= 220) {
        masked[1] = -1e9; // APPROACH_OPPONENT
        masked[2] = -1e9; // SPRINT_TO_OPPONENT
    }
    if (!obs.skillsReady[0] || !inMelee) masked[3] = -1e9; // USE_SKILL_1 (Decisive Strike)
    if (!obs.skillsReady[1] || !inMelee) masked[4] = -1e9; // USE_SKILL_2 (Imbalancing Strike)
    if (!obs.skillsReady[2]) masked[5] = -1e9;             // USE_SKILL_3 (Prepared Strike)
    if (!obs.skillsReady[3] || !inMelee) masked[6] = -1e9; // USE_SKILL_4 (On Guard)
    if (!safeNpc || obs.selfMp >= 35) masked[10] = -1e9;   // MANA_TAP_NPC (only when MP < 35)

    // While healthy (HP >= 50%) with an opponent in cell, stay engaged rather than fleeing the room or kiting out of melee range
    if (obs.selfHpRatio >= 0.50) {
        masked[7] = -1e9;  // KITE_AWAY
        masked[8] = -1e9;  // SPRINT_KITE_AWAY
        masked[11] = -1e9; // WALK_DOOR_FORWARD
        masked[12] = -1e9; // WALK_DOOR_BACKWARD
        masked[13] = -1e9; // SPRINT_DOOR_BACKWARD
    }
    // While Prepared Strike is active (next 2 Auto Attacks are unavoidable Crits), hold melee lock so Auto Attacks swing immediately
    if (prepAura && inMelee) {
        masked[9] = -1e9;  // JUKE_VERTICAL
    }

    let bestAct = 0;
    let bestScore = -Infinity;
    for (let i = 0; i < masked.length; i++) {
        if (masked[i]! > bestScore) {
            bestScore = masked[i]!;
            bestAct = i;
        }
    }

    const actName = policy.actionNames[bestAct] ?? "FIGHT_STAND";
    const screenSpec = BLUDRUT_16_SCREENS.find(
        (s) => s.cell.toLowerCase() === obs.currentCell.toLowerCase(),
    );

    switch (bestAct) {
        case 0: // FIGHT_STAND
            if (oppInCell && (!obs.hasTarget || obs.targetIsNpc || !obs.selfInCombat)) {
                return {
                    action: "TARGET_OPPONENT",
                    targetName: oppInCell.username,
                    reason: `[PPO-NN:${actName}] Engaging opponent ${oppInCell.username}.`,
                };
            }
            return {
                action: "IDLE",
                targetName: oppInCell?.username,
                reason: `[PPO-NN:${actName}] Holding position / auto-attacking.`,
            };

        case 1: // APPROACH_OPPONENT
        case 2: // SPRINT_TO_OPPONENT
            if (oppInCell) {
                return {
                    action: bestAct === 2 ? "SPRINT_TO_POINT" : "WALK_TO_POINT",
                    targetX: Math.round(oppInCell.x),
                    targetY: Math.round(oppInCell.y),
                    targetName: oppInCell.username,
                    reason: `[PPO-NN:${actName}] Closing distance to ${oppInCell.username} (${Math.round(dist)}px).`,
                };
            }
            if (anyOpp && screenSpec && screenSpec.doors.length > 0) {
                const door = screenSpec.doors[0]!;
                return {
                    action: bestAct === 2 ? "SPRINT_TO_DOOR" : "WALK_TO_DOOR",
                    targetCell: door.targetCell,
                    targetPad: door.targetPad,
                    targetX: door.aimX,
                    targetY: door.aimY,
                    reason: `[PPO-NN:${actName}] Advancing toward ${anyOpp.cell} via ${door.targetAlias}.`,
                };
            }
            return { action: "IDLE", reason: `[PPO-NN:${actName}] Waiting for opponent.` };

        case 3:
            return {
                action: "USE_SKILL_1",
                targetName: oppInCell?.username,
                reason: `[PPO-NN:${actName}] Firing Decisive Strike (190% weapon dmg).`,
            };
        case 4:
            return {
                action: "USE_SKILL_2",
                targetName: oppInCell?.username,
                reason: `[PPO-NN:${actName}] Firing Imbalancing Strike (4s Stun).`,
            };
        case 5:
            return {
                action: "USE_SKILL_3",
                targetName: oppInCell?.username,
                reason: `[PPO-NN:${actName}] Activating Prepared Strike (next 2 AAs unavoidable Crit).`,
            };
        case 6:
            return {
                action: "USE_SKILL_4",
                targetName: oppInCell?.username,
                reason: `[PPO-NN:${actName}] Activating On Guard (75% dmg + 10s 50% dmg reduction).`,
            };

        case 7: // KITE_AWAY
        case 8: { // SPRINT_KITE_AWAY
            // Center-biased tactical spacing: maintain ~180-240px distance while staying in the center arena
            // NEVER run to X=240 (left door trap) or Y=275 (ceiling/roofing trap)
            const oppX = oppInCell ? oppInCell.x : 480;
            const oppY = oppInCell ? oppInCell.y : 405;
            const centerPull = (480 - obs.selfX) * 0.25;
            const awayDx = obs.selfX >= oppX ? 180 : -180;
            const targetX = Math.round(Math.min(620, Math.max(340, obs.selfX + awayDx + centerPull)));
            const targetY = Math.round(Math.min(430, Math.max(380, oppY > 405 ? 385 : 425)));
            return {
                action: bestAct === 8 ? "SPRINT_TO_POINT" : "WALK_TO_POINT",
                targetX,
                targetY,
                reason: `[PPO-NN:${actName}] Tactical spacing inside center arena at (${targetX}, ${targetY}).`,
            };
        }

        case 9: { // JUKE_VERTICAL
            // Strictly stay on walkable ground Y in [380, 430] to eliminate the "roofing" bug
            const jukeY = obs.selfY > 405 ? 385 : 425;
            const targetX = Math.round(Math.min(620, Math.max(340, obs.selfX)));
            return {
                action: "WALK_TO_POINT",
                targetX,
                targetY: jukeY,
                reason: `[PPO-NN:${actName}] Vertical Y-axis juke on ground to Y=${jukeY}.`,
            };
        }

        case 10: // MANA_TAP_NPC
            if (safeNpc) {
                return {
                    action: "MANA_TAP_NPC",
                    npcMapId: safeNpc.mapId,
                    reason: `[PPO-NN:${actName}] Mana-tapping ${safeNpc.name} (HP=${safeNpc.hp}).`,
                };
            }
            return { action: "IDLE", reason: `[PPO-NN:${actName}] No safe NPC to mana-tap.` };

        case 11: // WALK_DOOR_FORWARD
        case 12: // WALK_DOOR_BACKWARD
        case 13: { // SPRINT_DOOR_BACKWARD
            if (screenSpec && screenSpec.doors.length > 0) {
                const nonSzDoors = screenSpec.doors.filter(
                    (d) => d.targetCell !== "Enter0" && d.targetCell !== "Enter1",
                );
                const door =
                    (bestAct === 11 ? nonSzDoors[nonSzDoors.length - 1] : nonSzDoors[0]) ??
                    screenSpec.doors[0]!;
                return {
                    action: bestAct === 13 ? "SPRINT_TO_DOOR" : "WALK_TO_DOOR",
                    targetCell: door.targetCell,
                    targetPad: door.targetPad,
                    targetX: door.aimX,
                    targetY: door.aimY,
                    reason: `[PPO-NN:${actName}] Transitioning to ${door.targetAlias} (${door.targetCell}).`,
                };
            }
            return { action: "IDLE", reason: `[PPO-NN:${actName}] No valid door available.` };
        }

        default:
            return { action: "IDLE", reason: `[PPO-NN] Default idle.` };
    }
}
