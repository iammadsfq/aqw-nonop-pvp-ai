/**
 * @file nn_inference.ts
 * @description Sub-millisecond Pure TypeScript Multi-Head Neural Network Inference Engine
 * for executing the Compound Atomic Policy (`COMPOUND_ATOMIC_POLICY_WEIGHTS`).
 *
 * Decouples human-level inputs into 3 orthogonal atomic channels:
 * 1. Combat Action: NONE, AUTO_ATTACK, USE_SKILL_1..4, MANA_TAP_NPC
 * 2. Movement Type: STAND, WALK, SPRINT
 * 3. Spatial Target: Continuous (X, Y) ground coordinates
 *
 * Runs simultaneously on every 100ms frame in pure TypeScript.
 */

import {
    BLUDRUT_16_SCREENS,
    checkTrajectoryDoorCollision,
    findNextDoorTowardCell,
    getDoorAimCoordinates,
} from "../config/constants";
import { findSafeManaTapNpc } from "../rules/shield";
import type {
    AtomicCombatAction,
    AtomicMovementType,
    CombatDecision,
    CombatObservation,
} from "../state/types";
import {
    COMPOUND_ATOMIC_POLICY_WEIGHTS,
    type CompoundAtomicPolicySpec,
} from "./trained_policy_weights";

const MELEE_RANGE = 301.0;

/**
 * Encodes the live Turu `CombatObservation` into the exact 26-dimensional normalized
 * observation vector expected by the neural network (`sim/bludrut_env.py`).
 */
export function buildObservationVector26(obs: CombatObservation): number[] {
    const oppInCell = obs.opponents.find(
        (p) => p.cell.toLowerCase() === obs.currentCell.toLowerCase() && p.alive,
    );
    const opp =
        oppInCell ??
        obs.opponents.find((p) => p.alive) ??
        obs.opponents[0];

    const sameCell = opp && opp.cell.toLowerCase() === obs.currentCell.toLowerCase() && opp.alive ? 1.0 : 0.0;
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

    const oppScreenIdx = opp
        ? Math.max(0, BLUDRUT_16_SCREENS.findIndex((s) => s.cell.toLowerCase() === opp.cell.toLowerCase()))
        : 7; // default mid (7)

    const anyOpp = obs.opponents.find(
        (p) =>
            p.alive &&
            p.cell.toLowerCase() !== "enter0" &&
            p.cell.toLowerCase() !== "enter1" &&
            p.cell.toLowerCase() !== "wait",
    );
    const targetNavCell = anyOpp ? anyOpp.cell : "Crosslower";
    const nextDoor = !sameCell ? findNextDoorTowardCell(obs.currentCell, targetNavCell) : null;

    let targetRelX = 0.0;
    let targetRelY = 0.0;
    if (sameCell && opp) {
        targetRelX = (opp.x - obs.selfX) / 960.0;
        targetRelY = (opp.y - obs.selfY) / 550.0;
    } else if (nextDoor) {
        const aim = getDoorAimCoordinates(nextDoor, obs.selfX, obs.selfY);
        targetRelX = (aim.x - obs.selfX) / 960.0;
        targetRelY = (aim.y - obs.selfY) / 550.0;
    }

    const safeNpc = findSafeManaTapNpc(obs);

    const vec = [
        obs.selfHpRatio,                                     // 0: self HP ratio
        obs.selfMpRatio,                                     // 1: self MP ratio
        obs.selfSpRatio,                                     // 2: self Stamina (intSP) ratio
        obs.canSprint ? 1.0 : 0.0,                           // 3: can sprint (SP >= 50)
        obs.selfInCombat ? 1.0 : 0.0,                        // 4: self in combat
        obs.selfInCombat ? 0.0 : 0.8,                        // 5: OOC timer progress
        screenIdx / 15.0,                                    // 6: normalized screen index
        Math.min(1.0, Math.max(0.0, obs.selfX / 960.0)),     // 7: self X
        Math.min(1.0, Math.max(0.0, obs.selfY / 550.0)),     // 8: self Y
        obs.selfAlive ? 1.0 : 0.0,                           // 9: self alive
        obs.skillsReady[0] ? 1.0 : 0.0,                      // 10: skill 1 ready
        obs.skillsReady[1] ? 1.0 : 0.0,                      // 11: skill 2 ready
        obs.skillsReady[2] ? 1.0 : 0.0,                      // 12: skill 3 ready
        obs.skillsReady[3] ? 1.0 : 0.0,                      // 13: skill 4 ready
        oppScreenIdx / 15.0,                                 // 14: opp screen index
        Math.min(1.0, Math.max(-1.0, targetRelX)),           // 15: primary target relative X (opp in room, or exit door when roaming)
        Math.min(1.0, Math.max(-1.0, targetRelY)),           // 16: primary target relative Y (opp in room, or exit door when roaming)
        0.0,                                                 // 17: mid timer progress ratio
        obs.selfAuras.length > 0 ? 1.0 : 0.0,                // 18: self has buff
        obs.selfStunned ? 1.0 : 0.0,                         // 19: self stunned
        opp && opp.alive ? 1.0 : 0.0,                        // 20: opp alive
        sameCell,                                            // 21: opp in same cell
        Math.min(1.0, dist / 1000.0),                        // 22: distance to opp
        inMelee,                                             // 23: in melee range
        opp ? opp.hp / Math.max(1, opp.maxHp) : 0.0,         // 24: opp HP ratio
        safeNpc ? 1.0 : 0.0,                                 // 25: safe NPC available
    ];

    return vec;
}

function denseTanh(x: readonly number[], w: readonly (readonly number[])[], b: readonly number[]): number[] {
    const out = new Array<number>(w.length);
    for (let i = 0; i < w.length; i++) {
        const row = w[i]!;
        let sum = b[i]!;
        for (let j = 0; j < row.length; j++) {
            sum += row[j]! * x[j]!;
        }
        out[i] = Math.tanh(sum);
    }
    return out;
}

function denseLinear(x: readonly number[], w: readonly (readonly number[])[], b: readonly number[]): number[] {
    const out = new Array<number>(w.length);
    for (let i = 0; i < w.length; i++) {
        const row = w[i]!;
        let sum = b[i]!;
        for (let j = 0; j < row.length; j++) {
            sum += row[j]! * x[j]!;
        }
        out[i] = sum;
    }
    return out;
}

function sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
}

/**
 * Runs forward inference through the Compound Atomic Policy MLP.
 * Simultaneously predicts:
 * 1. Atomic Combat Action (`USE_SKILL_1..4`, `AUTO_ATTACK`, `NONE`, `MANA_TAP_NPC`)
 * 2. Atomic Movement Decision (`STAND`, `WALK`, `SPRINT` to `targetX, targetY`)
 */
export function inferBattleWarriorDecision(
    obs: CombatObservation,
    policy: CompoundAtomicPolicySpec = COMPOUND_ATOMIC_POLICY_WEIGHTS,
): CombatDecision {
    const oppInCell = obs.opponents.find(
        (p) => p.cell.toLowerCase() === obs.currentCell.toLowerCase() && p.alive,
    );
    const dist = oppInCell ? Math.hypot(obs.selfX - oppInCell.x, obs.selfY - oppInCell.y) : 9999.0;
    const inMelee = oppInCell ? dist <= MELEE_RANGE : false;
    const safeNpc = findSafeManaTapNpc(obs);

    // Shared Neural Backbone forward pass
    const x = buildObservationVector26(obs);
    const h1 = denseTanh(x, policy.w1, policy.b1);
    const h2 = denseTanh(h1, policy.w2, policy.b2);

    // Head 1: Combat Action logits
    const rawCombatLogits = denseLinear(h2, policy.wCombat, policy.bCombat);
    const combatLogits = [...rawCombatLogits];

    // Mask physically impossible actions and rule violations only.
    // Pure learned policy: NEVER hardcode tactical combat behaviors or arbitrary logit boosts!
    if (!inMelee) {
        combatLogits[2] = -1e9; // S1 physical melee range limit
        combatLogits[3] = -1e9; // S2 physical melee range limit
        combatLogits[5] = -1e9; // S4 physical melee range limit
    }
    if (!obs.skillsReady[0]) combatLogits[2] = -1e9; // S1 on cooldown
    if (!obs.skillsReady[1]) combatLogits[3] = -1e9; // S2 on cooldown
    if (!obs.skillsReady[2]) combatLogits[4] = -1e9; // S3 on cooldown
    if (!obs.skillsReady[3]) combatLogits[5] = -1e9; // S4 on cooldown
    if (!safeNpc) combatLogits[6] = -1e9; // 1v1 RuleShield: cannot attack protected NPC if no safe NPC exists

    let bestCombatIdx = 0;
    let bestCombatScore = -Infinity;
    for (let i = 0; i < combatLogits.length; i++) {
        if (combatLogits[i]! > bestCombatScore) {
            bestCombatScore = combatLogits[i]!;
            bestCombatIdx = i;
        }
    }

    const combatActionMap: AtomicCombatAction[] = [
        "NONE",
        "AUTO_ATTACK",
        "USE_SKILL_1",
        "USE_SKILL_2",
        "USE_SKILL_3",
        "USE_SKILL_4",
    ];
    const chosenCombatAction: AtomicCombatAction = combatActionMap[bestCombatIdx] ?? "AUTO_ATTACK";

    // Head 2: Movement Type logits
    const rawMoveLogits = denseLinear(h2, policy.wMoveType, policy.bMoveType);
    const moveLogits = [...rawMoveLogits];

    if (!obs.canSprint) moveLogits[2] = -1e9; // Cannot sprint if SP < 50
    if (obs.selfStunned) {
        moveLogits[1] = -1e9; // Cannot walk while stunned
        moveLogits[2] = -1e9; // Cannot sprint while stunned
    }

    let bestMoveIdx = 0;
    let bestMoveScore = -Infinity;
    for (let i = 0; i < moveLogits.length; i++) {
        if (moveLogits[i]! > bestMoveScore) {
            bestMoveScore = moveLogits[i]!;
            bestMoveIdx = i;
        }
    }

    const moveTypeMap: AtomicMovementType[] = ["STAND", "WALK", "SPRINT"];
    const chosenMoveType: AtomicMovementType = moveTypeMap[bestMoveIdx] ?? "STAND";

    // Head 3: 100% Unified Atomic Target Coordinates regression (Sigmoid in [0, 1])
    const coordRaw = denseLinear(h2, policy.wCoord, policy.bCoord);
    const normX = sigmoid(coordRaw[0] ?? 0.0);
    const normY = sigmoid(coordRaw[1] ?? 0.0);

    let targetX = Math.round(40.0 + normX * 880.0);
    let targetY = Math.round(240.0 + normY * 260.0);

    // Movement & Roaming Navigation Resolution
    let targetCell: string | undefined = undefined;
    let targetPad: string | undefined = undefined;
    let finalMoveType: AtomicMovementType = chosenMoveType;

    if (!oppInCell) {
        // Roaming across screens: Navigate decisively toward opponent / Mid / Safezone
        const anyOpp = obs.opponents.find(
            (p) =>
                p.alive &&
                p.cell.toLowerCase() !== "enter0" &&
                p.cell.toLowerCase() !== "enter1" &&
                p.cell.toLowerCase() !== "wait",
        );
        const goalCell = obs.wave.canReturnToSafezone
            ? (obs.inSafezone ? obs.currentCell : (obs.currentCell.includes("0") ? "Enter0" : "Enter1"))
            : (anyOpp ? anyOpp.cell : "Crosslower");

        const nextDoor = findNextDoorTowardCell(obs.currentCell, goalCell);
        if (nextDoor) {
            targetX = nextDoor.aimX;
            targetY = nextDoor.aimY;
            targetCell = nextDoor.targetCell;
            targetPad = nextDoor.targetPad;
            finalMoveType = obs.canSprint ? "SPRINT" : "WALK";
        }
    } else {
        // In-room combat: Neural network controls 100% continuous spatial target (targetX, targetY)
        const hit = checkTrajectoryDoorCollision(obs.currentCell, obs.selfX, obs.selfY, targetX, targetY);
        if (hit && obs.wave.canReturnToSafezone) {
            targetCell = hit.door.targetCell;
            targetPad = hit.door.targetPad;
        }
    }

    return {
        combatAction: chosenCombatAction,
        movement: {
            type: finalMoveType,
            targetX,
            targetY,
            targetCell,
            targetPad,
        },
        targetName: oppInCell ? oppInCell.username : undefined,
        npcMapId: undefined,
        reason: oppInCell
            ? `[AtomicPolicy] Action: ${chosenCombatAction} | Move: ${finalMoveType} -> (${targetX}, ${targetY})`
            : `[AtomicPolicy] Roaming ${obs.currentCell} -> ${targetCell ?? "Crosslower"} (Move: ${finalMoveType})`,
    };
}
