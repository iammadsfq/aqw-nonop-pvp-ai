// src/ai/test_parity.ts
import { buildEntityObservation } from "./build_entity_obs.js";
import { TrackedEntity, CanonicalStats } from "../state/entity_tracker.js";

const PARITY_REFERENCE = {
    duel_same_cell: [
        1.0, 0.8, 0.9, 1.0, 1.0, 0.0, 0.3333, 0.4688, 0.6364, 1.0,
        1.0, 1.0, 1.0, 1.0, 0.5, 0.0,
        1.0, 1.0, 0.0, 1.0, 0.0, 0.0521, 0.0, 0.05, 1.0, 0.7096, 0.5, 1.0, 0.0, 1.15
    ],
    different_cells: [
        1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.3359, 0.6418, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0
    ]
};

class MockTracker {
    public selfEntity: TrackedEntity;
    public opponentEntity: TrackedEntity;

    constructor() {
        const stats: CanonicalStats = {
            inDmgMult: 1.0,
            outDmgMult: 1.0,
            manaCostMult: 1.0,
            critRate: 0.25,
            dodgeRate: 0.20,
            haste: 0.10,
        };

        this.selfEntity = {
            username: "uma",
            entId: 187461,
            team: "blue",
            isAlly: true,
            cell: "Enter0",
            x: 322.5,
            y: 353.0,
            currentHp: 8455,
            maxHp: 8455,
            hp_norm: 1.0,
            mp_norm: 1.0,
            sp_norm: 1.0,
            alive: true,
            stats: { ...stats },
            hardCcRem: 0.0,
            isStunned: false,
            roleCategory: 0,
            timeSinceLastAction: 0.0,
            lastSeenTs: Date.now(),
        };

        this.opponentEntity = {
            username: "master_ninja",
            entId: 188276,
            team: "red",
            isAlly: false,
            cell: "Enter1",
            x: 595.0,
            y: 353.0,
            currentHp: 8455,
            maxHp: 8455,
            hp_norm: 1.0,
            mp_norm: 1.0,
            sp_norm: 1.0,
            alive: true,
            stats: { ...stats },
            hardCcRem: 0.0,
            isStunned: false,
            roleCategory: 0,
            timeSinceLastAction: 0.0,
            lastSeenTs: Date.now(),
        };
    }

    public getSelf(): TrackedEntity { return this.selfEntity; }
    public getPrimaryOpponent(): TrackedEntity | null { return this.opponentEntity; }
}

function assertVectorClose(actual: Float32Array, expected: number[], tol: number = 2e-3, label: string = "") {
    if (actual.length !== expected.length) {
        throw new Error(`[${label}] Dimension mismatch: actual ${actual.length} vs expected ${expected.length}`);
    }
    for (let i = 0; i < actual.length; i++) {
        const diff = Math.abs(actual[i] - expected[i]);
        if (diff > tol) {
            throw new Error(`[${label}] Mismatch at index ${i}: actual=${actual[i]} vs expected=${expected[i]} (diff=${diff})`);
        }
    }
    console.log(`[PASS] ${label} - 30 dimensions match within tolerance ${tol}`);
}

export function runTypeScriptParityTest() {
    const tracker = new MockTracker();

    // 1. Skenario Duel Satu Layar
    tracker.selfEntity.cell = "Crosslower";
    tracker.selfEntity.x = 450.0;
    tracker.selfEntity.y = 350.0;
    tracker.selfEntity.mp_norm = 0.80;
    tracker.selfEntity.sp_norm = 0.90;
    tracker.selfEntity.stats.inDmgMult = 0.50;

    tracker.opponentEntity.cell = "Crosslower";
    tracker.opponentEntity.x = 500.0;
    tracker.opponentEntity.y = 350.0;
    tracker.opponentEntity.hp_norm = 6000.0 / 8455.0;
    tracker.opponentEntity.mp_norm = 0.50;
    tracker.opponentEntity.stats.outDmgMult = 1.15;

    const obsDuel = buildEntityObservation(tracker as any, [1, 1, 1, 1], true, 0.0);
    assertVectorClose(obsDuel, PARITY_REFERENCE.duel_same_cell, 2e-3, "Duel Same Cell");

    // 2. Skenario Beda Ruangan
    tracker.selfEntity.cell = "Enter0";
    tracker.selfEntity.x = 322.5;
    tracker.selfEntity.y = 353.0;
    tracker.selfEntity.mp_norm = 1.0;
    tracker.selfEntity.sp_norm = 1.0;
    tracker.selfEntity.stats.inDmgMult = 1.0;

    tracker.opponentEntity.cell = "Morale0A";
    tracker.opponentEntity.x = 787.5;
    tracker.opponentEntity.y = 351.0;
    tracker.opponentEntity.hp_norm = 1.0;
    tracker.opponentEntity.mp_norm = 1.0;
    tracker.opponentEntity.stats.outDmgMult = 1.0;

    const obsNav = buildEntityObservation(tracker as any, [1, 1, 1, 1], false, 6.0);
    assertVectorClose(obsNav, PARITY_REFERENCE.different_cells, 2e-3, "Different Cells Navigation");
}

runTypeScriptParityTest();