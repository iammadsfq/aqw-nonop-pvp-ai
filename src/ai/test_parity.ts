// src/ai/test_parity.ts
import { buildEntityObservation } from "./build_entity_obs";
import { TrackedEntity, CanonicalStats } from "../state/entity_tracker";

const PARITY_REFERENCE = {
    duel_same_cell: [
        1.0, 0.8, 0.9, 1.0, 1.0, 0.0, 0.4667, 0.4637, 0.3878, 1.0,
        1.0, 1.0, 1.0, 1.0, 0.25, 0.5, 0.0,
        1.0, 1.0, 0.0, 1.0, 0.4667, 0.0559, 0.0, 1.0, 0.7096, 0.5, 0.5, 0.0, 0.575
    ],
    different_cells: [
        1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 0.0, 0.3212, 0.4, 1.0,
        1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.0,
        1.0, 1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.5
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
            critRate: 0.2285,
            dodgeRate: 0.2029,
            haste: 0.1054,
        };

        this.selfEntity = {
            username: "uma",
            entId: 187461,
            team: "blue",
            pvpTeam: 0,
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
            activeAuras: new Map(),
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
            pvpTeam: 1,
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
            activeAuras: new Map(),
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

    // 1. Skenario Duel Satu Layar (Crosslower)
    tracker.selfEntity.cell = "Crosslower";
    tracker.selfEntity.x = 450.0;
    tracker.selfEntity.y = 350.0;
    tracker.selfEntity.mp_norm = 0.80;
    tracker.selfEntity.sp_norm = 0.90;
    tracker.selfEntity.stats.inDmgMult = 0.50; // Active shield On Guard (1.0 * 0.50)
    tracker.selfEntity.stats.outDmgMult = 1.0;

    tracker.opponentEntity.cell = "Crosslower";
    tracker.opponentEntity.x = 500.0;
    tracker.opponentEntity.y = 350.0;
    tracker.opponentEntity.hp_norm = 6000.0 / 8455.0;
    tracker.opponentEntity.mp_norm = 0.50;
    tracker.opponentEntity.stats.inDmgMult = 1.0;
    tracker.opponentEntity.stats.outDmgMult = 1.15; // Prepared Strike buff (1.0 * 1.15)

    const obsDuel = buildEntityObservation(tracker as any, [1, 1, 1, 1], true, 0.0);
    assertVectorClose(obsDuel, PARITY_REFERENCE.duel_same_cell, 2e-3, "Duel Same Cell");

    // 2. Skenario Beda Ruangan (Enter0 vs Morale0A)
    tracker.selfEntity.cell = "Enter0";
    tracker.selfEntity.x = 322.5;
    tracker.selfEntity.y = 353.0;
    tracker.selfEntity.hp_norm = 1.0;
    tracker.selfEntity.mp_norm = 1.0;
    tracker.selfEntity.sp_norm = 1.0;
    tracker.selfEntity.stats.inDmgMult = 1.0;
    tracker.selfEntity.stats.outDmgMult = 1.0;

    tracker.opponentEntity.cell = "Morale0A";
    tracker.opponentEntity.x = 787.5;
    tracker.opponentEntity.y = 351.0;
    tracker.opponentEntity.hp_norm = 1.0;
    tracker.opponentEntity.mp_norm = 1.0;
    tracker.opponentEntity.stats.inDmgMult = 1.0;
    tracker.opponentEntity.stats.outDmgMult = 1.0;

    const obsNav = buildEntityObservation(tracker as any, [1, 1, 1, 1], false, 5.0);
    assertVectorClose(obsNav, PARITY_REFERENCE.different_cells, 2e-3, "Different Cells Navigation");
}

runTypeScriptParityTest();
