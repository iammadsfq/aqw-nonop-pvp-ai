// src/ai/test_parity.ts
import { buildEntityObservation } from "./build_entity_obs.js";
import { TrackedEntity, CanonicalStats } from "../state/entity_tracker.js";

// Vektor referensi 30 dimensi yang dihasilkan secara deterministik dari sim/test_parity.py
const PARITY_REFERENCE = {
    // Skenario 1: Duel Satu Layar di Crosslower (In-Melee Range, Buff Aktif)
    duel_same_cell: [
        1.0,     // [0]  hp_norm (8455/8455)
        0.8,     // [1]  mp_norm (80/100)
        0.9,     // [2]  sp_norm (90/100)
        1.0,     // [3]  can_sprint
        1.0,     // [4]  in_combat
        0.0,     // [5]  ooc_progress (0.0 karena in_combat)
        0.3333,  // [6]  screen_idx (Crosslower = index 5 / 15)
        0.4688,  // [7]  norm_x (450 / 960)
        0.6364,  // [8]  norm_y (350 / 550)
        1.0,     // [9]  alive
        1.0,     // [10] s1_ready (cooldown 0)
        1.0,     // [11] s2_ready
        1.0,     // [12] s3_ready
        1.0,     // [13] s4_ready
        0.5,     // [14] in_dmg_mult (On Guard aktif)
        0.0,     // [15] is_hard_cc
        1.0,     // [16] is_valid
        1.0,     // [17] is_alive
        0.0,     // [18] is_ally (0 = lawan)
        1.0,     // [19] is_same_cell
        0.0,     // [20] cell_graph_dist (same cell = 0 hops)
        0.0521,  // [21] rel_x ((500 - 450) / 960 = 50 / 960)
        0.0,     // [22] rel_y ((350 - 350) / 550 = 0)
        0.05,    // [23] distance_norm (50 / 1000)
        1.0,     // [24] in_melee_range (dx=50 <= 160, dy=0 <= 38)
        0.7096,  // [25] hp_norm (6000 / 8455)
        0.5,     // [26] mp_norm (50 / 100)
        1.0,     // [27] in_dmg_mult musuh
        0.0,     // [28] is_hard_cc musuh
        1.15     // [29] out_dmg_mult musuh (Prepared Strike aktif)
    ],

    // Skenario 2: Beda Ruangan (Pure RL: rel_x & rel_y = 0.0, dist_norm = 1.0)
    different_cells: [
        1.0,     // [0]  hp_norm
        1.0,     // [1]  mp_norm
        1.0,     // [2]  sp_norm
        1.0,     // [3]  can_sprint
        0.0,     // [4]  in_combat
        0.5,     // [5]  ooc_progress (6.0s / 12.0s)
        0.0,     // [6]  screen_idx (Enter0 = index 0 / 15)
        0.3359,  // [7]  norm_x (322.5 / 960)
        0.6418,  // [8]  norm_y (353.0 / 550)
        1.0,     // [9]  alive
        1.0,     // [10] s1_ready
        1.0,     // [11] s2_ready
        1.0,     // [12] s3_ready
        1.0,     // [13] s4_ready
        1.0,     // [14] in_dmg_mult
        0.0,     // [15] is_hard_cc
        1.0,     // [16] is_valid
        1.0,     // [17] is_alive
        0.0,     // [18] is_ally
        0.0,     // [19] is_same_cell
        0.5,     // [20] cell_graph_dist (Enter0 -> Morale0A = 3 hops / 6.0)
        0.0,     // [21] rel_x (Murni RL: Beda room -> 0.0)
        0.0,     // [22] rel_y (Murni RL: Beda room -> 0.0)
        1.0,     // [23] distance_norm (Beda room -> 1.0)
        0.0,     // [24] in_melee_range
        1.0,     // [25] hp_norm
        1.0,     // [26] mp_norm
        1.0,     // [27] in_dmg_mult
        0.0,     // [28] is_hard_cc
        1.0      // [29] out_dmg_mult
    ]
};

class MockTracker {
    public selfEntity: TrackedEntity;
    public opponentEntity: TrackedEntity;

    constructor() {
        const defaultStats: CanonicalStats = {
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
            stats: { ...defaultStats },
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
            stats: { ...defaultStats },
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
    tracker.selfEntity.stats.inDmgMult = 0.50; // On Guard aktif

    tracker.opponentEntity.cell = "Crosslower";
    tracker.opponentEntity.x = 500.0;
    tracker.opponentEntity.y = 350.0;
    tracker.opponentEntity.hp_norm = 6000.0 / 8455.0;
    tracker.opponentEntity.mp_norm = 0.50;
    tracker.opponentEntity.stats.outDmgMult = 1.15; // Prepared Strike aktif

    const obsDuel = buildEntityObservation(tracker as any, 0.0, 0.0);
    assertVectorClose(obsDuel, PARITY_REFERENCE.duel_same_cell, 2e-3, "Duel Same Cell");

    // 2. Skenario Beda Ruangan (Pure RL)
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

    const obsNav = buildEntityObservation(tracker as any, 6.0, 0.0);
    assertVectorClose(obsNav, PARITY_REFERENCE.different_cells, 2e-3, "Different Cells Navigation");
}

runTypeScriptParityTest();