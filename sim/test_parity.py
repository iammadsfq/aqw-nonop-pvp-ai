// src/ai/test_parity.ts
import { buildEntityObservation, TOTAL_OBS_DIM } from "./build_entity_obs";
import { TrackedEntity, CanonicalStats } from "../state/entity_tracker";
import * as fs from "fs";

// Mock tracker sederhana untuk unit testing tanpa perlu koneksi client game
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

function assertVectorClose(actual: Float32Array, expected: number[], tol: number = 1e-3, label: string = "") {
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
    const rawRef = fs.readFileSync("sim/parity_reference.json", "utf-8");
    const refData = JSON.parse(rawRef);

    // Test 1: Skenario Duel Satu Layar
    tracker.selfEntity.cell = "Crosslower";
    tracker.selfEntity.x = 450.0;
    tracker.selfEntity.y = 350.0;
    tracker.selfEntity.stats.inDmgMult = 0.50; // On Guard

    tracker.opponentEntity.cell = "Crosslower";
    tracker.opponentEntity.x = 500.0;
    tracker.opponentEntity.y = 350.0;
    tracker.opponentEntity.hp_norm = 6000.0 / 8455.0;
    tracker.opponentEntity.mp_norm = 0.50;
    tracker.opponentEntity.stats.outDmgMult = 1.15; // Prepared Strike

    const obsDuel = buildEntityObservation(tracker as any, 0.0, 0.0);
    assertVectorClose(obsDuel, refData.duel_same_cell.obs0, 1e-3, "Duel Same Cell");

    // Test 2: Skenario Beda Ruangan (Pure RL)
    tracker.selfEntity.cell = "Enter0";
    tracker.selfEntity.x = 322.5;
    tracker.selfEntity.y = 353.0;
    tracker.selfEntity.stats.inDmgMult = 1.0;

    tracker.opponentEntity.cell = "Morale0A";
    tracker.opponentEntity.x = 787.5;
    tracker.opponentEntity.y = 351.0;
    tracker.opponentEntity.stats.outDmgMult = 1.0;

    const obsNav = buildEntityObservation(tracker as any, 6.0, 0.0);
    assertVectorClose(obsNav, refData.different_cells.obs0, 1e-3, "Different Cells Navigation");
}

if (require.main === module) {
    runTypeScriptParityTest();
}