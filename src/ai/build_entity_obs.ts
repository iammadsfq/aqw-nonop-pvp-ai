// src/ai/build_entity_obs.ts
import { EntityTracker, TrackedEntity } from "../state/entity_tracker.js";
import { SCREEN_INDICES, CELL_GRAPH_DIST } from "../config/constants.js";

export const SELF_OBS_DIM = 16;
export const ENTITY_SLOT_DIM = 14;
export const TOTAL_OBS_DIM = SELF_OBS_DIM + ENTITY_SLOT_DIM; // 30 float

function isMeleeRange(x1: number, y1: number, x2: number, y2: number): boolean {
    const dx = Math.abs(x1 - x2);
    const dy = Math.abs(y1 - y2);
    return dx <= 160 && dy <= 38;
}

/**
 * Membangun tensor observasi Float32Array 30-dimensi canonical (Pure Function, Zero Bot SDK Dependency).
 *
 * @param tracker Instance EntityTracker
 * @param skillsReady Status kesiapan skill [S1, S2, S3, S4] (1.0 = ready, 0.0 = cd/gcd)
 * @param inCombat Status in-combat karakter
 * @param outOfCombatTimer Timer durasi keluar combat dalam detik (0.0 s/d 12.0)
 */
export function buildEntityObservation(
    tracker: EntityTracker,
    skillsReady: number[] = [1, 1, 1, 1],
    inCombat: boolean = false,
    outOfCombatTimer: number = 0.0
): Float32Array {
    const obs = new Float32Array(TOTAL_OBS_DIM);

    // ==========================================
    // 1. Fitur Diri Sendiri (Self: 0..15)
    // ==========================================
    const self: TrackedEntity = tracker.getSelf();
    const myCell: string = self.cell || "Enter0";
    const myX: number = self.x;
    const myY: number = self.y;
    const cellIdx: number = SCREEN_INDICES[myCell] ?? 0;

    obs[0] = Math.min(1.0, Math.max(0.0, self.hp_norm));
    obs[1] = Math.min(1.0, Math.max(0.0, self.mp_norm));
    obs[2] = Math.min(1.0, Math.max(0.0, self.sp_norm));
    obs[3] = self.sp_norm >= 0.50 ? 1.0 : 0.0;
    obs[4] = inCombat ? 1.0 : 0.0;
    obs[5] = inCombat ? 0.0 : Math.min(1.0, outOfCombatTimer / 12.0);
    obs[6] = cellIdx / 15.0;
    obs[7] = Math.min(1.0, Math.max(0.0, myX / 960.0));
    obs[8] = Math.min(1.0, Math.max(0.0, myY / 550.0));
    obs[9] = self.alive ? 1.0 : 0.0;
    obs[10] = skillsReady[0] ?? 1.0;
    obs[11] = skillsReady[1] ?? 1.0;
    obs[12] = skillsReady[2] ?? 1.0;
    obs[13] = skillsReady[3] ?? 1.0;
    obs[14] = self.stats.inDmgMult;
    obs[15] = self.isStunned ? 1.0 : 0.0;

    // ==========================================
    // 2. Fitur Slot Lawan (Opponent: 16..29)
    // ==========================================
    const opp: TrackedEntity | null = tracker.getPrimaryOpponent();
    if (!opp) {
        return obs;
    }

    const sameCell: boolean = myCell === opp.cell;
    const graphKey = `${myCell}->${opp.cell}`;
    const hops: number = CELL_GRAPH_DIST[graphKey] ?? (sameCell ? 0 : 4);
    const cellGraphDistNorm: number = Math.min(1.0, hops / 6.0);

    let relX = 0.0;
    let relY = 0.0;
    let distNorm = 1.0;
    let inMelee = 0.0;

    if (sameCell) {
        relX = Math.min(1.0, Math.max(-1.0, (opp.x - myX) / 960.0));
        relY = Math.min(1.0, Math.max(-1.0, (opp.y - myY) / 550.0));
        distNorm = Math.min(1.0, Math.hypot(opp.x - myX, opp.y - myY) / 1000.0);
        inMelee = isMeleeRange(myX, myY, opp.x, opp.y) ? 1.0 : 0.0;
    } else {
        relX = 0.0;
        relY = 0.0;
        distNorm = 1.0;
        inMelee = 0.0;
    }

    obs[16] = 1.0;
    obs[17] = opp.alive ? 1.0 : 0.0;
    obs[18] = 0.0;
    obs[19] = sameCell ? 1.0 : 0.0;
    obs[20] = cellGraphDistNorm;
    obs[21] = relX;
    obs[22] = relY;
    obs[23] = distNorm;
    obs[24] = inMelee;
    obs[25] = Math.min(1.0, Math.max(0.0, opp.hp_norm));
    obs[26] = Math.min(1.0, Math.max(0.0, opp.mp_norm));
    obs[27] = opp.stats.inDmgMult;
    obs[28] = opp.isStunned ? 1.0 : 0.0;
    obs[29] = opp.stats.outDmgMult;

    return obs;
}