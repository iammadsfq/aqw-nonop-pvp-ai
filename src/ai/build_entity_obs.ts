// src/ai/build_entity_obs.ts
import { EntityTracker, TrackedEntity } from "../state/entity_tracker";
import { SCREEN_INDICES, CELL_GRAPH_DIST } from "../config/constants";

export const SELF_OBS_DIM = 17;
export const ENTITY_SLOT_DIM = 13;
export const TOTAL_OBS_DIM = SELF_OBS_DIM + ENTITY_SLOT_DIM; // 30 float

function isMeleeRange(x1: number, y1: number, x2: number, y2: number): boolean {
  // Rectangle combat range AQW: World.as abs(dx) <= 301*1.15 (~340px), abs(dy) <= 30*1.15 (~34px)
  const dx = Math.abs(x1 - x2);
  const dy = Math.abs(y1 - y2);
  return dx <= 340.0 && dy <= 34.0;
}

/**
 * Membangun tensor observasi Float32Array 30-dimensi canonical (Pure Function, Zero Bot SDK Dependency).
 *
 * @param tracker Instance EntityTracker
 * @param skillsReady Status kesiapan skill [S1, S2, S3, S4] (1.0 = ready, 0.0 = cd/gcd)
 * @param inCombat Status in-combat karakter
 * @param outOfCombatTimer Timer durasi keluar combat dalam detik (0.0 s/d 10.0)
 */
export function buildEntityObservation(
  tracker: EntityTracker,
  skillsReady: number[] = [1, 1, 1, 1],
  inCombat: boolean = false,
  outOfCombatTimer: number = 0.0
): Float32Array {
  const obs = new Float32Array(TOTAL_OBS_DIM);

  // ==========================================
  // 1. Fitur Diri Sendiri (Self: 0..16)
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
  obs[5] = inCombat ? 0.0 : Math.min(1.0, outOfCombatTimer / 10.0);
  obs[6] = cellIdx / 15.0;
  obs[7] = Math.min(1.0, Math.max(0.0, (myX - 35.0) / 895.0));
  obs[8] = Math.min(1.0, Math.max(0.0, (myY - 255.0) / 245.0));
  obs[9] = self.alive ? 1.0 : 0.0;
  obs[10] = skillsReady[0] ?? 1.0;
  obs[11] = skillsReady[1] ?? 1.0;
  obs[12] = skillsReady[2] ?? 1.0;
  obs[13] = skillsReady[3] ?? 1.0;
  obs[14] = Math.min(1.0, Math.max(0.0, (self.stats.inDmgMult ?? 1.0) / 2.0));
  obs[15] = Math.min(1.0, Math.max(0.0, (self.stats.outDmgMult ?? 1.0) / 2.0));
  obs[16] = self.isStunned ? 1.0 : 0.0;

  // ==========================================
  // 2. Fitur Slot Lawan (Opponent: 17..29)
  // ==========================================
  const opp: TrackedEntity | null = tracker.getPrimaryOpponent();

  // Jika lawan belum ada di memori tracker
  if (!opp) {
    obs[17] = 0.0;        // is_valid = 0
    obs[18] = 0.0;        // alive = 0
    obs[19] = 0.0;        // is_ally = 0
    obs[20] = 0.0;        // same_cell = 0
    obs[21] = 0.0;        // opp_cell_idx = 0
    obs[22] = 0.0;        // rel_x
    obs[23] = 0.0;        // rel_y
    obs[24] = 0.0;        // in_melee = 0
    obs[25] = 0.0;        // hp_norm
    obs[26] = 0.0;        // mp_norm
    obs[27] = 0.5;        // in_dmg_mult = 1.0 / 2.0 (neutral)
    obs[28] = 0.0;        // is_stunned
    obs[29] = 0.5;        // out_dmg_mult = 1.0 / 2.0 (neutral)
    return obs;
  }

  const sameCell: boolean = myCell === opp.cell && opp.alive;
  const oppCellIdx: number = SCREEN_INDICES[opp.cell] ?? 0;
  const oppCellIdxNorm: number = oppCellIdx / 15.0;

  let relX = 0.0;
  let relY = 0.0;
  let inMelee = 0.0;

  if (sameCell) {
    relX = Math.min(1.0, Math.max(-1.0, (opp.x - myX) / 895.0));
    relY = Math.min(1.0, Math.max(-1.0, (opp.y - myY) / 245.0));
    inMelee = isMeleeRange(myX, myY, opp.x, opp.y) ? 1.0 : 0.0;
  }

  obs[17] = 1.0;                                                        // is_valid
  obs[18] = opp.alive ? 1.0 : 0.0;
  obs[19] = 0.0;                                                        // is_ally
  obs[20] = sameCell ? 1.0 : 0.0;
  obs[21] = oppCellIdxNorm;
  obs[22] = relX;
  obs[23] = relY;
  obs[24] = inMelee;
  obs[25] = Math.min(1.0, Math.max(0.0, opp.hp_norm));
  obs[26] = Math.min(1.0, Math.max(0.0, opp.mp_norm));
  obs[27] = Math.min(1.0, Math.max(0.0, (opp.stats.inDmgMult ?? 1.0) / 2.0));
  obs[28] = opp.isStunned ? 1.0 : 0.0;
  obs[29] = Math.min(1.0, Math.max(0.0, (opp.stats.outDmgMult ?? 1.0) / 2.0));

  return obs;
}
