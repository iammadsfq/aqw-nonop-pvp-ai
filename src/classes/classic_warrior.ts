/**
 * @file classic_warrior.ts
 * @description Baseline Greedy Sparring Bot for Classic/Battle Warrior under Non-OP rules.
 * Digunakan sebagai lawan evaluasi (benchmark) bagi agent Pure RL.
 */

import { ROOM_DOORS, CELL_GRAPH_DIST, WALK_BOUNDS } from "../config/constants";
import { EntityTracker, TrackedEntity } from "../state/entity_tracker";
import type {
  CombatDecision,
  AtomicCombatAction,
  AtomicMovementType,
  AtomicMovementDecision,
} from "../state/types";

function isMeleeRange(x1: number, y1: number, x2: number, y2: number): boolean {
  // Rectangle combat range AQW: World.as abs(dx) <= 301*1.15 (~340px), abs(dy) <= 30*1.15 (~34px)
  return Math.abs(x1 - x2) <= 340.0 && Math.abs(y1 - y2) <= 34.0;
}

export function decideClassicWarriorGreedy(
  tracker: EntityTracker,
  skillsReady: number[] = [1, 1, 1, 1]
): CombatDecision {
  const self: TrackedEntity = tracker.getSelf();
  const opp: TrackedEntity | null = tracker.getPrimaryOpponent();

  let chosenCombat: AtomicCombatAction = "NONE";
  let moveType: AtomicMovementType = "STAND";
  let targetX = self.x;
  let targetY = self.y;
  let targetCell: string | undefined = undefined;
  let targetPad: string | undefined = undefined;
  let reason = "Idle";

  // 1. Jika ter-stun atau mati: Diam total
  if (self.isStunned || !self.alive) {
    return {
      combatAction: "NONE",
      movement: { type: "STAND", targetX: self.x, targetY: self.y },
      reason: "Stunned or dead",
    };
  }

  // 2. Evaluasi Navigasi Spasial (Pengejaran / Kiting)
  if (!opp || !opp.alive) {
    // Lawan belum ada / mati: Berdiam di tengah screen
    targetX = 480;
    targetY = 360;
    moveType = "WALK";
    reason = "No active opponent; holding room center";
  } else if (self.cell !== opp.cell) {
    // Lawan di cell lain: Cari pintu yang memendekkan jarak hop BFS menuju goalCell
    const goalCell =
      opp.cell === "Enter0"
        ? "Morale0C"
        : opp.cell === "Enter1"
        ? "Morale1C"
        : opp.cell;

    if (self.cell === goalCell) {
      targetX = 488;
      targetY = 351;
      moveType = "WALK";
      reason = "Waiting outside opponent safezone";
    } else {
      const currentDoors = ROOM_DOORS[self.cell] || [];
      let bestDoor = currentDoors[0];
      let minHops = 999;

      for (const door of currentDoors) {
        if (door.targetCell === "Enter0" || door.targetCell === "Enter1") {
          continue;
        }
        const graphKey = `${door.targetCell}->${goalCell}`;
        const hops = CELL_GRAPH_DIST[graphKey] ?? 99;
        if (hops < minHops) {
          minHops = hops;
          bestDoor = door;
        }
      }

      if (bestDoor) {
        targetX = Math.round((bestDoor.xMin + bestDoor.xMax) / 2);
        targetY = Math.round((bestDoor.yMin + bestDoor.yMax) / 2);
        // Jika SP cukup (>= 50), gunakan sprint untuk mengejar lawan
        moveType = self.sp_norm >= 0.50 ? "SPRINT" : "WALK";
        targetCell = bestDoor.targetCell;
        targetPad = bestDoor.targetPad;
        reason = `Moving toward door ${bestDoor.targetCell} to hunt opponent`;
      }
    }
  } else {
    // Lawan di cell yang sama: Bergerak mendekati posisi lawan untuk melee
    const inRange = isMeleeRange(self.x, self.y, opp.x, opp.y);

    if (!inRange) {
      targetX = opp.x;
      targetY = opp.y;
      moveType = self.sp_norm >= 0.50 ? "SPRINT" : "WALK";
      reason = "Approaching opponent into melee range";
    } else {
      // Sudah dalam range melee: Jaga spacing micro (jitter pelan atau stand)
      targetX = self.x;
      targetY = self.y;
      moveType = "STAND";
      reason = "In melee range; engaging in combat";
    }
  }

  // 3. Evaluasi Rotasi Combat Greedy
  const sameRoom = opp && self.cell === opp.cell && opp.alive;
  const inRange = sameRoom && isMeleeRange(self.x, self.y, opp.x, opp.y);
  const mp = self.mp_norm * 100.0;

  if (sameRoom) {
    // A. Defensive: On Guard (S4 - 50% Damage Reduction) jika HP < 70%
    if (inRange && skillsReady[3] && self.hp_norm < 0.70 && mp >= 30.0) {
      chosenCombat = "USE_SKILL_4";
      reason = "Defensive: Activating On Guard (S4)";
    }
    // B. CC Stun Lock: Imbalancing Strike (S2 - 4s Stun) jika lawan belum stun
    else if (inRange && skillsReady[1] && !opp.isStunned && mp >= 30.0) {
      chosenCombat = "USE_SKILL_2";
      reason = "CC: Landing Imbalancing Strike (S2)";
    }
    // C. Combo Priming: Prepared Strike (S3 - Guaranteed Crit) jika S1 siap
    else if (inRange && skillsReady[2] && skillsReady[0] && mp >= 35.0) {
      chosenCombat = "USE_SKILL_3";
      reason = "Combo: Priming Prepared Strike (S3)";
    }
    // D. Main Nuke / Strike: Decisive Strike (S1)
    else if (inRange && skillsReady[0] && mp >= 15.0) {
      chosenCombat = "USE_SKILL_1";
      reason = "Damage: Decisive Strike (S1)";
    }
    // E. Auto Attack berkelanjutan
    else if (inRange) {
      chosenCombat = "AUTO_ATTACK";
      reason = "Pressure: Auto Attack";
    }
  }

  // Clamp target coordinates ke batas stage fisik
  targetX = Math.round(Math.min(WALK_BOUNDS.xMax, Math.max(WALK_BOUNDS.xMin, targetX)));
  targetY = Math.round(Math.min(WALK_BOUNDS.yMax, Math.max(WALK_BOUNDS.yMin, targetY)));

  const movement: AtomicMovementDecision = {
    type: moveType,
    targetX,
    targetY,
    targetCell,
    targetPad,
  };

  return {
    combatAction: chosenCombat,
    movement,
    targetName: opp && sameRoom ? opp.username : undefined,
    reason: `[GreedyBot] ${reason} | Action: ${chosenCombat} | Move: ${moveType} -> (${targetX}, ${targetY})`,
  };
}