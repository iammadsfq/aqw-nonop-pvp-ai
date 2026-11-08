/**
 * @file types.ts
 * @description Pure RL State and Action Space definitions for 1v1 Bludrut Brawl.
 */

// =============================================================================
// 1. ATOMIC ACTION SPACE (SINKRON DENGAN SIMULATOR PYTHON)
// =============================================================================

export type AtomicCombatAction =
  | "NONE"
  | "AUTO_ATTACK"
  | "USE_SKILL_1"
  | "USE_SKILL_2"
  | "USE_SKILL_3"
  | "USE_SKILL_4";

export type AtomicMovementType = "STAND" | "WALK" | "SPRINT";

export interface AtomicMovementDecision {
  type: AtomicMovementType;
  /** Titik continuous X di lantai map (WALK_BOUNDS: 35..930) */
  targetX: number;
  /** Titik continuous Y di lantai map (WALK_BOUNDS: 255..500) */
  targetY: number;
  /** Diisi jika titik menabrak AABB pintu secara fisik */
  targetCell?: string;
  targetPad?: string;
}

export interface CombatDecision {
  combatAction: AtomicCombatAction;
  movement: AtomicMovementDecision;
  targetName?: string;
  reason: string;
}

// =============================================================================
// 2. MATCH & WAVE TELEMETRY
// =============================================================================

export interface WaveTrackerState {
  waveNumber: number;
  waveActive: boolean;
  opponentKilledInWave: boolean;
  teammateDiedInWave: boolean;
  canReturnToSafezone: boolean;
  teamKills: number;
  teamDeaths: number;
}