/**
 * @file nn_inference.ts
 * @description Pure RL Multi-Head Neural Network Inference Engine (Pure TypeScript Math).
 * Zero NPM dependencies — Fully native Turu runtime execution.
 */

import { ROOM_DOORS, WALK_BOUNDS } from "../config/constants";
import { EntityTracker, TrackedEntity } from "../state/entity_tracker";
import { buildEntityObservation } from "./build_entity_obs";
import { WEIGHTS } from "./weights";
import type {
  AtomicCombatAction,
  AtomicMovementType,
  CombatDecision,
} from "../state/types";

export const MAX_DISP_PX = 300.0;
const RNN_HIDDEN_DIM = 128;
const COORD_STD = (WEIGHTS["coord_logstd"] as number[]).map(Math.exp);

// Gunakan Float32Array<any> agar kompatibel dengan ArrayBuffer maupun SharedArrayBuffer
let activeHiddenState: Float32Array<any> = new Float32Array(RNN_HIDDEN_DIM);

export function resetPolicyState(): void {
  activeHiddenState.fill(0.0);
}

export function getPolicyState(): Float32Array<any> {
  return activeHiddenState;
}

/**
 * Pertahankan fungsi init untuk kompatibilitas signature main_1v1.ts
 */
export function initInferenceSession(): any {
  return WEIGHTS;
}

// =============================================================================
// NATIVE MATRIX & TENSOR OPERATIONS (Zero NPM)
// =============================================================================

function linear(
  x: ArrayLike<number>,
  weight: readonly (readonly number[])[],
  bias: readonly number[]
): Float32Array<any> {
  const outDim = weight.length;
  const inDim = x.length;
  const out = new Float32Array(outDim);
  for (let i = 0; i < outDim; i++) {
    let sum = bias[i] ?? 0;
    const row = weight[i]!;
    for (let j = 0; j < inDim; j++) {
      sum += (row[j] ?? 0) * x[j]!;
    }
    out[i] = sum;
  }
  return out;
}

function layerNorm(
  x: ArrayLike<number>,
  weight: readonly number[],
  bias: readonly number[],
  eps = 1e-5
): Float32Array<any> {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i]!;
  mean /= n;

  let variance = 0;
  for (let i = 0; i < n; i++) {
    const diff = x[i]! - mean;
    variance += diff * diff;
  }
  variance /= n;

  const invStd = 1.0 / Math.sqrt(variance + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = (x[i]! - mean) * invStd * (weight[i] ?? 1) + (bias[i] ?? 0);
  }
  return out;
}

function tanhAct(x: ArrayLike<number>): Float32Array<any> {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    out[i] = Math.tanh(x[i]!);
  }
  return out;
}

function sigmoid(val: number): number {
  return 1.0 / (1.0 + Math.exp(-val));
}

function gruStep(
  x: Float32Array<any>,
  hPrev: Float32Array<any>,
  wIh: readonly (readonly number[])[],
  wHh: readonly (readonly number[])[],
  bIh: readonly number[],
  bHh: readonly number[]
): Float32Array<any> {
  const hiddenDim = hPrev.length;
  const gi = linear(x, wIh, bIh);
  const gh = linear(hPrev, wHh, bHh);
  const hNext = new Float32Array(hiddenDim);

  for (let i = 0; i < hiddenDim; i++) {
    const r = sigmoid(gi[i]! + gh[i]!);
    const z = sigmoid(gi[i + hiddenDim]! + gh[i + hiddenDim]!);
    const n = Math.tanh(gi[i + 2 * hiddenDim]! + r * gh[i + 2 * hiddenDim]!);
    hNext[i] = (1.0 - z) * n + z * hPrev[i]!;
  }

  return hNext;
}

function checkSegmentDoorIntersection(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  door: { xMin: number; xMax: number; yMin: number; yMax: number }
): boolean {
  if (
    (x1 >= door.xMin && x1 <= door.xMax && y1 >= door.yMin && y1 <= door.yMax) ||
    (x2 >= door.xMin && x2 <= door.xMax && y2 >= door.yMin && y2 <= door.yMax)
  ) {
    return true;
  }

  const dx = x2 - x1;
  const dy = y2 - y1;
  const p = [-dx, dx, -dy, dy];
  const q = [x1 - door.xMin, door.xMax - x1, y1 - door.yMin, door.yMax - y1];
  let u1 = 0.0;
  let u2 = 1.0;

  for (let i = 0; i < 4; i++) {
    const pi = p[i]!;
    const qi = q[i]!;
    if (pi === 0) {
      if (qi < 0) return false;
    } else {
      const t = qi / pi;
      if (pi < 0) {
        if (t > u2) return false;
        if (t > u1) u1 = t;
      } else {
        if (t < u1) return false;
        if (t < u2) u2 = t;
      }
    }
  }
  return u1 <= u2;
}

function sampleFromLogits(logits: ArrayLike<number>, temperature: number): number {
  if (temperature <= 0.05) {
    let bestIdx = 0;
    let bestVal = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i]! > bestVal) {
        bestVal = logits[i]!;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i]! > -1e8 && logits[i]! > maxLogit) {
      maxLogit = logits[i]!;
    }
  }
  if (maxLogit === -Infinity) return 0;

  let sumExp = 0;
  const expScores = new Array<number>(logits.length);
  for (let i = 0; i < logits.length; i++) {
    if (logits[i]! <= -1e8) {
      expScores[i] = 0;
    } else {
      const e = Math.exp((logits[i]! - maxLogit) / temperature);
      expScores[i] = e;
      sumExp += e;
    }
  }

  if (sumExp <= 0) return 0;
  let r = Math.random() * sumExp;
  for (let i = 0; i < logits.length; i++) {
    r -= expScores[i]!;
    if (r <= 0) return i;
  }
  return logits.length - 1;
}

export interface InferenceOptions {
  temperature?: number;
  deterministic?: boolean;
}

/**
 * Inferensi Pure RL Multi-Head: Forward pass linear murni via bobot TypeScript
 */
export function inferBattleWarriorDecision(
  tracker: EntityTracker,
  _session: any,
  skillsReady: number[] = [1, 1, 1, 1],
  inCombat: boolean = false,
  outOfCombatTimer: number = 0.0,
  options: InferenceOptions = {}
): CombatDecision {
  const temperature = options.deterministic ? 0.0 : (options.temperature ?? 0.6);
  const self: TrackedEntity = tracker.getSelf();
  const opp: TrackedEntity | null = tracker.getPrimaryOpponent();

  const w = WEIGHTS;

  // 1. Ekstraksi Observasi Kanonikal 30-dim
  const obs = buildEntityObservation(tracker, skillsReady, inCombat, outOfCombatTimer);

  // 2. Forward Pass: Self Encoder (0..16)
  const selfInput = obs.subarray(0, 17);
  let selfFeat = linear(selfInput, w["self_encoder.0.weight"], w["self_encoder.0.bias"]);
  selfFeat = layerNorm(selfFeat, w["self_encoder.1.weight"], w["self_encoder.1.bias"]);
  selfFeat = tanhAct(selfFeat);

  // 3. Forward Pass: Opponent Encoder (17..29)
  const oppInput = obs.subarray(17, 30);
  let oppFeat = linear(oppInput, w["opp_encoder.0.weight"], w["opp_encoder.0.bias"]);
  oppFeat = layerNorm(oppFeat, w["opp_encoder.1.weight"], w["opp_encoder.1.bias"]);
  oppFeat = tanhAct(oppFeat);

  // 4. Fusion Layer (128 -> 128)
  const fusedInput = new Float32Array(128);
  fusedInput.set(selfFeat, 0);
  fusedInput.set(oppFeat, 64);
  let fused = linear(fusedInput, w["fusion.0.weight"], w["fusion.0.bias"]);
  fused = layerNorm(fused, w["fusion.1.weight"], w["fusion.1.bias"]);
  fused = tanhAct(fused);

  // 5. Recurrent GRU Step
  activeHiddenState = gruStep(
    fused,
    activeHiddenState,
    w["gru.weight_ih_l0"],
    w["gru.weight_hh_l0"],
    w["gru.bias_ih_l0"],
    w["gru.bias_hh_l0"]
  );

  // 6. Multi-Head Outputs
  const rawCombatLogits = linear(activeHiddenState, w["combat_head.weight"], w["combat_head.bias"]);
  const rawMoveLogits = linear(activeHiddenState, w["move_head.weight"], w["move_head.bias"]);
  const rawCoordMu = tanhAct(linear(activeHiddenState, w["coord_mu_head.weight"], w["coord_mu_head.bias"]));

  // 7. Action Masking - Head 1: Combat
  const combatLogits = new Float32Array(rawCombatLogits);
  const sameRoom = Boolean(opp && self.cell === opp.cell && opp.alive);
  const inRange = Boolean(sameRoom && Math.abs(self.x - opp!.x) <= 340.0 && Math.abs(self.y - opp!.y) <= 34.0);
  const selfMp = self.mp_norm * 100.0;

  if (!inRange) {
    combatLogits[1] = -1e9; // AA
    combatLogits[2] = -1e9; // S1
    combatLogits[3] = -1e9; // S2
    combatLogits[5] = -1e9; // S4 (On Guard membutuhkan target hostile dalam jangkauan)
  }
  if (!sameRoom) {
    combatLogits[1] = -1e9;
  }

  if (!skillsReady[0] || selfMp < 15.0) combatLogits[2] = -1e9;
  if (!skillsReady[1] || selfMp < 30.0) combatLogits[3] = -1e9;
  if (!skillsReady[2] || selfMp < 20.0) combatLogits[4] = -1e9;
  if (!skillsReady[3] || selfMp < 30.0) combatLogits[5] = -1e9;

  if (self.isStunned) {
    for (let i = 1; i < 6; i++) combatLogits[i] = -1e9;
  }

  const combatIdx = sampleFromLogits(combatLogits, temperature);
  const combatMap: AtomicCombatAction[] = [
    "NONE",
    "AUTO_ATTACK",
    "USE_SKILL_1",
    "USE_SKILL_2",
    "USE_SKILL_3",
    "USE_SKILL_4",
  ];
  const chosenCombat: AtomicCombatAction = combatMap[combatIdx] ?? "NONE";

  // 8. Action Masking - Head 2: Movement Type
  const moveLogits = new Float32Array(rawMoveLogits);
  if (self.sp_norm < 0.50) {
    moveLogits[2] = -1e9;
  }
  if (self.isStunned) {
    moveLogits[1] = -1e9;
    moveLogits[2] = -1e9;
  }

  const moveIdx = sampleFromLogits(moveLogits, temperature);
  const moveMap: AtomicMovementType[] = ["STAND", "WALK", "SPRINT"];
  const chosenMoveType: AtomicMovementType = moveMap[moveIdx] ?? "STAND";

  // 9. Action Head 3: Continuous Point Target Vector
  let normX = rawCoordMu[0]!;
  let normY = rawCoordMu[1]!;

  if (temperature > 0.05) {
    const u1 = Math.max(1e-6, Math.random());
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);

    normX = Math.min(1.0, Math.max(-1.0, normX + z0 * COORD_STD[0]! * temperature));
    normY = Math.min(1.0, Math.max(-1.0, normY + z1 * COORD_STD[1]! * temperature));
  }

  normX = Math.min(1.0, Math.max(-1.0, normX));
  normY = Math.min(1.0, Math.max(-1.0, normY));

  const targetX = Math.round(
    Math.min(WALK_BOUNDS.xMax, Math.max(WALK_BOUNDS.xMin, self.x + normX * MAX_DISP_PX))
  );
  const targetY = Math.round(
    Math.min(WALK_BOUNDS.yMax, Math.max(WALK_BOUNDS.yMin, self.y + normY * MAX_DISP_PX))
  );

  // 10. Segment Door Detection
  let targetCell: string | undefined = undefined;
  let targetPad: string | undefined = undefined;

  if (chosenMoveType !== "STAND") {
    const currentDoors = ROOM_DOORS[self.cell] || [];
    for (const door of currentDoors) {
      if (checkSegmentDoorIntersection(self.x, self.y, targetX, targetY, door)) {
        targetCell = door.targetCell;
        targetPad = door.targetPad;
        break;
      }
    }
  }

  return {
    combatAction: chosenCombat,
    movement: {
      type: chosenMoveType,
      targetX,
      targetY,
      targetCell,
      targetPad,
    },
    targetName: opp && sameRoom ? opp.username : undefined,
    reason: `[PureRL] Action: ${chosenCombat} | Move: ${chosenMoveType} -> (${targetX}, ${targetY})${
      targetCell ? ` [Door: ${targetCell}]` : ""
    }`,
  };
}