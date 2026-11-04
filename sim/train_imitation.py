#!/usr/bin/env python3
"""
Compound Multi-Head Behavioral Cloning / Imitation Learning Pipeline for AQW Non-OP 1v1 PvP.
Decouples human player decisions into 3 orthogonal atomic channels:
1. Combat Action: NONE (0), AA (1), Skill 1 (2), Skill 2 (3), Skill 3 (4), Skill 4 (5), MANA_TAP (6)
2. Movement Type: STAND (0), WALK (1), SPRINT (2)
3. Target Coordinates: Normalized screen point (X/960, Y/550) in [0, 1]^2

Trains a multi-head PyTorch neural network and exports pure weights to `src/ai/trained_policy_weights.ts`.
"""

import json
import math
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

OBS_DIM = 26
HIDDEN_DIM = 64
NUM_COMBAT_ACTIONS = 7
NUM_MOVE_TYPES = 3

COMBAT_ACTION_NAMES = [
    "NONE",
    "AUTO_ATTACK",
    "USE_SKILL_1",
    "USE_SKILL_2",
    "USE_SKILL_3",
    "USE_SKILL_4",
    "MANA_TAP_NPC",
]

MOVE_TYPE_NAMES = [
    "STAND",
    "WALK",
    "SPRINT",
]


class CompoundPolicyMLP(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.combat_head = nn.Linear(hidden_dim, NUM_COMBAT_ACTIONS)
        self.move_type_head = nn.Linear(hidden_dim, NUM_MOVE_TYPES)
        self.coord_head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))

        combat_logits = self.combat_head(h2)
        move_logits = self.move_type_head(h2)
        coords = torch.sigmoid(self.coord_head(h2))

        return combat_logits, move_logits, coords


def find_dataset_path() -> Path:
    candidates = [
        Path("/home/saf/.local/share/Turu/army/expert_trajectories.log"),
        Path(__file__).parent / "data" / "expert_trajectories.jsonl",
        Path(__file__).parent / "data" / "expert_trajectories.log",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    return candidates[0]


def load_demonstrations(file_path: Path):
    if not file_path.exists():
        print(f"[Error] Trajectory dataset not found at {file_path}")
        print("Please play a match in Turu GUI while running `src/tools/record_match.ts` first!")
        sys.exit(1)

    raw_records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                obs = rec.get("obsVector")
                if obs is None or len(obs) != OBS_DIM:
                    continue
                raw_records.append(rec)
            except Exception:
                continue

    if len(raw_records) < 10:
        print(f"[Warning] Only {len(raw_records)} samples found in {file_path}.")
        sys.exit(1)

    # Group into continuous cell segments to compute natural lookahead targets and movement velocity
    segments = []
    cur_seg = []
    for r in raw_records:
        if not cur_seg or cur_seg[-1].get("cell") == r.get("cell"):
            cur_seg.append(r)
        else:
            segments.append(cur_seg)
            cur_seg = [r]
    if cur_seg:
        segments.append(cur_seg)

    observations = []
    combat_targets = []
    move_targets = []
    coord_targets = []

    for s in segments:
        n = len(s)
        for i in range(n):
            r = s[i]
            obs = r["obsVector"]
            act_idx = r.get("actionIndex", 0)
            act_name = r.get("actionName", "")

            # 1. Determine Atomic Combat Action (decoupled from movement):
            # 0: NONE, 1: AA, 2: S1, 3: S2, 4: S3, 5: S4, 6: MANA_TAP
            if "SKILL_1" in act_name or act_idx == 3:
                combat_act = 2
            elif "SKILL_2" in act_name or act_idx == 4:
                combat_act = 3
            elif "SKILL_3" in act_name or act_idx == 5:
                combat_act = 4
            elif "SKILL_4" in act_name or act_idx == 6:
                combat_act = 5
            elif "MANA_TAP" in act_name or act_idx == 10:
                combat_act = 6
            elif obs[4] > 0.5:  # inCombat flag
                combat_act = 1  # AUTO_ATTACK
            else:
                combat_act = 0  # NONE

            # 2. Determine Movement Type based on true displacement & sprint status:
            next_idx = min(n - 1, i + 1)
            dx = float(s[next_idx].get("selfX", r.get("selfX", 0))) - float(r.get("selfX", 0))
            dy = float(s[next_idx].get("selfY", r.get("selfY", 0))) - float(r.get("selfY", 0))
            step_dist = math.hypot(dx, dy)

            is_moving = step_dist > 2.5
            is_sprint = bool(r.get("isSprint", False)) or step_dist > 12.0

            if is_sprint and is_moving:
                move_type = 2  # SPRINT
            elif is_moving:
                move_type = 1  # WALK
            else:
                move_type = 0  # STAND

            # 3. Determine Target Coordinates:
            # If moving, prefer ground-truth mouse click target (targetX, targetY) when recorded,
            # otherwise fallback to natural lookahead destination (5 frames / 0.5s into future)
            if is_moving:
                if "targetX" in r and "targetY" in r and r["targetX"] is not None:
                    target_x = float(r["targetX"])
                    target_y = float(r["targetY"])
                else:
                    lookahead_idx = min(n - 1, i + 5)
                    target_x = float(s[lookahead_idx].get("selfX", r.get("selfX", 480.0)))
                    target_y = float(s[lookahead_idx].get("selfY", r.get("selfY", 400.0)))
            else:
                target_x = float(r.get("selfX", 480.0))
                target_y = float(r.get("selfY", 400.0))

            # Normalize target coordinates to [0, 1]
            norm_x = min(1.0, max(0.0, target_x / 960.0))
            norm_y = min(1.0, max(0.0, target_y / 550.0))

            observations.append(obs)
            combat_targets.append(combat_act)
            move_targets.append(move_type)
            coord_targets.append([norm_x, norm_y])

    print(
        f"[Compound Imitation Learning] Loaded {len(observations)} frames across {len(segments)} segments from {file_path}."
    )
    return (
        np.array(observations, dtype=np.float32),
        np.array(combat_targets, dtype=np.int64),
        np.array(move_targets, dtype=np.int64),
        np.array(coord_targets, dtype=np.float32),
    )


def train_compound_policy(X, y_combat, y_move, y_coord, epochs=60, batch_size=32, lr=1.5e-3):
    device = torch.device("cpu")
    model = CompoundPolicyMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    criterion_combat = nn.CrossEntropyLoss()
    criterion_move = nn.CrossEntropyLoss()
    criterion_coord = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y_combat, dtype=torch.long),
        torch.tensor(y_move, dtype=torch.long),
        torch.tensor(y_coord, dtype=torch.float32),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    final_combat_acc = 0.0
    final_move_acc = 0.0
    final_coord_mse = 0.0

    for epoch in range(1, epochs + 1):
        total_combat_correct = 0
        total_move_correct = 0
        total_samples = 0
        total_coord_loss = 0.0

        for bx, b_comb, b_move, b_coord in dataloader:
            bx, b_comb, b_move, b_coord = (
                bx.to(device),
                b_comb.to(device),
                b_move.to(device),
                b_coord.to(device),
            )
            optimizer.zero_grad()

            pred_comb, pred_move, pred_coord = model(bx)

            l_comb = criterion_combat(pred_comb, b_comb)
            l_move = criterion_move(pred_move, b_move)
            l_coord = criterion_coord(pred_coord, b_coord)

            loss = l_comb + l_move + 3.0 * l_coord
            loss.backward()
            optimizer.step()

            total_combat_correct += (torch.argmax(pred_comb, dim=-1) == b_comb).sum().item()
            total_move_correct += (torch.argmax(pred_move, dim=-1) == b_move).sum().item()
            total_coord_loss += l_coord.item() * len(bx)
            total_samples += len(bx)

        combat_acc = (total_combat_correct / total_samples) * 100.0
        move_acc = (total_move_correct / total_samples) * 100.0
        coord_mse = total_coord_loss / total_samples

        final_combat_acc = combat_acc
        final_move_acc = move_acc
        final_coord_mse = coord_mse

        if epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:2d}/{epochs:2d}] Combat Acc: {combat_acc:.1f}% | Move Acc: {move_acc:.1f}% | Coord MSE: {coord_mse:.4f}"
            )

    return model, final_combat_acc, final_move_acc, final_coord_mse


def export_compound_weights_to_ts(
    model: CompoundPolicyMLP, combat_acc: float, move_acc: float, coord_mse: float, output_path: Path
):
    w1 = model.fc1.weight.detach().cpu().numpy().tolist()
    b1 = model.fc1.bias.detach().cpu().numpy().tolist()
    w2 = model.fc2.weight.detach().cpu().numpy().tolist()
    b2 = model.fc2.bias.detach().cpu().numpy().tolist()

    w_combat = model.combat_head.weight.detach().cpu().numpy().tolist()
    b_combat = model.combat_head.bias.detach().cpu().numpy().tolist()

    w_move = model.move_type_head.weight.detach().cpu().numpy().tolist()
    b_move = model.move_type_head.bias.detach().cpu().numpy().tolist()

    w_coord = model.coord_head.weight.detach().cpu().numpy().tolist()
    b_coord = model.coord_head.bias.detach().cpu().numpy().tolist()

    ts_content = f"""/**
 * @file trained_policy_weights.ts
 * @description Compound Multi-Head Behavioral Cloning Policy for AQW Non-OP 1v1 PvP.
 * Cloned directly from human pro-player demonstrations (`sim/train_imitation.py`).
 *
 * Architecture:
 * - Shared Backbone: 26 -> 64 (tanh) -> 64 (tanh)
 * - Head 1 (Combat Action, 7 classes): NONE, AUTO_ATTACK, SKILL_1..4, MANA_TAP_NPC
 * - Head 2 (Movement Type, 3 classes): STAND, WALK, SPRINT
 * - Head 3 (Target Point Regression, 2 coords): Normalized (X/960, Y/550) via Sigmoid
 *
 * Training Metrics:
 * - Combat Action Accuracy: {combat_acc:.1f}%
 * - Movement Type Accuracy: {move_acc:.1f}%
 * - Coordinate Target MSE: {coord_mse:.4f}
 */

export interface CompoundAtomicPolicySpec {{
    className: string;
    obsDim: number;
    hiddenDim: number;
    combatActions: readonly string[];
    moveTypes: readonly string[];
    metrics: {{
        combat_accuracy: number;
        move_accuracy: number;
        coord_mse: number;
    }};
    w1: readonly (readonly number[])[];
    b1: readonly number[];
    w2: readonly (readonly number[])[];
    b2: readonly number[];
    wCombat: readonly (readonly number[])[];
    bCombat: readonly number[];
    wMoveType: readonly (readonly number[])[];
    bMoveType: readonly number[];
    wCoord: readonly (readonly number[])[];
    bCoord: readonly number[];
}}

export const COMPOUND_ATOMIC_POLICY_WEIGHTS: CompoundAtomicPolicySpec = {json.dumps({
    "className": "Battle Warrior",
    "obsDim": OBS_DIM,
    "hiddenDim": HIDDEN_DIM,
    "combatActions": COMBAT_ACTION_NAMES,
    "moveTypes": MOVE_TYPE_NAMES,
    "metrics": {
        "combat_accuracy": round(combat_acc / 100.0, 3),
        "move_accuracy": round(move_acc / 100.0, 3),
        "coord_mse": round(coord_mse, 5),
    },
    "w1": w1,
    "b1": b1,
    "w2": w2,
    "b2": b2,
    "wCombat": w_combat,
    "bCombat": b_combat,
    "wMoveType": w_move,
    "bMoveType": b_move,
    "wCoord": w_coord,
    "bCoord": b_coord,
}, indent=4)};
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(ts_content)

    print(f"[Export] Successfully generated compound atomic weights at {output_path}!")


def main():
    dataset_path = find_dataset_path()
    print(f"[Compound Imitation] Training on trajectory data: {dataset_path}")
    X, y_combat, y_move, y_coord = load_demonstrations(dataset_path)

    print(f"[Compound Imitation] Training Compound Multi-Head MLP on {len(X)} frames...")
    model, comb_acc, move_acc, coord_mse = train_compound_policy(
        X, y_combat, y_move, y_coord, epochs=60, batch_size=32, lr=1.5e-3
    )

    out_ts = Path(__file__).parent.parent / "src" / "ai" / "trained_policy_weights.ts"
    export_compound_weights_to_ts(model, comb_acc, move_acc, coord_mse, out_ts)


if __name__ == "__main__":
    main()
