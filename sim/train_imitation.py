#!/usr/bin/env python3
"""
sim/train_imitation.py
Behavioral Cloning (Imitation Learning) dari Rekaman Nyata Pemain (aqw_combat_telemetry.log).
Mengekstrak 30-dim observation dan supervised targets (combat action, move type, normalized displacement).
Menghasilkan checkpoint bobot pre-trained agar agen PPO tidak mulai dari nol (tidak terjebak di safezone).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CUR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CUR_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.world.bludrut_map import SCREEN_INDICES, WALK_BOUNDS
from sim.train_kaggle_bundle import EntityCompoundActorCritic

FLOOR_X_MIN = WALK_BOUNDS["x_min"]  # 35.0
FLOOR_X_MAX = WALK_BOUNDS["x_max"]  # 930.0
FLOOR_Y_MIN = WALK_BOUNDS["y_min"]  # 255.0
FLOOR_Y_MAX = WALK_BOUNDS["y_max"]  # 500.0
FLOOR_WIDTH = FLOOR_X_MAX - FLOOR_X_MIN
FLOOR_HEIGHT = FLOOR_Y_MAX - FLOOR_Y_MIN
MAX_DISP_PX = 300.0


def extract_demonstrations(log_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ekstrak data log telemetri menjadi pasangan (obs, combat_label, move_label, coord_label).
    """
    if not log_path.exists():
        raise FileNotFoundError(f"File log telemetri tidak ditemukan di: {log_path}")

    observations: List[List[float]] = []
    combat_targets: List[int] = []
    move_targets: List[int] = []
    coord_targets: List[List[float]] = []

    SKILL_MAP = {"aa": 1, "a1": 2, "a2": 3, "a3": 4, "a4": 5}

    last_pending_skill = 0
    recent_x = 488.0
    recent_y = 350.0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            rtype = rec.get("type")
            cell = rec.get("cell", "Enter0")
            payload = rec.get("payload", {})
            cell_idx = float(SCREEN_INDICES.get(cell, 0)) / 15.0

            if rtype == "player_sent_action":
                pkt = payload.get("packet", "")
                parts = pkt.split("%")
                if len(parts) >= 7:
                    act_name = parts[6].split(">")[0]
                    last_pending_skill = SKILL_MAP.get(act_name, 0)
                continue

            if rtype == "movement_packet":
                from_x = float(payload.get("fromX", recent_x))
                from_y = float(payload.get("fromY", recent_y))
                tgt_x = float(payload.get("targetX", from_x))
                tgt_y = float(payload.get("targetY", from_y))
                is_sprint = bool(payload.get("isSprint", False))
                in_combat = bool(payload.get("inCombat", False))
                self_sp = float(payload.get("selfSp", 100.0))

                recent_x = tgt_x
                recent_y = tgt_y

                norm_x = min(1.0, max(0.0, (from_x - FLOOR_X_MIN) / FLOOR_WIDTH))
                norm_y = min(1.0, max(0.0, (from_y - FLOOR_Y_MIN) / FLOOR_HEIGHT))

                # Displacment target
                dx = tgt_x - from_x
                dy = tgt_y - from_y
                dist = math.hypot(dx, dy)

                if dist < 5.0:
                    move_type = 0  # Stand
                    norm_dx = 0.0
                    norm_dy = 0.0
                else:
                    move_type = 2 if is_sprint else 1  # Sprint or Walk
                    scale = max(1.0, dist / MAX_DISP_PX)
                    norm_dx = float(np.clip(dx / (MAX_DISP_PX * scale), -1.0, 1.0))
                    norm_dy = float(np.clip(dy / (MAX_DISP_PX * scale), -1.0, 1.0))

                # Sintesis 30-dim observation
                # 17 self features
                self_obs = [
                    1.0,                       # hp/max_hp (default sehat)
                    0.8,                       # mp/max_mp
                    self_sp / 100.0,           # sp/max_sp
                    1.0 if self_sp >= 50.0 else 0.0,
                    1.0 if in_combat else 0.0,
                    0.0 if in_combat else 1.0,
                    cell_idx,
                    norm_x,
                    norm_y,
                    1.0,                       # is_alive
                    1.0, 1.0, 1.0, 1.0,        # ready skills [S1..S4]
                    0.50,                      # inc dmg mult (1.0 baseline)
                    0.50,                      # out dmg mult (1.0 baseline)
                    0.0                        # is_stunned
                ]

                # 13 opp features (approximated / general)
                opp_obs = [
                    1.0,                       # valid
                    1.0,                       # opp is_alive
                    0.0,                       # is_ally
                    1.0 if in_combat else 0.0, # same_cell approx
                    cell_idx,                  # opp_cell approx
                    0.0, 0.0,                  # rel_x, rel_y
                    1.0 if in_combat else 0.0, # in_melee
                    0.8,                       # opp hp
                    0.8,                       # opp mp
                    0.50,                      # opp inc dmg
                    0.0,                       # opp stunned
                    0.50                       # opp out dmg
                ]

                full_obs = self_obs + opp_obs
                observations.append(full_obs)
                move_targets.append(move_type)
                coord_targets.append([norm_dx, norm_dy])
                combat_targets.append(last_pending_skill)
                last_pending_skill = 0  # Reset setelah dikonsumsi

    obs_np = np.array(observations, dtype=np.float32)
    c_tgt_np = np.array(combat_targets, dtype=np.int64)
    m_tgt_np = np.array(move_targets, dtype=np.int64)
    coord_tgt_np = np.array(coord_targets, dtype=np.float32)

    return obs_np, c_tgt_np, m_tgt_np, coord_tgt_np


def train_behavioral_cloning(
    epochs: int = 35,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_dir: Optional[Path] = None
) -> Path:
    log_path = Path.home() / ".local/share/Turu/army/aqw_combat_telemetry.log"
    print(f"[1/4] Membaca dataset dari: {log_path}")

    obs, c_tgt, m_tgt, coord_tgt = extract_demonstrations(log_path)
    n_samples = len(obs)
    print(f"      Total sampel terekstrak: {n_samples:,} frame demonstrasi")

    if n_samples < 50:
        raise ValueError("Data demonstrasi terlalu sedikit untuk behavioral cloning!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[2/4] Menginisialisasi Model Actor-Critic di device: {device}")

    model = EntityCompoundActorCritic(hidden_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    t_obs = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(1)  # [N, 1, 30]
    t_c = torch.tensor(c_tgt, dtype=torch.long, device=device)
    t_m = torch.tensor(m_tgt, dtype=torch.long, device=device)
    t_coord = torch.tensor(coord_tgt, dtype=torch.float32, device=device)

    dataset_indices = np.arange(n_samples)

    print(f"[3/4] Melatih Behavioral Cloning selama {epochs} epoch...")
    model.train()
    for ep in range(1, epochs + 1):
        np.random.shuffle(dataset_indices)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            mb = dataset_indices[start:start + batch_size]
            b_obs = t_obs[mb]
            b_c = t_c[mb]
            b_m = t_m[mb]
            b_coord = t_coord[mb]

            h_0 = torch.zeros(1, len(mb), 128, device=device)
            c_logits, m_logits, coord_mu, _, _ = model.forward(b_obs, h_0)

            c_loss = ce_loss_fn(c_logits.squeeze(1), b_c)
            m_loss = ce_loss_fn(m_logits.squeeze(1), b_m)
            coord_loss = mse_loss_fn(coord_mu.squeeze(1), b_coord)

            loss = c_loss + m_loss + 2.0 * coord_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if ep % 5 == 0 or ep == 1:
            avg_loss = total_loss / max(1, n_batches)
            print(f"      Epoch {ep:02d}/{epochs:02d} | Loss: {avg_loss:.4f}")

    if output_dir is None:
        output_dir = CUR_DIR / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "imitation_warrior_base.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[4/4] Pre-trained weights berhasil disimpan ke: {ckpt_path}")

    # Export juga langsung ke src/ai/weights.ts
    ts_path = PROJECT_ROOT / "src/ai/weights.ts"
    state = model.state_dict()
    with open(ts_path, "w") as f:
        f.write("/**\n")
        f.write(" * @file weights.ts\n")
        f.write(" * Pre-trained from Real Pro Player Demonstrations via Behavioral Cloning\n")
        f.write(" */\n\n")
        f.write("export const WEIGHTS: Record<string, any> = {\n")
        for k, v in state.items():
            arr = v.cpu().numpy().tolist()
            f.write(f'  "{k}": {arr},\n')
        f.write("};\n")
    print(f"      Bobot langsung terpasang di runtime: {ts_path}")

    return ckpt_path


if __name__ == "__main__":
    train_behavioral_cloning()
