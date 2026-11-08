#!/usr/bin/env python3
"""
sim/train_kaggle_bundle.py
Self-contained PPO Training Pipeline untuk Kaggle GPU & Local Training.
Mendukung:
  1. Continuous 10-Kill Match dengan 4.0s Respawn Delay
  2. Bounding Box Melee Range Flash AQW (dx <= 340.0, dy <= 34.0 px)
  3. Dynamic ActiveAura domain model & passive buffs
  4. Regulasi Safezone (penalti masuk ilegal vs izin keluar pasca kill)
  5. 30-dim observation parity (17 self + 13 opp)
  6. Recurrent Multi-Head Actor-Critic (Compound Actions)
  7. Export bobot langsung ke src/ai/weights.ts (zero npm runtime)
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Setup path agar import lokal selalu aman di Kaggle maupun lokal
CUR_DIR = Path(__file__).resolve().parent
if str(CUR_DIR) not in sys.path:
    sys.path.insert(0, str(CUR_DIR))
ROOT_DIR = CUR_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from sim.bludrut_env import (
        BludrutBrawlEnv,
        ActionSpec,
        Entity,
        ActiveAura,
        MELEE_RANGE_X,
        MELEE_RANGE_Y,
        MAX_DISP_PX,
        SP_SPRINT_COST,
        WARRIOR_PROFILE,
        FLOOR_WIDTH,
        FLOOR_HEIGHT,
        ROOM_DOORS,
        CELL_GRAPH_DIST,
    )
except ImportError:
    from bludrut_env import (
        BludrutBrawlEnv,
        ActionSpec,
        Entity,
        ActiveAura,
        MELEE_RANGE_X,
        MELEE_RANGE_Y,
        MAX_DISP_PX,
        SP_SPRINT_COST,
        WARRIOR_PROFILE,
        FLOOR_WIDTH,
        FLOOR_HEIGHT,
        ROOM_DOORS,
        CELL_GRAPH_DIST,
    )


# ==============================================================================
# 1. NEURAL NETWORK ARCHITECTURE
# ==============================================================================

class EntityCompoundActorCritic(nn.Module):
    """
    Recurrent Compound Actor-Critic (Decoupled Self & Opponent Encoders -> GRU -> Decoupled Heads):
    - Self Encoder: 17 float -> 64
    - Opponent Encoder: 13 float -> 64
    - Fusion: 128 -> 128
    - Recurrent GRU: 128 -> 128
    - Actor Combat Head: 128 -> 6 logits
    - Actor Move Type Head: 128 -> 3 logits
    - Actor Egocentric Coord Head: 128 -> 2 continuous offsets [-1, 1]
    - Critic Value Head: 128 -> 1
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.self_encoder = nn.Sequential(
            nn.Linear(17, 64),
            nn.LayerNorm(64),
            nn.Tanh()
        )
        self.opp_encoder = nn.Sequential(
            nn.Linear(13, 64),
            nn.LayerNorm(64),
            nn.Tanh()
        )

        self.fusion = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Multi-Head Outputs
        self.combat_head = nn.Linear(hidden_dim, 6)    # None, AA, S1..S4
        self.move_head = nn.Linear(hidden_dim, 3)      # Stand, Walk, Sprint
        self.coord_mu_head = nn.Linear(hidden_dim, 2)  # continuous dx, dy
        self.critic_head = nn.Linear(hidden_dim, 1)

        self.coord_logstd = nn.Parameter(torch.ones(2) * -0.5)

    def forward(
        self,
        obs: torch.Tensor,                       # [B, T, 30]
        h_in: torch.Tensor,                      # [1, B, H]
        dones: Optional[torch.Tensor] = None,    # [B, T]; 1 => obs[:, t] adalah awal episode baru
    ):
        self_feat = self.self_encoder(obs[:, :, :17])
        opp_feat = self.opp_encoder(obs[:, :, 17:])
        fused = self.fusion(torch.cat([self_feat, opp_feat], dim=-1))

        if dones is None:
            gru_out, h_out = self.gru(fused, h_in)
        else:
            outs = []
            h = h_in
            for t in range(fused.size(1)):
                h = h * (1.0 - dones[:, t]).view(1, -1, 1)
                o, h = self.gru(fused[:, t:t + 1], h)
                outs.append(o)
            gru_out = torch.cat(outs, dim=1)
            h_out = h

        combat_logits = self.combat_head(gru_out)
        move_logits = self.move_head(gru_out)
        coord_mu = torch.tanh(self.coord_mu_head(gru_out))
        value = self.critic_head(gru_out)
        return combat_logits, move_logits, coord_mu, value, h_out

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        h_in: torch.Tensor,
        c_mask: torch.Tensor,
        m_mask: torch.Tensor,
        c_act: Optional[torch.Tensor] = None,
        m_act: Optional[torch.Tensor] = None,
        coord_act: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
    ):
        combat_logits, move_logits, coord_mu, value, h_out = self.forward(obs, h_in, dones)

        if c_mask.dim() == 2:
            c_mask = c_mask.unsqueeze(1)
        if m_mask.dim() == 2:
            m_mask = m_mask.unsqueeze(1)

        c_dist = torch.distributions.Categorical(logits=combat_logits.masked_fill(~c_mask, -1e9))
        m_dist = torch.distributions.Categorical(logits=move_logits.masked_fill(~m_mask, -1e9))
        coord_std = torch.exp(self.coord_logstd).view(1, 1, 2)
        coord_dist = torch.distributions.Normal(coord_mu, coord_std)

        if c_act is None:
            c_act = c_dist.sample()
        if m_act is None:
            m_act = m_dist.sample()
        if coord_act is None:
            coord_act = coord_dist.sample()

        if c_act.dim() == 1:
            c_act = c_act.unsqueeze(1)
            m_act = m_act.unsqueeze(1)
            coord_act = coord_act.unsqueeze(1)

        log_prob = (
            c_dist.log_prob(c_act)
            + m_dist.log_prob(m_act)
            + coord_dist.log_prob(coord_act).sum(dim=-1)
        )
        entropy = c_dist.entropy() + m_dist.entropy() + coord_dist.entropy().sum(dim=-1)
        value = value.squeeze(-1)  # [B, T]

        if obs.size(1) == 1:
            return (
                c_act.squeeze(1), m_act.squeeze(1), coord_act.squeeze(1),
                log_prob.squeeze(1), entropy.squeeze(1), value.squeeze(1), h_out,
            )
        return c_act, m_act, coord_act, log_prob, entropy, value, h_out


# ==============================================================================
# 2. ACTION MASKS & GREEDY SPARRING BOT
# ==============================================================================

def compute_masks(ent: Entity, opp: Entity) -> Tuple[np.ndarray, np.ndarray]:
    combat_mask = np.ones(6, dtype=bool)
    move_mask = np.ones(3, dtype=bool)

    if ent.is_stunned or not ent.is_alive:
        combat_mask[1:] = False
        move_mask[1:] = False
        return combat_mask, move_mask

    if ent.sp < SP_SPRINT_COST:
        move_mask[2] = False  # cannot sprint

    same_room = ent.cell == opp.cell
    in_range = same_room and opp.is_alive and (abs(ent.x - opp.x) <= MELEE_RANGE_X and abs(ent.y - opp.y) <= MELEE_RANGE_Y)

    for act_idx in range(1, 6):
        skill = ent.skills.get(act_idx)
        if not skill:
            combat_mask[act_idx] = False
            continue
        if ent.cooldowns[act_idx] > 0.0 or (act_idx > 1 and ent.gcd_rem > 0.0) or ent.mp < skill.mana_cost:
            combat_mask[act_idx] = False
            continue
        if not skill.is_self_buff and not in_range:
            combat_mask[act_idx] = False

    return combat_mask, move_mask


def select_greedy_sparring_action(ent: Entity, opp: Entity) -> ActionSpec:
    c_mask, m_mask = compute_masks(ent, opp)
    if not ent.is_alive or ent.is_stunned:
        return ActionSpec(0, 0, 0.0, 0.0)

    same_room = ent.cell == opp.cell and opp.is_alive
    move_type = 0
    target_dx = 0.0
    target_dy = 0.0

    if same_room:
        dist_x = opp.x - ent.x
        dist_y = opp.y - ent.y
        in_range = abs(dist_x) <= MELEE_RANGE_X and abs(dist_y) <= MELEE_RANGE_Y
        if not in_range:
            move_type = 2 if m_mask[2] else 1
            scale = max(1.0, math.hypot(dist_x, dist_y) / MAX_DISP_PX)
            target_dx = float(dist_x / (MAX_DISP_PX * scale))
            target_dy = float(dist_y / (MAX_DISP_PX * scale))
    else:
        # Navigasi Invarian menuju Mid (Crosslower):
        # Selama lawan tidak berada di ruangan yang sama (not same_room), setiap ruangan di luar
        # Crosslower selalu mengarahkan agen melalui rute terpendek menuju Mid (Crosslower) tanpa
        # pernah tersesat ke Captain0/1 atau berbalik arah saat baru respawn.
        if ent.cell != "Crosslower":
            goal_cell = "Crosslower"
        elif not opp.is_alive or opp.cell in ("Enter0", "Enter1", "Captain0", "Captain1"):
            goal_cell = "Crosslower"
        else:
            goal_cell = opp.cell

        if ent.cell == goal_cell:
            # Sudah di cell tujuan (misal menunggu lawan di Crosslower)
            dist_x = 488.0 - ent.x
            dist_y = 351.0 - ent.y
            if math.hypot(dist_x, dist_y) > 25.0:
                move_type = 1
                scale = max(1.0, math.hypot(dist_x, dist_y) / MAX_DISP_PX)
                target_dx = float(dist_x / (MAX_DISP_PX * scale))
                target_dy = float(dist_y / (MAX_DISP_PX * scale))
        else:
            doors = ROOM_DOORS.get(ent.cell, [])
            best_door = None
            min_hops = 999
            for door in doors:
                # Jangan masuk safezone atau ruang Captain saat menuju Mid/lawan
                if door.target_cell in ("Enter0", "Enter1"):
                    continue
                if door.target_cell in ("Captain0", "Captain1") and goal_cell != door.target_cell:
                    continue
                hops = CELL_GRAPH_DIST.get((door.target_cell, goal_cell), 99)
                if hops < min_hops:
                    min_hops = hops
                    best_door = door

            if best_door is not None:
                door_x = 0.5 * (best_door.x_min + best_door.x_max)
                door_y = 0.5 * (best_door.y_min + best_door.y_max)
                # Pintu bawah penuh (Captain0, Captain1, Crossupper -> y=[455..500]):
                # cukup turun vertikal dengan komponen dy tajam menuju y=495
                if best_door.y_min >= 450.0:
                    door_x = float(np.clip(ent.x, 260.0, 700.0))
                    door_y = 495.0
                    dist_x = door_x - ent.x
                    dist_y = (door_y - ent.y) * 2.2
                # Hindari menyenggol pintu Captain0/Captain1 (x=[401..573], y=[275..300]) saat melintasi Morale0C/1C
                # tanpa pernah memberi vektor dy > 0 (turun): cukup jaga y >= 306 sampai melewati x pintu Captain.
                elif ent.cell == "Morale0C" and best_door.target_cell == "Morale0B" and ent.x < 575.0:
                    door_x = 650.0
                    door_y = min(ent.y, 306.0)
                    dist_x = door_x - ent.x
                    dist_y = door_y - ent.y
                elif ent.cell == "Morale1C" and best_door.target_cell == "Morale1B" and ent.x > 395.0:
                    door_x = 320.0
                    door_y = min(ent.y, 306.0)
                    dist_x = door_x - ent.x
                    dist_y = door_y - ent.y
                else:
                    # Pintu koridor atas (y=[255..300]): tarik vertikal lebih kuat (2.2x) ke y=268
                    # agar agen sudah berada di y=268..285 sebelum mencapai batas x pintu (765 / 220).
                    door_y = 268.0
                    dist_x = door_x - ent.x
                    dist_y = (door_y - ent.y) * 2.2
                move_type = 2 if m_mask[2] else 1
                scale = max(1.0, math.hypot(dist_x, dist_y) / MAX_DISP_PX)
                target_dx = float(dist_x / (MAX_DISP_PX * scale))
                target_dy = float(dist_y / (MAX_DISP_PX * scale))

    combat_act = 0
    if c_mask[5] and (ent.hp / max(1.0, ent.max_hp)) < 0.70:
        combat_act = 5  # On Guard (S4)
    elif c_mask[3] and not opp.is_stunned:
        combat_act = 3  # Imbalancing Strike (S2 - Stun)
    elif c_mask[4] and c_mask[2]:
        combat_act = 4  # Prepared Strike (S3 - Crit buff)
    elif c_mask[2]:
        combat_act = 2  # Decisive Strike (S1)
    elif c_mask[1]:
        combat_act = 1  # AA

    return ActionSpec(combat_act, move_type, target_dx, target_dy)


# ==============================================================================
# ==============================================================================
# 3. TRAINING PIPELINE & PARAMETER CONFIGURATION
# ==============================================================================
# Parameter yang dapat diterima (via CLI flags maupun Environment Variables):
#   --matches / BB_MATCHES         : Target jumlah match selesai untuk menghentikan training (default: 500; 0 = gunakan --updates)
#   --kills / BB_TARGET_KILLS      : Target kill per match untuk menang (default: 10 kill)
#   --max-time / BB_MAX_MATCH_TIME : Batas durasi maksimum 1 match dalam detik (default: 1200.0s / 20 menit)
#   --updates / BB_UPDATES         : Batas maksimum iterasi PPO update (default: 5000 sebagai safety cap)
#   --envs / BB_NUM_ENVS           : Jumlah environment paralel (default: 16; minimal 4)
#   --rollout-steps / BB_ROLLOUT   : Jumlah step per rollout buffer sebelum PPO update (default: 128)
#   --chunk-len / BB_CHUNK_LEN     : Panjang sekuens BPTT GRU per chunk (default: 16)
#   --lr / BB_LR                   : Learning rate Adam optimizer (default: 3e-4)
#   --gamma / BB_GAMMA             : Discount factor GAE/PPO (default: 0.99)
#   --ent-coef / BB_ENT_COEF       : Koefisien entropy bonus untuk eksplorasi (default: 0.01)
#   --seed / BB_SEED               : Random seed (default: 0)

import argparse


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AQW Bludrut Brawl PPO Training Pipeline")
    parser.add_argument("--matches", type=int, default=int(os.environ.get("BB_MATCHES", 500)),
                        help="Target total match selesai (0 = hanya berhenti berdasarkan --updates)")
    parser.add_argument("--kills", type=int, default=int(os.environ.get("BB_TARGET_KILLS", 10)),
                        help="Target kill untuk memenangkan 1 match (default: 10)")
    parser.add_argument("--max-time", type=float, default=float(os.environ.get("BB_MAX_MATCH_TIME", 1200.0)),
                        help="Batas waktu maksimum per match dalam detik (default: 1200.0)")
    parser.add_argument("--updates", type=int, default=int(os.environ.get("BB_UPDATES", 5000)),
                        help="Batas maksimum PPO update iterations (default: 5000)")
    parser.add_argument("--envs", type=int, default=int(os.environ.get("BB_NUM_ENVS", 16)),
                        help="Jumlah environment paralel (default: 16)")
    parser.add_argument("--rollout-steps", type=int, default=int(os.environ.get("BB_ROLLOUT", 128)),
                        help="Jumlah step per rollout (default: 128)")
    parser.add_argument("--chunk-len", type=int, default=int(os.environ.get("BB_CHUNK_LEN", 16)),
                        help="Panjang chunk BPTT GRU (default: 16)")
    parser.add_argument("--lr", type=float, default=float(os.environ.get("BB_LR", 3e-4)),
                        help="Learning rate Adam (default: 3e-4)")
    parser.add_argument("--gamma", type=float, default=float(os.environ.get("BB_GAMMA", 0.99)),
                        help="Discount factor gamma (default: 0.99)")
    parser.add_argument("--ent-coef", type=float, default=float(os.environ.get("BB_ENT_COEF", 0.01)),
                        help="Entropy coefficient (default: 0.01)")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("BB_SEED", 0)),
                        help="Random seed (default: 0)")
    args, _ = parser.parse_known_args()
    return args


def train():
    args = parse_train_args()
    NUM_ENVS = max(4, args.envs)
    n_quarter = max(1, NUM_ENVS // 4)
    # 25% Blue vs Greedy, 25% Red vs Greedy, 50% Self-Play (Blue + Red)
    STREAM_MAP: List[Tuple[int, int]] = (
        [(i, 0) for i in range(0, n_quarter)]
        + [(i, 1) for i in range(n_quarter, 2 * n_quarter)]
        + [(i, 0) for i in range(2 * n_quarter, NUM_ENVS)]
        + [(i, 1) for i in range(2 * n_quarter, NUM_ENVS)]
    )
    NUM_STREAMS = len(STREAM_MAP)
    STREAM_IDX_LOOKUP: Dict[Tuple[int, int], int] = {
        pair: idx for idx, pair in enumerate(STREAM_MAP)
    }

    ROLLOUT_STEPS = args.rollout_steps
    CHUNK_LEN = args.chunk_len
    TARGET_MATCHES = args.matches
    TOTAL_UPDATES = args.updates
    LR = args.lr
    GAMMA = args.gamma
    GAE_LAMBDA = 0.95
    CLIP_COEF = 0.2
    ENT_COEF = args.ent_coef
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    UPDATE_EPOCHS = 4
    TARGET_KILLS = args.kills
    MAX_MATCH_TIME = args.max_time
    HIDDEN = 128
    assert ROLLOUT_STEPS % CHUNK_LEN == 0

    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = Path("/kaggle/working/weights") if os.path.exists("/kaggle/working") else (CUR_DIR / "weights")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    envs = [
        BludrutBrawlEnv(
            seed=42 + SEED + i,
            target_kills=TARGET_KILLS,
            max_match_time=MAX_MATCH_TIME,
        )
        for i in range(NUM_ENVS)
    ]
    policy = EntityCompoundActorCritic(hidden_dim=HIDDEN).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR, eps=1e-5)

    obs_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS, 30), device=DEVICE)
    c_act_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS), dtype=torch.long, device=DEVICE)
    m_act_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS), dtype=torch.long, device=DEVICE)
    coord_act_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS, 2), device=DEVICE)
    c_mask_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS, 6), dtype=torch.bool, device=DEVICE)
    m_mask_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS, 3), dtype=torch.bool, device=DEVICE)
    logprob_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS), device=DEVICE)
    rew_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS), device=DEVICE)
    done_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS), device=DEVICE)
    val_buf = torch.zeros((ROLLOUT_STEPS, NUM_STREAMS), device=DEVICE)
    h_buf = torch.zeros((ROLLOUT_STEPS, 1, NUM_STREAMS, HIDDEN), device=DEVICE)

    env_init_obs = [env.reset() for env in envs]
    init_stream_obs = [env_init_obs[env_i][side] for (env_i, side) in STREAM_MAP]
    next_obs = torch.tensor(np.array(init_stream_obs), dtype=torch.float32, device=DEVICE)
    next_done = torch.zeros(NUM_STREAMS, device=DEVICE)
    next_h = torch.zeros(1, NUM_STREAMS, HIDDEN, device=DEVICE)

    ep_ret = np.zeros(NUM_STREAMS, dtype=np.float64)
    ep_len = np.zeros(NUM_STREAMS, dtype=np.int64)
    recent_ret: deque = deque(maxlen=100)
    recent_len: deque = deque(maxlen=100)
    recent_win: deque = deque(maxlen=100)
    recent_trunc: deque = deque(maxlen=100)

    start_time = time.time()
    total_steps = 0
    completed_matches = 0

    print("======================================================================")
    print(" [Multi-Head PPO Compound Policy] Dual-Side & Self-Play Training")
    print(f" Device: {DEVICE} | Envs: {NUM_ENVS} (Streams: {NUM_STREAMS}) | Target Kills: {TARGET_KILLS}")
    print(f" Target Matches: {TARGET_MATCHES if TARGET_MATCHES > 0 else 'Unlimited'} | Max Updates: {TOTAL_UPDATES} | Max Time: {MAX_MATCH_TIME}s")
    print(f" Target Output Directory: {OUT_DIR}")
    print("======================================================================\n")

    for update in range(1, TOTAL_UPDATES + 1):
        if TARGET_MATCHES > 0:
            prog = min(0.95, completed_matches / max(1, TARGET_MATCHES))
            frac = 1.0 - prog
        else:
            frac = 1.0 - (update - 1.0) / TOTAL_UPDATES
        optimizer.param_groups[0]["lr"] = max(1e-5, frac * LR)

        policy.eval()
        for step in range(ROLLOUT_STEPS):
            total_steps += NUM_ENVS
            obs_buf[step] = next_obs
            done_buf[step] = next_done
            h_buf[step] = next_h

            masks = [
                compute_masks(envs[env_i].p0, envs[env_i].p1)
                if side == 0
                else compute_masks(envs[env_i].p1, envs[env_i].p0)
                for (env_i, side) in STREAM_MAP
            ]
            c_masks = torch.tensor(np.array([m[0] for m in masks]), dtype=torch.bool, device=DEVICE)
            m_masks = torch.tensor(np.array([m[1] for m in masks]), dtype=torch.bool, device=DEVICE)
            c_mask_buf[step] = c_masks
            m_mask_buf[step] = m_masks

            with torch.no_grad():
                c_act, m_act, coord_act, logp, _, val, next_h = policy.get_action_and_value(
                    next_obs.unsqueeze(1), next_h, c_masks, m_masks
                )

            c_act_buf[step] = c_act
            m_act_buf[step] = m_act
            coord_act_buf[step] = coord_act
            logprob_buf[step] = logp
            val_buf[step] = val

            c_np = c_act.cpu().numpy()
            m_np = m_act.cpu().numpy()
            xy_np = coord_act.cpu().numpy()

            def build_stream_act(s_idx: int) -> ActionSpec:
                return ActionSpec(
                    combat_act=int(c_np[s_idx]),
                    move_type=int(m_np[s_idx]),
                    target_dx=float(np.clip(xy_np[s_idx, 0], -1.0, 1.0)),
                    target_dy=float(np.clip(xy_np[s_idx, 1], -1.0, 1.0)),
                )

            def build_noisy_greedy_act(me_ent: Entity, opp_ent: Entity) -> ActionSpec:
                g = select_greedy_sparring_action(me_ent, opp_ent)
                if g.move_type == 0:
                    return g
                return ActionSpec(
                    combat_act=g.combat_act,
                    move_type=g.move_type,
                    target_dx=float(np.clip(g.target_dx + rng.normal(0.0, 0.05), -1.0, 1.0)),
                    target_dy=float(np.clip(g.target_dy + rng.normal(0.0, 0.05), -1.0, 1.0)),
                )

            step_rewards = [0.0] * NUM_STREAMS
            step_dones = [0.0] * NUM_STREAMS
            next_obs_list: List[Optional[np.ndarray]] = [None] * NUM_STREAMS
            reset_stream_idx: List[int] = []

            for i, env in enumerate(envs):
                if i < n_quarter:
                    # Agent (Blue p0) vs Greedy (Red p1)
                    s0 = STREAM_IDX_LOOKUP[(i, 0)]
                    act0 = build_stream_act(s0)
                    act1 = build_noisy_greedy_act(env.p1, env.p0)
                    (o0, _), (r0, _), done, info = env.step(act0, act1)
                    if done:
                        completed_matches += 1
                        o0, _ = env.reset()
                        recent_win.append(1.0 if info["p0_kills"] > info["p1_kills"] else 0.0)
                        recent_trunc.append(1.0 if info["match_time"] >= env.max_match_time else 0.0)
                    ep_ret[s0] += r0
                    ep_len[s0] += 1
                    if done:
                        recent_ret.append(ep_ret[s0])
                        recent_len.append(ep_len[s0])
                        ep_ret[s0] = 0.0
                        ep_len[s0] = 0
                        reset_stream_idx.append(s0)
                    step_rewards[s0] = r0
                    step_dones[s0] = 1.0 if done else 0.0
                    next_obs_list[s0] = o0

                elif i < 2 * n_quarter:
                    # Greedy (Blue p0) vs Agent (Red p1)
                    s1 = STREAM_IDX_LOOKUP[(i, 1)]
                    act0 = build_noisy_greedy_act(env.p0, env.p1)
                    act1 = build_stream_act(s1)
                    (_, o1), (_, r1), done, info = env.step(act0, act1)
                    if done:
                        completed_matches += 1
                        _, o1 = env.reset()
                        recent_win.append(1.0 if info["p1_kills"] > info["p0_kills"] else 0.0)
                        recent_trunc.append(1.0 if info["match_time"] >= env.max_match_time else 0.0)
                    ep_ret[s1] += r1
                    ep_len[s1] += 1
                    if done:
                        recent_ret.append(ep_ret[s1])
                        recent_len.append(ep_len[s1])
                        ep_ret[s1] = 0.0
                        ep_len[s1] = 0
                        reset_stream_idx.append(s1)
                    step_rewards[s1] = r1
                    step_dones[s1] = 1.0 if done else 0.0
                    next_obs_list[s1] = o1

                else:
                    # Self-Play: Agent (Blue p0) vs Agent (Red p1)
                    s0 = STREAM_IDX_LOOKUP[(i, 0)]
                    s1 = STREAM_IDX_LOOKUP[(i, 1)]
                    act0 = build_stream_act(s0)
                    act1 = build_stream_act(s1)
                    (o0, o1), (r0, r1), done, info = env.step(act0, act1)
                    if done:
                        completed_matches += 1
                        o0, o1 = env.reset()
                        recent_trunc.append(1.0 if info["match_time"] >= env.max_match_time else 0.0)
                    for s_idx, r_val, o_val in ((s0, r0, o0), (s1, r1, o1)):
                        ep_ret[s_idx] += r_val
                        ep_len[s_idx] += 1
                        if done:
                            recent_ret.append(ep_ret[s_idx])
                            recent_len.append(ep_len[s_idx])
                            ep_ret[s_idx] = 0.0
                            ep_len[s_idx] = 0
                            reset_stream_idx.append(s_idx)
                        step_rewards[s_idx] = r_val
                        step_dones[s_idx] = 1.0 if done else 0.0
                        next_obs_list[s_idx] = o_val

            if reset_stream_idx:
                next_h[:, reset_stream_idx, :] = 0.0

            rew_buf[step] = torch.tensor(step_rewards, dtype=torch.float32, device=DEVICE)
            next_obs = torch.tensor(np.array(next_obs_list), dtype=torch.float32, device=DEVICE)
            next_done = torch.tensor(step_dones, dtype=torch.float32, device=DEVICE)

        # ---------------------------------------------------------- GAE
        with torch.no_grad():
            _, _, _, next_val, _ = policy.forward(next_obs.unsqueeze(1), next_h)
            next_val = next_val.view(-1)

        advantages = torch.zeros_like(rew_buf)
        lastgaelam = 0.0
        for t in reversed(range(ROLLOUT_STEPS)):
            if t == ROLLOUT_STEPS - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_val
            else:
                nextnonterminal = 1.0 - done_buf[t + 1]
                nextvalues = val_buf[t + 1]
            delta = rew_buf[t] + GAMMA * nextvalues * nextnonterminal - val_buf[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns = advantages + val_buf

        # ----------------------------------------------- batch -> chunks
        policy.train()
        n_chunks = ROLLOUT_STEPS // CHUNK_LEN
        total_chunks = n_chunks * NUM_STREAMS

        def to_chunks(x: torch.Tensor) -> torch.Tensor:
            tail = x.shape[2:]
            x = x.view(n_chunks, CHUNK_LEN, NUM_STREAMS, *tail)
            perm = (0, 2, 1) + tuple(range(3, 3 + len(tail)))
            return x.permute(*perm).reshape(total_chunks, CHUNK_LEN, *tail)

        b_obs = to_chunks(obs_buf)
        b_c_act = to_chunks(c_act_buf)
        b_m_act = to_chunks(m_act_buf)
        b_coord_act = to_chunks(coord_act_buf)
        b_c_mask = to_chunks(c_mask_buf)
        b_m_mask = to_chunks(m_mask_buf)
        b_logp = to_chunks(logprob_buf)
        b_adv = to_chunks(advantages)
        b_ret = to_chunks(returns)
        b_done = to_chunks(done_buf)
        b_hin = h_buf[0::CHUNK_LEN].permute(1, 0, 2, 3).reshape(1, total_chunks, HIDDEN)

        chunk_indices = np.arange(total_chunks)
        minibatch_chunks = max(1, total_chunks // 4)

        pg_l = v_l = ent_l = kl_l = clipf = 0.0
        n_mb = 0
        for _ in range(UPDATE_EPOCHS):
            rng.shuffle(chunk_indices)
            for start in range(0, total_chunks, minibatch_chunks):
                mb = torch.as_tensor(chunk_indices[start:start + minibatch_chunks],
                                     dtype=torch.long, device=DEVICE)

                _, _, _, new_logp, entropy, v_pred, _ = policy.get_action_and_value(
                    b_obs[mb],
                    b_hin[:, mb, :].contiguous(),
                    b_c_mask[mb],
                    b_m_mask[mb],
                    b_c_act[mb],
                    b_m_act[mb],
                    b_coord_act[mb],
                    dones=b_done[mb],
                )

                logratio = new_logp - b_logp[mb]
                ratio = torch.exp(logratio)

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(
                    -mb_adv * ratio,
                    -mb_adv * torch.clamp(ratio, 1.0 - CLIP_COEF, 1.0 + CLIP_COEF)
                ).mean()
                v_loss = 0.5 * ((v_pred - b_ret[mb]) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimizer.step()

                with torch.no_grad():
                    pg_l += pg_loss.item()
                    v_l += v_loss.item()
                    ent_l += entropy.mean().item()
                    kl_l += ((ratio - 1.0) - logratio).mean().item()
                    clipf += ((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()
                    n_mb += 1

        if update % 10 == 0 or update == 1 or (TARGET_MATCHES > 0 and completed_matches >= TARGET_MATCHES):
            mret = float(np.mean(recent_ret)) if recent_ret else float("nan")
            mlen = float(np.mean(recent_len)) if recent_len else float("nan")
            wr = float(np.mean(recent_win)) if recent_win else float("nan")
            tr = float(np.mean(recent_trunc)) if recent_trunc else float("nan")
            sps = total_steps / max(1e-6, time.time() - start_time)
            match_str = f"{completed_matches}/{TARGET_MATCHES}" if TARGET_MATCHES > 0 else f"{completed_matches}"
            print(
                f"[Update {update:04d} | Matches: {match_str}] Steps: {total_steps:,} ({sps:,.0f}/s) | "
                f"EpRet: {mret:.3f} | EpLen: {mlen:.0f} | WinRate: {wr:.2f} | Trunc: {tr:.2f} | "
                f"pg: {pg_l / n_mb:.4f} | v: {v_l / n_mb:.4f} | ent: {ent_l / n_mb:.3f} | "
                f"kl: {kl_l / n_mb:.4f} | clipfrac: {clipf / n_mb:.3f}"
            )

        if update % 20 == 0:
            torch.save(policy.state_dict(), OUT_DIR / "bludrut_ppo_warrior_ckpt.pt")

        if TARGET_MATCHES > 0 and completed_matches >= TARGET_MATCHES:
            print(f"\n[✓] Target {TARGET_MATCHES} matches tercapai ({completed_matches} matches selesai pada update {update}).")
            break

    # 1. Simpan Checkpoint PyTorch
    pt_path = OUT_DIR / "bludrut_ppo_warrior.pt"
    torch.save(policy.state_dict(), pt_path)
    print(f"\n[✓] Checkpoint saved: {pt_path}")

    # 2. Export ke TypeScript Module
    ts_path = OUT_DIR / "weights.ts"
    policy.eval()
    state = policy.state_dict()

    with open(ts_path, "w") as f:
        f.write("/**\n")
        f.write(" * @file weights.ts\n")
        f.write(" * Auto-generated by sim/train_kaggle_bundle.py\n")
        f.write(" * Pure weight matrices for native in-memory inference\n")
        f.write(" */\n\n")
        f.write("export const WEIGHTS: Record<string, any> = {\n")
        for k, v in state.items():
            arr = v.cpu().numpy().tolist()
            f.write(f'  "{k}": {arr},\n')
        f.write("};\n")

    if os.path.exists("/kaggle/working") and OUT_DIR != Path("/kaggle/working"):
        import shutil
        shutil.copy2(pt_path, "/kaggle/working/bludrut_ppo_warrior.pt")
        shutil.copy2(ts_path, "/kaggle/working/weights.ts")

    print(f"[✓] Pure TypeScript weights exported: {ts_path}")


if __name__ == "__main__":
    train()
