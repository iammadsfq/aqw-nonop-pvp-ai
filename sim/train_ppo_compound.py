#!/usr/bin/env python3
"""
Compound Multi-Head PPO Training Engine for AQW Non-OP 1v1 PvP (/bludrutbrawl).
Optimizes:
1. Combat Action: Categorical(7 logits) with physical range/CD masking
2. Movement Type: Categorical(3 logits: STAND, WALK, SPRINT) with SP/Stun masking
3. Continuous Coordinates: Gaussian distribution N(mu, sigma) with PPO policy gradient
Combined with Class-Agnostic ROI Reward and live streaming to port 8088.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader, TensorDataset

# Ensure local imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bludrut_env import (
    MAX_HP,
    OBS_DIM,
    SCREEN_BY_CELL,
    Bludrut1v1Env,
)
from train_imitation import (
    find_dataset_path,
    load_demonstrations,
)
from visualizer_server import (
    start_visualizer_server,
    update_training_status,
    record_eval_match,
)

TS_WEIGHTS_PATH = ROOT_DIR / "src" / "ai" / "trained_policy_weights.ts"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "ppo_compound_policy.pt"

NUM_COMBAT_ACTIONS = 7  # NONE, AUTO_ATTACK, S1, S2, S3, S4, MANA_TAP_NPC
NUM_MOVE_TYPES = 3       # STAND, WALK, SPRINT
COORD_DIM = 2            # norm_x, norm_y

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


class CompoundActorCritic(nn.Module):
    """
    Decoupled multi-head policy matching `src/ai/nn_inference.ts`:
    - Shared Backbone: 26 -> 64 (tanh) -> 64 (tanh)
    - Actor Combat Head: 64 -> 7
    - Actor Move Type Head: 64 -> 3
    - Actor Coord Head: 64 -> 2 (Continuous Gaussian distribution over [0, 1]^2)
    - Critic Head: 64 -> 1
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.combat_head = nn.Linear(hidden_dim, NUM_COMBAT_ACTIONS)
        self.move_type_head = nn.Linear(hidden_dim, NUM_MOVE_TYPES)
        self.coord_head = nn.Linear(hidden_dim, COORD_DIM)
        # Learnable log standard deviation for continuous spatial target exploration
        self.coord_log_std = nn.Parameter(torch.ones(COORD_DIM) * -1.2)

        self.critic_head = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.combat_head.weight, gain=0.01)
        nn.init.orthogonal_(self.move_type_head.weight, gain=0.01)
        nn.init.orthogonal_(self.coord_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return h2

    def forward_policy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h2 = self.forward_backbone(x)
        combat_logits = self.combat_head(h2)
        move_logits = self.move_type_head(h2)
        coords_mu = torch.sigmoid(self.coord_head(h2))
        return combat_logits, move_logits, coords_mu

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        h2 = self.forward_backbone(x)
        return self.critic_head(h2).squeeze(-1)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        combat_mask: torch.Tensor,
        move_mask: torch.Tensor,
        combat_act: torch.Tensor | None = None,
        move_act: torch.Tensor | None = None,
        coords_act: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        h2 = self.forward_backbone(x)

        raw_combat = self.combat_head(h2)
        raw_move = self.move_type_head(h2)
        coords_mu = torch.sigmoid(self.coord_head(h2))

        # Mask physically invalid actions
        masked_combat = torch.where(combat_mask, raw_combat, torch.tensor(-1e9, device=x.device))
        masked_move = torch.where(move_mask, raw_move, torch.tensor(-1e9, device=x.device))

        dist_combat = Categorical(logits=masked_combat)
        dist_move = Categorical(logits=masked_move)

        coord_std = self.coord_log_std.exp().expand_as(coords_mu)
        dist_coord = Normal(coords_mu, coord_std)

        if deterministic:
            c_chosen = masked_combat.argmax(dim=-1)
            m_chosen = masked_move.argmax(dim=-1)
            coords_chosen = coords_mu
        else:
            c_chosen = dist_combat.sample() if combat_act is None else combat_act
            m_chosen = dist_move.sample() if move_act is None else move_act
            coords_chosen = (
                torch.clamp(dist_coord.sample(), 0.0, 1.0)
                if coords_act is None
                else coords_act
            )

        logp_combat = dist_combat.log_prob(c_chosen)
        logp_move = dist_move.log_prob(m_chosen)
        logp_coord = dist_coord.log_prob(coords_chosen).sum(dim=-1)

        total_logp = logp_combat + logp_move + logp_coord

        entropy = (
            dist_combat.entropy()
            + dist_move.entropy()
            + 0.1 * dist_coord.entropy().sum(dim=-1)
        )
        value = self.critic_head(h2).squeeze(-1)

        return c_chosen, m_chosen, coords_chosen, total_logp, entropy, value


def compute_masks(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes boolean physical action masks for combat (shape 7) and move (shape 3).
    Only masks physically on-cooldown / out-of-range skills and insufficient stamina.
    """
    combat_mask = np.ones(NUM_COMBAT_ACTIONS, dtype=bool)
    move_mask = np.ones(NUM_MOVE_TYPES, dtype=bool)

    in_melee = obs[23] > 0.5
    s1_ready = obs[10] > 0.5
    s2_ready = obs[11] > 0.5
    s3_ready = obs[12] > 0.5
    s4_ready = obs[13] > 0.5
    has_safe_npc = obs[25] > 0.0
    can_sprint = obs[3] > 0.5
    stunned = obs[19] > 0.5

    # Combat masks:
    if not in_melee:
        combat_mask[2] = False  # S1 requires melee
        combat_mask[3] = False  # S2 requires melee
        combat_mask[5] = False  # S4 requires melee
    if not s1_ready:
        combat_mask[2] = False
    if not s2_ready:
        combat_mask[3] = False
    if not s3_ready:
        combat_mask[4] = False
    if not s4_ready:
        combat_mask[5] = False
    if not has_safe_npc:
        combat_mask[6] = False

    # Move masks:
    if not can_sprint:
        move_mask[2] = False
    if stunned:
        move_mask[1] = False
        move_mask[2] = False

    return combat_mask, move_mask


def load_pretrained_imitation_weights(model: CompoundActorCritic, ts_path: Path) -> bool:
    """Warm-starts model weights from `src/ai/trained_policy_weights.ts` if available."""
    if not ts_path.exists():
        return False

    try:
        content = ts_path.read_text(encoding="utf-8")
        json_start = content.find("= {")
        if json_start == -1:
            return False
        json_raw = content[json_start + 2 :].strip()
        last_brace = json_raw.rfind("}")
        if last_brace == -1:
            return False
        json_str = json_raw[: last_brace + 1]

        data = json.loads(json_str)

        with torch.no_grad():
            model.fc1.weight.copy_(torch.tensor(data["w1"], dtype=torch.float32))
            model.fc1.bias.copy_(torch.tensor(data["b1"], dtype=torch.float32))
            model.fc2.weight.copy_(torch.tensor(data["w2"], dtype=torch.float32))
            model.fc2.bias.copy_(torch.tensor(data["b2"], dtype=torch.float32))
            model.combat_head.weight.copy_(torch.tensor(data["wCombat"], dtype=torch.float32))
            model.combat_head.bias.copy_(torch.tensor(data["bCombat"], dtype=torch.float32))
            model.move_type_head.weight.copy_(torch.tensor(data["wMoveType"], dtype=torch.float32))
            model.move_type_head.bias.copy_(torch.tensor(data["bMoveType"], dtype=torch.float32))
            model.coord_head.weight.copy_(torch.tensor(data["wCoord"], dtype=torch.float32))
            model.coord_head.bias.copy_(torch.tensor(data["bCoord"], dtype=torch.float32))

        print(f"[WarmStart] Successfully transferred pretrained imitation weights from {ts_path}!")
        return True
    except Exception as e:
        print(f"[WarmStart WARNING] Could not parse existing weights: {e}")
        return False


def export_compound_weights_to_ts(
    model: CompoundActorCritic,
    metrics: dict[str, float],
    output_path: Path,
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

    payload = {
        "className": "Battle Warrior",
        "obsDim": OBS_DIM,
        "hiddenDim": 64,
        "combatActions": [
            "NONE",
            "AUTO_ATTACK",
            "USE_SKILL_1",
            "USE_SKILL_2",
            "USE_SKILL_3",
            "USE_SKILL_4",
            "MANA_TAP_NPC",
        ],
        "moveTypes": [
            "STAND",
            "WALK",
            "SPRINT",
        ],
        "metrics": metrics,
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
    }

    ts_content = f"""/**
 * @file trained_policy_weights.ts
 * @description PPO-Demo Hybrid Policy for AQW Non-OP 1v1 PvP (/bludrutbrawl).
 * Pre-trained on human pro-player demonstrations and optimized via PPO Self-Play.
 *
 * Architecture:
 * - Shared Backbone: 26 -> 64 (tanh) -> 64 (tanh)
 * - Head 1 (Combat Action, 7 classes): NONE, AUTO_ATTACK, SKILL_1..4, MANA_TAP_NPC
 * - Head 2 (Movement Type, 3 classes): STAND, WALK, SPRINT
 * - Head 3 (Target Point Regression, 2 coords): Normalized (X/960, Y/550) via Sigmoid
 *
 * Training Metrics:
 * - Win Rate vs Curriculum: {metrics.get('win_rate', 0.0) * 100:.1f}%
 * - Average Win HP: {metrics.get('avg_win_hp_ratio', 0.0) * 100:.1f}%
 */

export interface CompoundAtomicPolicySpec {{
    className: string;
    obsDim: number;
    hiddenDim: number;
    combatActions: readonly string[];
    moveTypes: readonly string[];
    metrics: {{
        combat_accuracy?: number;
        move_accuracy?: number;
        coord_mse?: number;
        win_rate?: number;
        avg_win_hp_ratio?: number;
        loss_rate?: number;
        [key: string]: number | undefined;
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

export const COMPOUND_ATOMIC_POLICY_WEIGHTS: CompoundAtomicPolicySpec = {json.dumps(payload, indent=4)};
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ts_content, encoding="utf-8")
    print(f"[Export] Successfully generated compound atomic weights at {output_path}!")


def evaluate_and_record_match(
    model: CompoundActorCritic,
    env: Bludrut1v1Env,
    update_num: int,
    episodes: int = 30,
) -> dict[str, float]:
    """
    Evaluates policy across `episodes` and records frame-by-frame telemetry
    of Match #1 for live broadcast to the 2D visualizer.
    """
    wins = 0
    losses = 0
    draws = 0
    win_hps = []

    model.eval()
    match_frames = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        record_this = ep == 0
        step_idx = 0

        while not done:
            step_idx += 1
            c_mask, m_mask = compute_masks(obs)
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            t_cmask = torch.as_tensor(c_mask, dtype=torch.bool).unsqueeze(0)
            t_mmask = torch.as_tensor(m_mask, dtype=torch.bool).unsqueeze(0)

            with torch.no_grad():
                c_act, m_act, coords, _, _, _ = model.get_action_and_value(
                    t_obs, t_cmask, t_mmask, deterministic=True
                )

            norm_x = float(coords[0, 0].item())
            norm_y = float(coords[0, 1].item())
            target_px_x = int(norm_x * 960)
            target_px_y = int(norm_y * 520)

            c_act_name = COMBAT_ACTION_NAMES[c_act.item()]
            m_act_name = MOVE_TYPE_NAMES[m_act.item()]

            prev_agent_cell = env.agent.cell
            prev_enemy_cell = env.enemy.cell
            prev_opp_hp = env.enemy.hp
            prev_ag_hp = env.agent.hp

            action = (c_act.item(), m_act.item(), norm_x, norm_y)
            next_obs, _, term, trunc, info = env.step(action)
            done = term or trunc

            if record_this:
                dmg_to_opp = max(0.0, prev_opp_hp - env.enemy.hp)
                dmg_to_ag = max(0.0, prev_ag_hp - env.agent.hp)

                log_msg = ""
                if prev_agent_cell != env.agent.cell:
                    alias = SCREEN_BY_CELL[env.agent.cell].alias if env.agent.cell in SCREEN_BY_CELL else env.agent.cell
                    log_msg = f"🚪 Blue crossed door into {env.agent.cell} ({alias})"
                elif prev_enemy_cell != env.enemy.cell:
                    alias = SCREEN_BY_CELL[env.enemy.cell].alias if env.enemy.cell in SCREEN_BY_CELL else env.enemy.cell
                    log_msg = f"🚪 Red crossed door into {env.enemy.cell} ({alias})"
                elif c_act.item() == 2:
                    log_msg = f"Blue CAST Decisive Strike (S1) -> dealt {dmg_to_opp:.0f} dmg"
                elif c_act.item() == 3:
                    log_msg = f"Blue CAST Imbalancing Strike (S2 Stun 4s) -> Opponent STUNNED!"
                elif c_act.item() == 4:
                    log_msg = f"Blue CAST Prepared Strike (S3) -> Crit buff activated"
                elif c_act.item() == 5:
                    log_msg = f"Blue CAST On Guard (S4 Shield) -> -50% Barrier activated!"
                elif c_act.item() == 1 and dmg_to_opp > 0:
                    log_msg = f"Blue AUTO ATTACK -> dealt {dmg_to_opp:.0f} dmg"
                elif dmg_to_ag > 0:
                    log_msg = f"Red attacked Blue -> dealt {dmg_to_ag:.0f} dmg"

                agent_alias = SCREEN_BY_CELL[env.agent.cell].alias if env.agent.cell in SCREEN_BY_CELL else env.agent.cell
                enemy_alias = SCREEN_BY_CELL[env.enemy.cell].alias if env.enemy.cell in SCREEN_BY_CELL else env.enemy.cell

                match_frames.append({
                    "t": round(step_idx * 0.1, 2),
                    "room": env.agent.cell,
                    "room_alias": agent_alias,
                    "enemy_room": env.enemy.cell,
                    "enemy_room_alias": enemy_alias,
                    "agent": {
                        "x": round(env.agent.x, 1),
                        "y": round(env.agent.y, 1),
                        "hp": round(env.agent.hp, 1),
                        "max_hp": env.agent.max_hp,
                        "mp": round(env.agent.mp, 1),
                        "max_mp": env.agent.max_mp,
                        "sp": round(env.agent.sp, 1),
                        "action": c_act_name,
                        "move": m_act_name,
                        "target_x": target_px_x,
                        "target_y": target_px_y,
                        "stun": round(env.agent.stun_rem, 1),
                        "on_guard": round(env.agent.on_guard_rem, 1),
                        "cds": [0.0] + [round(env.agent.cooldowns[i], 1) for i in range(1, 5)],
                    },
                    "enemy": {
                        "x": round(env.enemy.x, 1),
                        "y": round(env.enemy.y, 1),
                        "hp": round(env.enemy.hp, 1),
                        "max_hp": env.enemy.max_hp,
                        "mp": round(env.enemy.mp, 1),
                        "max_mp": env.enemy.max_mp,
                        "sp": round(env.enemy.sp, 1),
                        "action": "AUTO_ATTACK" if env.enemy.in_combat else "NONE",
                        "move": "STAND",
                        "stun": round(env.enemy.stun_rem, 1),
                        "on_guard": round(env.enemy.on_guard_rem, 1),
                    },
                    "log": log_msg,
                })

            if done:
                winner = info.get("winner", "enemy")
                if winner == "agent":
                    wins += 1
                    win_hps.append(env.agent.hp / MAX_HP)
                elif winner == "enemy":
                    losses += 1
                else:
                    draws += 1
            else:
                obs = next_obs

    model.train()
    total = max(1, episodes)
    metrics = {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "avg_win_hp_ratio": (sum(win_hps) / max(1, len(win_hps))) if win_hps else 0.0,
    }

    winner_str = "Blue Agent" if wins > losses else ("Red Bot" if losses > wins else "Draw")
    record_eval_match({
        "update": update_num,
        "winner": winner_str,
        "total_frames": len(match_frames),
        "frames": match_frames,
    })

    return metrics


def train_ppo_compound(
    num_updates: int = 60,
    steps_per_update: int = 2048,
    minibatch_size: int = 256,
    ppo_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.03,
    vf_coef: float = 0.5,
):
    print("======================================================================")
    print(" [PPO Continuous] Training Compound PvP Policy (/bludrutbrawl)")
    print("======================================================================")

    # 1. Start Live Telemetry Visualizer Server
    start_visualizer_server(port=8088)

    # 2. Build Model & Warm Start from Imitation Weights
    model = CompoundActorCritic()
    load_pretrained_imitation_weights(model, TS_WEIGHTS_PATH)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    env = Bludrut1v1Env(max_steps=600, seed=42)

    # Pre-training Baseline Evaluation
    print("\n[Baseline Evaluation] Evaluating initial policy against curriculum...")
    init_metrics = evaluate_and_record_match(model, env, update_num=0, episodes=20)
    print(
        f"[Baseline Evaluation] WinRate: {init_metrics['win_rate']*100:.1f}%, "
        f"LossRate: {init_metrics['loss_rate']*100:.1f}%, "
        f"AvgWinHP: {init_metrics['avg_win_hp_ratio']*100:.1f}%"
    )

    update_training_status(
        update=0,
        total_updates=num_updates,
        step=0,
        total_steps=num_updates * steps_per_update,
        metrics=init_metrics,
        status="training",
    )

    best_win_rate = init_metrics["win_rate"]

    obs_buf = np.zeros((steps_per_update, OBS_DIM), dtype=np.float32)
    cmask_buf = np.ones((steps_per_update, NUM_COMBAT_ACTIONS), dtype=bool)
    mmask_buf = np.ones((steps_per_update, NUM_MOVE_TYPES), dtype=bool)
    c_act_buf = np.zeros(steps_per_update, dtype=np.int64)
    m_act_buf = np.zeros(steps_per_update, dtype=np.int64)
    coord_buf = np.zeros((steps_per_update, COORD_DIM), dtype=np.float32)
    logp_buf = np.zeros(steps_per_update, dtype=np.float32)
    rew_buf = np.zeros(steps_per_update, dtype=np.float32)
    done_buf = np.zeros(steps_per_update, dtype=np.float32)
    val_buf = np.zeros(steps_per_update, dtype=np.float32)

    obs, _ = env.reset(seed=42)
    done = False

    print("\n[PPO Continuous Training] Starting rollout updates with live visualizer streaming...")
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # 1. Rollout Collection
        for step in range(steps_per_update):
            c_mask, m_mask = compute_masks(obs)

            obs_buf[step] = obs
            cmask_buf[step] = c_mask
            mmask_buf[step] = m_mask
            done_buf[step] = float(done)

            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            t_cmask = torch.as_tensor(c_mask, dtype=torch.bool).unsqueeze(0)
            t_mmask = torch.as_tensor(m_mask, dtype=torch.bool).unsqueeze(0)

            with torch.no_grad():
                c_act_t, m_act_t, coords_t, logp_t, _, val_t = model.get_action_and_value(
                    t_obs, t_cmask, t_mmask, deterministic=False
                )

            c_act_buf[step] = c_act_t.item()
            m_act_buf[step] = m_act_t.item()
            coord_buf[step] = coords_t.squeeze(0).numpy()
            logp_buf[step] = logp_t.item()
            val_buf[step] = val_t.item()

            action = (c_act_t.item(), m_act_t.item(), coords_t[0, 0].item(), coords_t[0, 1].item())
            next_obs, reward, term, trunc, _ = env.step(action)

            rew_buf[step] = reward
            done = term or trunc
            if done:
                obs, _ = env.reset()
                done = False
            else:
                obs = next_obs

        # 2. GAE Computation
        with torch.no_grad():
            next_val = float(
                model.forward_critic(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).item()
            )

        adv_buf = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(steps_per_update)):
            if t == steps_per_update - 1:
                next_non_terminal = 1.0 - float(done)
                next_values = next_val
            else:
                next_non_terminal = 1.0 - done_buf[t + 1]
                next_values = val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_values * next_non_terminal - val_buf[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        # 3. PPO Optimization (Direct Continuous Spatial Policy Gradient)
        b_obs = torch.as_tensor(obs_buf, dtype=torch.float32)
        b_cmask = torch.as_tensor(cmask_buf, dtype=torch.bool)
        b_mmask = torch.as_tensor(mmask_buf, dtype=torch.bool)
        b_cact = torch.as_tensor(c_act_buf, dtype=torch.int64)
        b_mact = torch.as_tensor(m_act_buf, dtype=torch.int64)
        b_coord = torch.as_tensor(coord_buf, dtype=torch.float32)
        b_logp = torch.as_tensor(logp_buf, dtype=torch.float32)
        b_adv = torch.as_tensor(adv_buf, dtype=torch.float32)
        b_ret = torch.as_tensor(ret_buf, dtype=torch.float32)

        inds = np.arange(steps_per_update)

        for _ in range(ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, steps_per_update, minibatch_size):
                mb_inds = inds[start : start + minibatch_size]

                _, _, _, new_logp, entropy, new_val = model.get_action_and_value(
                    b_obs[mb_inds],
                    b_cmask[mb_inds],
                    b_mmask[mb_inds],
                    b_cact[mb_inds],
                    b_mact[mb_inds],
                    b_coord[mb_inds],
                    deterministic=False,
                )

                logratio = new_logp - b_logp[mb_inds]
                ratio = logratio.exp()

                mb_adv = b_adv[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((new_val - b_ret[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # 4. Periodic Evaluation, Visualizer Broadcast, & Weight Exporting
        if update % 5 == 0 or update == num_updates:
            metrics = evaluate_and_record_match(model, env, update_num=update, episodes=30)
            elapsed = time.time() - start_time
            print(
                f"[Update {update:2d}/{num_updates:2d}] "
                f"WinRate={metrics['win_rate']*100:.1f}% | "
                f"AvgWinHP={metrics['avg_win_hp_ratio']*100:.1f}% | "
                f"LossRate={metrics['loss_rate']*100:.1f}% | "
                f"Elapsed={elapsed:.1f}s"
            )

            update_training_status(
                update=update,
                total_updates=num_updates,
                step=update * steps_per_update,
                total_steps=num_updates * steps_per_update,
                metrics=metrics,
                status="finished" if update == num_updates else "training",
            )

            if metrics["win_rate"] >= best_win_rate:
                best_win_rate = metrics["win_rate"]
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                export_compound_weights_to_ts(model, metrics, TS_WEIGHTS_PATH)

    print("\n======================================================================")
    print(f"[Training Finished] Best PPO Continuous WinRate: {best_win_rate*100:.1f}%")
    print("Visualizer server remains active on http://localhost:8088.")
    print("======================================================================")


if __name__ == "__main__":
    train_ppo_compound()
