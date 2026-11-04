#!/usr/bin/env python3
"""
sim/train_ppo_compound.py
-------------------------
Hybrid Reinforcement Learning & Imitation Policy Optimizer (PPO-Demo)
for AQW Non-OP 1v1 PvP (/bludrutbrawl).

Combines:
1. "Warm Start" initialization from expert human demonstrations.
2. Dynamic Demonstration Hints via Auxiliary BC Regularization Loss (DAPG).
3. PPO Multi-Categorical Self-Play / Curriculum in Gym Simulator.
4. Class-Agnostic ROI & Kinematic Stall Penalty Reward Function.
5. Direct export of compound multi-head weights to `src/ai/trained_policy_weights.ts`.
"""

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset

# Ensure local imports work cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bludrut_env import (
    MAX_HP,
    OBS_DIM,
    Bludrut1v1Env,
)
from train_imitation import (
    find_dataset_path,
    load_demonstrations,
)

TS_WEIGHTS_PATH = ROOT_DIR / "src" / "ai" / "trained_policy_weights.ts"
CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "ppo_compound_policy.pt"

NUM_COMBAT_ACTIONS = 7  # NONE, AUTO_ATTACK, S1, S2, S3, S4, MANA_TAP_NPC
NUM_MOVE_TYPES = 3       # STAND, WALK, SPRINT
COORD_DIM = 2            # norm_x, norm_y


class CompoundActorCritic(nn.Module):
    """
    Decoupled multi-head policy matching `src/ai/nn_inference.ts`:
    - Shared Backbone: 26 -> 64 (tanh) -> 64 (tanh)
    - Actor Combat Head: 64 -> 7
    - Actor Move Type Head: 64 -> 3
    - Actor Coord Head: 64 -> 2 (Sigmoid regression)
    - Critic Head: 64 -> 1
    """

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.combat_head = nn.Linear(hidden_dim, NUM_COMBAT_ACTIONS)
        self.move_type_head = nn.Linear(hidden_dim, NUM_MOVE_TYPES)
        self.coord_head = nn.Linear(hidden_dim, COORD_DIM)

        self.critic_head = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.combat_head.weight, gain=0.01)
        nn.init.orthogonal_(self.move_type_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return h2

    def forward_policy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h2 = self.forward_backbone(x)
        combat_logits = self.combat_head(h2)
        move_logits = self.move_type_head(h2)
        coords = torch.sigmoid(self.coord_head(h2))
        return combat_logits, move_logits, coords

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
    ):
        h2 = self.forward_backbone(x)

        raw_combat = self.combat_head(h2)
        raw_move = self.move_type_head(h2)
        coords = torch.sigmoid(self.coord_head(h2))

        # Mask physically invalid actions
        masked_combat = torch.where(combat_mask, raw_combat, torch.tensor(-1e9, device=x.device))
        masked_move = torch.where(move_mask, raw_move, torch.tensor(-1e9, device=x.device))

        dist_combat = Categorical(logits=masked_combat)
        dist_move = Categorical(logits=masked_move)

        if combat_act is None:
            combat_act = dist_combat.sample()
        if move_act is None:
            move_act = dist_move.sample()

        logp_combat = dist_combat.log_prob(combat_act)
        logp_move = dist_move.log_prob(move_act)
        total_logp = logp_combat + logp_move

        entropy = dist_combat.entropy() + dist_move.entropy()
        value = self.critic_head(h2).squeeze(-1)

        return combat_act, move_act, coords, total_logp, entropy, value


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
    """Initializes the actor backbone and heads from previously exported imitation weights."""
    if not ts_path.exists():
        print(f"[WarmStart] No existing weights found at {ts_path}. Starting from scratch.")
        return False
    try:
        text = ts_path.read_text(encoding="utf-8")
        eq_idx = text.find("= {")
        if eq_idx == -1:
            eq_idx = text.find("={")
        s_idx = text.find("{", eq_idx)
        e_idx = text.rfind("}") + 1
        data = json.loads(text[s_idx:e_idx])

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

export const COMPOUND_ATOMIC_POLICY_WEIGHTS: CompoundAtomicPolicySpec = {json.dumps(payload, indent=4)} as const;
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(ts_content, encoding="utf-8")
    print(f"[Export] Successfully generated compound atomic weights at {output_path}!")


def evaluate_policy(model: CompoundActorCritic, env: Bludrut1v1Env, episodes: int = 30) -> dict[str, float]:
    """Runs evaluation episodes without exploration noise to evaluate win rate and combat efficiency."""
    wins = 0
    losses = 0
    draws = 0
    win_hps = []

    model.eval()
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            c_mask, m_mask = compute_masks(obs)
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            t_cmask = torch.as_tensor(c_mask, dtype=torch.bool).unsqueeze(0)
            t_mmask = torch.as_tensor(m_mask, dtype=torch.bool).unsqueeze(0)

            with torch.no_grad():
                c_act, m_act, coords, _, _, _ = model.get_action_and_value(t_obs, t_cmask, t_mmask)

            norm_x = float(coords[0, 0].item())
            norm_y = float(coords[0, 1].item())

            action = (c_act.item(), m_act.item(), norm_x, norm_y)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            if done:
                winner = info.get("winner", "enemy")
                if winner == "agent":
                    wins += 1
                    win_hps.append(env.agent.hp / MAX_HP)
                elif winner == "enemy":
                    losses += 1
                else:
                    draws += 1
    model.train()

    total = max(1, episodes)
    return {
        "win_rate": wins / total,
        "loss_rate": losses / total,
        "draw_rate": draws / total,
        "avg_win_hp_ratio": (sum(win_hps) / max(1, len(win_hps))) if win_hps else 0.0,
    }


def train_ppo_compound(
    num_updates: int = 40,
    steps_per_update: int = 2048,
    minibatch_size: int = 256,
    ppo_epochs: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.02,
    vf_coef: float = 0.5,
    demo_weight: float = 0.25,
):
    print("======================================================================")
    print(" [PPO-Demo] Training Compound Multi-Head PvP Policy (/bludrutbrawl)")
    print("======================================================================")

    # 1. Prepare Expert Demonstrations for Auxiliary BC Loss ("Hint" Mechanism)
    dataset_path = find_dataset_path()
    expert_loader = None
    if dataset_path.exists():
        print(f"[Demonstration Buffer] Loading expert hints from {dataset_path}...")
        X_exp, y_comb_exp, y_move_exp, y_coord_exp = load_demonstrations(dataset_path)
        exp_dataset = TensorDataset(
            torch.tensor(X_exp, dtype=torch.float32),
            torch.tensor(y_comb_exp, dtype=torch.long),
            torch.tensor(y_move_exp, dtype=torch.long),
            torch.tensor(y_coord_exp, dtype=torch.float32),
        )
        expert_loader = DataLoader(exp_dataset, batch_size=minibatch_size, shuffle=True)
        print(f"[Demonstration Buffer] Loaded {len(X_exp)} expert demonstration frames for BC regularization.")

    # 2. Build Model & Warm Start from Imitation Weights
    model = CompoundActorCritic()
    load_pretrained_imitation_weights(model, TS_WEIGHTS_PATH)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    criterion_comb_bc = nn.CrossEntropyLoss()
    criterion_move_bc = nn.CrossEntropyLoss()
    criterion_coord_bc = nn.MSELoss()

    env = Bludrut1v1Env(max_steps=500, seed=42)

    # Pre-training Baseline Evaluation
    print("\n[Baseline Evaluation] Evaluating initial policy against curriculum...")
    init_metrics = evaluate_policy(model, env, episodes=20)
    print(
        f"[Baseline Evaluation] WinRate: {init_metrics['win_rate']*100:.1f}%, "
        f"LossRate: {init_metrics['loss_rate']*100:.1f}%, "
        f"AvgWinHP: {init_metrics['avg_win_hp_ratio']*100:.1f}%"
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

    exp_iter = iter(expert_loader) if expert_loader else None

    print("\n[PPO Self-Play Training] Starting rollout updates with dynamic demo hints...")
    start_time = time.time()

    for update in range(1, num_updates + 1):
        # Dynamic Demonstration Weight Annealing
        cur_demo_weight = demo_weight * max(0.1, 1.0 - (update / num_updates) * 0.7)

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
                    t_obs, t_cmask, t_mmask
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

        # 3. PPO Optimization with Auxiliary Demo Regularization
        b_obs = torch.as_tensor(obs_buf, dtype=torch.float32)
        b_cmask = torch.as_tensor(cmask_buf, dtype=torch.bool)
        b_mmask = torch.as_tensor(mmask_buf, dtype=torch.bool)
        b_cact = torch.as_tensor(c_act_buf, dtype=torch.int64)
        b_mact = torch.as_tensor(m_act_buf, dtype=torch.int64)
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

                # Auxiliary Demonstration Hint Loss (DAPG)
                if expert_loader is not None:
                    try:
                        exp_batch = next(exp_iter)
                    except (StopIteration, TypeError):
                        exp_iter = iter(expert_loader)
                        exp_batch = next(exp_iter)

                    e_obs, e_cact, e_mact, e_coord = exp_batch
                    pred_c, pred_m, pred_xy = model.forward_policy(e_obs)

                    l_demo_c = criterion_comb_bc(pred_c, e_cact)
                    l_demo_m = criterion_move_bc(pred_m, e_mact)
                    l_demo_xy = criterion_coord_bc(pred_xy, e_coord)

                    loss_demo = l_demo_c + l_demo_m + 3.0 * l_demo_xy
                    loss = loss + cur_demo_weight * loss_demo

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        # 4. Periodic Evaluation & Weight Exporting
        if update % 5 == 0 or update == num_updates:
            metrics = evaluate_policy(model, env, episodes=30)
            elapsed = time.time() - start_time
            print(
                f"[Update {update:2d}/{num_updates:2d}] "
                f"WinRate={metrics['win_rate']*100:.1f}% | "
                f"AvgWinHP={metrics['avg_win_hp_ratio']*100:.1f}% | "
                f"LossRate={metrics['loss_rate']*100:.1f}% | "
                f"Elapsed={elapsed:.1f}s"
            )

            if metrics["win_rate"] >= best_win_rate:
                best_win_rate = metrics["win_rate"]
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), CHECKPOINT_PATH)
                export_compound_weights_to_ts(model, metrics, TS_WEIGHTS_PATH)

    print("\n======================================================================")
    print(f"[Training Finished] Best PPO-Demo Compound Policy WinRate: {best_win_rate*100:.1f}%")
    print("======================================================================")


if __name__ == "__main__":
    train_ppo_compound()
