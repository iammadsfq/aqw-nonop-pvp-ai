#!/usr/bin/env python3
"""
PPO Self-Play + Tactical Curriculum Trainer for `Battle Warrior` 1v1 in `/bludrutbrawl`.

Trains an Actor-Critic Neural Network (`26 -> 64 -> 64 -> 14` with `Tanh` activations)
on `Bludrut1v1Env` (`sim/bludrut_env.py`) and exports the best trained policy weights
directly to TypeScript (`src/ai/trained_policy_weights.ts`) for sub-millisecond
real-time execution inside Turu Headless.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from bludrut_env import (
    ACTION_NAMES,
    NUM_ACTIONS,
    OBS_DIM,
    Bludrut1v1Env,
    compute_action_mask_from_obs,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = Path(__file__).resolve().parent / "data" / "ppo_battle_warrior_1v1.pt"
TS_WEIGHTS_PATH = ROOT_DIR / "src" / "ai" / "trained_policy_weights.ts"


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 64, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.actor_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, num_actions)

        self.critic_fc1 = nn.Linear(obs_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_out.weight, gain=0.01)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.actor_fc1(x))
        h = torch.tanh(self.actor_fc2(h))
        return self.actor_out(h)

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.critic_fc1(x))
        h = torch.tanh(self.critic_fc2(h))
        return self.critic_out(h)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward_actor(x)
        masked_logits = logits.masked_fill(~mask, -1e9)
        dist = Categorical(logits=masked_logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.forward_critic(x).squeeze(-1)

    @torch.no_grad()
    def act_greedy(self, obs_np: np.ndarray) -> int:
        x = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        mask_np = compute_action_mask_from_obs(obs_np)
        mask_t = torch.as_tensor(mask_np, dtype=torch.bool).unsqueeze(0)
        logits = self.forward_actor(x).masked_fill(~mask_t, -1e9)
        return int(torch.argmax(logits, dim=-1).item())


def evaluate_policy(model: ActorCritic, episodes: int = 80, seed: int = 1000) -> dict[str, float]:
    env = Bludrut1v1Env(max_steps=500, opponent_policy=None, seed=seed)
    wins = 0
    losses = 0
    draws = 0
    total_hp_rem = 0.0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        info: dict[str, Any] = {}
        while not done:
            action = model.act_greedy(obs)
            obs, _rew, term, trunc, info = env.step(action)
            done = term or trunc

        winner = info.get("winner")
        if winner == "agent":
            wins += 1
            total_hp_rem += env.agent.hp / env.agent.max_hp
        elif winner == "enemy":
            losses += 1
        else:
            draws += 1

    return {
        "win_rate": wins / episodes,
        "loss_rate": losses / episodes,
        "draw_rate": draws / episodes,
        "avg_win_hp_ratio": (total_hp_rem / max(1, wins)),
    }


def export_weights_to_typescript(model: ActorCritic, metrics: dict[str, float]) -> None:
    TS_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    def to_list(t: torch.Tensor) -> list[Any]:
        return np.round(t.detach().cpu().numpy(), 6).tolist()

    payload = {
        "className": "Battle Warrior",
        "obsDim": OBS_DIM,
        "hiddenDim": 64,
        "numActions": NUM_ACTIONS,
        "actionNames": list(ACTION_NAMES),
        "metrics": metrics,
        "w1": to_list(model.actor_fc1.weight),
        "b1": to_list(model.actor_fc1.bias),
        "w2": to_list(model.actor_fc2.weight),
        "b2": to_list(model.actor_fc2.bias),
        "wOut": to_list(model.actor_out.weight),
        "bOut": to_list(model.actor_out.bias),
    }

    ts_code = f"""/**
 * @file trained_policy_weights.ts
 * @description Auto-generated PPO Actor Neural Network weights for `Battle Warrior` 1v1
 * trained on the 1:1 calibrated `/bludrutbrawl` Gymnasium simulator (`sim/bludrut_env.py`).
 *
 * Evaluation vs Tactical Heuristic Baseline:
 * - Win Rate: {metrics['win_rate'] * 100:.1f}%
 * - Loss Rate: {metrics['loss_rate'] * 100:.1f}%
 * - Avg Remaining HP on Win: {metrics['avg_win_hp_ratio'] * 100:.1f}%
 */

export interface TrainedMlpPolicySpec {{
    className: string;
    obsDim: number;
    hiddenDim: number;
    numActions: number;
    actionNames: readonly string[];
    metrics: {{
        win_rate: number;
        loss_rate: number;
        draw_rate: number;
        avg_win_hp_ratio: number;
    }};
    w1: readonly (readonly number[])[];
    b1: readonly number[];
    w2: readonly (readonly number[])[];
    b2: readonly number[];
    wOut: readonly (readonly number[])[];
    bOut: readonly number[];
}}

export const BATTLE_WARRIOR_1V1_POLICY_WEIGHTS: TrainedMlpPolicySpec = {json.dumps(payload, indent=4)} as const;
"""
    TS_WEIGHTS_PATH.write_text(ts_code, encoding="utf-8")
    print(f"[Export] Exported trained TypeScript policy weights to {TS_WEIGHTS_PATH}")


def train_ppo(
    num_updates: int = 60,
    steps_per_update: int = 2048,
    minibatch_size: int = 256,
    ppo_epochs: int = 4,
    lr: float = 1e-3,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.015,
    vf_coef: float = 0.5,
) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    # League opponent pool (starts with tactical heuristic, adds historical self-play checkpoints)
    opponent_snapshots: list[ActorCritic] = []
    env = Bludrut1v1Env(max_steps=500, seed=42)

    best_win_rate = -1.0
    best_model_state = copy.deepcopy(model.state_dict())

    obs_buf = np.zeros((steps_per_update, OBS_DIM), dtype=np.float32)
    mask_buf = np.ones((steps_per_update, NUM_ACTIONS), dtype=bool)
    act_buf = np.zeros(steps_per_update, dtype=np.int64)
    logp_buf = np.zeros(steps_per_update, dtype=np.float32)
    rew_buf = np.zeros(steps_per_update, dtype=np.float32)
    done_buf = np.zeros(steps_per_update, dtype=np.float32)
    val_buf = np.zeros(steps_per_update, dtype=np.float32)

    obs, _ = env.reset(seed=42)
    done = False
    start_time = time.time()

    print("======================================================================")
    print(" Starting PPO Self-Play Training: Battle Warrior 1v1 (/bludrutbrawl)")
    print("======================================================================")

    for update in range(1, num_updates + 1):
        # Curriculum: 65% against tactical heuristic, 35% against historical self-play pool
        if opponent_snapshots and np.random.random() < 0.35:
            snap = opponent_snapshots[np.random.randint(0, len(opponent_snapshots))]
            env.opponent_policy = snap.act_greedy
        else:
            env.opponent_policy = None

        for step in range(steps_per_update):
            obs_buf[step] = obs
            mask_np = compute_action_mask_from_obs(obs)
            mask_buf[step] = mask_np
            done_buf[step] = float(done)

            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                mask_t = torch.as_tensor(mask_np, dtype=torch.bool).unsqueeze(0)
                action_t, logp_t, _ent_t, val_t = model.get_action_and_value(obs_t, mask_t)

            act = int(action_t.item())
            act_buf[step] = act
            logp_buf[step] = float(logp_t.item())
            val_buf[step] = float(val_t.item())

            next_obs, reward, term, trunc, _info = env.step(act)
            rew_buf[step] = reward
            done = term or trunc
            if done:
                obs, _ = env.reset()
                done = False
            else:
                obs = next_obs

        # Compute GAE advantages
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

        # Optimize policy & value network for `ppo_epochs`
        b_obs = torch.as_tensor(obs_buf, dtype=torch.float32)
        b_mask = torch.as_tensor(mask_buf, dtype=torch.bool)
        b_act = torch.as_tensor(act_buf, dtype=torch.int64)
        b_logp = torch.as_tensor(logp_buf, dtype=torch.float32)
        b_adv = torch.as_tensor(adv_buf, dtype=torch.float32)
        b_ret = torch.as_tensor(ret_buf, dtype=torch.float32)

        inds = np.arange(steps_per_update)
        for _ in range(ppo_epochs):
            np.random.shuffle(inds)
            for start in range(0, steps_per_update, minibatch_size):
                mb_inds = inds[start : start + minibatch_size]
                _, new_logp, entropy, new_val = model.get_action_and_value(
                    b_obs[mb_inds], b_mask[mb_inds], b_act[mb_inds]
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

        if update % 5 == 0 or update == num_updates:
            metrics = evaluate_policy(model, episodes=80, seed=2000 + update * 100)
            elapsed = time.time() - start_time
            print(
                f"[Update {update:02d}/{num_updates}] "
                f"WinRate={metrics['win_rate']*100:5.1f}% | "
                f"LossRate={metrics['loss_rate']*100:5.1f}% | "
                f"AvgWinHP={metrics['avg_win_hp_ratio']*100:5.1f}% | "
                f"Time={elapsed:5.1f}s"
            )
            if metrics["win_rate"] >= best_win_rate:
                best_win_rate = metrics["win_rate"]
                best_model_state = copy.deepcopy(model.state_dict())
                CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_state, CHECKPOINT_PATH)
                export_weights_to_typescript(model, metrics)

            if metrics["win_rate"] >= 0.55 and len(opponent_snapshots) < 8:
                snap_copy = ActorCritic()
                snap_copy.load_state_dict(copy.deepcopy(model.state_dict()))
                opponent_snapshots.append(snap_copy)

    # Restore best model and run final 120-episode evaluation
    model.load_state_dict(best_model_state)
    final_metrics = evaluate_policy(model, episodes=120, seed=99999)
    print("======================================================================")
    print(
        f"[Finished] Best Battle Warrior 1v1 Policy: "
        f"WinRate={final_metrics['win_rate']*100:.1f}%, "
        f"LossRate={final_metrics['loss_rate']*100:.1f}%, "
        f"AvgWinHP={final_metrics['avg_win_hp_ratio']*100:.1f}%"
    )
    print("======================================================================")
    export_weights_to_typescript(model, final_metrics)


if __name__ == "__main__":
    train_ppo()
