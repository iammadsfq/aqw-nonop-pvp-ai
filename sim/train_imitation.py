#!/usr/bin/env python3
"""
sim/train_imitation.py
Behavioral Cloning + Interactive DAgger (Imitation Learning) untuk Bludrut Brawl 1v1.
Menggabungkan:
  1. Ekstraksi & interpolasi lintasan in-transit (10 Hz) dari rekaman bermain nyata
     (aqw_combat_telemetry.log, termasuk match 3-5 terbaru) dengan:
     - Pembersihan otomatis paket tug-of-war bot (plafon y<=258, tarik mundur menjauhi pintu).
     - Deduplikasi berbasis waktu nyata (dt_ms) untuk spam-klik pintu agar kecepatan &
       status sprint/walk tidak terdistorsi.
  2. Rollout sekuensial multi-step di SELURUH 16 ruangan BludrutBrawl (termasuk Captain0/1,
     Crossupper, Resource0/1 A/B, Enter0/1, Morale0/1 C/B/A, Crosslower) agar hidden state
     GRU (128-dim) stabil untuk horizon panjang dan mampu keluar dari ruangan mana pun.
  3. Interactive DAgger (Dataset Aggregation): membiarkan model menyetir sendiri di
     simulator lalu mengoreksi setiap penyimpangan lintasan / hidden state GRU.
  4. Class-Weighted Cross-Entropy Loss agar model tidak bias ke aksi STAND (0).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

CUR_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CUR_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sim.bludrut_env import (
    BludrutBrawlEnv,
    ActionSpec,
    FLOOR_X_MIN,
    FLOOR_X_MAX,
    FLOOR_Y_MIN,
    FLOOR_Y_MAX,
    FLOOR_WIDTH,
    FLOOR_HEIGHT,
    MAX_DISP_PX,
    SCREEN_INDICES,
    SCREEN_NAMES,
)
from sim.train_kaggle_bundle import (
    EntityCompoundActorCritic,
    compute_masks,
    select_greedy_sparring_action,
)


def extract_clean_human_sequences(
    log_path: Path,
    seq_len: int = 32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mengekstrak dan menginterpolasi lintasan gerakan & aksi combat manusia dari
    aqw_combat_telemetry.log menjadi sekuens sepanjang `seq_len` untuk pelatihan GRU.
    Menyaring paket tug-of-war bot (plafon y<=258, arah terbalik dari pintu) dan
    menggabungkan spam-klik pintu menggunakan selisih timestamp nyata (dt).
    """
    if not log_path.exists():
        return (
            np.zeros((0, seq_len, 30), dtype=np.float32),
            np.zeros((0, seq_len), dtype=np.int64),
            np.zeros((0, seq_len), dtype=np.int64),
            np.zeros((0, seq_len, 2), dtype=np.float32),
        )

    raw_records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                raw_records.append(json.loads(raw_line))
            except Exception:
                continue

    # Pre-index timestamp paket movement berikutnya untuk deduplikasi spam-klik & batas durasi interpolasi
    next_move_info = {}
    last_move_idx = None
    for idx, rec in enumerate(raw_records):
        if rec.get("type") == "movement_packet":
            if last_move_idx is not None:
                next_move_info[last_move_idx] = (
                    rec.get("ts", 0),
                    rec.get("cell", ""),
                    rec.get("payload", {}),
                )
            last_move_idx = idx

    skill_map = {"aa": 1, "a1": 2, "a2": 3, "a3": 4, "a4": 5}
    frames_obs: List[List[float]] = []
    frames_c: List[int] = []
    frames_m: List[int] = []
    frames_xy: List[List[float]] = []

    pending_skill = 0
    cur_hp_norm = 1.0
    cur_mp_norm = 1.0
    cur_sp_norm = 1.0
    opp_hp_norm = 1.0
    opp_mp_norm = 1.0
    sprint_rem_sec = 0.0
    last_pkt_ts = 0

    bottom_door_cells = {"Captain0", "Captain1", "Crossupper"}
    right_door_cells = {"Enter0", "Morale0C", "Morale0B", "Morale0A", "Resource0B", "Resource0A"}
    left_door_cells = {"Enter1", "Morale1C", "Morale1B", "Morale1A", "Resource1B", "Resource1A"}

    for idx, rec in enumerate(raw_records):
        rtype = rec.get("type")
        cell = rec.get("cell", "Enter0")
        if cell not in SCREEN_INDICES:
            continue
        cell_idx = float(SCREEN_INDICES[cell]) / 15.0
        payload = rec.get("payload", {})
        cur_ts = int(rec.get("ts", 0))

        if last_pkt_ts > 0 and cur_ts > last_pkt_ts:
            sprint_rem_sec = max(0.0, sprint_rem_sec - (cur_ts - last_pkt_ts) / 1000.0)
        if cur_ts > 0:
            last_pkt_ts = cur_ts

        if rtype == "combat_trace":
            p_dict = payload.get("p", {})
            if isinstance(p_dict, dict):
                for uname, stats in p_dict.items():
                    if not isinstance(stats, dict):
                        continue
                    uname_low = str(uname).lower()
                    if uname_low == "master_ninja":
                        if "intHP" in stats:
                            cur_hp_norm = float(np.clip(stats["intHP"] / 8455.0, 0.0, 1.0))
                        if "intMP" in stats:
                            cur_mp_norm = float(np.clip(stats["intMP"] / 100.0, 0.0, 1.0))
                        if "intSP" in stats:
                            cur_sp_norm = float(np.clip(stats["intSP"] / 100.0, 0.0, 1.0))
                    else:
                        if "intHP" in stats:
                            opp_hp_norm = float(np.clip(stats["intHP"] / 8455.0, 0.0, 1.0))
                        if "intMP" in stats:
                            opp_mp_norm = float(np.clip(stats["intMP"] / 100.0, 0.0, 1.0))
            continue

        if rtype == "player_sent_action":
            pkt = str(payload.get("packet", ""))
            parts = pkt.split("%")
            if len(parts) >= 7:
                act_name = parts[6].split(">")[0]
                pending_skill = skill_map.get(act_name, 0)
            if "selfHp" in payload:
                cur_hp_norm = float(np.clip(float(payload["selfHp"]) / 8455.0, 0.0, 1.0))
            if "selfMp" in payload:
                cur_mp_norm = float(np.clip(float(payload["selfMp"]) / 100.0, 0.0, 1.0))
            if "selfSp" in payload:
                cur_sp_norm = float(np.clip(float(payload["selfSp"]) / 100.0, 0.0, 1.0))
            continue

        if rtype == "movement_packet":
            from_x = float(payload.get("fromX", 0.0))
            from_y = float(payload.get("fromY", 0.0))
            tgt_x = float(payload.get("targetX", from_x))
            tgt_y = float(payload.get("targetY", from_y))
            is_sprint = bool(payload.get("isSprint", False))
            in_combat = bool(payload.get("inCombat", False))
            self_sp = float(payload.get("selfSp", 100.0))
            cur_sp_norm = float(np.clip(self_sp / 100.0, 0.0, 1.0))

            if from_x <= 1.0 or from_y <= 1.0:
                continue

            # 1. Bersihkan paket wall-clamping dari bot lama:
            #    - Plafon atas (tgt_y <= 258.0) adalah 100% artefak clamp WALK_BOUNDS.yMin (255) dari bot lama!
            #    - Tembok kiri/kanan (tgt_x <= 42 atau tgt_x >= 920).
            #    - Lantai bawah (tgt_y >= 490) kecuali di Captain0/1 & Crossupper yang pintunya memang di bawah.
            if tgt_y <= 258.0 or tgt_x <= 42.0 or tgt_x >= 920.0:
                continue
            if cell not in bottom_door_cells and tgt_y >= 490.0:
                continue

            dist = math.hypot(tgt_x - from_x, tgt_y - from_y)
            # Abaikan paket stop-sync Flash (fromX == targetX) saat navigasi
            if dist < 15.0 and not in_combat:
                continue

            # 2. Bersihkan artefak "rebutan kontrol" (tug-of-war) & paket di Crosslower tanpa koordinat lawan:
            #    - Di Crosslower, movement_packet tidak memuat (opp_x, opp_y) sehingga rel_dx=0, rel_dy=0
            #      akan mengaburkan arah gerak. Aksi skill di Crosslower sudah direkam oleh player_sent_action.
            #    - Di luar Crosslower, pastikan hanya mengambil gerakan murni menuju pintu keluar ke arah Mid
            #      (terlepas dari apakah NPC Brawler/Captain membuat inCombat=True).
            if cell == "Crosslower":
                continue
            if cell in right_door_cells:
                # Harus bergerak ke kanan-atas menuju pintu (x >= 720, y <= 315, dy <= 0)
                if tgt_x < 720.0 or tgt_x < from_x + 15.0 or tgt_y > 315.0 or tgt_y > from_y + 2.0:
                    continue
                if cell == "Morale0C" and 395.0 <= tgt_x <= 585.0 and tgt_y <= 310.0:
                    continue
            elif cell in left_door_cells:
                if tgt_x > 240.0 or tgt_x > from_x - 15.0 or tgt_y > 315.0 or tgt_y > from_y + 2.0:
                    continue
                if cell == "Morale1C" and 395.0 <= tgt_x <= 585.0 and tgt_y <= 310.0:
                    continue
            elif cell in bottom_door_cells:
                # Di Captain0/1 & Crossupper, pintu keluar ke Mid ada di bawah (y >= 450).
                # Buang paket bot yang menarik ke atas (tgt_y < 440)!
                if tgt_y < 440.0 or tgt_y < from_y + 10.0:
                    continue

            # 3. Deduplikasi Spam-Klik Pintu berdasarkan selisih waktu nyata (dt_sec):
            max_steps_by_time = 12
            if idx in next_move_info:
                next_ts, next_cell, next_pay = next_move_info[idx]
                dt_sec = max(0.0, (next_ts - cur_ts) / 1000.0)
                next_tx = float(next_pay.get("targetX", 0.0))
                next_ty = float(next_pay.get("targetY", 0.0))
                # Jika klik beruntun di ruangan yang sama dalam < 85ms ke titik yang hampir sama (spam klik),
                # lewati paket antara ini agar tidak menduplikasi frame diam.
                if next_cell == cell and dt_sec < 0.085 and math.hypot(next_tx - tgt_x, next_ty - tgt_y) < 40.0:
                    continue
                if next_cell == cell and dt_sec > 0.0:
                    max_steps_by_time = max(1, min(12, int(round(dt_sec / 0.10))))

            if is_sprint and cur_sp_norm >= 0.50 and sprint_rem_sec <= 0.0:
                sprint_rem_sec = 1.0  # 1x bayar 50 SP = 1.0 detik sprint status

            # Interpolasi in-transit sepanjang lintasan sesuai kecepatan nyata (48 px/tick sprint vs 16 px/tick walk)
            effective_sprint = is_sprint and (cur_sp_norm >= 0.50 or sprint_rem_sec > 0.0)
            step_px = 48.0 if effective_sprint else 16.0
            n_interp = max(1, min(max_steps_by_time, int(dist // step_px) + 1))

            for k in range(n_interp):
                alpha = k / max(1, n_interp)
                cur_x = from_x + alpha * (tgt_x - from_x)
                cur_y = from_y + alpha * (tgt_y - from_y)
                rem_dx = tgt_x - cur_x
                rem_dy = tgt_y - cur_y
                rem_dist = math.hypot(rem_dx, rem_dy)

                if rem_dist < 5.0:
                    m_type = 0
                    ndx, ndy = 0.0, 0.0
                else:
                    m_type = 2 if (effective_sprint and (cur_sp_norm >= 0.50 or sprint_rem_sec > 0.0)) else 1
                    scale = max(1.0, rem_dist / MAX_DISP_PX)
                    ndx = float(np.clip(rem_dx / (MAX_DISP_PX * scale), -1.0, 1.0))
                    ndy = float(np.clip(rem_dy / (MAX_DISP_PX * scale), -1.0, 1.0))

                norm_x = float(np.clip((cur_x - FLOOR_X_MIN) / FLOOR_WIDTH, 0.0, 1.0))
                norm_y = float(np.clip((cur_y - FLOOR_Y_MIN) / FLOOR_HEIGHT, 0.0, 1.0))

                opp_cell_idx = 0.4667 if cell_idx < 0.45 else 0.0
                obs_vec = [
                    cur_hp_norm,
                    cur_mp_norm,
                    cur_sp_norm,
                    1.0 if cur_sp_norm >= 0.5 else 0.0,
                    0.0,
                    1.0,
                    cell_idx,
                    norm_x,
                    norm_y,
                    1.0,
                    1.0, 1.0, 1.0, 1.0,
                    0.50,
                    0.50,
                    0.0,
                    # Opponent slot (13-dim): navigasi keluar pintu menuju Crosslower (same_cell = 0.0)
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    opp_cell_idx,
                    0.0,
                    0.0,
                    0.0,
                    opp_hp_norm,
                    opp_mp_norm,
                    0.50,
                    0.0,
                    0.50,
                ]
                frames_obs.append(obs_vec)
                frames_c.append(pending_skill if k == 0 else 0)
                frames_m.append(m_type)
                frames_xy.append([ndx, ndy])
                sprint_rem_sec = max(0.0, sprint_rem_sec - 0.10)
            pending_skill = 0

    total = (len(frames_obs) // seq_len) * seq_len
    if total == 0:
        return (
            np.zeros((0, seq_len, 30), dtype=np.float32),
            np.zeros((0, seq_len), dtype=np.int64),
            np.zeros((0, seq_len), dtype=np.int64),
            np.zeros((0, seq_len, 2), dtype=np.float32),
        )

    obs_seq = np.array(frames_obs[:total], dtype=np.float32).reshape(-1, seq_len, 30)
    c_seq = np.array(frames_c[:total], dtype=np.int64).reshape(-1, seq_len)
    m_seq = np.array(frames_m[:total], dtype=np.int64).reshape(-1, seq_len)
    xy_seq = np.array(frames_xy[:total], dtype=np.float32).reshape(-1, seq_len, 2)
    return obs_seq, c_seq, m_seq, xy_seq


def mask_unseen_opponent(obs: np.ndarray) -> np.ndarray:
    """
    Mensimulasikan kondisi di client AQW saat lawan belum terdeteksi di memori tracker
    (obs[17..29] bernilai default kosong sesuai build_entity_obs.ts baris 65-79).
    """
    out = obs.copy()
    out[17:27] = 0.0
    out[27] = 0.5
    out[28] = 0.0
    out[29] = 0.5
    return out


def collect_expert_and_dagger_sequences(
    model: Optional[EntityCompoundActorCritic],
    device: torch.device,
    num_episodes: int = 64,
    steps_per_ep: int = 256,
    seq_len: int = 32,
    beta_expert: float = 1.0,
    seed: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mengumpulkan sekuens penuh di SELURUH 16 ruangan BludrutBrawl (Blue & Red side,
    termasuk Captain0/1, Crossupper, Resource0/1 A/B, Enter0/1, Morale0/1 C/B/A, Crosslower).
    Jika `beta_expert < 1.0`, menggunakan kebijakan model saat ini (DAgger) untuk mengeksplorasi
    state & hidden state GRU nyata, sambil merekam label aksi pakar (expert action).
    """
    rng = np.random.default_rng(seed)
    all_obs: List[np.ndarray] = []
    all_c: List[np.ndarray] = []
    all_m: List[np.ndarray] = []
    all_xy: List[np.ndarray] = []

    if model is not None:
        model.eval()

    all_blue_start_cells = [
        "Enter0", "Morale0C", "Morale0B", "Morale0A",
        "Captain0", "Resource0B", "Resource0A", "Crossupper", "Crosslower",
    ]
    all_red_start_cells = [
        "Enter1", "Morale1C", "Morale1B", "Morale1A",
        "Captain1", "Resource1B", "Resource1A", "Crossupper", "Crosslower",
    ]

    for ep in range(num_episodes):
        env = BludrutBrawlEnv(seed=seed + ep, target_kills=10, max_match_time=1200.0)
        o0, o1 = env.reset()

        # Variasikan posisi awal di seluruh 16 ruangan (termasuk area pinggir x=[40..925] jika agen kebablasan pintu)
        if ep % 3 != 0:
            b_cell = all_blue_start_cells[ep % len(all_blue_start_cells)]
            r_cell = all_red_start_cells[(ep + 4) % len(all_red_start_cells)]
            env.p0.cell = b_cell
            env.p0.x = float(rng.uniform(45.0, 920.0))
            b_ymin = 308.0 if (b_cell in ("Morale0C", "Morale1C") and 395.0 <= env.p0.x <= 580.0) else 260.0
            env.p0.y = float(rng.uniform(b_ymin, 450.0))
            env.p1.cell = r_cell
            env.p1.x = float(rng.uniform(45.0, 920.0))
            r_ymin = 308.0 if (r_cell in ("Morale0C", "Morale1C") and 395.0 <= env.p1.x <= 580.0) else 260.0
            env.p1.y = float(rng.uniform(r_ymin, 450.0))
            env.p0.time_since_combat = float(rng.uniform(0.0, 15.0))
            env.p1.time_since_combat = float(rng.uniform(0.0, 15.0))
            o0 = env._get_obs(env.p0, env.p1)
            o1 = env._get_obs(env.p1, env.p0)

        # Simulasikan juga kondisi baru respawn di Enter0/Enter1 sementara lawan masih di
        # Morale0A, Captain0, Resource0A, atau Crossupper (seperti di match 3-5!)
        if ep % 5 == 0:
            env.p0.cell = "Enter0"
            env.p0.x = float(rng.choice([322.5, rng.uniform(750.0, 925.0)]))
            env.p0.y = float(rng.uniform(275.0, 390.0))
            env.p1.cell = SCREEN_NAMES[int(rng.integers(1, 15))]
            env.p1.x = float(rng.uniform(250.0, 700.0))
            env.p1.y = float(rng.uniform(280.0, 420.0))
            o0 = env._get_obs(env.p0, env.p1)
            o1 = env._get_obs(env.p1, env.p0)

        # 35% episode mensimulasikan lawan belum terdeteksi saat masih di koridor sendiri
        simulate_unseen_opp = (ep % 5 in (0, 1))
        opp_mode = ep % 3
        if opp_mode == 1 and ep % 3 == 0:
            env.p1.cell = "Crosslower" if (ep % 2 == 0) else "Enter1"
            env.p1.x = 550.0
            env.p1.y = 350.0
            o0 = env._get_obs(env.p0, env.p1)
            o1 = env._get_obs(env.p1, env.p0)
        elif opp_mode == 2 and ep % 3 == 0:
            env.p0.cell = "Crosslower" if (ep % 2 == 0) else "Enter0"
            env.p0.x = 420.0
            env.p0.y = 350.0
            o0 = env._get_obs(env.p0, env.p1)
            o1 = env._get_obs(env.p1, env.p0)

        h0 = torch.zeros(1, 1, 128, device=device)
        h1 = torch.zeros(1, 1, 128, device=device)

        ep_obs_0, ep_c_0, ep_m_0, ep_xy_0 = [], [], [], []
        ep_obs_1, ep_c_1, ep_m_1, ep_xy_1 = [], [], [], []

        for t in range(steps_per_ep):
            obs0_in = (
                mask_unseen_opponent(o0)
                if (simulate_unseen_opp and env.p0.cell != env.p1.cell)
                else o0
            )
            obs1_in = (
                mask_unseen_opponent(o1)
                if (simulate_unseen_opp and env.p0.cell != env.p1.cell)
                else o1
            )

            # Variasikan obs[5] (outOfCombatTimer) saat tidak in_combat agar model tidak sensitif
            # terhadap nilai timer out-of-combat
            if not env.p0.in_combat and (ep + t) % 4 == 0:
                obs0_in = obs0_in.copy()
                obs0_in[5] = float(rng.uniform(0.0, 1.0))
            if not env.p1.in_combat and (ep + t) % 4 == 0:
                obs1_in = obs1_in.copy()
                obs1_in[5] = float(rng.uniform(0.0, 1.0))

            # Gunakan target navigasi Mid saat lawan berada di koridor kita tetapi kita masih di Safezone,
            # agar keluar dari Safezone selalu menuju pintu keluar koridor.
            exp0 = select_greedy_sparring_action(env.p0, env.p1)
            exp1 = select_greedy_sparring_action(env.p1, env.p0)

            ep_obs_0.append(obs0_in)
            ep_c_0.append(exp0.combat_act)
            ep_m_0.append(exp0.move_type)
            ep_xy_0.append([exp0.target_dx, exp0.target_dy])

            ep_obs_1.append(obs1_in)
            ep_c_1.append(exp1.combat_act)
            ep_m_1.append(exp1.move_type)
            ep_xy_1.append([exp1.target_dx, exp1.target_dy])

            if model is not None and rng.random() > beta_expert:
                with torch.no_grad():
                    t_o0 = torch.tensor(obs0_in, dtype=torch.float32, device=device).view(1, 1, 30)
                    t_o1 = torch.tensor(obs1_in, dtype=torch.float32, device=device).view(1, 1, 30)
                    c_m0, m_m0 = compute_masks(env.p0, env.p1)
                    c_m1, m_m1 = compute_masks(env.p1, env.p0)

                    c_l0, m_l0, xy0, _, h0 = model.forward(t_o0, h0)
                    c_l1, m_l1, xy1, _, h1 = model.forward(t_o1, h1)

                    c_l0 = c_l0.squeeze().masked_fill(~torch.tensor(c_m0, device=device), -1e9)
                    m_l0 = m_l0.squeeze().masked_fill(~torch.tensor(m_m0, device=device), -1e9)
                    c_l1 = c_l1.squeeze().masked_fill(~torch.tensor(c_m1, device=device), -1e9)
                    m_l1 = m_l1.squeeze().masked_fill(~torch.tensor(m_m1, device=device), -1e9)

                    dx0, dy0 = xy0.squeeze().cpu().numpy()
                    dx1, dy1 = xy1.squeeze().cpu().numpy()
                    act0 = ActionSpec(int(torch.argmax(c_l0).item()), int(torch.argmax(m_l0).item()), float(dx0), float(dy0))
                    act1 = ActionSpec(int(torch.argmax(c_l1).item()), int(torch.argmax(m_l1).item()), float(dx1), float(dy1))
            else:
                if model is not None:
                    with torch.no_grad():
                        t_o0 = torch.tensor(obs0_in, dtype=torch.float32, device=device).view(1, 1, 30)
                        t_o1 = torch.tensor(obs1_in, dtype=torch.float32, device=device).view(1, 1, 30)
                        _, _, _, _, h0 = model.forward(t_o0, h0)
                        _, _, _, _, h1 = model.forward(t_o1, h1)
                act0 = exp0
                act1 = exp1

            if opp_mode == 1 and env.p0.cell != env.p1.cell:
                act1 = ActionSpec(0, 0, 0.0, 0.0)
            elif opp_mode == 2 and env.p0.cell != env.p1.cell:
                act0 = ActionSpec(0, 0, 0.0, 0.0)

            prev_alive_0 = env.p0.is_alive
            prev_alive_1 = env.p1.is_alive
            prev_cell_0 = env.p0.cell
            prev_cell_1 = env.p1.cell
            (o0, o1), _, done, _ = env.step(act0, act1)

            # Pada episode khusus latihan multi-ruangan (ep % 3 == 2), jika agen sudah sampai di Crosslower,
            # tempatkan kembali di ruangan non-Mid (Captain0/1, Crossupper, Resource0/1, Morale0C/1C, Enter0/1)
            # agar seluruh lintasan keluar dari setiap ruangan menuju Crosslower terekam utuh.
            if ep % 3 == 2 and t > 0 and (env.p0.cell == "Crosslower" or t % 32 == 0):
                special_blue = ["Captain0", "Crossupper", "Resource0B", "Resource0A", "Morale0C", "Enter0", "Morale0B", "Morale0A"]
                special_red = ["Captain1", "Crossupper", "Resource1B", "Resource1A", "Morale1C", "Enter1", "Morale1B", "Morale1A"]
                env.p0.cell = special_blue[(ep + t // 16) % len(special_blue)]
                env.p0.x = float(rng.uniform(50.0, 915.0))
                b_ymin2 = 308.0 if (env.p0.cell in ("Morale0C", "Morale1C") and 395.0 <= env.p0.x <= 580.0) else 265.0
                env.p0.y = float(rng.uniform(b_ymin2, 440.0))
                env.p1.cell = special_red[(ep + t // 16 + 3) % len(special_red)]
                if env.p1.cell == env.p0.cell:
                    env.p1.cell = "Crosslower"
                env.p1.x = float(rng.uniform(50.0, 915.0))
                r_ymin2 = 308.0 if (env.p1.cell in ("Morale0C", "Morale1C") and 395.0 <= env.p1.x <= 580.0) else 265.0
                env.p1.y = float(rng.uniform(r_ymin2, 440.0))
                h0.zero_()
                h1.zero_()
                o0 = env._get_obs(env.p0, env.p1)
                o1 = env._get_obs(env.p1, env.p0)

            if not env.p0.is_alive and prev_alive_0:
                h0.zero_()
            if not env.p1.is_alive and prev_alive_1:
                h1.zero_()
            if done:
                o0, o1 = env.reset()
                h0.zero_()
                h1.zero_()

        all_obs.append(np.array(ep_obs_0, dtype=np.float32).reshape(-1, seq_len, 30))
        all_c.append(np.array(ep_c_0, dtype=np.int64).reshape(-1, seq_len))
        all_m.append(np.array(ep_m_0, dtype=np.int64).reshape(-1, seq_len))
        all_xy.append(np.array(ep_xy_0, dtype=np.float32).reshape(-1, seq_len, 2))

        all_obs.append(np.array(ep_obs_1, dtype=np.float32).reshape(-1, seq_len, 30))
        all_c.append(np.array(ep_c_1, dtype=np.int64).reshape(-1, seq_len))
        all_m.append(np.array(ep_m_1, dtype=np.int64).reshape(-1, seq_len))
        all_xy.append(np.array(ep_xy_1, dtype=np.float32).reshape(-1, seq_len, 2))

    return (
        np.concatenate(all_obs, axis=0),
        np.concatenate(all_c, axis=0),
        np.concatenate(all_m, axis=0),
        np.concatenate(all_xy, axis=0),
    )


def train_behavioral_cloning(
    seq_len: int = 32,
    batch_size: int = 64,
    lr: float = 1e-3,
    output_dir: Optional[Path] = None,
) -> Path:
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_path = Path.home() / ".local/share/Turu/army/aqw_combat_telemetry.log"
    print(f"[1/5] Mengekstrak & menginterpolasi demonstrasi bersih dari: {log_path}")
    h_obs, h_c, h_m, h_xy = extract_clean_human_sequences(log_path, seq_len=seq_len)
    print(f"      -> Sekuens telemetri manusia bersih: {len(h_obs)} sekuens ({len(h_obs) * seq_len:,} steps)")

    model = EntityCompoundActorCritic(hidden_dim=128).to(device)
    # Inisialisasi basis termometer berpasangan (+/-) pada self_encoder.0 (fitur obs[6] = cell_idx/15,
    # obs[7] = norm_x, obs[8] = norm_y) agar rata-rata (mean) pada LayerNorm(64) tetap nol dan variansnya
    # seimbang di seluruh 64 neuron, sehingga ke-16 ruangan dan batas koordinat pintu tajam sejak epoch pertama.
    with torch.no_grad():
        model.coord_logstd.fill_(-2.2)
        model.self_encoder[0].weight.mul_(2.0)
        # 15 pasangan neuron (+/-) untuk 15 batas antar 16 ruangan (neuron 0..29)
        for k in range(15):
            thresh = (k + 0.5) / 15.0
            n_pos = 2 * k
            n_neg = 2 * k + 1
            model.self_encoder[0].weight[n_pos].zero_()
            model.self_encoder[0].weight[n_pos, 6] = 6.0
            model.self_encoder[0].bias[n_pos] = -6.0 * thresh
            model.self_encoder[0].weight[n_neg].zero_()
            model.self_encoder[0].weight[n_neg, 6] = -6.0
            model.self_encoder[0].bias[n_neg] = 6.0 * thresh

        # Nolkan bobot terhadap opp_cell_idx (index 4 pada opp_encoder = obs[21])
        # agar navigasi saat tidak satu ruangan (same_cell == 0) 100% invarian terhadap lokasi ruangan lawan!
        model.opp_encoder[0].weight[:, 4].zero_()

        # 6 pasangan neuron (+/-) untuk batas geometri pintu (norm_x = obs[7], norm_y = obs[8]) pada neuron 30..41
        spatial_thresholds = [
            (7, 0.8715, 6.0),   # Batas pusat pintu kanan (x = 815)
            (7, 0.1508, 6.0),   # Batas pusat pintu kiri (x = 170)
            (7, 0.6011, 6.0),   # Batas kanan pintu Captain0 di Morale0C (x = 573)
            (7, 0.4089, 6.0),   # Batas kiri pintu Captain1 di Morale1C (x = 401)
            (8, 0.1837, 6.0),   # Batas bawah pintu koridor atas (y = 300)
            (8, 0.8163, 6.0),   # Batas atas pintu bawah Captain/Crossupper (y = 455)
        ]
        for idx_s, (feat_idx, th_val, slope) in enumerate(spatial_thresholds):
            n_pos = 30 + 2 * idx_s
            n_neg = 30 + 2 * idx_s + 1
            model.self_encoder[0].weight[n_pos].zero_()
            model.self_encoder[0].weight[n_pos, feat_idx] = slope
            model.self_encoder[0].bias[n_pos] = -slope * th_val
            model.self_encoder[0].weight[n_neg].zero_()
            model.self_encoder[0].weight[n_neg, feat_idx] = -slope
            model.self_encoder[0].bias[n_neg] = slope * th_val

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Class-Weighted Cross-Entropy Loss:
    # Bobot tinggi untuk WALK (1) dan SPRINT (2) agar tidak bias ke STAND (0)
    move_weights = torch.tensor([0.35, 1.50, 1.65], dtype=torch.float32, device=device)
    combat_weights = torch.tensor([0.30, 1.00, 1.40, 1.50, 1.40, 1.50], dtype=torch.float32, device=device)
    ce_move_fn = nn.CrossEntropyLoss(weight=move_weights)
    ce_combat_fn = nn.CrossEntropyLoss(weight=combat_weights)
    mse_coord_fn = nn.MSELoss()

    obs_list = [h_obs] if len(h_obs) > 0 else []
    c_list = [h_c] if len(h_c) > 0 else []
    m_list = [h_m] if len(h_m) > 0 else []
    xy_list = [h_xy] if len(h_xy) > 0 else []

    # 3 Putaran DAgger dengan cakupan seluruh 16 ruangan BludrutBrawl
    dagger_schedule = [
        (54, 1.0, 20),  # Round 1: Pure Expert (16 rooms) + Clean Human Telemetry
        (36, 0.3, 20),  # Round 2: Interactive DAgger (70% Model-driven states)
        (36, 0.0, 25),  # Round 3: Full Student-driven states corrected by Expert
    ]

    for round_idx, (n_eps, beta, n_epochs) in enumerate(dagger_schedule, start=1):
        print(f"\n[2.{round_idx}/5] Mengumpulkan {n_eps} episode rollout sekuensial 16-room (beta_expert={beta:.1f})...")
        d_obs, d_c, d_m, d_xy = collect_expert_and_dagger_sequences(
            model=model if round_idx > 1 else None,
            device=device,
            num_episodes=n_eps,
            steps_per_ep=256,
            seq_len=seq_len,
            beta_expert=beta,
            seed=1000 * round_idx,
        )
        obs_list.append(d_obs)
        c_list.append(d_c)
        m_list.append(d_m)
        xy_list.append(d_xy)

        all_obs = torch.tensor(np.concatenate(obs_list, axis=0), dtype=torch.float32, device=device)
        all_c = torch.tensor(np.concatenate(c_list, axis=0), dtype=torch.long, device=device)
        all_m = torch.tensor(np.concatenate(m_list, axis=0), dtype=torch.long, device=device)
        all_xy = torch.tensor(np.concatenate(xy_list, axis=0), dtype=torch.float32, device=device)

        n_seqs = all_obs.size(0)
        indices = np.arange(n_seqs)
        print(f"      Total dataset kumulatif: {n_seqs:,} sekuens ({n_seqs * seq_len:,} steps). Melatih {n_epochs} epoch...")

        model.train()
        for ep in range(1, n_epochs + 1):
            np.random.shuffle(indices)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n_seqs, batch_size):
                mb = indices[start:start + batch_size]
                b_obs = all_obs[mb]     # [B, T, 30]
                b_c = all_c[mb]         # [B, T]
                b_m = all_m[mb]         # [B, T]
                b_xy = all_xy[mb]       # [B, T, 2]

                h_0 = torch.zeros(1, len(mb), 128, device=device)
                if ep % 2 == 0:
                    # Warm-up hidden state GRU dengan paruh pertama sekuens
                    with torch.no_grad():
                        _, _, _, _, h_0 = model.forward(b_obs[:, :seq_len // 2, :], h_0)

                c_logits, m_logits, coord_mu, _, _ = model.forward(b_obs, h_0)

                c_loss = ce_combat_fn(c_logits.reshape(-1, 6), b_c.reshape(-1))
                m_loss = ce_move_fn(m_logits.reshape(-1, 3), b_m.reshape(-1))
                active_mask = (b_m > 0).unsqueeze(-1).expand_as(b_xy)
                if active_mask.any():
                    coord_loss = mse_coord_fn(coord_mu[active_mask], b_xy[active_mask])
                else:
                    coord_loss = mse_coord_fn(coord_mu, b_xy)

                loss = c_loss + 1.5 * m_loss + 4.0 * coord_loss

                optimizer.zero_grad()
                loss.backward()
                if model.opp_encoder[0].weight.grad is not None:
                    model.opp_encoder[0].weight.grad[:, 4].zero_()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    model.opp_encoder[0].weight[:, 4].zero_()

                total_loss += loss.item()
                n_batches += 1

            if ep % 5 == 0 or ep == 1 or ep == n_epochs:
                print(f"      [Round {round_idx} | Epoch {ep:02d}/{n_epochs:02d}] Loss: {total_loss / max(1, n_batches):.4f}")

    with torch.no_grad():
        model.coord_logstd.fill_(-2.2)

    if output_dir is None:
        output_dir = CUR_DIR / "weights"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / "imitation_warrior_base.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n[4/5] Pre-trained weights disimpan ke: {ckpt_path}")

    ts_path = PROJECT_ROOT / "src/ai/weights.ts"
    state = model.state_dict()
    with open(ts_path, "w", encoding="utf-8") as f:
        f.write("/**\n")
        f.write(" * @file weights.ts\n")
        f.write(" * Pre-trained via Sequence Behavioral Cloning + Interactive DAgger\n")
        f.write(" */\n\n")
        f.write("export const WEIGHTS: Record<string, any> = {\n")
        for k, v in state.items():
            arr = v.cpu().numpy().tolist()
            f.write(f'  "{k}": {arr},\n')
        f.write("};\n")
    print(f"[5/5] Bobot TypeScript terpasang di: {ts_path}")
    return ckpt_path


if __name__ == "__main__":
    train_behavioral_cloning()
