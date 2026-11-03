#!/usr/bin/env python3
"""
1:1 Calibrated AQW `/bludrutbrawl` Non-OP PvP Gymnasium Environment (`Battle Warrior`).

Calibrated directly from real server telemetry (`sim/data/aqw_classes_spec.json`):
- Class: `Battle Warrior` (Level 100, Lucky + Platinum PvP Amulet +5000, MaxHP=8455, MaxMP=100, MaxSP=100)
- Exact Skill Kit (`Battle Warrior` id=82486):
  - `aa` (Auto Attack): 1.0x dmg, 2.0s base CD, 0 MP, range 301 (+6 MP hit, +9 MP crit)
  - `a1` (Decisive Strike): 1.9x dmg, 2.0s base CD, 15 MP, range 301
  - `a2` (Imbalancing Strike): 0.6x dmg, 10.0s base CD, 30 MP, range 301 -> 4.0s Stun (`Off Balance`)
  - `a3` (Prepared Strike): 15.0s base CD, 20 MP, self -> 5.0s buff (next 2 Auto Attacks are unavoidable Crits)
  - `a4` (On Guard): 0.75x dmg, 30.0s base CD, 30 MP, range 301 -> 10.0s buff (-50% incoming damage)
- Exact 16-screen `/bludrutbrawl` topology (no `R2 <-> Capt` door), 2D `DoorTriggerArea` bounding boxes,
  and Liang-Barsky trajectory raycast (`check_trajectory_door_collision`).
- Exact Sprint (`-50 SP`, 2x walkSpeed `16` px/frame) & 6.0s Server Regen Ticks (`+25 SP`, `+15% MaxHP`, `+15 MP`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

SPEC_PATH = Path(__file__).resolve().parent / "data" / "aqw_classes_spec.json"

# --- Calibrated Constants ---
DT_SEC = 0.20  # 200ms decision step (5 Hz)
FPS = 24.0
WALK_SPEED_PX_PER_SEC = 8.0 * FPS   # 192 px/sec normal walk
SPRINT_SPEED_PX_PER_SEC = 16.0 * FPS  # 384 px/sec sprint (2x speed)
SPRINT_SP_COST = 50.0
MAX_SP = 100.0
MAX_MP = 100.0
MAX_HP = 8455.0

SERVER_REGEN_TICK_SEC = 6.0
STAMINA_REGEN_PER_TICK = 25.0
OOC_FIRST_TICK_MP = 3.0
OOC_FULL_TICK_HP_RATIO = 0.15  # +1269 HP per 6s tick
OOC_FULL_TICK_MP = 15.0

AA_HIT_MP_GAIN = 6.0
AA_CRIT_MP_GAIN = 9.0
ON_HIT_TAKEN_MP_GAIN = 3.0

MELEE_RANGE = 301.0


@dataclass(frozen=True)
class DoorTrigger:
    target_cell: str
    target_pad: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    aim_x: float
    aim_y: float
    spawn_x: float
    spawn_y: float


@dataclass(frozen=True)
class ScreenSpec:
    index: int
    alias: str
    cell: str
    side: str
    role: str
    npc_max_hp: float
    doors: tuple[DoorTrigger, ...]


# Complete 16-screen BludrutBrawl topology calibrated from `aqw_combat_telemetry.log`
BLUDRUT_SCREENS: tuple[ScreenSpec, ...] = (
    ScreenSpec(
        index=0,
        alias="Blue SZ",
        cell="Enter0",
        side="blue",
        role="safezone",
        npc_max_hp=0.0,
        doors=(
            DoorTrigger("Morale0C", "Left", 750, 960, 240, 420, 770, 275, 173, 342),
        ),
    ),
    ScreenSpec(
        index=1,
        alias="Blue B3",
        cell="Morale0C",
        side="blue",
        role="brawler",
        npc_max_hp=10000.0,
        doors=(
            DoorTrigger("Enter0", "Right", 0, 220, 240, 400, 180, 275, 789, 354),
            DoorTrigger("Morale0B", "Left", 750, 960, 240, 400, 770, 275, 191, 352),
            DoorTrigger("Captain0", "Spawn", 400, 540, 220, 285, 465, 275, 468, 272),
        ),
    ),
    ScreenSpec(
        index=2,
        alias="Blue B2",
        cell="Morale0B",
        side="blue",
        role="brawler",
        npc_max_hp=10000.0,
        doors=(
            DoorTrigger("Morale0C", "Right", 0, 225, 240, 400, 205, 278, 787, 354),
            DoorTrigger("Morale0A", "Left", 735, 960, 240, 400, 760, 285, 192, 349),
        ),
    ),
    ScreenSpec(
        index=3,
        alias="Blue B1",
        cell="Morale0A",
        side="blue",
        role="brawler",
        npc_max_hp=10000.0,
        doors=(
            DoorTrigger("Morale0B", "Right", 0, 225, 240, 400, 205, 282, 787, 350),
            DoorTrigger("Crosslower", "Left", 745, 960, 240, 400, 765, 278, 186, 350),
        ),
    ),
    ScreenSpec(
        index=4,
        alias="Blue Capt",
        cell="Captain0",
        side="blue",
        role="captain",
        npc_max_hp=15000.0,
        doors=(
            DoorTrigger("Morale0C", "Top", 360, 580, 445, 530, 485, 460, 483, 472),
        ),
    ),
    ScreenSpec(
        index=5,
        alias="Blue R2",
        cell="Resource0B",
        side="blue",
        role="restorer",
        npc_max_hp=5000.0,
        doors=(
            DoorTrigger("Resource0A", "Left", 780, 960, 240, 400, 815, 280, 188, 351),
        ),
    ),
    ScreenSpec(
        index=6,
        alias="Blue R1",
        cell="Resource0A",
        side="blue",
        role="restorer",
        npc_max_hp=5000.0,
        doors=(
            DoorTrigger("Resource0B", "Right", 0, 235, 240, 400, 215, 275, 785, 350),
            DoorTrigger("Crossupper", "Left", 755, 960, 240, 400, 775, 275, 192, 351),
        ),
    ),
    ScreenSpec(
        index=7,
        alias="Mid",
        cell="Crosslower",
        side="neutral",
        role="mid",
        npc_max_hp=0.0,
        doors=(
            DoorTrigger("Morale0A", "Right", 0, 230, 240, 400, 210, 285, 789, 349),
            DoorTrigger("Morale1A", "Left", 745, 960, 240, 400, 765, 278, 186, 354),
            DoorTrigger("Crossupper", "Bottom", 400, 530, 240, 305, 465, 280, 488, 429),
        ),
    ),
    ScreenSpec(
        index=8,
        alias="Top",
        cell="Crossupper",
        side="neutral",
        role="top",
        npc_max_hp=0.0,
        doors=(
            DoorTrigger("Resource0A", "Right", 0, 230, 240, 400, 215, 300, 785, 350),
            DoorTrigger("Resource1A", "Left", 745, 960, 240, 400, 770, 297, 185, 347),
            DoorTrigger("Crosslower", "Middle", 120, 900, 450, 530, 485, 465, 490, 342),
        ),
    ),
    ScreenSpec(
        index=9,
        alias="Red R1",
        cell="Resource1A",
        side="red",
        role="restorer",
        npc_max_hp=5000.0,
        doors=(
            DoorTrigger("Crossupper", "Right", 0, 185, 240, 400, 160, 295, 787, 351),
            DoorTrigger("Resource1B", "Left", 745, 960, 240, 400, 770, 280, 188, 350),
        ),
    ),
    ScreenSpec(
        index=10,
        alias="Red R2",
        cell="Resource1B",
        side="red",
        role="restorer",
        npc_max_hp=5000.0,
        doors=(
            DoorTrigger("Resource1A", "Right", 0, 225, 240, 400, 205, 280, 785, 350),
        ),
    ),
    ScreenSpec(
        index=11,
        alias="Red Capt",
        cell="Captain1",
        side="red",
        role="captain",
        npc_max_hp=15000.0,
        doors=(
            DoorTrigger("Morale1C", "Top", 360, 580, 445, 530, 410, 460, 409, 471),
        ),
    ),
    ScreenSpec(
        index=12,
        alias="Red B1",
        cell="Morale1A",
        side="red",
        role="brawler",
        npc_max_hp=10000.0,
        doors=(
            DoorTrigger("Crosslower", "Right", 0, 230, 240, 400, 210, 270, 787, 349),
            DoorTrigger("Morale1B", "Left", 745, 960, 240, 400, 765, 280, 190, 350),
        ),
    ),
    ScreenSpec(
        index=13,
        alias="Red B2",
        cell="Morale1B",
        side="red",
        role="brawler",
        npc_max_hp=10000.0,
        doors=(
            DoorTrigger("Morale1A", "Right", 0, 230, 240, 400, 215, 288, 785, 350),
            DoorTrigger("Morale1C", "Left", 745, 960, 240, 400, 760, 286, 190, 350),
        ),
    ),
    ScreenSpec(
        index=14,
        alias="Red B3",
        cell="Morale1C",
        side="red",
        role="brawler",
        npc_max_hp=10000.0,
        doors=(
            DoorTrigger("Morale1B", "Right", 0, 235, 240, 400, 220, 287, 785, 350),
            DoorTrigger("Captain1", "Spawn", 390, 520, 220, 285, 430, 275, 437, 271),
            DoorTrigger("Enter1", "Left", 750, 960, 240, 400, 775, 280, 185, 350),
        ),
    ),
    ScreenSpec(
        index=15,
        alias="Red SZ",
        cell="Enter1",
        side="red",
        role="safezone",
        npc_max_hp=0.0,
        doors=(
            DoorTrigger("Morale1C", "Right", 0, 220, 240, 420, 195, 280, 785, 350),
        ),
    ),
)

SCREEN_BY_CELL: dict[str, ScreenSpec] = {s.cell: s for s in BLUDRUT_SCREENS}


def check_trajectory_door_collision(
    cell: str, x0: float, y0: float, x1: float, y1: float
) -> tuple[DoorTrigger, float, float, float] | None:
    """Liang-Barsky line-segment vs AABB collision check for 2D door activation areas."""
    screen = SCREEN_BY_CELL.get(cell)
    if screen is None:
        return None

    dx = x1 - x0
    dy = y1 - y0
    best_hit: tuple[DoorTrigger, float, float, float] | None = None

    for door in screen.doors:
        t_min = 0.0
        t_max = 1.0
        p = (-dx, dx, -dy, dy)
        q = (x0 - door.x_min, door.x_max - x0, y0 - door.y_min, door.y_max - y0)
        intersects = True

        for pi, qi in zip(p, q):
            if abs(pi) < 1e-8:
                if qi < 0.0:
                    intersects = False
                    break
            else:
                t = qi / pi
                if pi < 0.0:
                    if t > t_max:
                        intersects = False
                        break
                    if t > t_min:
                        t_min = t
                else:
                    if t < t_min:
                        intersects = False
                        break
                    if t < t_max:
                        t_max = t

        if intersects and t_min <= t_max:
            if best_hit is None or t_min < best_hit[1]:
                best_hit = (door, t_min, x0 + dx * t_min, y0 + dy * t_min)

    return best_hit


@dataclass
class FighterState:
    name: str
    team: str  # "blue" or "red"
    cell: str
    x: float
    y: float
    hp: float = MAX_HP
    max_hp: float = MAX_HP
    mp: float = MAX_MP
    max_mp: float = MAX_MP
    sp: float = MAX_SP
    max_sp: float = MAX_SP

    # Combat & timers
    in_combat: bool = False
    time_since_last_combat_action: float = 12.0
    regen_Clock: float = 0.0

    # Cooldowns remaining (seconds): [aa, a1, a2, a3, a4]
    cooldowns: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    # Active Auras
    stun_rem: float = 0.0            # Off Balance (4s stun from Imbalancing Strike)
    prepared_strike_rem: float = 0.0 # Prepared Strike (5s window)
    prepared_strike_stacks: int = 0  # Up to 2 auto-attacks guaranteed unavoidable crit
    on_guard_rem: float = 0.0        # On Guard (10s -50% incoming damage)

    # Target state: "opponent", "npc", or "none"
    target_kind: str = "opponent"


# Discrete Action Space (14 actions, mapped 1:1 with TypeScript inference engine)
ACTION_NAMES: tuple[str, ...] = (
    "FIGHT_STAND",           # 0: Target opponent & auto-attack in place
    "APPROACH_OPPONENT",     # 1: Walk toward opponent (melee range)
    "SPRINT_TO_OPPONENT",    # 2: Sprint (50 SP) toward opponent
    "USE_SKILL_1",           # 3: Decisive Strike (15 MP, 1.9x dmg, 2s CD)
    "USE_SKILL_2",           # 4: Imbalancing Strike (30 MP, 0.6x dmg, 4s Stun, 10s CD)
    "USE_SKILL_3",           # 5: Prepared Strike (20 MP, next 2 AAs unavoidable Crit, 15s CD)
    "USE_SKILL_4",           # 6: On Guard (30 MP, 0.75x dmg + 10s 50% dmg reduction, 30s CD)
    "KITE_AWAY",             # 7: Walk away from opponent (keep inside current cell)
    "SPRINT_KITE_AWAY",      # 8: Sprint (50 SP) away from opponent (keep inside current cell)
    "JUKE_VERTICAL",         # 9: Walk vertically (up/down y-axis) to dodge/kite
    "MANA_TAP_NPC",          # 10: Walk to & auto-attack cell NPC for safe mana regen
    "WALK_DOOR_FORWARD",     # 11: Walk through door toward opponent / enemy side
    "WALK_DOOR_BACKWARD",    # 12: Walk through door toward own side (escape/reset combat)
    "SPRINT_DOOR_BACKWARD",  # 13: Sprint (50 SP) through door toward own side (fast escape/regen)
)

OBS_DIM = 26
NUM_ACTIONS = len(ACTION_NAMES)


def compute_action_mask_from_obs(obs: np.ndarray) -> np.ndarray:
    """
    Computes a boolean validity mask of shape `(14,)` directly from the 26-dim observation vector.
    Identical 1:1 with the action mask in `src/ai/nn_inference.ts`!
    """
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    mp = obs[1] * 100.0
    can_sprint = obs[3] > 0.5
    a1_cd = obs[10]
    a2_cd = obs[11]
    a3_cd = obs[12]
    a4_cd = obs[13]
    in_melee = obs[19] > 0.5
    has_safe_npc = obs[25] > 0.5

    if not can_sprint:
        mask[2] = False   # SPRINT_TO_OPPONENT
        mask[8] = False   # SPRINT_KITE_AWAY
        mask[13] = False  # SPRINT_DOOR_BACKWARD

    if a1_cd > 1e-4 or mp < 15.0 or not in_melee:
        mask[3] = False   # USE_SKILL_1 (Decisive Strike)
    if a2_cd > 1e-4 or mp < 30.0 or not in_melee:
        mask[4] = False   # USE_SKILL_2 (Imbalancing Strike)
    if a3_cd > 1e-4 or mp < 20.0:
        mask[5] = False   # USE_SKILL_3 (Prepared Strike)
    if a4_cd > 1e-4 or mp < 30.0 or not in_melee:
        mask[6] = False   # USE_SKILL_4 (On Guard)
    if not has_safe_npc:
        mask[10] = False  # MANA_TAP_NPC

    return mask


class Bludrut1v1Env(gym.Env):
    """
    1:1 Calibrated Battle Warrior 1v1 Environment in `/bludrutbrawl`.
    Supports training against a curriculum of opponents (heuristic, aggressive, kiter, or self-play policy).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 600, opponent_policy: Any = None, seed: int | None = None):
        super().__init__()
        self.max_steps = max_steps
        self.opponent_policy = opponent_policy
        self.rng = np.random.default_rng(seed)

        # Load calibrated Battle Warrior stats from `aqw_classes_spec.json`
        self._load_calibrated_spec()

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.agent: FighterState = FighterState("Uma", "blue", "Crosslower", 340.0, 340.0)
        self.enemy: FighterState = FighterState("Opponent", "red", "Crosslower", 620.0, 340.0)
        self.npc_hp: dict[str, float] = {}
        self.step_count = 0

    def _load_calibrated_spec(self) -> None:
        if SPEC_PATH.exists():
            with SPEC_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            bw = data.get("classes", {}).get("Battle Warrior") or data.get("classes", {}).get("Warrior", {})
            cstats = bw.get("computedStats", {})
            raw_sta = bw.get("rawSta", {})
            self.haste = float(cstats.get("haste", 0.1327))
            self.crit_chance = float(cstats.get("criticalChance", 0.2763))
            self.crit_mult = float(cstats.get("criticalMultiplier", 3.754))
            self.dodge_chance = float(cstats.get("evasionChance", 0.2253))
            self.cai = float(raw_sta.get("$cai", 0.9))
            self.cpo = float(raw_sta.get("$cpo", 1.1))
        else:
            self.haste = 0.1327
            self.crit_chance = 0.2763
            self.crit_mult = 3.754
            self.dodge_chance = 0.2253
            self.cai = 0.9
            self.cpo = 1.1

        # Calibrated base Auto Attack hit from Uma's telemetry (`sarsa`: normal aa hit ~ 390-498)
        self.base_aa_damage = 445.0
        cd_scale = 1.0 - min(0.5, self.haste)
        # [aa, a1, a2, a3, a4] cooldowns in seconds for Battle Warrior
        self.skill_cds = [
            2.0 * cd_scale,
            2.0 * cd_scale,
            10.0 * cd_scale,
            15.0 * cd_scale,
            30.0 * cd_scale,
        ]
        self.skill_mp = [0.0, 15.0, 30.0, 20.0, 30.0]
        self.skill_dmg_mult = [1.0, 1.9, 0.6, 0.0, 0.75]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        # Start duel in Mid (Crosslower) or adjacent B1 screen with slight position jitter
        start_cell = "Crosslower"
        ax = float(self.rng.uniform(280.0, 420.0))
        ay = float(self.rng.uniform(300.0, 390.0))
        ex = float(self.rng.uniform(540.0, 680.0))
        ey = float(self.rng.uniform(300.0, 390.0))

        self.agent = FighterState("Uma", "blue", start_cell, ax, ay)
        self.enemy = FighterState("Opponent", "red", start_cell, ex, ey)
        self.npc_hp = {s.cell: s.npc_max_hp for s in BLUDRUT_SCREENS if s.npc_max_hp > 0}

        return self._get_obs(self.agent, self.enemy), {}

    def _dist(self, a: FighterState, b: FighterState) -> float:
        if a.cell != b.cell:
            return 9999.0
        return math.hypot(a.x - b.x, a.y - b.y)

    def _get_obs(self, me: FighterState, opp: FighterState) -> np.ndarray:
        same_cell = 1.0 if me.cell == opp.cell else 0.0
        dist = math.hypot(me.x - opp.x, me.y - opp.y) if same_cell else 1000.0
        in_melee = 1.0 if (same_cell and dist <= MELEE_RANGE) else 0.0

        me_screen = SCREEN_BY_CELL[me.cell]
        opp_screen = SCREEN_BY_CELL[opp.cell]
        npc_cur_hp = self.npc_hp.get(me.cell, 0.0)
        npc_max = me_screen.npc_max_hp
        has_safe_npc = 1.0 if (npc_max > 0 and npc_cur_hp >= 2000.0 and (npc_cur_hp / npc_max) >= 0.35) else 0.0

        obs = np.array(
            [
                me.hp / me.max_hp,                                  # 0: self HP ratio
                me.mp / me.max_mp,                                  # 1: self MP ratio
                me.sp / me.max_sp,                                  # 2: self Stamina (intSP) ratio
                1.0 if me.sp >= SPRINT_SP_COST else 0.0,            # 3: can sprint
                1.0 if me.in_combat else 0.0,                       # 4: self in combat
                min(1.0, me.time_since_last_combat_action / 12.0),  # 5: OOC timer progress
                (SERVER_REGEN_TICK_SEC - me.regen_Clock) / SERVER_REGEN_TICK_SEC,  # 6: time to next 6s server tick
                me.x / 960.0,                                       # 7: self X normalized
                me.y / 540.0,                                       # 8: self Y normalized
                me_screen.index / 15.0,                             # 9: self cell index
                min(1.0, me.cooldowns[1] / self.skill_cds[1]),      # 10: a1 CD ratio
                min(1.0, me.cooldowns[2] / self.skill_cds[2]),      # 11: a2 CD ratio
                min(1.0, me.cooldowns[3] / self.skill_cds[3]),      # 12: a3 CD ratio
                min(1.0, me.cooldowns[4] / self.skill_cds[4]),      # 13: a4 CD ratio
                min(1.0, me.stun_rem / 4.0),                        # 14: self stun remaining
                min(1.0, me.prepared_strike_rem / 5.0),             # 15: self Prepared Strike remaining
                me.prepared_strike_stacks / 2.0,                    # 16: self Prepared Strike stacks
                min(1.0, me.on_guard_rem / 10.0),                   # 17: self On Guard remaining
                same_cell,                                          # 18: opponent in same screen
                in_melee,                                           # 19: opponent within melee range (301px)
                min(1.0, dist / 960.0),                             # 20: 2D distance to opponent
                opp.hp / opp.max_hp,                                # 21: opponent HP ratio
                opp.mp / opp.max_mp,                                # 22: opponent MP ratio
                min(1.0, opp.stun_rem / 4.0),                       # 23: opponent stun remaining
                min(1.0, opp.on_guard_rem / 10.0),                  # 24: opponent On Guard (-50% dmg) remaining
                has_safe_npc,                                       # 25: safe NPC available in cell for mana tap
            ],
            dtype=np.float32,
        )
        return np.clip(obs, -1.0, 1.0)

    def _move_fighter(
        self, f: FighterState, target_x: float, target_y: float, sprint: bool, allow_door: bool
    ) -> bool:
        """
        Moves fighter toward `(target_x, target_y)` for `DT_SEC`.
        If `allow_door` is False, clamps before any `DoorTriggerArea` (matching `applyRuleShield`).
        If a door trigger is intersected, executes the screen transition!
        Returns True if a door transition occurred.
        """
        if f.stun_rem > 0.0:
            return False

        use_sprint = sprint and f.sp >= SPRINT_SP_COST
        if use_sprint:
            f.sp = max(0.0, f.sp - SPRINT_SP_COST)

        speed = SPRINT_SPEED_PX_PER_SEC if use_sprint else WALK_SPEED_PX_PER_SEC
        max_step_dist = speed * DT_SEC

        dx = target_x - f.x
        dy = target_y - f.y
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return False

        ratio = min(1.0, max_step_dist / dist)
        nx = float(np.clip(f.x + dx * ratio, 60.0, 900.0))
        ny = float(np.clip(f.y + dy * ratio, 245.0, 495.0))

        hit = check_trajectory_door_collision(f.cell, f.x, f.y, nx, ny)
        if hit is not None:
            door, hit_t, _hx, _hy = hit
            # Block illegal mid-wave entry into Safezone (`Enter0` / `Enter1`)
            if door.target_cell in ("Enter0", "Enter1"):
                safe_t = max(0.0, hit_t - 0.15)
                f.x = f.x + (nx - f.x) * safe_t
                f.y = f.y + (ny - f.y) * safe_t
                return False

            if not allow_door:
                # RuleShield clamp: stop right before the door trigger box
                safe_t = max(0.0, hit_t - 0.15)
                f.x = f.x + (nx - f.x) * safe_t
                f.y = f.y + (ny - f.y) * safe_t
                return False

            # Trigger door transition to target screen!
            f.cell = door.target_cell
            f.x = door.spawn_x + float(self.rng.uniform(-8.0, 8.0))
            f.y = door.spawn_y + float(self.rng.uniform(-6.0, 6.0))
            f.in_combat = False
            f.target_kind = "none"
            return True

        f.x = nx
        f.y = ny
        return False

    def _resolve_attack(
        self,
        attacker: FighterState,
        defender: FighterState,
        skill_idx: int,
    ) -> float:
        """Executes skill `skill_idx` (0=aa, 1=a1, 2=a2, 4=a4) from `attacker` onto `defender`."""
        mult = self.skill_dmg_mult[skill_idx]
        unavoidable_crit = False
        if skill_idx == 0 and attacker.prepared_strike_stacks > 0 and attacker.prepared_strike_rem > 0.0:
            unavoidable_crit = True
            attacker.prepared_strike_stacks -= 1
            if attacker.prepared_strike_stacks <= 0:
                attacker.prepared_strike_rem = 0.0

        # Check dodge (unless unavoidable crit from Prepared Strike or defender is stunned)
        if not unavoidable_crit and defender.stun_rem <= 0.0:
            if float(self.rng.random()) < self.dodge_chance:
                return 0.0

        is_crit = unavoidable_crit or (float(self.rng.random()) < self.crit_chance)
        raw_dmg = self.base_aa_damage * mult * float(self.rng.uniform(0.95, 1.05))
        if is_crit:
            raw_dmg *= self.crit_mult

        # Apply outgoing & incoming damage modifiers ($cpo = 1.1, $cai = 0.9, On Guard = -50%)
        dmg = raw_dmg * self.cpo * self.cai
        if defender.on_guard_rem > 0.0:
            dmg *= 0.50

        defender.hp = max(0.0, defender.hp - dmg)

        # Mana gain on Auto Attack hit/crit and defender on-hit-taken MP gain
        if skill_idx == 0:
            mp_gain = AA_CRIT_MP_GAIN if is_crit else AA_HIT_MP_GAIN
            attacker.mp = min(attacker.max_mp, attacker.mp + mp_gain)
        defender.mp = min(defender.max_mp, defender.mp + ON_HIT_TAKEN_MP_GAIN)

        # Apply Imbalancing Strike 4s Stun (`a2`)
        if skill_idx == 2:
            defender.stun_rem = max(defender.stun_rem, 4.0)

        return dmg

    def _apply_action(self, f: FighterState, opp: FighterState, action: int) -> tuple[float, float]:
        """
        Applies discrete action for fighter `f`.
        Returns `(damage_dealt_to_opp, rule_penalty)`.
        """
        dmg_dealt = 0.0
        penalty = 0.0

        if f.stun_rem > 0.0:
            return 0.0, 0.0

        same_cell = f.cell == opp.cell
        dist = math.hypot(f.x - opp.x, f.y - opp.y) if same_cell else 9999.0

        # Movement actions
        if action in (1, 2):  # APPROACH_OPPONENT / SPRINT_TO_OPPONENT
            sprint = action == 2
            if same_cell:
                f.target_kind = "opponent"
                if dist > MELEE_RANGE * 0.75:
                    self._move_fighter(f, opp.x, opp.y, sprint=sprint, allow_door=False)
            else:
                # Move through door toward opponent's cell
                self._step_toward_cell(f, opp.cell, sprint=sprint)

        elif action in (7, 8):  # KITE_AWAY / SPRINT_KITE_AWAY (stay inside screen)
            sprint = action == 8
            if same_cell:
                away_x = 180.0 if opp.x > f.x else 780.0
                away_y = 270.0 if opp.y > 360.0 else 430.0
                self._move_fighter(f, away_x, away_y, sprint=sprint, allow_door=False)

        elif action == 9:  # JUKE_VERTICAL (up/down y-axis juke inside screen)
            target_y = 265.0 if f.y > 355.0 else 435.0
            self._move_fighter(f, f.x, target_y, sprint=False, allow_door=False)

        elif action == 10:  # MANA_TAP_NPC
            screen = SCREEN_BY_CELL[f.cell]
            npc_cur = self.npc_hp.get(f.cell, 0.0)
            if screen.npc_max_hp > 0 and npc_cur >= 2000.0 and (npc_cur / screen.npc_max_hp) >= 0.35:
                f.target_kind = "npc"
                # Move near center of room where Brawler/Restorer stands (480, 330)
                self._move_fighter(f, 480.0, 330.0, sprint=False, allow_door=False)
            else:
                f.target_kind = "opponent"

        elif action == 11:  # WALK_DOOR_FORWARD (toward opponent / enemy side)
            target_cell = opp.cell if not same_cell else ("Morale1A" if f.team == "blue" else "Morale0A")
            self._step_toward_cell(f, target_cell, sprint=False)

        elif action in (12, 13):  # WALK_DOOR_BACKWARD / SPRINT_DOOR_BACKWARD (retreat to adjacent screen for OOC regen)
            sprint = action == 13
            retreat_cell = "Morale0C" if f.team == "blue" else "Morale1C"
            if f.cell == retreat_cell:
                retreat_cell = "Morale0B" if f.team == "blue" else "Morale1B"
            self._step_toward_cell(f, retreat_cell, sprint=sprint)

        # Skill activations (3=a1, 4=a2, 5=a3, 6=a4)
        elif action in (3, 4, 5, 6):
            skill_idx = action - 2  # 1, 2, 3, 4
            if f.cooldowns[skill_idx] <= 0.0 and f.mp >= self.skill_mp[skill_idx]:
                if skill_idx == 3:
                    # a3: Prepared Strike (self-cast)
                    f.mp -= self.skill_mp[3]
                    f.cooldowns[3] = self.skill_cds[3]
                    f.prepared_strike_rem = 5.0
                    f.prepared_strike_stacks = 2
                else:
                    # Hostile melee skills (a1, a2, a4) require opponent in melee range
                    if same_cell and dist <= MELEE_RANGE:
                        f.target_kind = "opponent"
                        f.mp -= self.skill_mp[skill_idx]
                        f.cooldowns[skill_idx] = self.skill_cds[skill_idx]
                        f.in_combat = True
                        f.time_since_last_combat_action = 0.0
                        opp.in_combat = True
                        opp.time_since_last_combat_action = 0.0
                        if skill_idx == 4:
                            # a4: On Guard applies 10s -50% incoming dmg buff to self AND deals 0.75x damage
                            f.on_guard_rem = 10.0
                        dmg_dealt += self._resolve_attack(f, opp, skill_idx)

        # Automatic Auto Attack (`aa`) when cooldown ready
        if f.cooldowns[0] <= 0.0 and f.stun_rem <= 0.0:
            if f.target_kind == "npc":
                screen = SCREEN_BY_CELL[f.cell]
                npc_cur = self.npc_hp.get(f.cell, 0.0)
                if screen.npc_max_hp > 0 and npc_cur >= 2000.0 and math.hypot(f.x - 480.0, f.y - 330.0) <= MELEE_RANGE:
                    f.cooldowns[0] = self.skill_cds[0]
                    f.in_combat = True
                    f.time_since_last_combat_action = 0.0
                    is_crit = float(self.rng.random()) < self.crit_chance
                    npc_dmg = self.base_aa_damage * (self.crit_mult if is_crit else 1.0)
                    self.npc_hp[f.cell] = max(1500.0, npc_cur - npc_dmg)
                    f.mp = min(f.max_mp, f.mp + (AA_CRIT_MP_GAIN if is_crit else AA_HIT_MP_GAIN))
                    # Brawler hits back slightly, giving +3 MP
                    f.hp = max(1.0, f.hp - 190.0 * (0.5 if f.on_guard_rem > 0 else 1.0))
                    f.mp = min(f.max_mp, f.mp + ON_HIT_TAKEN_MP_GAIN)
            elif same_cell and dist <= MELEE_RANGE:
                f.target_kind = "opponent"
                f.cooldowns[0] = self.skill_cds[0]
                f.in_combat = True
                f.time_since_last_combat_action = 0.0
                opp.in_combat = True
                opp.time_since_last_combat_action = 0.0
                dmg_dealt += self._resolve_attack(f, opp, 0)

        return dmg_dealt, penalty

    def _step_toward_cell(self, f: FighterState, goal_cell: str, sprint: bool) -> None:
        """Finds the door in `f.cell` leading toward `goal_cell` (BFS shortest path) and walks/sprints to its `aim_x, aim_y`."""
        if f.cell == goal_cell:
            return
        screen = SCREEN_BY_CELL[f.cell]
        chosen_door = screen.doors[0]
        # Simple BFS to find which immediate neighbor is closest to `goal_cell`
        from collections import deque
        queue = deque([(d, d.target_cell) for d in screen.doors if d.target_cell not in ("Enter0", "Enter1")])
        visited = {f.cell}
        while queue:
            first_door, curr = queue.popleft()
            if curr == goal_cell:
                chosen_door = first_door
                break
            if curr in visited:
                continue
            visited.add(curr)
            for nd in SCREEN_BY_CELL[curr].doors:
                if nd.target_cell not in visited and nd.target_cell not in ("Enter0", "Enter1"):
                    queue.append((first_door, nd.target_cell))

        self._move_fighter(f, chosen_door.aim_x, chosen_door.aim_y, sprint=sprint, allow_door=True)

    def _tick_timers_and_regen(self, f: FighterState) -> None:
        for i in range(5):
            f.cooldowns[i] = max(0.0, f.cooldowns[i] - DT_SEC)
        f.stun_rem = max(0.0, f.stun_rem - DT_SEC)
        f.prepared_strike_rem = max(0.0, f.prepared_strike_rem - DT_SEC)
        if f.prepared_strike_rem <= 0.0:
            f.prepared_strike_stacks = 0
        f.on_guard_rem = max(0.0, f.on_guard_rem - DT_SEC)

        f.time_since_last_combat_action += DT_SEC
        if f.in_combat and f.time_since_last_combat_action >= 4.5:
            f.in_combat = False

        # Server 6.0-second tick (`ct` p.uma update)
        f.regen_Clock += DT_SEC
        if f.regen_Clock >= SERVER_REGEN_TICK_SEC:
            f.regen_Clock -= SERVER_REGEN_TICK_SEC
            # Stamina (`intSP`) regenerates +25 per 6s server tick when out of combat
            if not f.in_combat:
                f.sp = min(f.max_sp, f.sp + STAMINA_REGEN_PER_TICK)
                if f.time_since_last_combat_action >= 9.0:
                    # Full Out-of-Combat server regen tick (+15% MaxHP, +15 MP)
                    f.hp = min(f.max_hp, f.hp + f.max_hp * OOC_FULL_TICK_HP_RATIO)
                    f.mp = min(f.max_mp, f.mp + OOC_FULL_TICK_MP)
                else:
                    # First transition tick (+3 MP)
                    f.mp = min(f.max_mp, f.mp + OOC_FIRST_TICK_MP)

    def _heuristic_opponent_action(self, opp: FighterState, ag: FighterState) -> int:
        """Strong tactical Battle Warrior opponent baseline for curriculum & evaluation."""
        obs = self._get_obs(opp, ag)
        if self.opponent_policy is not None:
            return int(self.opponent_policy(obs))

        same_cell = opp.cell == ag.cell
        dist = math.hypot(opp.x - ag.x, opp.y - ag.y) if same_cell else 9999.0

        # Retreat through door for OOC regen if HP is low and opponent is stunned or far
        if opp.hp / opp.max_hp < 0.32 and (ag.stun_rem > 1.0 or dist > 260.0):
            return 13 if opp.sp >= 50.0 else 12

        # Stay out of combat to catch the 6s regen tick if recovering HP/MP
        if not same_cell and opp.hp / opp.max_hp < 0.70 and not opp.in_combat:
            return 0

        if not same_cell:
            return 11

        # Kite away if agent has On Guard (-50% damage wall) and opponent doesn't
        if ag.on_guard_rem > 2.0 and opp.on_guard_rem <= 0.0 and dist < 350.0:
            return 8 if opp.sp >= 50.0 else 7

        if dist > MELEE_RANGE:
            return 2 if (opp.sp >= 50.0 and dist > 480.0) else 1

        # In melee range:
        if opp.cooldowns[4] <= 0.0 and opp.mp >= 30.0 and opp.hp / opp.max_hp < 0.88:
            return 6  # On Guard (a4)
        if opp.cooldowns[2] <= 0.0 and opp.mp >= 30.0 and ag.stun_rem <= 0.2:
            return 4  # Imbalancing Strike (a2 4s stun)
        if opp.cooldowns[3] <= 0.0 and opp.mp >= 35.0 and opp.prepared_strike_stacks == 0:
            return 5  # Prepared Strike (a3)
        if opp.cooldowns[1] <= 0.0 and opp.mp >= 15.0 and ag.on_guard_rem <= 1.0:
            return 3  # Decisive Strike (a1)
        return 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        opp_action = self._heuristic_opponent_action(self.enemy, self.agent)

        prev_agent_hp = self.agent.hp
        prev_enemy_hp = self.enemy.hp

        dmg_by_agent, pen_agent = self._apply_action(self.agent, self.enemy, int(action))
        dmg_by_enemy, _ = self._apply_action(self.enemy, self.agent, int(opp_action))

        self._tick_timers_and_regen(self.agent)
        self._tick_timers_and_regen(self.enemy)

        # Dense reward shaping + terminal win/loss reward
        hp_change_agent = self.agent.hp - prev_agent_hp
        hp_change_enemy = self.enemy.hp - prev_enemy_hp
        reward = (( -hp_change_enemy) - (-hp_change_agent)) / MAX_HP * 2.5 - pen_agent

        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        if self.enemy.hp <= 0.0 and self.agent.hp <= 0.0:
            terminated = True
            info["winner"] = "draw"
        elif self.enemy.hp <= 0.0:
            terminated = True
            reward += 5.0 + 2.0 * (self.agent.hp / self.agent.max_hp)
            info["winner"] = "agent"
        elif self.agent.hp <= 0.0:
            terminated = True
            reward -= 5.0
            info["winner"] = "enemy"
        elif self.step_count >= self.max_steps:
            truncated = True
            # Advantage evaluation on timeout
            hp_diff = (self.agent.hp / self.agent.max_hp) - (self.enemy.hp / self.enemy.max_hp)
            reward += hp_diff * 1.5
            info["winner"] = "timeout"

        return self._get_obs(self.agent, self.enemy), float(reward), terminated, truncated, info
