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


L_XMIN, L_XMAX = 0.0, 90.0
R_XMIN, R_XMAX = 870.0, 960.0
Y_MIN, Y_MAX = 220.0, 510.0
L_AIM, R_AIM = 45.0, 915.0
Y_AIM = 350.0
L_SPAWN, R_SPAWN = 180.0, 780.0

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
            DoorTrigger("Morale0C", "Left", 750.0, 960.0, 245.0, 315.0, 850.0, 262.0, 173.0, 342.0),
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
            DoorTrigger("Enter0", "Right", 0.0, 220.0, 245.0, 312.0, 120.0, 265.0, 789.0, 354.0),
            DoorTrigger("Morale0B", "Left", 750.0, 960.0, 245.0, 312.0, 855.0, 265.0, 191.0, 352.0),
            DoorTrigger("Captain0", "Spawn", 400.0, 540.0, 220.0, 285.0, 465.0, 255.0, 468.0, 272.0),
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
            DoorTrigger("Morale0C", "Right", 0.0, 225.0, 245.0, 312.0, 120.0, 268.0, 787.0, 354.0),
            DoorTrigger("Morale0A", "Left", 735.0, 960.0, 245.0, 312.0, 855.0, 268.0, 192.0, 349.0),
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
            DoorTrigger("Morale0B", "Right", 0.0, 225.0, 245.0, 312.0, 120.0, 270.0, 787.0, 350.0),
            DoorTrigger("Crosslower", "Left", 745.0, 960.0, 245.0, 312.0, 855.0, 266.0, 186.0, 350.0),
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
            DoorTrigger("Morale0C", "Top", 360.0, 580.0, 445.0, 530.0, 485.0, 480.0, 483.0, 472.0),
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
            DoorTrigger("Resource0A", "Left", 780.0, 960.0, 245.0, 312.0, 855.0, 268.0, 188.0, 351.0),
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
            DoorTrigger("Resource0B", "Right", 0.0, 235.0, 245.0, 312.0, 120.0, 268.0, 785.0, 350.0),
            DoorTrigger("Crossupper", "Left", 755.0, 960.0, 245.0, 312.0, 855.0, 270.0, 192.0, 351.0),
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
            DoorTrigger("Morale0A", "Right", 0.0, 230.0, 245.0, 312.0, 120.0, 270.0, 789.0, 349.0),
            DoorTrigger("Morale1A", "Left", 745.0, 960.0, 245.0, 312.0, 855.0, 268.0, 186.0, 354.0),
            DoorTrigger("Crossupper", "Bottom", 400.0, 530.0, 240.0, 305.0, 465.0, 258.0, 488.0, 429.0),
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
            DoorTrigger("Resource0A", "Right", 0.0, 230.0, 245.0, 315.0, 120.0, 275.0, 785.0, 350.0),
            DoorTrigger("Resource1A", "Left", 745.0, 960.0, 245.0, 315.0, 855.0, 275.0, 185.0, 347.0),
            DoorTrigger("Crosslower", "Middle", 120.0, 900.0, 450.0, 530.0, 485.0, 482.0, 490.0, 342.0),
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
            DoorTrigger("Crossupper", "Right", 0.0, 185.0, 245.0, 315.0, 115.0, 275.0, 787.0, 351.0),
            DoorTrigger("Resource1B", "Left", 745.0, 960.0, 245.0, 312.0, 855.0, 268.0, 188.0, 350.0),
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
            DoorTrigger("Resource1A", "Right", 0.0, 225.0, 245.0, 312.0, 120.0, 268.0, 785.0, 350.0),
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
            DoorTrigger("Morale1C", "Top", 360.0, 580.0, 445.0, 530.0, 410.0, 480.0, 409.0, 471.0),
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
            DoorTrigger("Crosslower", "Right", 0.0, 230.0, 245.0, 312.0, 120.0, 268.0, 787.0, 349.0),
            DoorTrigger("Morale1B", "Left", 745.0, 960.0, 245.0, 312.0, 855.0, 268.0, 190.0, 350.0),
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
            DoorTrigger("Morale1A", "Right", 0.0, 230.0, 245.0, 312.0, 120.0, 270.0, 785.0, 350.0),
            DoorTrigger("Morale1C", "Left", 745.0, 960.0, 245.0, 312.0, 855.0, 270.0, 190.0, 350.0),
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
            DoorTrigger("Morale1B", "Right", 0.0, 235.0, 245.0, 312.0, 120.0, 270.0, 785.0, 350.0),
            DoorTrigger("Captain1", "Spawn", 390.0, 520.0, 220.0, 285.0, 430.0, 255.0, 437.0, 271.0),
            DoorTrigger("Enter1", "Left", 750.0, 960.0, 245.0, 312.0, 855.0, 268.0, 185.0, 350.0),
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
            DoorTrigger("Morale1C", "Right", 0.0, 220.0, 245.0, 315.0, 120.0, 268.0, 785.0, 350.0),
        ),
    ),
)


SCREEN_BY_CELL: dict[str, ScreenSpec] = {s.cell: s for s in BLUDRUT_SCREENS}


def compute_cell_graph_distances() -> dict[tuple[str, str], int]:
    """Precomputes BFS shortest path screen hop distances between all 16 Bludrut Brawl cells."""
    from collections import deque
    cells = [s.cell for s in BLUDRUT_SCREENS]
    dist_map = {}
    for c1 in cells:
        queue = deque([(c1, 0)])
        visited = {c1}
        while queue:
            curr, d = queue.popleft()
            dist_map[(c1, curr)] = d
            screen = SCREEN_BY_CELL.get(curr)
            if screen:
                for door in screen.doors:
                    if door.target_cell not in visited:
                        visited.add(door.target_cell)
                        queue.append((door.target_cell, d + 1))
    return dist_map


CELL_GRAPH_DIST: dict[tuple[str, str], int] = compute_cell_graph_distances()


def find_door_toward_cell(from_cell: str, goal_cell: str) -> DoorTrigger | None:
    """Finds the DoorTrigger in `from_cell` that leads along the BFS shortest path to `goal_cell`."""
    if from_cell == goal_cell:
        return None
    screen = SCREEN_BY_CELL.get(from_cell)
    if not screen or not screen.doors:
        return None
    from collections import deque
    queue = deque([(d, d.target_cell) for d in screen.doors])
    visited = {from_cell}
    while queue:
        first_door, curr = queue.popleft()
        if curr == goal_cell:
            return first_door
        if curr in visited:
            continue
        visited.add(curr)
        curr_screen = SCREEN_BY_CELL.get(curr)
        if curr_screen:
            for nd in curr_screen.doors:
                if nd.target_cell not in visited:
                    queue.append((first_door, nd.target_cell))
    return screen.doors[0]


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
        eff_y_max = max(door.y_max, 420.0) if (door.x_min == 0.0 or door.x_max >= 950.0) else door.y_max
        q = (x0 - door.x_min, door.x_max - x0, y0 - door.y_min, eff_y_max - y0)
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
    door_cooldown: float = 0.0       # Screen transition immunity to prevent instant ping-pong

    # Event telemetry for accurate logging
    last_combat_event: str = ""
    last_aa_event: str = ""

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

OBS_DIM = 28
NUM_ACTIONS = len(ACTION_NAMES)


def compute_action_mask_from_obs(obs: np.ndarray) -> np.ndarray:
    """
    Computes a boolean validity mask of shape `(14,)` directly from the 28-dim observation vector.
    Identical 1:1 with the action mask in `src/ai/nn_inference.ts`!
    """
    mask = np.ones(NUM_ACTIONS, dtype=bool)
    mp = obs[1] * 100.0
    can_sprint = obs[3] > 0.5
    a1_cd = obs[10]
    a2_cd = obs[11]
    a3_cd = obs[12]
    a4_cd = obs[13]
    in_melee = obs[25] > 0.5
    has_safe_npc = obs[27] > 0.5
    same_cell = obs[23] > 0.5

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
    # S4 (On Guard) can hit opponent in melee, OR hit safe NPC if opponent is in another room
    s4_can_hit = in_melee or (has_safe_npc and not same_cell)
    if a4_cd > 1e-4 or mp < 30.0 or not s4_can_hit:
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

    def __init__(self, max_steps: int = 6000, target_kills: int = 10, opponent_policy: Any = None, seed: int | None = None):
        super().__init__()
        self.max_steps = max_steps
        self.target_kills = target_kills
        self.opponent_policy = opponent_policy
        self.rng = np.random.default_rng(seed)

        # Load calibrated Battle Warrior stats from `aqw_classes_spec.json`
        self._load_calibrated_spec()

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.agent: FighterState = FighterState("Uma", "blue", "Enter0", 350.0, 360.0)
        self.enemy: FighterState = FighterState("Opponent", "red", "Enter1", 600.0, 360.0)
        self.npc_hp: dict[str, float] = {}
        self.step_count = 0
        self.wave_won = False
        self.blue_kills = 0
        self.red_kills = 0
        self.enemy_returning_to_sz = False
        self.last_kill_event = ""
        self.last_opp_action = 0
        self.agent_sz_time: float = 0.0
        self.agent_has_exited_sz: bool = False
        self.agent_attempted_illegal_sz_entry: bool = False
        self.agent_no_combat_away_from_mid_time: float = 0.0
        self.agent_entered_mid: bool = False

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
        self.wave_won = False
        self.blue_kills = 0
        self.red_kills = 0
        self.enemy_returning_to_sz = False
        self.last_kill_event = ""
        self.last_opp_action = 0

        # Curriculum / Full-Map Spawn Distribution across all 16 screens:
        # Symmetrically sample agent team: 50% Blue, 50% Red
        agent_team = "blue" if self.rng.random() < 0.5 else "red"
        enemy_team = "red" if agent_team == "blue" else "blue"
        sz_agent = "Enter0" if agent_team == "blue" else "Enter1"
        sz_enemy = "Enter1" if agent_team == "blue" else "Enter0"
        sz_agent_x = 350.0 if agent_team == "blue" else 600.0
        sz_enemy_x = 600.0 if agent_team == "blue" else 350.0

        spawn_type = float(self.rng.random())
        if spawn_type < 0.35:
            self.agent = FighterState(
                "Uma", agent_team, sz_agent,
                sz_agent_x + float(self.rng.uniform(-30, 30)),
                360.0 + float(self.rng.uniform(-20, 20))
            )
            self.enemy = FighterState(
                "Opponent", enemy_team, sz_enemy,
                sz_enemy_x + float(self.rng.uniform(-30, 30)),
                360.0 + float(self.rng.uniform(-20, 20))
            )
        elif spawn_type < 0.65:
            corridor_rooms = ["Morale0C", "Morale0B", "Morale0A", "Crosslower", "Morale1A", "Morale1B", "Morale1C"]
            c_agent = str(self.rng.choice(corridor_rooms))
            c_opp = c_agent if self.rng.random() < 0.5 else str(self.rng.choice(corridor_rooms))
            ax = float(self.rng.uniform(240.0, 420.0))
            ay = float(self.rng.uniform(300.0, 420.0))
            ex = float(self.rng.uniform(540.0, 720.0))
            ey = float(self.rng.uniform(300.0, 420.0))
            self.agent = FighterState("Uma", agent_team, c_agent, ax, ay)
            self.enemy = FighterState("Opponent", enemy_team, c_opp, ex, ey)
        elif spawn_type < 0.80:
            upper_rooms = ["Resource0B", "Resource0A", "Crossupper", "Resource1A", "Resource1B"]
            c_agent = str(self.rng.choice(upper_rooms))
            c_opp = c_agent if self.rng.random() < 0.5 else str(self.rng.choice(upper_rooms))
            self.agent = FighterState("Uma", agent_team, c_agent, float(self.rng.uniform(240.0, 420.0)), float(self.rng.uniform(300.0, 400.0)))
            self.enemy = FighterState("Opponent", enemy_team, c_opp, float(self.rng.uniform(540.0, 720.0)), float(self.rng.uniform(300.0, 400.0)))
        elif spawn_type < 0.90:
            c = str(self.rng.choice(["Captain0", "Captain1"]))
            self.agent = FighterState("Uma", agent_team, c, 380.0, 360.0)
            self.enemy = FighterState("Opponent", enemy_team, c, 580.0, 360.0)
        else:
            pairs = [("Crosslower", "Crossupper"), ("Crossupper", "Crosslower"), ("Morale0C", "Captain0"), ("Morale1C", "Captain1")]
            c_ag, c_en = self.rng.choice(pairs)
            self.agent = FighterState("Uma", agent_team, str(c_ag), 480.0, 360.0)
            self.enemy = FighterState("Opponent", enemy_team, str(c_en), 480.0, 360.0)

        self.npc_hp = {s.cell: s.npc_max_hp for s in BLUDRUT_SCREENS if s.npc_max_hp > 0}
        self.agent_sz_time = 0.0
        self.agent_has_exited_sz = (self.agent.cell not in ("Enter0", "Enter1"))
        self.agent_attempted_illegal_sz_entry = False
        self.agent_no_combat_away_from_mid_time = 0.0
        self.agent_entered_mid = (self.agent.cell == "Crosslower")

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

        # Strategic Navigation Objective (Symmetric for Blue & Red):
        own_sz = "Enter0" if me.team == "blue" else "Enter1"
        opp_in_arena = (opp.cell not in ("Enter0", "Enter1", "Wait") and opp.hp > 0.0)

        if self.wave_won:
            target_nav_cell = own_sz
        elif opp_in_arena:
            target_nav_cell = opp.cell
        else:
            target_nav_cell = "Crosslower"

        next_door = find_door_toward_cell(me.cell, target_nav_cell) if not same_cell else None

        if same_cell:
            target_rel_x = (opp.x - me.x) / 960.0
            target_rel_y = (opp.y - me.y) / 550.0
        elif next_door:
            target_rel_x = (next_door.aim_x - me.x) / 960.0
            target_rel_y = (next_door.aim_y - me.y) / 550.0
        else:
            target_rel_x = 0.0
            target_rel_y = 0.0

        mid_timer_ratio = min(1.0, self.agent_no_combat_away_from_mid_time / 60.0)

        # Universal Combat Invariants (Class-Agnostic):
        self_in_dmg_multiplier = 0.50 if (me.on_guard_rem > 0.0) else 1.0
        self_hard_cc = 1.0 if (me.stun_rem > 0.0) else 0.0
        opp_in_dmg_multiplier = 0.50 if (opp.on_guard_rem > 0.0) else 1.0
        opp_hard_cc = 1.0 if (opp.stun_rem > 0.0) else 0.0

        obs = np.array(
            [
                me.hp / me.max_hp,                                  # 0: self HP ratio
                me.mp / me.max_mp,                                  # 1: self MP ratio
                me.sp / me.max_sp,                                  # 2: self Stamina (intSP) ratio
                1.0 if me.sp >= SPRINT_SP_COST else 0.0,            # 3: can sprint (SP >= 50)
                1.0 if me.in_combat else 0.0,                       # 4: self in combat
                0.0 if me.in_combat else min(1.0, me.time_since_last_combat_action / 12.0),  # 5: OOC timer progress
                me_screen.index / 15.0,                             # 6: normalized screen index
                min(1.0, max(0.0, me.x / 960.0)),                   # 7: self X
                min(1.0, max(0.0, me.y / 550.0)),                   # 8: self Y
                1.0 if me.hp > 0.0 else 0.0,                        # 9: self alive
                1.0 if me.cooldowns[1] <= 0.0 else 0.0,             # 10: skill 1 ready
                1.0 if me.cooldowns[2] <= 0.0 else 0.0,             # 11: skill 2 ready
                1.0 if me.cooldowns[3] <= 0.0 else 0.0,             # 12: skill 3 ready
                1.0 if me.cooldowns[4] <= 0.0 else 0.0,             # 13: skill 4 ready
                opp_screen.index / 15.0,                            # 14: opp screen index
                float(np.clip(target_rel_x, -1.0, 1.0)),            # 15: primary target relative X (opp in room, or exit door when roaming)
                float(np.clip(target_rel_y, -1.0, 1.0)),            # 16: primary target relative Y (opp in room, or exit door when roaming)
                mid_timer_ratio,                                    # 17: 1-minute tournament rule progress
                self_in_dmg_multiplier,                             # 18: self incoming damage multiplier (0.50 if barrier, 1.0 normal)
                self_hard_cc,                                       # 19: self stunned / disabled
                opp_in_dmg_multiplier,                              # 20: opp incoming damage multiplier (0.50 if barrier, 1.0 normal)
                opp_hard_cc,                                        # 21: opp stunned / disabled
                1.0 if opp.hp > 0.0 else 0.0,                       # 22: opp alive
                same_cell,                                          # 23: opp in same cell
                min(1.0, dist / 1000.0),                            # 24: distance to opp
                in_melee,                                           # 25: in melee range
                opp.hp / max(1.0, opp.max_hp),                      # 26: opp HP ratio
                has_safe_npc,                                       # 27: safe NPC available
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
        nx = float(np.clip(f.x + dx * ratio, 20.0, 940.0))
        ny = float(np.clip(f.y + dy * ratio, 240.0, 500.0))

        hit = check_trajectory_door_collision(f.cell, f.x, f.y, nx, ny)
        if hit is not None:
            door, hit_t, _hx, _hy = hit
            # Safezone door entry rules:
            # Block entry to enemy safezone, or own safezone during active wave
            if door.target_cell in ("Enter0", "Enter1"):
                can_enter_sz = False
                if door.target_cell == "Enter0" and f.team == "blue" and self.wave_won:
                    can_enter_sz = True
                elif door.target_cell == "Enter1" and f.team == "red" and (self.wave_won or self.enemy_returning_to_sz):
                    can_enter_sz = True

                if not can_enter_sz:
                    if f == self.agent:
                        self.agent_attempted_illegal_sz_entry = True
                    safe_t = max(0.0, hit_t - 0.15)
                    f.x = f.x + (nx - f.x) * safe_t
                    f.y = f.y + (ny - f.y) * safe_t
                    return False

            if not allow_door or f.door_cooldown > 0.0:
                # RuleShield clamp: stop right before the door trigger box
                safe_t = max(0.0, hit_t - 0.15)
                f.x = f.x + (nx - f.x) * safe_t
                f.y = f.y + (ny - f.y) * safe_t
                return False

            # Trigger door transition to target screen!
            f.cell = door.target_cell
            f.x = door.spawn_x + float(self.rng.uniform(-6.0, 6.0))
            f.y = door.spawn_y + float(self.rng.uniform(-4.0, 4.0))
            f.door_cooldown = 1.0  # 1.0s cooldown before next door transition
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
                return 0.0, 0.0

        is_crit = unavoidable_crit or (float(self.rng.random()) < self.crit_chance)
        raw_dmg = self.base_aa_damage * mult * float(self.rng.uniform(0.95, 1.05))
        if is_crit:
            raw_dmg *= self.crit_mult

        # Apply outgoing & incoming damage modifiers ($cpo = 1.1, $cai = 0.9, On Guard = -50%)
        dmg = raw_dmg * self.cpo * self.cai
        mitigated = 0.0
        if defender.on_guard_rem > 0.0:
            mitigated = dmg * 0.50
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

        return dmg, mitigated

    def _apply_action(self, f: FighterState, opp: FighterState, action: int) -> tuple[float, float, float]:
        """
        Applies discrete action for fighter `f`.
        Returns `(damage_dealt_to_opp, rule_penalty, damage_mitigated)`.
        """
        dmg_dealt = 0.0
        penalty = 0.0
        dmg_mitigated = 0.0

        if f.stun_rem > 0.0:
            return 0.0, 0.0, 0.0

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
            sprint = (action == 8) and (f.sp >= 50.0)
            if sprint:
                f.sp = max(0.0, f.sp - 5.0)
            if same_cell:
                away_x = 100.0 if opp.x > f.x else 860.0
                away_y = 280.0 if opp.y > 360.0 else 440.0
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

        elif action in (12, 13):  # WALK_DOOR_BACKWARD / SPRINT_DOOR_BACKWARD (retreat to Safezone)
            sprint = (action == 13) and (f.sp >= 50.0)
            if sprint:
                f.sp = max(0.0, f.sp - 5.0)
            target_sz = "Enter0" if f.team == "blue" else "Enter1"
            self._step_toward_cell(f, target_sz, sprint=sprint)

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
                        d, m = self._resolve_attack(f, opp, skill_idx)
                        dmg_dealt += d
                        dmg_mitigated += m

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
                d, m = self._resolve_attack(f, opp, 0)
                dmg_dealt += d
                dmg_mitigated += m

        return dmg_dealt, penalty, dmg_mitigated

    def _apply_compound_action(
        self,
        f: FighterState,
        opp: FighterState,
        action: tuple | list | np.ndarray,
    ) -> tuple[float, float, float]:
        """
        Applies atomic compound action:
        combat_act: 0=NONE, 1=AA, 2=S1, 3=S2, 4=S3, 5=S4, 6=MANA_TAP
        move_act: 0=STAND, 1=WALK, 2=SPRINT
        norm_x, norm_y: optional target coordinates in [0, 1]
        Returns (damage_dealt, stall_penalty, damage_mitigated).
        """
        dmg_dealt = 0.0
        stall_penalty = 0.0
        dmg_mitigated = 0.0

        f.last_combat_event = ""
        f.last_aa_event = ""

        if f.stun_rem > 0.0:
            return 0.0, 0.0, 0.0

        combat_act = int(action[0])
        move_act = int(action[1])
        norm_x = float(action[2]) if len(action) > 2 else None
        norm_y = float(action[3]) if len(action) > 3 else None

        same_cell = f.cell == opp.cell
        dist = math.hypot(f.x - opp.x, f.y - opp.y) if same_cell else 9999.0

        # 1. Execute Unified 100% Atomic Continuous Spatial Movement
        if move_act in (1, 2):
            sprint = (move_act == 2)
            if sprint and f.sp < 50.0:
                sprint = False

            start_x, start_y = f.x, f.y

            # Map normalized [0, 1] continuous coordinates to full arena walkable canvas [40, 920] x [240, 500]
            if norm_x is not None and norm_y is not None:
                target_x = 40.0 + float(np.clip(norm_x, 0.0, 1.0)) * 880.0
                target_y = 240.0 + float(np.clip(norm_y, 0.0, 1.0)) * 260.0
            else:
                target_x, target_y = f.x, f.y

            # allow_door=True: Passing through door trigger physically transitions screens!
            self._move_fighter(f, target_x, target_y, sprint=sprint, allow_door=True)

            disp = math.hypot(f.x - start_x, f.y - start_y)
            if sprint:
                f.sp = max(0.0, f.sp - 5.0)
                # Kinematic stall penalty: burning sprint with near-zero displacement
                if disp < 2.0:
                    stall_penalty = 0.05 * (1.0 - disp / 16.0)

        # 2. Recalculate spatial distance post-movement:
        same_cell = f.cell == opp.cell
        dist = math.hypot(f.x - opp.x, f.y - opp.y) if same_cell else 9999.0
        if same_cell:
            f.target_kind = "opponent"

        if combat_act in (2, 3, 4, 5):
            skill_idx = combat_act - 1  # 1=a1, 2=a2, 3=a3, 4=a4
            if f.cooldowns[skill_idx] <= 0.0 and f.mp >= self.skill_mp[skill_idx]:
                if skill_idx == 3:
                    f.mp -= self.skill_mp[3]
                    f.cooldowns[3] = self.skill_cds[3]
                    f.prepared_strike_rem = 5.0
                    f.prepared_strike_stacks = 2
                    f.last_combat_event = "S3_BUFF"
                else:
                    screen = SCREEN_BY_CELL.get(f.cell)
                    has_safe_npc = screen and screen.npc_max_hp > 0 and self.npc_hp.get(f.cell, 0.0) >= 2000.0

                    if same_cell and dist <= MELEE_RANGE:
                        f.target_kind = "opponent"
                        f.mp -= self.skill_mp[skill_idx]
                        f.cooldowns[skill_idx] = self.skill_cds[skill_idx]
                        f.in_combat = True
                        f.time_since_last_combat_action = 0.0
                        opp.in_combat = True
                        opp.time_since_last_combat_action = 0.0
                        if skill_idx == 4:
                            f.on_guard_rem = 10.0
                        d, m = self._resolve_attack(f, opp, skill_idx)
                        dmg_dealt += d
                        dmg_mitigated += m
                        if d == 0.0:
                            f.last_combat_event = f"S{skill_idx}_DODGED"
                        else:
                            if skill_idx == 2:
                                f.last_combat_event = f"S2_STUN:{d:.0f}"
                            elif skill_idx == 4:
                                f.last_combat_event = f"S4_GUARD:{d:.0f}"
                            else:
                                f.last_combat_event = f"S{skill_idx}_HIT:{d:.0f}"
                    elif skill_idx == 4 and has_safe_npc and not same_cell:
                        # Pre-buff On Guard on Brawler/Restorer NPC before entering combat!
                        f.target_kind = "npc"
                        f.mp -= self.skill_mp[4]
                        f.cooldowns[4] = self.skill_cds[4]
                        f.on_guard_rem = 10.0  # 10s 50% defense shield activated!
                        f.in_combat = True
                        f.time_since_last_combat_action = 0.0
                        f.last_combat_event = "S4_NPC_PREBUFF"
                    else:
                        f.last_combat_event = f"S{skill_idx}_OUT_OF_RANGE"
            elif f.cooldowns[skill_idx] > 0.0:
                f.last_combat_event = f"S{skill_idx}_ON_CD"
            elif f.mp < self.skill_mp[skill_idx]:
                f.last_combat_event = f"S{skill_idx}_LOW_MP"

        # Automatic Auto Attack when ready
        if f.cooldowns[0] <= 0.0 and f.stun_rem <= 0.0:
            if same_cell and dist <= MELEE_RANGE:
                f.target_kind = "opponent"
                f.cooldowns[0] = self.skill_cds[0]
                f.in_combat = True
                f.time_since_last_combat_action = 0.0
                opp.in_combat = True
                opp.time_since_last_combat_action = 0.0
                d, m = self._resolve_attack(f, opp, 0)
                dmg_dealt += d
                dmg_mitigated += m
                if d == 0.0:
                    f.last_aa_event = "AA_DODGED"
                else:
                    f.last_aa_event = f"AA_HIT:{d:.0f}"

        return dmg_dealt, stall_penalty, dmg_mitigated

    def _step_toward_cell(self, f: FighterState, goal_cell: str, sprint: bool) -> None:
        """Finds the door in `f.cell` leading toward `goal_cell` (BFS shortest path) and walks/sprints to its `aim_x, aim_y`."""
        if f.cell == goal_cell:
            return
        door = find_door_toward_cell(f.cell, goal_cell)
        if door is None:
            return
        self._move_fighter(f, door.aim_x, door.aim_y, sprint=sprint, allow_door=True)

    def _tick_timers_and_regen(self, f: FighterState) -> None:
        for i in range(5):
            f.cooldowns[i] = max(0.0, f.cooldowns[i] - DT_SEC)
        f.stun_rem = max(0.0, f.stun_rem - DT_SEC)
        f.prepared_strike_rem = max(0.0, f.prepared_strike_rem - DT_SEC)
        if f.prepared_strike_rem <= 0.0:
            f.prepared_strike_stacks = 0
        f.on_guard_rem = max(0.0, f.on_guard_rem - DT_SEC)
        f.door_cooldown = max(0.0, f.door_cooldown - DT_SEC)

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

        # Fast Safezone Rest Recovery (AQW /rest in safezone restores ~20% per sec)
        if f.cell in ("Enter0", "Enter1") and not f.in_combat:
            f.hp = min(f.max_hp, f.hp + f.max_hp * 0.20 * DT_SEC)
            f.mp = min(f.max_mp, f.mp + f.max_mp * 0.20 * DT_SEC)
            f.sp = min(f.max_sp, f.sp + f.max_sp * 0.20 * DT_SEC)

    def _heuristic_opponent_action(self, opp: FighterState, ag: FighterState) -> int:
        """
        Greedy Tactical 3-4-5-2 Opponent:
        1. Greedy Approach: Always walks towards Blue Agent (NO sprint during approach to save SP!).
        2. In Melee: Executes combo priority:
           - 3 (Skill 2: Imbalancing Strike 4s Stun "Off Balance")
           - 4 (Skill 3: Prepared Strike Crit Buff)
           - 5 (Skill 4: On Guard 50% Barrier)
           - 2 (Skill 1: Decisive Strike Damage Burst)
           - AA auto-fires
        3. Tactical Retreat / Kite: When Blue Agent is NO LONGER stunned (stun expired)
           AND Off Balance is still on cooldown:
           - Retreats / kites away from Blue using SPRINT (if SP >= 50) until Off Balance is ready again!
           - Sprint is strictly reserved for this retreat only!
        4. Post-Kill Safezone Rest: When Red eliminates Blue:
           - Walks directly back to Safezone (Enter1).
           - Rests until HP and MP are 100% full.
           - Once full, departs immediately back to combat!
        """
        if self.opponent_policy is not None:
            obs = self._get_obs(opp, ag)
            return int(self.opponent_policy(obs))

        # 1. Post-Kill Behavior: Walk back to own Safezone and rest until full
        opp_sz = "Enter1" if opp.team == "red" else "Enter0"
        if self.enemy_returning_to_sz:
            if opp.cell != opp_sz:
                self._step_toward_cell(opp, opp_sz, sprint=False)
                return 0
            else:
                # Inside Safezone: Rest until 100% full HP and MP
                if opp.hp >= opp.max_hp - 1.0 and opp.mp >= opp.max_mp - 1.0:
                    self.enemy_returning_to_sz = False
                return 0  # Rest in place in safezone

        same_cell = opp.cell == ag.cell
        dist = math.hypot(opp.x - ag.x, opp.y - ag.y) if same_cell else 9999.0

        # 2. Inter-screen Navigation & Mid Control:
        if not same_cell:
            opp_territory = (
                ("Morale1A", "Morale1B", "Morale1C", "Resource1A", "Resource1B", "Captain1")
                if opp.team == "red"
                else ("Morale0A", "Morale0B", "Morale0C", "Resource0A", "Resource0B", "Captain0")
            )
            # If agent invaded opponent's territory, intercept and defend!
            if ag.cell in opp_territory:
                self._step_toward_cell(opp, ag.cell, sprint=False)
                return 0

            # If opponent is not yet in Mid (Crosslower), advance forward toward Mid
            if opp.cell != "Crosslower":
                self._step_toward_cell(opp, "Crosslower", sprint=False)
                return 0

            # In Mid (Crosslower): hold position near opponent corridor entrance
            target_hold_x, target_hold_y = (260.0, 350.0) if opp.team == "red" else (700.0, 350.0)
            dist_to_hold = math.hypot(opp.x - target_hold_x, opp.y - target_hold_y)
            if dist_to_hold > 30.0:
                self._move_fighter(opp, target_hold_x, target_hold_y, sprint=False, allow_door=False)
            return 0  # Wait in combat stance at Mid doorway

        # 3. Intra-screen Combat:
        # Check if Blue is no longer stunned AND Red's Stun is on cooldown
        s2_ready = (opp.cooldowns[2] <= 0.0) and (opp.mp >= 30.0)

        if ag.stun_rem <= 0.0 and not s2_ready:
            # Stun has worn off and Off Balance is on cooldown -> MUNDUR / KITE!
            # Use sprint if SP >= 50!
            return 8 if opp.sp >= 50.0 else 7

        # 4. If Blue IS stunned OR Red is ready to land Stun:
        if dist > MELEE_RANGE:
            # Approach Blue into melee, WALK ONLY (preserve SP for retreat)
            return 1

        # 5. In Melee Range: Execute combo 3 - 4 - 5 - 2
        # Priority 1: Key 3 (Skill 2: Imbalancing Strike 4s Stun "Off Balance", CD=10s, MP=30)
        if opp.cooldowns[2] <= 0.0 and opp.mp >= 30.0:
            return 4  # USE_SKILL_2

        # Priority 2: Key 4 (Skill 3: Prepared Strike Crit Buff, CD=15s, MP=20)
        if opp.cooldowns[3] <= 0.0 and opp.mp >= 20.0 and opp.prepared_strike_stacks == 0:
            return 5  # USE_SKILL_3

        # Priority 3: Key 5 (Skill 4: On Guard 50% Barrier, CD=30s, MP=30)
        if opp.cooldowns[4] <= 0.0 and opp.mp >= 30.0 and opp.on_guard_rem <= 0.0:
            return 6  # USE_SKILL_4

        # Priority 4: Key 2 (Skill 1: Decisive Strike Damage, CD=2s, MP=15)
        if opp.cooldowns[1] <= 0.0 and opp.mp >= 15.0:
            return 3  # USE_SKILL_1

        return 0  # FIGHT_STAND (Auto Attack)

    def _respawn_fighter(self, f: FighterState, cell: str, x: float, y: float) -> None:
        """Respawns fighter at designated safezone pad matching AQW /bludrutbrawl rules."""
        f.hp = f.max_hp
        f.mp = f.max_mp
        f.sp = f.max_sp
        f.cell = cell
        f.x = x
        f.y = y
        f.in_combat = False
        f.time_since_last_combat_action = 12.0
        f.cooldowns = [0.0, 0.0, 0.0, 0.0, 0.0]
        f.stun_rem = 0.0
        f.prepared_strike_rem = 0.0
        f.prepared_strike_stacks = 0
        f.on_guard_rem = 0.0
        f.door_cooldown = 1.0
        f.target_kind = "opponent"
        if f == self.agent:
            self.agent_no_combat_away_from_mid_time = 0.0
            self.agent_entered_mid = False

    def step(self, action: int | tuple | list | np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        opp_action = self._heuristic_opponent_action(self.enemy, self.agent)
        self.last_opp_action = opp_action

        prev_agent_cell = self.agent.cell
        prev_enemy_cell = self.enemy.cell
        prev_agent_x = self.agent.x
        prev_agent_y = self.agent.y
        prev_agent_hp = self.agent.hp
        prev_enemy_hp = self.enemy.hp
        prev_agent_mp = self.agent.mp
        prev_agent_sp = self.agent.sp

        combat_act = int(action[0]) if isinstance(action, (tuple, list, np.ndarray)) else int(action)
        move_act = int(action[1]) if isinstance(action, (tuple, list, np.ndarray)) else 0

        if isinstance(action, (tuple, list, np.ndarray)):
            dmg_by_agent, pen_agent, agent_mit = self._apply_compound_action(self.agent, self.enemy, action)
        else:
            dmg_by_agent, pen_agent, agent_mit = self._apply_action(self.agent, self.enemy, int(action))

        dmg_by_enemy, _, enemy_mit = self._apply_action(self.enemy, self.agent, int(opp_action))

        self._tick_timers_and_regen(self.agent)
        self._tick_timers_and_regen(self.enemy)

        # 1. Risk-Sensitive Damage Trade:
        # Taking damage when low on HP is up to 3x more lethal/punishing than when healthy.
        # This naturally incentivizes low-HP agents to EVADE / KITE / STALL, while healthy agents pursue!
        dmg_dealt = max(0.0, prev_enemy_hp - self.enemy.hp)
        dmg_taken = max(0.0, prev_agent_hp - self.agent.hp)
        hp_vuln_mult = 1.0 + 2.0 * max(0.0, 1.0 - (self.agent.hp / self.agent.max_hp))
        r_ehp = (dmg_dealt / MAX_HP) * 2.0 - (dmg_taken / MAX_HP) * 2.0 * hp_vuln_mult

        # 2. Damage Mitigation Dividend (absorbing damage with shields/buffs)
        r_mitigation = (agent_mit / MAX_HP) * 2.0

        # 3. Crowd Control Dividend (incapacitating the opponent in-combat)
        r_stun = 0.04 if (self.enemy.stun_rem > 0.0 and self.enemy.cell == self.agent.cell) else 0.0

        # 4. Navigation Guidance across 16 screens:
        # Only reward positive progress towards target room; NEVER penalize disengaging/kiting!
        if self.agent.cell != self.enemy.cell:
            target_nav_cell = self.enemy.cell if (self.enemy.cell not in ("Enter0", "Enter1")) else "Crosslower"
            prev_hop_dist = CELL_GRAPH_DIST.get((prev_agent_cell, target_nav_cell), 0)
            curr_hop_dist = CELL_GRAPH_DIST.get((self.agent.cell, target_nav_cell), 0)
            r_nav = max(0.0, (prev_hop_dist - curr_hop_dist) * 0.30)

            # Intra-room spatial door guidance: guide (targetX, targetY) clicks toward exit door
            target_door = find_door_toward_cell(self.agent.cell, target_nav_cell)
            if target_door:
                prev_d_door = math.hypot(prev_agent_x - target_door.aim_x, prev_agent_y - target_door.aim_y)
                curr_d_door = math.hypot(self.agent.x - target_door.aim_x, self.agent.y - target_door.aim_y)
                r_nav += max(0.0, (prev_d_door - curr_d_door) / 500.0 * 0.05)
        else:
            r_nav = 0.0

        # 5. Dynamic Concave Mana Potential (Logarithmic Diminishing Marginal Utility)
        def mana_potential(mp: float) -> float:
            return 0.5 * math.log(1.0 + 4.0 * max(0.0, mp) / MAX_MP)

        f_mana = mana_potential(self.agent.mp) - mana_potential(prev_agent_mp)

        # 6. Dynamic Stamina Potential
        def stamina_potential(sp: float) -> float:
            return 0.2 * (max(0.0, sp) / MAX_SP)

        f_stamina = stamina_potential(self.agent.sp) - stamina_potential(prev_agent_sp)

        # 7. Step penalty (Time efficiency to encourage decisive win)
        r_time = -0.002

        # 8. Safezone Rules & Tournament Enforcement
        # Rule A: Camping in Safezone > 15s violates match regulations: heavy escalating penalty
        # Rule B: Re-entering or retreating into Safezone before opponent is eliminated is strictly illegal
        r_sz = 0.0
        if self.agent.cell in ("Enter0", "Enter1"):
            self.agent_sz_time += DT_SEC
            if self.agent_sz_time > 15.0:
                overstay = self.agent_sz_time - 15.0
                r_sz -= (0.50 + min(1.50, overstay * 0.1))

            if self.agent_has_exited_sz and not self.wave_won:
                r_sz -= 1.0  # Illegal presence in Safezone mid-fight
        else:
            if not self.agent_has_exited_sz:
                self.agent_has_exited_sz = True
                r_sz += 2.0  # Bonus for breaking out of Safezone to engage
            self.agent_sz_time = 0.0

        if self.agent_attempted_illegal_sz_entry:
            r_sz -= 5.0  # Severe penalty for attempting illegal retreat into Safezone while enemy is alive
            self.agent_attempted_illegal_sz_entry = False

        # 8b. Intra-Room Tactical Spacing & Stutter-Step Dynamics
        r_tactical = 0.0
        dist = math.hypot(self.agent.x - self.enemy.x, self.agent.y - self.enemy.y) if self.agent.cell == self.enemy.cell else 9999.0
        if self.agent.cell == self.enemy.cell:
            # Spacing / Kiting dividend: when opponent has On Guard barrier (-50%), reward kiting away (>200px)
            if self.enemy.on_guard_rem > 0.0:
                if dist > 200.0:
                    r_tactical += 0.05  # Reward maintaining safe distance while enemy has barrier
                else:
                    r_tactical -= 0.02  # Disincentivize face-tanking into enemy 50% barrier
            # Stutter-step dividend: reward moving simultaneously while attacking
            if move_act in (1, 2) and combat_act in (1, 2, 3, 4, 5):
                r_tactical += 0.02
            # Anti-corner penalty: penalize hugging room edges / corners during fight
            if self.agent.x < 70.0 or self.agent.x > 890.0 or self.agent.y < 265.0 or self.agent.y > 475.0:
                r_tactical -= 0.02

        # 8c. Mid Objective & Anti-Stalling / 1-Minute Tournament Rule Enforcement
        # In competitive BludrutBrawl, players must contest Mid (Crosslower) within 60s if out of combat.
        # Camping in B3/B2/SZ is toxic and penalized heavily!
        r_mid = 0.0
        r_stall = 0.0
        is_combat = (
            self.agent.in_combat or
            self.enemy.in_combat or
            dmg_dealt > 0.0 or
            dmg_taken > 0.0 or
            (self.agent.cell == self.enemy.cell and dist <= 300.0)
        )

        if self.agent.cell == "Crosslower":
            self.agent_no_combat_away_from_mid_time = 0.0
            if not self.agent_entered_mid:
                self.agent_entered_mid = True
                r_mid += 1.0  # Big arrival dividend for reaching & contesting Mid
            r_mid += 0.01  # Mid presence control dividend
        elif is_combat:
            self.agent_no_combat_away_from_mid_time = 0.0
        else:
            self.agent_entered_mid = False
            self.agent_no_combat_away_from_mid_time += DT_SEC

            # Stalling penalty:
            # 0..20s: Grace traversal period from Safezone/B3 to Mid
            # 20..60s: Escalating anti-camping shaping penalty
            if self.agent_no_combat_away_from_mid_time > 20.0:
                elapsed_excess = self.agent_no_combat_away_from_mid_time - 20.0
                r_stall += 0.03 * min(1.0, elapsed_excess / 40.0)

            # > 60s: CRITICAL 1-MINUTE TOURNAMENT RULE VIOLATION (TOXIC CAMPING)
            if self.agent_no_combat_away_from_mid_time > 60.0:
                r_stall += 0.50  # -5.0 penalty per second!
                if int(self.agent_no_combat_away_from_mid_time * 10) % 50 == 0:
                    self.last_kill_event = f"⚠️ [1-MIN RULE VIOLATION] Blue agent camping away from Mid ({self.agent_no_combat_away_from_mid_time:.1f}s)! Heavy stalling penalty applied."

        reward = r_ehp + r_mitigation + r_stun + r_nav + f_mana + f_stamina + r_time + r_sz + r_tactical + r_mid - r_stall - pen_agent

        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        # Check deaths & respawns (Target: 10 Kills)
        self.last_kill_event = ""
        agent_sz = "Enter0" if self.agent.team == "blue" else "Enter1"
        enemy_sz = "Enter1" if self.agent.team == "blue" else "Enter0"
        agent_sz_x = 350.0 if self.agent.team == "blue" else 600.0
        enemy_sz_x = 600.0 if self.agent.team == "blue" else 350.0

        if self.enemy.hp <= 0.0 and self.agent.hp <= 0.0:
            self.blue_kills += 1
            self.red_kills += 1
            self.last_kill_event = f"⚖️ MUTUAL KILL! Score: Blue {self.blue_kills} - {self.red_kills} Red"
            reward += 5.0
            if self.blue_kills >= self.target_kills or self.red_kills >= self.target_kills:
                terminated = True
                ag_score = self.blue_kills if self.agent.team == "blue" else self.red_kills
                en_score = self.red_kills if self.agent.team == "blue" else self.blue_kills
                info["winner"] = "draw" if ag_score == en_score else ("agent" if ag_score > en_score else "enemy")
            else:
                self._respawn_fighter(self.agent, agent_sz, agent_sz_x, 360.0)
                self._respawn_fighter(self.enemy, enemy_sz, enemy_sz_x, 360.0)
                self.agent_has_exited_sz = False
                self.agent_sz_time = 0.0
                self.wave_won = True
                self.enemy_returning_to_sz = False

        elif self.enemy.hp <= 0.0:
            if self.agent.team == "blue":
                self.blue_kills += 1
            else:
                self.red_kills += 1
            self.last_kill_event = f"🏆 {self.agent.team.upper()} ELIMINATED {self.enemy.team.upper()}! Score: Blue {self.blue_kills} - {self.red_kills} Red"
            reward += 10.0 + 5.0 * (self.agent.hp / self.agent.max_hp)
            self.wave_won = True
            ag_score = self.blue_kills if self.agent.team == "blue" else self.red_kills
            if ag_score >= self.target_kills:
                terminated = True
                info["winner"] = "agent"
            else:
                self._respawn_fighter(self.enemy, enemy_sz, enemy_sz_x, 360.0)
                self.enemy_returning_to_sz = False

        elif self.agent.hp <= 0.0:
            if self.enemy.team == "blue":
                self.blue_kills += 1
            else:
                self.red_kills += 1
            self.last_kill_event = f"💀 {self.enemy.team.upper()} ELIMINATED {self.agent.team.upper()}! Score: Blue {self.blue_kills} - {self.red_kills} Red"
            reward -= 10.0
            en_score = self.blue_kills if self.enemy.team == "blue" else self.red_kills
            if en_score >= self.target_kills:
                terminated = True
                info["winner"] = "enemy"
            else:
                self._respawn_fighter(self.agent, agent_sz, agent_sz_x, 360.0)
                self.agent_has_exited_sz = False
                self.agent_sz_time = 0.0
                self.enemy_returning_to_sz = True

        elif self.step_count >= self.max_steps:
            truncated = True
            ag_score = self.blue_kills if self.agent.team == "blue" else self.red_kills
            en_score = self.red_kills if self.agent.team == "blue" else self.blue_kills
            info["winner"] = "agent" if ag_score > en_score else ("enemy" if en_score > ag_score else "draw")

        info["blue_kills"] = self.blue_kills
        info["red_kills"] = self.red_kills
        info["kill_event"] = self.last_kill_event
        info["stalling_time"] = self.agent_no_combat_away_from_mid_time

        return self._get_obs(self.agent, self.enemy), float(reward), terminated, truncated, info
