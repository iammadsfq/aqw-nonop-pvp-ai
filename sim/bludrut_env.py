#!/usr/bin/env python3
"""
sim/bludrut_env.py
Arena Orchestrator Bludrut Brawl 16-Screen Map & Continuous Point Physics.
Mendukung 3-Channel Decoupled Compound Actions.
Dikalibrasi 1:1 dengan Topologi Lantai AQW & Engine Aturan Non-OP PvP.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union

import numpy as np

try:
    from sim.world.bludrut_map import (
        SCREEN_NAMES,
        SCREEN_INDICES,
        CELL_GRAPH_DIST,
        WALK_BOUNDS,
        ROOM_DOORS,
        SAFEZONE_SPAWN,
        BLUDRUT_NPCS,
        DoorTrigger,
    )
    from sim.combat.mrm import ManaRecoveryModel, RegenerationEngine
except ModuleNotFoundError:
    try:
        from world.bludrut_map import (
            SCREEN_NAMES,
            SCREEN_INDICES,
            CELL_GRAPH_DIST,
            WALK_BOUNDS,
            ROOM_DOORS,
            SAFEZONE_SPAWN,
            BLUDRUT_NPCS,
            DoorTrigger,
        )
        from combat.mrm import ManaRecoveryModel, RegenerationEngine
    except ModuleNotFoundError:
        import sys
        from pathlib import Path
        _sim_dir = Path(__file__).resolve().parent
        if str(_sim_dir) not in sys.path:
            sys.path.insert(0, str(_sim_dir))
        from world.bludrut_map import (
            SCREEN_NAMES,
            SCREEN_INDICES,
            CELL_GRAPH_DIST,
            WALK_BOUNDS,
            ROOM_DOORS,
            SAFEZONE_SPAWN,
            BLUDRUT_NPCS,
            DoorTrigger,
        )
        from combat.mrm import ManaRecoveryModel, RegenerationEngine


# ==============================================================================
# 1. CONSTANTS, BOUNDS & SIMULATION HYPERPARAMETERS
# ==============================================================================
# Dimensi lantai yang valid diinjak avatar
FLOOR_X_MIN = WALK_BOUNDS["x_min"]  # 35.0
FLOOR_X_MAX = WALK_BOUNDS["x_max"]  # 930.0
FLOOR_Y_MIN = WALK_BOUNDS["y_min"]  # 255.0
FLOOR_Y_MAX = WALK_BOUNDS["y_max"]  # 500.0

FLOOR_WIDTH = FLOOR_X_MAX - FLOOR_X_MIN    # 895.0 px (Dynamic Range X penuh)
FLOOR_HEIGHT = FLOOR_Y_MAX - FLOOR_Y_MIN  # 245.0 px (Dynamic Range Y penuh)
MAX_DIAG_DIST = math.hypot(FLOOR_WIDTH, FLOOR_HEIGHT)  # ~927.9 px

# Parameter combat & jangkauan serangan (Persegi Panjang Standar AQW World.as:9398)
# abs(dx) <= 301 * 1.15 = 346.15 px (tolerance ~340.0 px)
# abs(dy) <= 30 * 1.15 = 34.5 px (vertical elevation limit ~34.0 px)
MELEE_RANGE_X = 340.0
MELEE_RANGE_Y = 34.0
BASE_AUTO_DAMAGE = 150.0

# Kecepatan & mobilitas
BASE_SPEED_PX = 8.0 * 20.0 * 0.10   # 16.0 px/tick (160 px/s)
SPRINT_MULT = 3.0                   # Telemetri AQW: 24 / 8 = 3.0x
SP_SPRINT_COST = 50.0               # Konsumsi SP untuk sprint
MAX_DISP_PX = 300.0                 # Jarak target relatif maksimum per step

# Siklus simulasi & match
SIM_TIMESTEP = 0.10                 # 10 Hz simulation loop
DEFAULT_MAX_MATCH_TIME = 1200.0     # 20 Menit (12000 tick) batas maksimum server PvP AQW


# ==============================================================================
# 2. ACTION SPECIFICATION
# ==============================================================================
@dataclass(frozen=True)
class ActionSpec:
    combat_act: int = 0         # 0: None, 1: AA, 2..5: S1..S4
    move_type: int = 0          # 0: Stand, 1: Walk, 2: Sprint
    target_dx: float = 0.0      # Continuous normalized displacement [-1.0, 1.0]
    target_dy: float = 0.0      # Continuous normalized displacement [-1.0, 1.0]


RawActionInput = Union[ActionSpec, Tuple[int, int, float, float], List[Union[int, float]]]


def _parse_action(raw: RawActionInput) -> ActionSpec:
    if isinstance(raw, ActionSpec):
        return raw
    return ActionSpec(
        combat_act=int(raw[0]),
        move_type=int(raw[1]),
        target_dx=float(np.clip(raw[2], -1.0, 1.0)),
        target_dy=float(np.clip(raw[3], -1.0, 1.0)),
    )


# ==============================================================================
# 3. COMBAT SPEC & DATA-DRIVEN PROFILES
# ==============================================================================
@dataclass(frozen=True)
class SkillSpec:
    name: str
    key: str
    cooldown: float
    mana_cost: float
    damage_mult: float
    is_heal: bool = False
    is_self_buff: bool = False
    applies_stun: bool = False
    stun_duration: float = 0.0
    applies_shield: bool = False
    shield_duration: float = 0.0
    applies_charge_crit: bool = False


WARRIOR_PROFILE: Dict[int, SkillSpec] = {
    1: SkillSpec("Auto Attack", "aa", cooldown=2.0, mana_cost=0.0, damage_mult=1.0),
    2: SkillSpec("Decisive Strike", "a1", cooldown=2.0, mana_cost=15.0, damage_mult=1.9),
    3: SkillSpec("Imbalancing Strike", "a2", cooldown=10.0, mana_cost=30.0, damage_mult=0.6, applies_stun=True, stun_duration=4.0),
    4: SkillSpec("Prepared Strike", "a3", cooldown=15.0, mana_cost=20.0, damage_mult=0.0, is_self_buff=True, applies_charge_crit=True),
    5: SkillSpec("On Guard", "a4", cooldown=30.0, mana_cost=30.0, damage_mult=0.75, applies_shield=True, shield_duration=10.0),
}

HEALER_PROFILE: Dict[int, SkillSpec] = {
    1: SkillSpec("Auto Attack", "aa", cooldown=2.0, mana_cost=0.0, damage_mult=0.8),
    2: SkillSpec("Healing Word", "a1", cooldown=3.0, mana_cost=20.0, damage_mult=2.0, is_heal=True),
    3: SkillSpec("Radiance", "a2", cooldown=12.0, mana_cost=30.0, damage_mult=0.0, is_self_buff=True),
    4: SkillSpec("Energy Flow", "a3", cooldown=10.0, mana_cost=15.0, damage_mult=0.5),
    5: SkillSpec("Clear Mind", "a4", cooldown=20.0, mana_cost=20.0, damage_mult=0.0, is_self_buff=True),
}


# ==============================================================================
# 4. AURA & ENTITY DOMAIN MODEL
# ==============================================================================
@dataclass
class ActiveAura:
    """Generic Data-Driven Aura representation based on AQW 'aura+p' packets."""
    name: str
    duration_rem: float
    is_stun: bool = False
    crit_charges: int = 0
    effects: Dict[str, float] = field(default_factory=dict)


@dataclass
class Entity:
    ent_id: str
    name: str
    team: int
    is_player: bool
    cell: str
    x: float
    y: float
    hp: float = 8455.0
    max_hp: float = 8455.0
    mp: float = 100.0
    max_mp: float = 100.0
    sp: float = 100.0
    max_sp: float = 100.0
    is_alive: bool = True
    role_name: str = "Warrior"
    skills: Dict[int, SkillSpec] = field(default_factory=lambda: WARRIOR_PROFILE)
    gcd_rem: float = 0.0
    cooldowns: List[float] = field(default_factory=lambda: [0.0] * 6)
    time_since_combat: float = 15.0
    in_combat: bool = False

    # Dynamic Aura Management
    active_auras: Dict[str, ActiveAura] = field(default_factory=dict)

    # Base Stats (Class-agnostic baseline 1.0 untuk in/out damage multiplier)
    dodge_rate: float = 0.2029
    crit_rate: float = 0.2285
    crit_mult: float = 2.9032
    base_haste: float = 0.1054
    base_cpo: float = 1.00
    base_cai: float = 1.00

    # Safezone & Navigation States
    can_enter_sz: bool = False
    touched_mid: bool = False
    illegal_sz_entered: bool = False
    used_sprint: bool = False
    sprint_rem: float = 0.0
    respawn_timer: float = 0.0

    @property
    def is_stunned(self) -> bool:
        return any(a.is_stun and a.duration_rem > 0.0 for a in self.active_auras.values())

    @property
    def incoming_dmg_mult(self) -> float:
        mult = self.base_cai
        for a in self.active_auras.values():
            if a.duration_rem > 0.0 and "cai" in a.effects:
                mult *= (1.0 + a.effects["cai"])
        return max(0.0, mult)

    @property
    def outgoing_dmg_mult(self) -> float:
        mult = self.base_cpo
        for a in self.active_auras.values():
            if a.duration_rem > 0.0 and "cpo" in a.effects:
                mult *= (1.0 + a.effects["cpo"])
        return max(0.0, mult)

    def reset(self, cell: str, x: float, y: float) -> None:
        self.cell = cell
        self.x = x
        self.y = y
        self.hp = self.max_hp
        self.mp = self.max_mp
        self.sp = self.max_sp
        self.is_alive = True
        self.respawn_timer = 0.0
        self.gcd_rem = 0.0
        self.cooldowns = [0.0] * 6
        self.time_since_combat = 15.0
        self.in_combat = False
        self.active_auras.clear()
        self.can_enter_sz = False
        self.touched_mid = False
        self.illegal_sz_entered = False
        self.used_sprint = False
        self.sprint_rem = 0.0

    def tick_status(self, dt: float, is_regen_tick: bool) -> None:
        if not self.is_alive:
            self.sprint_rem = 0.0
            self.respawn_timer = max(0.0, self.respawn_timer - dt)
            if self.respawn_timer <= 1e-4:
                self.is_alive = True
                self.respawn_timer = 0.0
                self.hp = self.max_hp
                self.mp = self.max_mp
                self.sp = self.max_sp
                self.gcd_rem = 0.0
                self.cooldowns = [0.0] * 6
                self.time_since_combat = 15.0
                self.in_combat = False
                self.active_auras.clear()
                self.can_enter_sz = False
                self.touched_mid = False
                self.illegal_sz_entered = False
                self.sprint_rem = 0.0
                sz_cell = "Enter0" if self.team == 0 else "Enter1"
                self.cell = sz_cell
                self.x = SAFEZONE_SPAWN[sz_cell]["x"]
                self.y = SAFEZONE_SPAWN[sz_cell]["y"]
            return

        self.gcd_rem = max(0.0, self.gcd_rem - dt)
        self.sprint_rem = max(0.0, self.sprint_rem - dt)
        for i in range(len(self.cooldowns)):
            self.cooldowns[i] = max(0.0, self.cooldowns[i] - dt)

        # Dynamic Aura Decay
        for aura in self.active_auras.values():
            aura.duration_rem = max(0.0, aura.duration_rem - dt)

        self.active_auras = {
            name: aura for name, aura in self.active_auras.items() if aura.duration_rem > 0.0
        }

        self.time_since_combat += dt
        self.in_combat = self.time_since_combat < 10.0

        if is_regen_tick and self.is_player:
            if self.in_combat:
                self.sp = RegenerationEngine.process_in_combat_sp_regen(self.sp, self.max_sp)
            else:
                self.hp, self.sp = RegenerationEngine.process_ooc_regen(
                    self.hp, self.max_hp, self.sp, self.max_sp
                )
                self.mp = min(self.max_mp, self.mp + 15.0)


# ==============================================================================
# 5. GEOMETRY UTILITIES
# ==============================================================================
def _intersects_aabb(x1: float, y1: float, x2: float, y2: float, door: DoorTrigger) -> bool:
    """Liang-Barsky segment clipping terhadap bounding box AABB pintu."""
    if door.contains(x1, y1) or door.contains(x2, y2):
        return True

    dx = x2 - x1
    dy = y2 - y1
    p = [-dx, dx, -dy, dy]
    q = [x1 - door.x_min, door.x_max - x1, y1 - door.y_min, door.y_max - y1]
    u1, u2 = 0.0, 1.0

    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return False
        else:
            t = qi / pi
            if pi < 0:
                if t > u2:
                    return False
                if t > u1:
                    u1 = t
            else:
                if t < u1:
                    return False
                if t < u2:
                    u2 = t
    return u1 <= u2


# ==============================================================================
# 6. ENVIRONMENT ORCHESTRATOR
# ==============================================================================
class BludrutBrawlEnv:
    DT = SIM_TIMESTEP
    REGEN_INTERVAL = RegenerationEngine.TICK_INTERVAL
    TOTAL_OBS_DIM = 30
    DEFAULT_TARGET_KILLS = 10
    RESPAWN_DELAY = 10.0

    def __init__(
        self,
        p0_role: str = "Warrior",
        p1_role: str = "Warrior",
        seed: Optional[int] = None,
        target_kills: int = 10,
        max_match_time: float = DEFAULT_MAX_MATCH_TIME,
    ):
        self.rng = random.Random(seed)
        self.match_time = 0.0
        self.regen_timer = 0.0
        self.target_kills = target_kills
        self.max_match_time = max_match_time
        self.p0_kills = 0
        self.p1_kills = 0
        self.p0_deaths = 0
        self.p1_deaths = 0

        self.p0 = Entity(
            ent_id="p:187461",
            name="uma",
            team=0,
            is_player=True,
            cell="Enter0",
            x=SAFEZONE_SPAWN["Enter0"]["x"],
            y=SAFEZONE_SPAWN["Enter0"]["y"],
            role_name=p0_role,
            skills=WARRIOR_PROFILE if p0_role == "Warrior" else HEALER_PROFILE,
        )
        self.p1 = Entity(
            ent_id="p:188276",
            name="master_ninja",
            team=1,
            is_player=True,
            cell="Enter1",
            x=SAFEZONE_SPAWN["Enter1"]["x"],
            y=SAFEZONE_SPAWN["Enter1"]["y"],
            role_name=p1_role,
            skills=WARRIOR_PROFILE if p1_role == "Warrior" else HEALER_PROFILE,
        )
        self.npcs: Dict[int, Entity] = {}
        self._init_npcs()

    def _init_npcs(self) -> None:
        self.npcs.clear()
        for defn in BLUDRUT_NPCS:
            self.npcs[defn.mon_map_id] = Entity(
                ent_id=f"m:{defn.mon_map_id}",
                name=defn.name,
                team=defn.team,
                is_player=False,
                cell=defn.cell,
                x=488.0,
                y=350.0,
                hp=defn.max_hp,
                max_hp=defn.max_hp,
                mp=100.0,
                max_mp=100.0,
                sp=100.0,
                max_sp=100.0,
                dodge_rate=0.05,
                crit_rate=0.05,
            )

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.match_time = 0.0
        self.regen_timer = 0.0
        self.p0_kills = 0
        self.p1_kills = 0
        self.p0_deaths = 0
        self.p1_deaths = 0
        self.p0.reset("Enter0", SAFEZONE_SPAWN["Enter0"]["x"], SAFEZONE_SPAWN["Enter0"]["y"])
        self.p1.reset("Enter1", SAFEZONE_SPAWN["Enter1"]["x"], SAFEZONE_SPAWN["Enter1"]["y"])
        self._init_npcs()
        return self._get_obs(self.p0, self.p1), self._get_obs(self.p1, self.p0)

    # --------------------------------------------------------------------------
    # OBSERVATION EXTRACTORS (FULL DYNAMIC RANGE SCALING)
    # --------------------------------------------------------------------------
    def _extract_self_features(self, me: Entity) -> List[float]:
        cell_idx = float(SCREEN_INDICES.get(me.cell, 0)) / 15.0
        norm_x = min(1.0, max(0.0, (me.x - FLOOR_X_MIN) / FLOOR_WIDTH))
        norm_y = min(1.0, max(0.0, (me.y - FLOOR_Y_MIN) / FLOOR_HEIGHT))

        return [
            me.hp / me.max_hp,
            me.mp / me.max_mp,
            me.sp / me.max_sp,
            1.0 if me.sp >= SP_SPRINT_COST else 0.0,
            1.0 if me.in_combat else 0.0,
            0.0 if me.in_combat else min(1.0, me.time_since_combat / 10.0),
            cell_idx,
            norm_x,
            norm_y,
            1.0 if me.is_alive else 0.0,
            1.0 if (me.cooldowns[2] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            1.0 if (me.cooldowns[3] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            1.0 if (me.cooldowns[4] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            1.0 if (me.cooldowns[5] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            me.incoming_dmg_mult / 2.0, # Baseline 1.0 -> 0.5; range [0.0, 1.0]
            me.outgoing_dmg_mult / 2.0, # ini sebelumnya hilang
            1.0 if me.is_stunned else 0.0,
        ]

    def _extract_opponent_features(self, me: Entity, opp: Entity) -> List[float]:
        same_cell = 1.0 if (me.cell == opp.cell and opp.is_alive) else 0.0
        opp_cell_idx = float(SCREEN_INDICES.get(opp.cell, 0)) / 15.0

        if same_cell:
            rel_x = float(np.clip((opp.x - me.x) / FLOOR_WIDTH, -1.0, 1.0))
            rel_y = float(np.clip((opp.y - me.y) / FLOOR_HEIGHT, -1.0, 1.0))
            in_melee = 1.0 if (abs(opp.x - me.x) <= MELEE_RANGE_X and abs(opp.y - me.y) <= MELEE_RANGE_Y) else 0.0
        else:
            rel_x, rel_y = 0.0, 0.0
            in_melee = 0.0

        return [
            1.0,  # is_valid
            1.0 if opp.is_alive else 0.0,
            0.0,  # is_ally
            same_cell,
            opp_cell_idx,
            rel_x,
            rel_y,
            in_melee,
            opp.hp / opp.max_hp,
            opp.mp / opp.max_mp,
            opp.incoming_dmg_mult / 2.0,
            1.0 if opp.is_stunned else 0.0,
            opp.outgoing_dmg_mult / 2.0, # Baseline 1.1x -> 0.55; 1.265x -> 0.6325
        ]

    def _get_obs(self, me: Entity, opp: Entity) -> np.ndarray:
        obs = np.array(self._extract_self_features(me) + self._extract_opponent_features(me, opp), dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    # --------------------------------------------------------------------------
    # STEP EXECUTION
    # --------------------------------------------------------------------------
    def step(
        self, a0: RawActionInput, a1: RawActionInput
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, dict]:
        act0 = _parse_action(a0)
        act1 = _parse_action(a1)

        self.match_time += self.DT
        self.regen_timer += self.DT
        is_regen_tick = self.regen_timer >= self.REGEN_INTERVAL
        if is_regen_tick:
            self.regen_timer = 0.0

        # Snapshot State t-1
        prev_p0 = {
            "hp": self.p0.hp,
            "mp": self.p0.mp,
            "cell": self.p0.cell,
            "x": self.p0.x,
            "y": self.p0.y,
            "alive": self.p0.is_alive,
        }
        prev_p1 = {
            "hp": self.p1.hp,
            "mp": self.p1.mp,
            "cell": self.p1.cell,
            "x": self.p1.x,
            "y": self.p1.y,
            "alive": self.p1.is_alive,
        }

        # 1. Update Domain Model Status (termasuk respawn timer countdown)
        self.p0.tick_status(self.DT, is_regen_tick)
        self.p1.tick_status(self.DT, is_regen_tick)
        for npc in self.npcs.values():
            npc.tick_status(self.DT, is_regen_tick)

        # 2. Movement Phase
        self._process_movement(self.p0, act0)
        self._process_movement(self.p1, act1)

        # 3. Combat Phase
        self._process_combat(self.p0, self.p1, act0.combat_act)
        self._process_combat(self.p1, self.p0, act1.combat_act)

        # 4. Check lethal events occurring in this step
        p0_died = prev_p0["alive"] and not self.p0.is_alive
        p1_died = prev_p1["alive"] and not self.p1.is_alive

        if p0_died:
            self.p1_kills += 1
            self.p0_deaths += 1
        if p1_died:
            self.p0_kills += 1
            self.p1_deaths += 1

        # 4b. Update Zone Mode Triggers (Mode 1: Respawn/Push vs Mode 2: Mid Contested / Met Opponent)
        if p0_died or p1_died:
            self.p0.touched_mid = False
            self.p1.touched_mid = False
            if p0_died:
                self.p0.can_enter_sz = False
            if p1_died:
                self.p1.can_enter_sz = False
        elif self.p0.is_alive and self.p1.is_alive:
            if self.p0.cell == "Crosslower":
                self.p0.touched_mid = True
                self.p1.can_enter_sz = False
            if self.p1.cell == "Crosslower":
                self.p1.touched_mid = True
                self.p0.can_enter_sz = False
            if self.p0.cell == self.p1.cell:
                self.p0.touched_mid = True
                self.p1.touched_mid = True
                self.p0.can_enter_sz = False
                self.p1.can_enter_sz = False

        # Hitung jarak tempuh real step
        p0_disp = math.hypot(self.p0.x - prev_p0["x"], self.p0.y - prev_p0["y"])
        p1_disp = math.hypot(self.p1.x - prev_p1["x"], self.p1.y - prev_p1["y"])

        # 5. Compute Markovian Rewards
        r0 = self._compute_markovian_reward(
            me=self.p0,
            opp=self.p1,
            prev_me_hp=prev_p0["hp"],
            prev_opp_hp=prev_p1["hp"],
            prev_me_mp=prev_p0["mp"],
            prev_opp_mp=prev_p1["mp"],
            prev_me_cell=prev_p0["cell"],
            me_disp=p0_disp,
        )
        r1 = self._compute_markovian_reward(
            me=self.p1,
            opp=self.p0,
            prev_me_hp=prev_p1["hp"],
            prev_opp_hp=prev_p0["hp"],
            prev_me_mp=prev_p1["mp"],
            prev_opp_mp=prev_p0["mp"],
            prev_me_cell=prev_p1["cell"],
            me_disp=p1_disp,
        )

        # Kill & Death event incentives
        if p1_died:
            r0 += 10.0
            r1 -= 10.0
        if p0_died:
            r1 += 10.0
            r0 -= 10.0

        # 6. Termination Check (10 Kills Target or Max Match Time)
        done = False
        if self.p0_kills >= self.target_kills:
            r0 += 50.0
            r1 -= 50.0
            done = True
        elif self.p1_kills >= self.target_kills:
            r1 += 50.0
            r0 -= 50.0
            done = True
        elif self.match_time >= self.max_match_time:
            if self.p0_kills > self.p1_kills:
                r0 += 50.0
                r1 -= 50.0
            elif self.p1_kills > self.p0_kills:
                r1 += 50.0
                r0 -= 50.0
            done = True

        info = {
            "p0_cell": self.p0.cell,
            "p1_cell": self.p1.cell,
            "p0_hp": self.p0.hp,
            "p1_hp": self.p1.hp,
            "p0_kills": self.p0_kills,
            "p1_kills": self.p1_kills,
            "p0_deaths": self.p0_deaths,
            "p1_deaths": self.p1_deaths,
            "match_time": self.match_time,
        }
        return (self._get_obs(self.p0, self.p1), self._get_obs(self.p1, self.p0)), (r0, r1), done, info

    # --------------------------------------------------------------------------
    # REWARD ENGINE
    # --------------------------------------------------------------------------
    def _compute_zone_reward(self, me: Entity, opp: Entity) -> float:
        """
        2-Mode Zone Reward/Penalty System:
        - Skala: berdiam di B1 (Morale A, rasio -2.0) selama 50s = -10.0 (setara 1 death).
          Artinya rate per detik = ratio * 0.10, dan rate per tick (0.1s) = ratio * 0.10 * DT.
        - Mode 1 (opp.touched_mid == False): awal match / baru ada yg respawn & belum ketemu.
        - Mode 2 (opp.touched_mid == True): musuh sudah menyentuh Crosslower atau sudah ketemu di cell yg sama.
        """
        is_mode_2 = opp.touched_mid
        cell = me.cell
        my_side = "0" if me.team == 0 else "1"
        opp_side = "1" if me.team == 0 else "0"

        if cell == "Crosslower":
            ratio = 0.0
        elif cell == "Crossupper":
            ratio = -0.5
        elif cell in ("Enter0", "Enter1", "Captain0", "Captain1", "Resource0B", "Resource1B"):
            ratio = -5.0
        elif cell in ("Resource0A", "Resource1A"):
            ratio = -2.0
        elif cell == f"Morale{my_side}C":
            ratio = -4.0
        elif cell == f"Morale{my_side}B":
            ratio = -3.0
        elif cell == f"Morale{my_side}A":
            ratio = -2.0
        elif cell == f"Morale{opp_side}A":
            ratio = -2.0 if is_mode_2 else 0.5
        elif cell == f"Morale{opp_side}B":
            ratio = -3.0 if is_mode_2 else 1.0
        elif cell == f"Morale{opp_side}C":
            ratio = -4.0 if is_mode_2 else 1.5
        else:
            ratio = 0.0

        # Jika can_enter_sz aktif (di Mode 1 pasca kill), penalti negatif tidak berlaku,
        # tetapi reward positif jika tetap maju menekan ke wilayah lawan tetap berlaku.
        if not is_mode_2 and me.can_enter_sz and ratio < 0.0:
            ratio = 0.0

        return ratio * 0.10 * self.DT

    def _compute_markovian_reward(
        self,
        me: Entity,
        opp: Entity,
        prev_me_hp: float,
        prev_opp_hp: float,
        prev_me_mp: float,
        prev_opp_mp: float,
        prev_me_cell: str,
        me_disp: float,
    ) -> float:
        # Jika avatar mati dan menunggu respawn, reward nol
        if not me.is_alive:
            return 0.0

        # 1. EHP Trade
        dmg_dealt = max(0.0, prev_opp_hp - opp.hp)
        dmg_taken = max(0.0, prev_me_hp - me.hp)
        r_ehp = ((dmg_dealt - dmg_taken) / me.max_hp) * 2.0

        # 2. Resource Trade
        delta_my_mp = me.mp - prev_me_mp
        delta_opp_mp = opp.mp - prev_opp_mp
        r_mana = ((delta_my_mp - delta_opp_mp) / 100.0) * 0.10

        # 3. Spacing Dynamics
        same_room = me.cell == opp.cell
        in_melee = same_room and opp.is_alive and (abs(me.x - opp.x) <= MELEE_RANGE_X and abs(me.y - opp.y) <= MELEE_RANGE_Y)
        my_ready_skills = sum(1 for i in range(2, 6) if me.cooldowns[i] <= 0.0 and me.gcd_rem <= 0.0)
        all_cds_spent = all(me.cooldowns[i] > 1.0 for i in range(2, 6))

        r_spacing = 0.0
        if in_melee:
            if my_ready_skills >= 2 or opp.is_stunned:
                r_spacing += 0.005
            elif all_cds_spent:
                r_spacing -= 0.003

        # 4. Sprint Penalty
        r_kinetic = 0.0
        if me.used_sprint and me_disp < 10.0:
            r_kinetic -= 0.05

        # 5. 2-Mode Zone Reward/Penalty & Safezone Regulations
        r_regulations = self._compute_zone_reward(me, opp)
        if me.illegal_sz_entered:
            r_regulations -= 10.0
            me.illegal_sz_entered = False

        return r_ehp + r_mana + r_spacing + r_kinetic + r_regulations

    # --------------------------------------------------------------------------
    # PHYSICS & COMBAT INTERNAL ENGINES
    # --------------------------------------------------------------------------
    def _process_movement(self, ent: Entity, act: ActionSpec) -> None:
        ent.used_sprint = False
        if not ent.is_alive or ent.is_stunned or act.move_type == 0:
            ent.sprint_rem = 0.0
            return

        target_x = float(np.clip(ent.x + act.target_dx * MAX_DISP_PX, FLOOR_X_MIN, FLOOR_X_MAX))
        target_y = float(np.clip(ent.y + act.target_dy * MAX_DISP_PX, FLOOR_Y_MIN, FLOOR_Y_MAX))
        dx = target_x - ent.x
        dy = target_y - ent.y
        dist = math.hypot(dx, dy)

        if dist < 1e-4:
            ent.sprint_rem = 0.0
            return

        speed = BASE_SPEED_PX
        if act.move_type == 2:  # SPRINT
            if ent.sprint_rem > 0.0:
                speed *= SPRINT_MULT
                ent.used_sprint = True
            elif ent.sp >= SP_SPRINT_COST:
                ent.sp -= SP_SPRINT_COST
                ent.sprint_rem = 1.0  # Telemetri AQW: 1x bayar 50 SP = status sprint (speed=24) selama 1.0 detik (~480 px)
                speed *= SPRINT_MULT
                ent.used_sprint = True
        elif ent.sprint_rem > 0.0:
            speed *= SPRINT_MULT

        step_dist = min(dist, speed)
        next_x = ent.x + (dx / dist) * step_dist
        next_y = ent.y + (dy / dist) * step_dist

        # Door Crossing Check
        doors = ROOM_DOORS.get(ent.cell, [])
        for door in doors:
            # Cegah masuk safezone milik tim lawan
            if door.target_cell == "Enter0" and ent.team != 0:
                continue
            if door.target_cell == "Enter1" and ent.team != 1:
                continue

            if _intersects_aabb(ent.x, ent.y, next_x, next_y, door):
                # 1. Validasi Masuk Safezone Ilegal
                if door.target_cell in ("Enter0", "Enter1"):
                    if not ent.can_enter_sz:
                        # Masuk SZ saat duel masih aktif dikenai penalti setara 1 kematian (-10.0)
                        ent.illegal_sz_entered = True

                prev_cell = ent.cell

                ent.cell = door.target_cell
                ent.x = door.spawn_x
                ent.y = door.spawn_y
                ent.sprint_rem = 0.0

                if prev_cell in ("Enter0", "Enter1") and ent.cell not in ("Enter0", "Enter1"):
                    ent.can_enter_sz = False
                return

        ent.x = float(np.clip(next_x, FLOOR_X_MIN, FLOOR_X_MAX))
        ent.y = float(np.clip(next_y, FLOOR_Y_MIN, FLOOR_Y_MAX))

    def _process_combat(self, atk: Entity, tgt: Entity, combat_act: int) -> Tuple[float, bool]:
        if not atk.is_alive or atk.is_stunned or combat_act < 1 or combat_act > 5:
            return 0.0, False

        skill = atk.skills.get(combat_act)
        if not skill or atk.mp < skill.mana_cost or atk.cooldowns[combat_act] > 0.0 or atk.gcd_rem > 0.0:
            return 0.0, False

        # Self-buff
        if skill.is_self_buff:
            atk.mp -= skill.mana_cost
            atk.cooldowns[combat_act] = skill.cooldown
            atk.gcd_rem = 1.5
            if skill.applies_charge_crit:
                atk.active_auras["Prepared Strike"] = ActiveAura(
                    name="Prepared Strike",
                    duration_rem=5.0,
                    crit_charges=2,
                    effects={"cpo": 0.15}
                )
            return 0.0, False

        # Target Attack
        same_room = atk.cell == tgt.cell
        in_range = same_room and (abs(atk.x - tgt.x) <= MELEE_RANGE_X and abs(atk.y - tgt.y) <= MELEE_RANGE_Y)
        if not in_range or not tgt.is_alive:
            return 0.0, False

        atk.mp -= skill.mana_cost
        atk.cooldowns[combat_act] = skill.cooldown
        if combat_act > 1:
            atk.gcd_rem = 1.5  # Auto Attack tidak memicu GCD

        # Check Prepared Strike charges ("Your next two auto attacks taken within 5 seconds
        # will be powerful critical hits which cannot be avoided.")
        prep = atk.active_auras.get("Prepared Strike")
        if combat_act == 1 and prep and prep.duration_rem > 0.0 and prep.crit_charges > 0:
            is_dodge = False
            is_crit = True
            prep.crit_charges -= 1
            if prep.crit_charges <= 0:
                del atk.active_auras["Prepared Strike"]
        else:
            is_dodge = self.rng.random() < tgt.dodge_rate
            is_crit = (not is_dodge) and (self.rng.random() < atk.crit_rate)

        damage = 0.0
        applied_cc = False

        if not is_dodge:
            out_mult = atk.outgoing_dmg_mult
            in_mult = tgt.incoming_dmg_mult
            crit_mult = atk.crit_mult if is_crit else 1.0
            base_dmg = BASE_AUTO_DAMAGE * skill.damage_mult

            damage = base_dmg * out_mult * in_mult * crit_mult * self.rng.uniform(0.95, 1.05)
            tgt.hp = max(0.0, tgt.hp - damage)
            atk.time_since_combat = 0.0
            tgt.time_since_combat = 0.0
            atk.in_combat = True
            tgt.in_combat = True

            if tgt.hp <= 0.0:
                tgt.hp = 0.0
                tgt.is_alive = False
                tgt.respawn_timer = self.RESPAWN_DELAY
                atk.can_enter_sz = True  # Lawan tereliminasi, pemenang boleh kembali ke safezone
                atk.in_combat = False
                atk.time_since_combat = 15.0

            if skill.applies_shield:
                atk.active_auras["On Guard"] = ActiveAura(
                    name="On Guard",
                    duration_rem=skill.shield_duration,
                    effects={"cai": -0.50}
                )
            if skill.applies_stun:
                tgt.active_auras["Off Balance"] = ActiveAura(
                    name="Off Balance",
                    duration_rem=skill.stun_duration,
                    is_stun=True
                )
                applied_cc = True

            atk.mp = min(atk.max_mp, atk.mp + ManaRecoveryModel.calc_on_hit(atk.role_name, is_crit))
            tgt.mp = min(tgt.max_mp, tgt.mp + ManaRecoveryModel.calc_on_struck(tgt.role_name))

        return damage, applied_cc