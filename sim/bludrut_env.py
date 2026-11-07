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

# Parameter combat & jangkauan serangan
MELEE_RANGE_X = 160.0
MELEE_RANGE_Y = 55.0
BASE_AUTO_DAMAGE = 150.0

# Kecepatan & mobilitas
BASE_SPEED_PX = 8.0 * 20.0 * 0.10   # 16.0 px/tick (160 px/s)
SPRINT_MULT = 3.0                   # Telemetri AQW: 24 / 8 = 3.0x
SP_SPRINT_COST = 50.0               # Konsumsi SP untuk sprint
MAX_DISP_PX = 300.0                 # Jarak target relatif maksimum per step

# Siklus simulasi & match
SIM_TIMESTEP = 0.10                 # 10 Hz simulation loop
DEFAULT_MAX_MATCH_TIME = 600.0      # 10 Menit (6000 tick) horizon PvP


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
    3: SkillSpec("Imbalancing Strike", "a2", cooldown=10.0, mana_cost=30.0, damage_mult=0.7, applies_stun=True, stun_duration=4.0),
    4: SkillSpec("Prepared Strike", "a3", cooldown=15.0, mana_cost=20.0, damage_mult=0.0, is_self_buff=True, applies_charge_crit=True),
    5: SkillSpec("On Guard", "a4", cooldown=30.0, mana_cost=30.0, damage_mult=0.8, applies_shield=True, shield_duration=10.0),
}

HEALER_PROFILE: Dict[int, SkillSpec] = {
    1: SkillSpec("Auto Attack", "aa", cooldown=2.0, mana_cost=0.0, damage_mult=0.8),
    2: SkillSpec("Healing Word", "a1", cooldown=3.0, mana_cost=20.0, damage_mult=2.0, is_heal=True),
    3: SkillSpec("Radiance", "a2", cooldown=12.0, mana_cost=30.0, damage_mult=0.0, is_self_buff=True),
    4: SkillSpec("Energy Flow", "a3", cooldown=10.0, mana_cost=15.0, damage_mult=0.5),
    5: SkillSpec("Clear Mind", "a4", cooldown=20.0, mana_cost=20.0, damage_mult=0.0, is_self_buff=True),
}


# ==============================================================================
# 4. ENTITY DOMAIN MODEL
# ==============================================================================
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
    shield_rem: float = 0.0
    stun_rem: float = 0.0
    crit_charge_rem: float = 0.0
    crit_charges: int = 0
    inhibition_rem: float = 0.0
    dodge_rate: float = 0.20
    crit_rate: float = 0.25
    crit_mult: float = 2.0
    base_haste: float = 0.10
    used_sprint: bool = False

    def reset(self, cell: str, x: float, y: float) -> None:
        self.cell = cell
        self.x = x
        self.y = y
        self.hp = self.max_hp
        self.mp = self.max_mp
        self.sp = self.max_sp
        self.is_alive = True
        self.gcd_rem = 0.0
        self.cooldowns = [0.0] * 6
        self.time_since_combat = 15.0
        self.in_combat = False
        self.shield_rem = 0.0
        self.stun_rem = 0.0
        self.crit_charge_rem = 0.0
        self.crit_charges = 0
        self.inhibition_rem = 0.0
        self.used_sprint = False

    def tick_status(self, dt: float, is_regen_tick: bool) -> None:
        if not self.is_alive:
            return

        self.gcd_rem = max(0.0, self.gcd_rem - dt)
        for i in range(len(self.cooldowns)):
            self.cooldowns[i] = max(0.0, self.cooldowns[i] - dt)

        if self.shield_rem > 0.0:
            self.shield_rem = max(0.0, self.shield_rem - dt)
        if self.stun_rem > 0.0:
            self.stun_rem = max(0.0, self.stun_rem - dt)
        if self.inhibition_rem > 0.0:
            self.inhibition_rem = max(0.0, self.inhibition_rem - dt)
        if self.crit_charge_rem > 0.0:
            self.crit_charge_rem = max(0.0, self.crit_charge_rem - dt)
            if self.crit_charge_rem == 0.0:
                self.crit_charges = 0

        if self.in_combat:
            self.time_since_combat = 0.0
            if is_regen_tick and self.is_player:
                self.sp = RegenerationEngine.process_in_combat_sp_regen(self.sp, self.max_sp)
        else:
            self.time_since_combat += dt
            if is_regen_tick and self.is_player:
                if self.time_since_combat >= 12.0:
                    self.hp, self.sp = RegenerationEngine.process_ooc_regen(
                        self.hp, self.max_hp, self.sp, self.max_sp
                    )
                    self.mp = min(self.max_mp, self.mp + 15.0)
                else:
                    self.sp = min(self.max_sp, self.sp + 10.0)


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

    def __init__(self, p0_role: str = "Warrior", p1_role: str = "Warrior", seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.match_time = 0.0
        self.regen_timer = 0.0
        self.max_match_time = DEFAULT_MAX_MATCH_TIME

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
            0.0 if me.in_combat else min(1.0, me.time_since_combat / 12.0),
            cell_idx,
            norm_x,
            norm_y,
            1.0 if me.is_alive else 0.0,
            1.0 if (me.cooldowns[1] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            1.0 if (me.cooldowns[2] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            1.0 if (me.cooldowns[3] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            1.0 if (me.cooldowns[4] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            0.50 if (me.shield_rem > 0.0) else 1.0,
            1.0 if (me.stun_rem > 0.0) else 0.0,
        ]

    def _extract_opponent_features(self, me: Entity, opp: Entity) -> List[float]:
        same_cell = 1.0 if me.cell == opp.cell else 0.0
        cell_hops = float(CELL_GRAPH_DIST.get((me.cell, opp.cell), 4)) / 6.0

        if same_cell:
            rel_x = float(np.clip((opp.x - me.x) / FLOOR_WIDTH, -1.0, 1.0))
            rel_y = float(np.clip((opp.y - me.y) / FLOOR_HEIGHT, -1.0, 1.0))
            dist = math.hypot(opp.x - me.x, opp.y - me.y)
            dist_norm = min(1.0, dist / MAX_DIAG_DIST)
            in_melee = 1.0 if (abs(opp.x - me.x) <= MELEE_RANGE_X and abs(opp.y - me.y) <= MELEE_RANGE_Y) else 0.0
        else:
            rel_x, rel_y = 0.0, 0.0
            dist_norm = 1.0
            in_melee = 0.0

        if opp.crit_charge_rem > 0.0:
            opp_out_mult = 1.15
        elif opp.inhibition_rem > 0.0:
            opp_out_mult = 0.70
        else:
            opp_out_mult = 1.0

        return [
            1.0,  # is_valid
            1.0 if opp.is_alive else 0.0,
            0.0,  # is_ally
            same_cell,
            cell_hops,
            rel_x,
            rel_y,
            dist_norm,
            in_melee,
            opp.hp / opp.max_hp,
            opp.mp / opp.max_mp,
            0.50 if (opp.shield_rem > 0.0) else 1.0,
            1.0 if (opp.stun_rem > 0.0) else 0.0,
            opp_out_mult,
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
        prev_states = {
            self.p0: {"hp": self.p0.hp, "mp": self.p0.mp, "cell": self.p0.cell, "x": self.p0.x, "y": self.p0.y},
            self.p1: {"hp": self.p1.hp, "mp": self.p1.mp, "cell": self.p1.cell, "x": self.p1.x, "y": self.p1.y},
        }

        # 1. Update Domain Model Status
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

        # Hitung jarak tempuh real step
        p0_disp = math.hypot(self.p0.x - prev_states[self.p0]["x"], self.p0.y - prev_states[self.p0]["y"])
        p1_disp = math.hypot(self.p1.x - prev_states[self.p1]["x"], self.p1.y - prev_states[self.p1]["y"])

        # 4. Compute Markovian Rewards
        r0 = self._compute_markovian_reward(
            me=self.p0,
            opp=self.p1,
            prev_me_hp=prev_states[self.p0]["hp"],
            prev_opp_hp=prev_states[self.p1]["hp"],
            prev_me_mp=prev_states[self.p0]["mp"],
            prev_opp_mp=prev_states[self.p1]["mp"],
            prev_me_cell=prev_states[self.p0]["cell"],
            me_disp=p0_disp,
        )
        r1 = self._compute_markovian_reward(
            me=self.p1,
            opp=self.p0,
            prev_me_hp=prev_states[self.p1]["hp"],
            prev_opp_hp=prev_states[self.p0]["hp"],
            prev_me_mp=prev_states[self.p1]["mp"],
            prev_opp_mp=prev_states[self.p0]["mp"],
            prev_me_cell=prev_states[self.p1]["cell"],
            me_disp=p1_disp,
        )

        # 5. Termination Check (Non-OP PvP Player-Centric)
        done = False
        if not self.p0.is_alive or self.p0.hp <= 0.0:
            r0 -= 10.0
            r1 += 10.0
            done = True
        elif not self.p1.is_alive or self.p1.hp <= 0.0:
            r0 += 10.0
            r1 -= 10.0
            done = True
        elif self.match_time >= self.max_match_time:
            done = True

        info = {
            "p0_cell": self.p0.cell,
            "p1_cell": self.p1.cell,
            "p0_hp": self.p0.hp,
            "p1_hp": self.p1.hp,
            "match_time": self.match_time,
        }
        return (self._get_obs(self.p0, self.p1), self._get_obs(self.p1, self.p0)), (r0, r1), done, info

    # --------------------------------------------------------------------------
    # REWARD ENGINE
    # --------------------------------------------------------------------------
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
        # 1. EHP Trade
        dmg_dealt = max(0.0, prev_opp_hp - opp.hp)
        dmg_taken = max(0.0, prev_me_hp - me.hp)
        r_ehp = ((dmg_dealt - dmg_taken) / me.max_hp) * 2.0

        # 2. Resource Trade
        delta_my_mp = me.mp - prev_me_mp
        delta_opp_mp = opp.mp - prev_opp_mp
        r_mana = ((delta_my_mp - delta_opp_mp) / 100.0) * 0.10

        # 3. PBRS Navigation
        r_nav = 0.0
        if me.cell != opp.cell:
            target_cell = opp.cell if opp.cell not in ("Enter0", "Enter1") else "Crosslower"
            phi_old = -float(CELL_GRAPH_DIST.get((prev_me_cell, target_cell), 4))
            phi_new = -float(CELL_GRAPH_DIST.get((me.cell, target_cell), 4))
            gamma = 0.99
            r_nav = (gamma * phi_new - phi_old) * 0.25

        # 4. Spacing Dynamics
        same_room = me.cell == opp.cell
        in_melee = same_room and (abs(me.x - opp.x) <= MELEE_RANGE_X and abs(me.y - opp.y) <= MELEE_RANGE_Y)
        my_ready_skills = sum(1 for i in range(1, 5) if me.cooldowns[i] <= 0.0 and me.gcd_rem <= 0.0)
        all_cds_spent = all(me.cooldowns[i] > 1.0 for i in range(1, 5))

        r_spacing = 0.0
        if in_melee:
            if my_ready_skills >= 2 or opp.stun_rem > 0.0:
                r_spacing += 0.005
            elif all_cds_spent:
                r_spacing -= 0.003

        # 5. Sprint Penalty
        r_kinetic = 0.0
        if me.used_sprint and me_disp < 10.0:
            r_kinetic -= 0.05

        # 6. Step Time Penalty
        r_time = -0.005

        # 7. Safezone Regulations
        r_regulations = 0.0
        if me.cell in ("Enter0", "Enter1"):
            if self.match_time > 15.0:
                r_regulations -= 0.05
        else:
            if not me.in_combat and me.cell != "Crosslower":
                ooc_ratio = min(1.0, me.time_since_combat / 12.0)
                if me.hp / me.max_hp > 0.85:
                    r_regulations -= 0.005 * ooc_ratio

        return r_ehp + r_mana + r_nav + r_spacing + r_kinetic + r_time + r_regulations

    # --------------------------------------------------------------------------
    # PHYSICS & COMBAT INTERNAL ENGINES
    # --------------------------------------------------------------------------
    def _process_movement(self, ent: Entity, act: ActionSpec) -> None:
        ent.used_sprint = False
        if not ent.is_alive or ent.stun_rem > 0.0 or act.move_type == 0:
            return

        target_x = float(np.clip(ent.x + act.target_dx * MAX_DISP_PX, FLOOR_X_MIN, FLOOR_X_MAX))
        target_y = float(np.clip(ent.y + act.target_dy * MAX_DISP_PX, FLOOR_Y_MIN, FLOOR_Y_MAX))
        dx = target_x - ent.x
        dy = target_y - ent.y
        dist = math.hypot(dx, dy)

        if dist < 1e-4:
            return

        speed = BASE_SPEED_PX
        if act.move_type == 2:  # SPRINT
            if ent.sp >= SP_SPRINT_COST:
                ent.sp -= SP_SPRINT_COST
                speed *= SPRINT_MULT
                ent.used_sprint = True

        step_dist = min(dist, speed)
        next_x = ent.x + (dx / dist) * step_dist
        next_y = ent.y + (dy / dist) * step_dist

        # Door Crossing Check
        doors = ROOM_DOORS.get(ent.cell, [])
        for door in doors:
            if _intersects_aabb(ent.x, ent.y, next_x, next_y, door):
                # Larangan kembali ke Safezone jika duel masih berjalan
                if door.target_cell in ("Enter0", "Enter1") and self.p0.is_alive and self.p1.is_alive:
                    ent.x = float(np.clip(ent.x, FLOOR_X_MIN + 10.0, FLOOR_X_MAX - 10.0))
                    return

                ent.cell = door.target_cell
                ent.x = door.spawn_x
                ent.y = door.spawn_y
                return

        ent.x = float(np.clip(next_x, FLOOR_X_MIN, FLOOR_X_MAX))
        ent.y = float(np.clip(next_y, FLOOR_Y_MIN, FLOOR_Y_MAX))

    def _process_combat(self, atk: Entity, tgt: Entity, combat_act: int) -> Tuple[float, bool]:
        if not atk.is_alive or atk.stun_rem > 0.0 or combat_act < 1 or combat_act > 5:
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
                atk.crit_charge_rem = 5.0
                atk.crit_charges = 2
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

        is_dodge = self.rng.random() < tgt.dodge_rate
        damage = 0.0
        applied_cc = False

        if not is_dodge:
            if atk.crit_charges > 0 and atk.crit_charge_rem > 0.0:
                is_crit = True
                atk.crit_charges -= 1
                if atk.crit_charges <= 0:
                    atk.crit_charge_rem = 0.0
                out_mult = 1.15
            else:
                is_crit = self.rng.random() < atk.crit_rate
                out_mult = 1.0

            if atk.inhibition_rem > 0.0:
                out_mult *= 0.70

            in_mult = 0.50 if tgt.shield_rem > 0.0 else 1.0
            crit_mult = atk.crit_mult if is_crit else 1.0
            base_dmg = BASE_AUTO_DAMAGE * skill.damage_mult

            damage = base_dmg * out_mult * in_mult * crit_mult * self.rng.uniform(0.95, 1.05)
            tgt.hp = max(0.0, tgt.hp - damage)
            atk.in_combat = True
            tgt.in_combat = True

            if tgt.hp <= 0.0:
                tgt.is_alive = False

            if skill.applies_shield:
                atk.shield_rem = skill.shield_duration
            if skill.applies_stun:
                tgt.stun_rem = skill.stun_duration
                applied_cc = True

            atk.mp = min(atk.max_mp, atk.mp + ManaRecoveryModel.calc_on_hit(atk.role_name, is_crit))
            tgt.mp = min(tgt.max_mp, tgt.mp + ManaRecoveryModel.calc_on_struck(tgt.role_name))

        return damage, applied_cc