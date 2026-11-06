#!/usr/bin/env python3
"""
sim/bludrut_env.py
Arena Orchestrator Bludrut Brawl 16-Screen Map & Ground Truth Physics.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

from sim.world.bludrut_map import (
    SCREEN_NAMES,
    SCREEN_INDICES,
    CELL_GRAPH_DIST,
    WALK_BOUNDS,
    ROOM_DOORS,
    SAFEZONE_SPAWN,
    BLUDRUT_NPCS
)
from sim.combat.mrm import ManaRecoveryModel, RegenerationEngine

# ==============================================================================
# COMBAT SPEC & DATA-DRIVEN PROFILES
# ==============================================================================

@dataclass
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

    def reset(self, cell: str, x: float, y: float):
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


# ==============================================================================
# ENVIRONMENT ORCHESTRATOR
# ==============================================================================

class BludrutBrawlEnv:
    DT = 0.10  # 10 Hz
    REGEN_INTERVAL = RegenerationEngine.TICK_INTERVAL  # 2.0s
    TOTAL_OBS_DIM = 30

    def __init__(self, p0_role: str = "Warrior", p1_role: str = "Warrior", seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.match_time = 0.0
        self.regen_timer = 0.0
        self.max_match_time = 300.0

        # Inisialisasi awal pemain tepat di titik safezone masing-masing
        self.p0 = Entity(
            ent_id="p:187461",
            name="uma",
            team=0,
            is_player=True,
            cell="Enter0",
            x=SAFEZONE_SPAWN["Enter0"]["x"],
            y=SAFEZONE_SPAWN["Enter0"]["y"],
            role_name=p0_role,
            skills=WARRIOR_PROFILE if p0_role == "Warrior" else HEALER_PROFILE
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
            skills=WARRIOR_PROFILE if p1_role == "Warrior" else HEALER_PROFILE
        )

        self.npcs: Dict[int, Entity] = {}
        self._init_npcs()

    def _init_npcs(self):
        self.npcs.clear()
        for defn in BLUDRUT_NPCS:
            npc_ent = Entity(
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
                crit_rate=0.05
            )
            self.npcs[defn.mon_map_id] = npc_ent

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.match_time = 0.0
        self.regen_timer = 0.0
        self.p0.reset("Enter0", SAFEZONE_SPAWN["Enter0"]["x"], SAFEZONE_SPAWN["Enter0"]["y"])
        self.p1.reset("Enter1", SAFEZONE_SPAWN["Enter1"]["x"], SAFEZONE_SPAWN["Enter1"]["y"])
        self._init_npcs()
        return self._get_obs(self.p0, self.p1), self._get_obs(self.p1, self.p0)

    def _get_obs(self, me: Entity, opp: Entity) -> np.ndarray:
        same_cell = 1.0 if me.cell == opp.cell else 0.0
        cell_hops = float(CELL_GRAPH_DIST.get((me.cell, opp.cell), 4)) / 6.0
        cell_idx = float(SCREEN_INDICES.get(me.cell, 0)) / 15.0

        if same_cell:
            rel_x = float(np.clip((opp.x - me.x) / 960.0, -1.0, 1.0))
            rel_y = float(np.clip((opp.y - me.y) / 550.0, -1.0, 1.0))
            dist = math.hypot(opp.x - me.x, opp.y - me.y)
            dist_norm = min(1.0, dist / 1000.0)
            in_melee = 1.0 if (abs(opp.x - me.x) <= 160.0 and abs(opp.y - me.y) <= 38.0) else 0.0
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

        self_feats = [
            me.hp / me.max_hp,
            me.mp / me.max_mp,
            me.sp / me.max_sp,
            1.0 if me.sp >= 50.0 else 0.0,
            1.0 if me.in_combat else 0.0,
            0.0 if me.in_combat else min(1.0, me.time_since_combat / 12.0),
            cell_idx,
            min(1.0, max(0.0, me.x / 960.0)),
            min(1.0, max(0.0, me.y / 550.0)),
            1.0 if me.is_alive else 0.0,
            1.0 if (me.cooldowns[1] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            1.0 if (me.cooldowns[2] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            1.0 if (me.cooldowns[3] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            1.0 if (me.cooldowns[4] <= 0.0 and me.gcd_rem <= 0.0) else 0.0,
            0.50 if (me.shield_rem > 0.0) else 1.0,
            1.0 if (me.stun_rem > 0.0) else 0.0,
        ]

        opp_slot = [
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

        obs = np.array(self_feats + opp_slot, dtype=np.float32)
        return np.clip(obs, -1.0, 1.0)

    def step(self, a0: int, a1: int) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[float, float], bool, dict]:
        self.match_time += self.DT
        self.regen_timer += self.DT
        is_regen_tick = self.regen_timer >= self.REGEN_INTERVAL
        if is_regen_tick:
            self.regen_timer = 0.0

        self._tick_entity(self.p0, is_regen_tick)
        self._tick_entity(self.p1, is_regen_tick)
        for npc in self.npcs.values():
            self._tick_entity(npc, is_regen_tick)

        self._process_movement(self.p0, a0)
        self._process_movement(self.p1, a1)

        dmg0, _ = self._process_combat(self.p0, self.p1, a0)
        dmg1, _ = self._process_combat(self.p1, self.p0, a1)

        r0 = (dmg0 * 0.001) - (dmg1 * 0.001)
        r1 = (dmg1 * 0.001) - (dmg0 * 0.001)

        hops = CELL_GRAPH_DIST.get((self.p0.cell, self.p1.cell), 4)
        if hops == 0:
            r0 += 0.002
            r1 += 0.002

        done = False
        cap0 = self.npcs[1]
        cap1 = self.npcs[16]

        if not self.p0.is_alive or self.p0.hp <= 0.0:
            r0 -= 1.0
            r1 += 1.0
            done = True
        elif not self.p1.is_alive or self.p1.hp <= 0.0:
            r0 += 1.0
            r1 -= 1.0
            done = True
        elif not cap0.is_alive or not cap1.is_alive:
            done = True
        elif self.match_time >= self.max_match_time:
            done = True

        return (self._get_obs(self.p0, self.p1), self._get_obs(self.p1, self.p0)), (r0, r1), done, {}

    def _tick_entity(self, ent: Entity, is_regen_tick: bool):
        if not ent.is_alive:
            return

        ent.gcd_rem = max(0.0, ent.gcd_rem - self.DT)
        for i in range(len(ent.cooldowns)):
            ent.cooldowns[i] = max(0.0, ent.cooldowns[i] - self.DT)

        if ent.shield_rem > 0.0:
            ent.shield_rem = max(0.0, ent.shield_rem - self.DT)
        if ent.stun_rem > 0.0:
            ent.stun_rem = max(0.0, ent.stun_rem - self.DT)
        if ent.inhibition_rem > 0.0:
            ent.inhibition_rem = max(0.0, ent.inhibition_rem - self.DT)

        if ent.crit_charge_rem > 0.0:
            ent.crit_charge_rem = max(0.0, ent.crit_charge_rem - self.DT)
            if ent.crit_charge_rem == 0.0:
                ent.crit_charges = 0

        if ent.in_combat:
            ent.time_since_combat = 0.0
            if is_regen_tick and ent.is_player:
                ent.sp = RegenerationEngine.process_in_combat_sp_regen(ent.sp, ent.max_sp)
        else:
            ent.time_since_combat += self.DT
            if is_regen_tick and ent.is_player:
                if ent.time_since_combat >= 12.0:
                    ent.hp, ent.sp = RegenerationEngine.process_ooc_regen(ent.hp, ent.max_hp, ent.sp, ent.max_sp)
                    ent.mp = min(ent.max_mp, ent.mp + 15.0)
                else:
                    ent.sp = min(ent.max_sp, ent.sp + 10.0)

    def _process_movement(self, ent: Entity, act: int):
        if not ent.is_alive or ent.stun_rem > 0.0 or act < 6 or act > 9:
            return

        speed = 8.0 * 20.0 * self.DT  # ~16 px / tick
        if ent.sp >= 50.0:
            speed *= 1.5
            ent.sp = max(0.0, ent.sp - (10.0 * self.DT))

        dx, dy = 0.0, 0.0
        if act == 6: dx = -speed
        elif act == 7: dx = speed
        elif act == 8: dy = -speed
        elif act == 9: dy = speed

        ent.x += dx
        ent.y += dy

        # Evaluasi AABB Pintu Ruangan
        doors = ROOM_DOORS.get(ent.cell, [])
        for door in doors:
            if door.contains(ent.x, ent.y):
                ent.cell = door.target_cell
                ent.x = door.spawn_x
                ent.y = door.spawn_y
                return

        # Clamping ke batas fisik valid move
        ent.x = max(WALK_BOUNDS["x_min"], min(WALK_BOUNDS["x_max"], ent.x))
        ent.y = max(WALK_BOUNDS["y_min"], min(WALK_BOUNDS["y_max"], ent.y))

    def _process_combat(self, atk: Entity, tgt: Entity, act: int) -> Tuple[float, bool]:
        if not atk.is_alive or atk.stun_rem > 0.0 or act < 1 or act > 5:
            return 0.0, False

        skill = atk.skills.get(act)
        if not skill or atk.mp < skill.mana_cost or atk.cooldowns[act] > 0.0 or atk.gcd_rem > 0.0:
            return 0.0, False

        if skill.is_self_buff:
            atk.mp -= skill.mana_cost
            atk.cooldowns[act] = skill.cooldown
            atk.gcd_rem = 1.5
            if skill.applies_charge_crit:
                atk.crit_charge_rem = 5.0
                atk.crit_charges = 2
            return 0.0, False

        same_room = atk.cell == tgt.cell
        in_range = same_room and (abs(atk.x - tgt.x) <= 160.0 and abs(atk.y - tgt.y) <= 38.0)
        if not in_range or not tgt.is_alive:
            return 0.0, False

        atk.mp -= skill.mana_cost
        atk.cooldowns[act] = skill.cooldown
        atk.gcd_rem = 1.5

        is_dodge = self.rng.random() < tgt.dodge_rate
        damage = 0.0
        applied_cc = False

        if not is_dodge:
            is_crit = False
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

            base_dmg = 150.0 * skill.damage_mult
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