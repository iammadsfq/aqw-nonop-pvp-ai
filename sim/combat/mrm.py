#!/usr/bin/env python3
"""
sim/combat/mrm.py
Ground Truth Formula Regenerasi HP, Mana, dan SP AQW dari Hasil Intercept Paket.
"""

class ManaRecoveryModel:
    @staticmethod
    def calc_on_hit(role: str, is_crit: bool, has_energy_flow: bool = False) -> float:
        """Kalkulasi pemulihan mana saat pemain menyerang target."""
        if role == "Warrior":
            return 12.0 if is_crit else 6.0
        elif role == "Healer":
            # Caster MRM: AA biasa +1, crit +5. Jika Energy Flow aktif bertambah +2.
            base = 5.0 if is_crit else 1.0
            return base + (2.0 if has_energy_flow else 0.0)
        elif role == "Rogue":
            return 15.0 if is_crit else 3.0
        return 4.0

    @staticmethod
    def calc_on_struck(role: str) -> float:
        """Kalkulasi pemulihan mana saat terkena pukulan musuh."""
        if role == "Warrior":
            return 3.0
        elif role == "Healer":
            return 1.0  # Dari log: intMP selalu +1 saat terkena hit
        return 0.0

class RegenerationEngine:
    TICK_INTERVAL = 2.0  # Server AQW mengirim tick regen setiap 2.0 detik

    @staticmethod
    def process_ooc_regen(current_hp: float, max_hp: float, current_sp: float, max_sp: float) -> tuple[float, float]:
        """
        Regen out-of-combat (Rest):
        - HP pulih 15% dari Max HP per tick (2 detik).
        - SP pulih +25 per tick (2 detik).
        """
        new_hp = min(max_hp, current_hp + (max_hp * 0.15))
        new_sp = min(max_sp, current_sp + 25.0)
        return new_hp, new_sp

    @staticmethod
    def process_in_combat_sp_regen(current_sp: float, max_sp: float) -> float:
        """
        Regen SP saat in-combat:
        - SP tetap pulih tetapi melambat (+5 SP per tick 2 detik).
        """
        return min(max_sp, current_sp + 5.0)