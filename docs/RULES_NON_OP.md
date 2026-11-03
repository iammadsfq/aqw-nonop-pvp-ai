# Classic (Non-OP) PvP Ruleset & Code Enforcement Specification

This document maps the official **AQW Classic (Non-OP) PvP Rules** to their concrete enforcement mechanisms in this repository.

---

## 1. Allowed Classes

### Standard Tier (Allowed in 1v1, 2v2, 3v3, 4v4)
- Battle Warrior / Classic Warrior (`src/classes/classic_warrior.ts`)
- Classic Ninja
- Classic Dragonlord
- Classic Starlord
- Classic Paladin
- Battle Bard
- Classic Defender *(Static Weapon Range enforced)*
- Classic Guardian
- Barber
- Rustbucket
- UndeadSlayer
- Classic DoomKnight
- No Class *(Hobo Highlord & Obsidian No Class)*
- Clawsuit

### Exceptional Tier (Allowed ONLY in 3v3 & 4v4)
- Battle Healer / PvP Test Healer

> **Code Enforcement:** Validated in [`src/rules/preflight.ts`](../src/rules/preflight.ts) against `STANDARD_NON_OP_CLASSES` and `EXCEPTIONAL_3V3_4V4_CLASSES` based on `config.allowBattleHealer`.

---

## 2. Combat & Map Rules

| Rule Category | Official Non-OP Rule | Code Enforcement Mechanism |
| :--- | :--- | :--- |
| **NPCs** | No killing NPCs (Brawlers/Restorers/Captains). You may damage them to regenerate mana, but cannot kill them. | [`enforceNpcProtectionGuard`](../src/rules/shield.ts) blocks offensive skills (`1..4`) on NPCs and cancels basic auto-attacks immediately if NPC HP drops below `2000 HP` or `35%`. |
| **Returning to SZ** | Not allowed unless:<br>• **1v1:** You kill your opponent.<br>• **2v2+:** A teammate or opponent dies. | [`WaveTrackerState`](../src/state/types.ts) and [`applyRuleShield`](../src/rules/shield.ts) mask out `RETREAT_TO_SAFEZONE` until `canReturnToSafezone` is unlocked by a verified wave kill/death. |
| **Leaving Midwave** | Not allowed. (Leaving the map is fine only if you're in a safezone.) | Map transitions are restricted unless `obs.inSafezone === true`. |
| **Boosts & Enhancements** | No damage-boosted gear, potions, scrolls, or special enhancements (Awe & Forge). | [`runNonOpPreflightCheck`](../src/rules/preflight.ts) inspects `bot.inventory.items` and rejects Awe/Forge pattern IDs (`23..32`). |
| **PvP Amulet** | Platinum PvP Amulet +5000 or Diamond PvP Amulet +5500 required. | [`runNonOpPreflightCheck`](../src/rules/preflight.ts) verifies an equipped `+5000` or `+5500` amulet before combat begins. |
| **Kill Targets** | • **1v1:** 10 kills<br>• **2v2:** 20 kills<br>• **3v3:** 30 kills<br>• **4v4:** 40 kills | Configured in [`KILL_TARGETS_BY_FORMAT`](../src/config/match_config.ts); match loop automatically terminates once `wave.teamKills >= config.killTarget`. |
