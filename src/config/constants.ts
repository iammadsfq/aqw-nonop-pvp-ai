/**
 * @file constants.ts
 * @description Official Classic (Non-OP) PvP ruleset constants, allowed classes,
 * enhancement IDs, amulet requirements, and BludrutBrawl map metadata.
 */

/** Standard Tier classes allowed in all Non-OP formats (1v1, 2v2, 3v3, 4v4). */
export const STANDARD_NON_OP_CLASSES: readonly string[] = [
    "Warrior",
    "Classic Warrior",
    "Battle Warrior",
    "Ninja",
    "Classic Ninja",
    "Dragonlord",
    "Classic Dragonlord",
    "Starlord",
    "Classic Starlord",
    "Paladin",
    "Classic Paladin",
    "Battle Bard",
    "Bard",
    "Defender",
    "Classic Defender",
    "Guardian",
    "Classic Guardian",
    "Barber",
    "Rustbucket",
    "UndeadSlayer",
    "DoomKnight",
    "Classic DoomKnight",
    "No Class",
    "Hobo Highlord",
    "Obsidian No Class",
    "Clawsuit",
] as const;

/** Exceptional Tier classes allowed ONLY in 3v3 and 4v4 formats. */
export const EXCEPTIONAL_3V3_4V4_CLASSES: readonly string[] = [
    "Healer",
    "Battle Healer",
    "PvP Test Healer",
] as const;

/** Classes that strictly require static melee weapon range enforcement. */
export const STATIC_RANGE_CLASSES: readonly string[] = [
    "Defender",
    "Classic Defender",
] as const;

/** Required PvP Amulets under Non-OP rules (+5000 or +5500 HP). */
export const ALLOWED_PVP_AMULETS: readonly string[] = [
    "Platinum PvP Amulet +5000",
    "Diamond PvP Amulet +5500",
] as const;

/**
 * Standard vanilla AQW Enhancement Pattern IDs.
 * 1: Adventurer, 2: Fighter, 3: Thief, 4: Armsman, 5: Hybrid, 6: Wizard, 7: Healer, 8: Spellbreaker, 9: Lucky
 */
export const ALLOWED_ENHANCEMENT_PATTERN_IDS: ReadonlySet<number> = new Set([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
]);

/**
 * Forbidden Special Enhancements (Awe & Forge Pattern IDs in AQW).
 * 23: Awe, 24: Vainglory, 25: Vim, 26: Examen, 27: Pneuma, 28: Anima, 29: Penitence, 30: Lament, 32: Hearty
 */
export const FORBIDDEN_ENHANCEMENT_PATTERN_IDS: ReadonlySet<number> = new Set([
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
]);

/** BludrutBrawl (/bludrutbrawl) Safezone cells for Team A (Blue) and Team B (Red). */
export const BLUDRUT_SAFEZONE_CELLS: ReadonlySet<string> = new Set([
    "Enter0",
    "Enter1",
]);

/** Protected NPC keywords in /bludrutbrawl that MUST NOT be killed (mana tap only). */
export const PROTECTED_NPC_KEYWORDS: readonly string[] = [
    "brawler",
    "restorer",
    "captain",
    "healer",
] as const;

/**
 * Minimum NPC HP thresholds required to allow mana-tapping an NPC.
 * If an NPC's HP drops below either threshold, the Rule Shield immediately cancels the attack.
 */
export const NPC_MANA_TAP_SAFE_HP_MIN = 2000;
export const NPC_MANA_TAP_SAFE_HP_RATIO = 0.35;
