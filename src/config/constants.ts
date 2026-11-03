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

/**
 * Complete 16-screen topology of `/bludrutbrawl` (7 Blue + 7 Red + 2 Center).
 * Maps standard community PvP screen names to AQW internal SWF cell names
 * and defines the undirected/directed neighbor connectivity graph.
 */
export interface BludrutScreenSpec {
    /** Community PvP alias (e.g., "Blue SZ", "Blue B3", "Mid", "Top"). */
    alias: string;
    /** Internal AQW cell name in `/bludrutbrawl` (e.g., "Enter0", "Morale0A", "Center"). */
    cell: string;
    /** Team side ownership ("blue" | "red" | "neutral"). */
    side: "blue" | "red" | "neutral";
    /** Screen role in BludrutBrawl. */
    role: "safezone" | "brawler" | "restorer" | "captain" | "mid" | "top";
    /** Directed door connections keyed by neighbor alias -> target cell. */
    neighbors: Readonly<Record<string, string>>;
}

export const BLUDRUT_16_SCREENS: readonly BludrutScreenSpec[] = [
    // --- BLUE SIDE (LEFT, 7 SCREENS) ---
    {
        alias: "Blue SZ",
        cell: "Enter0",
        side: "blue",
        role: "safezone",
        neighbors: { "Blue B3": "Morale0A" },
    },
    {
        alias: "Blue B3",
        cell: "Morale0A",
        side: "blue",
        role: "brawler",
        neighbors: {
            "Blue SZ": "Enter0",
            "Blue B2": "Morale0B",
            "Blue Capt": "Captain0",
        },
    },
    {
        alias: "Blue B2",
        cell: "Morale0B",
        side: "blue",
        role: "brawler",
        neighbors: {
            "Blue B3": "Morale0A",
            "Blue B1": "Morale0C",
        },
    },
    {
        alias: "Blue B1",
        cell: "Morale0C",
        side: "blue",
        role: "brawler",
        neighbors: {
            "Blue B2": "Morale0B",
            Mid: "Center",
        },
    },
    {
        alias: "Blue Capt",
        cell: "Captain0",
        side: "blue",
        role: "captain",
        neighbors: {
            "Blue B3": "Morale0A",
            "Blue R2": "Resource0B",
        },
    },
    {
        alias: "Blue R2",
        cell: "Resource0B",
        side: "blue",
        role: "restorer",
        neighbors: {
            "Blue Capt": "Captain0",
            "Blue R1": "Resource0A",
        },
    },
    {
        alias: "Blue R1",
        cell: "Resource0A",
        side: "blue",
        role: "restorer",
        neighbors: {
            "Blue R2": "Resource0B",
            Top: "Crossupper",
        },
    },

    // --- CENTER (2 SCREENS) ---
    {
        alias: "Mid",
        cell: "Center",
        side: "neutral",
        role: "mid",
        neighbors: {
            "Blue B1": "Morale0C",
            "Red B1": "Morale1C",
            Top: "Crossupper",
        },
    },
    {
        alias: "Top",
        cell: "Crossupper",
        side: "neutral",
        role: "top",
        neighbors: {
            "Blue R1": "Resource0A",
            "Red R1": "Resource1A",
            Mid: "Center",
        },
    },

    // --- RED SIDE (RIGHT, 7 SCREENS, MIRRORED) ---
    {
        alias: "Red R1",
        cell: "Resource1A",
        side: "red",
        role: "restorer",
        neighbors: {
            Top: "Crossupper",
            "Red R2": "Resource1B",
        },
    },
    {
        alias: "Red R2",
        cell: "Resource1B",
        side: "red",
        role: "restorer",
        neighbors: {
            "Red R1": "Resource1A",
            "Red Capt": "Captain1",
        },
    },
    {
        alias: "Red Capt",
        cell: "Captain1",
        side: "red",
        role: "captain",
        neighbors: {
            "Red R2": "Resource1B",
            "Red B3": "Morale1A",
        },
    },
    {
        alias: "Red B1",
        cell: "Morale1C",
        side: "red",
        role: "brawler",
        neighbors: {
            Mid: "Center",
            "Red B2": "Morale1B",
        },
    },
    {
        alias: "Red B2",
        cell: "Morale1B",
        side: "red",
        role: "brawler",
        neighbors: {
            "Red B1": "Morale1C",
            "Red B3": "Morale1A",
        },
    },
    {
        alias: "Red B3",
        cell: "Morale1A",
        side: "red",
        role: "brawler",
        neighbors: {
            "Red B2": "Morale1B",
            "Red Capt": "Captain1",
            "Red SZ": "Enter1",
        },
    },
    {
        alias: "Red SZ",
        cell: "Enter1",
        side: "red",
        role: "safezone",
        neighbors: { "Red B3": "Morale1A" },
    },
] as const;

/** Movement & Stamina (`intSP`) constants in AQW client (`Game4000r7.swf`). */
export const AQW_MAX_STAMINA = 100;
export const AQW_SPRINT_STAMINA_COST = 50;
export const AQW_DEFAULT_WALK_SPEED = 8;
export const AQW_SPRINT_WALK_SPEED = 16;

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
