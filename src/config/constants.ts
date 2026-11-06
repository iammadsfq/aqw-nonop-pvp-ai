/**
 * @file constants.ts
 * @description Official Classic (Non-OP) PvP ruleset constants, allowed classes,
 * enhancement IDs, amulet requirements, and BludrutBrawl map metadata.
 * Sesuai 1:1 dengan sim/world/bludrut_map.py.
 */

// =============================================================================
// 1. CLASS & RULESET CONSTANTS
// =============================================================================

/** Standard Tier classes allowed in all Non-OP formats (1v1, 2v2, 3v3, 4v4). */
export const STANDARD_NON_OP_CLASSES: readonly string[] = [
    "Warrior", "Classic Warrior", "Battle Warrior",
    "Ninja", "Classic Ninja",
    "Dragonlord", "Classic Dragonlord",
    "Starlord", "Classic Starlord",
    "Paladin", "Classic Paladin",
    "Battle Bard", "Bard",
    "Defender", "Classic Defender",
    "Guardian", "Classic Guardian",
    "Barber", "Rustbucket", "UndeadSlayer",
    "DoomKnight", "Classic DoomKnight",
    "No Class", "Hobo Highlord", "Obsidian No Class", "Clawsuit",
] as const;

/** Exceptional Tier classes allowed ONLY in 3v3 and 4v4 formats. */
export const EXCEPTIONAL_3V3_4V4_CLASSES: readonly string[] = [
    "Healer", "Battle Healer", "PvP Test Healer",
] as const;

/** Classes that strictly require static melee weapon range enforcement. */
export const STATIC_RANGE_CLASSES: readonly string[] = [
    "Defender", "Classic Defender",
] as const;

/** Required PvP Amulets under Non-OP rules (+5000 or +5500 HP). */
export const ALLOWED_PVP_AMULETS: readonly string[] = [
    "Platinum PvP Amulet +5000",
    "Diamond PvP Amulet +5500",
] as const;

export const ALLOWED_ENHANCEMENT_PATTERN_IDS: ReadonlySet<number> = new Set([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
]);

export const FORBIDDEN_ENHANCEMENT_PATTERN_IDS: ReadonlySet<number> = new Set([
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
]);

export const BLUDRUT_SAFEZONE_CELLS: ReadonlySet<string> = new Set([
    "Enter0", "Enter1", "Wait",
]);

// =============================================================================
// 2. MAP TOPOLOGY & GRAPH DISTANCES (16 PLAYABLE SCREENS)
// =============================================================================

export const SCREEN_NAMES: readonly string[] = [
    // Blue Side
    "Enter0", "Morale0C", "Captain0", "Morale0B", "Morale0A",
    // Mid Hub
    "Crosslower", "Crossupper",
    // Upper Resource Corridor (Buff Lane)
    "Resource0B", "Resource0A", "Resource1A", "Resource1B",
    // Red Side
    "Morale1A", "Morale1B", "Morale1C", "Captain1", "Enter1"
] as const;

export const SCREEN_INDICES: Readonly<Record<string, number>> = Object.freeze(
    SCREEN_NAMES.reduce((acc, name, idx) => {
        acc[name] = idx;
        return acc;
    }, {} as Record<string, number>)
);

export const ADJACENCY: Readonly<Record<string, readonly string[]>> = Object.freeze({
    "Enter0": ["Morale0C"],
    "Morale0C": ["Enter0", "Captain0", "Morale0B"],
    "Captain0": ["Morale0C"],
    "Morale0B": ["Morale0C", "Morale0A"],
    "Morale0A": ["Morale0B", "Crosslower"],
    "Crosslower": ["Morale0A", "Morale1A", "Crossupper"],
    "Crossupper": ["Crosslower", "Resource0A", "Resource1A"],
    "Resource0B": ["Resource0A"],
    "Resource0A": ["Resource0B", "Crossupper"],
    "Resource1A": ["Crossupper", "Resource1B"],
    "Resource1B": ["Resource1A"],
    "Morale1A": ["Crosslower", "Morale1B"],
    "Morale1B": ["Morale1A", "Morale1C"],
    "Morale1C": ["Morale1B", "Captain1", "Enter1"],
    "Captain1": ["Morale1C"],
    "Enter1": ["Morale1C"],
});

function computeBfsDistances(): Record<string, number> {
    const distMap: Record<string, number> = {};
    for (const start of SCREEN_NAMES) {
        const queue: [string, number][] = [[start, 0]];
        const visited = new Set<string>([start]);
        while (queue.length > 0) {
            const [curr, d] = queue.shift()!;
            distMap[`${start}->${curr}`] = d;
            const neighbors = ADJACENCY[curr] || [];
            for (const nxt of neighbors) {
                if (!visited.has(nxt)) {
                    visited.add(nxt);
                    queue.push([nxt, d + 1]);
                }
            }
        }
    }
    return distMap;
}

export const CELL_GRAPH_DIST: Readonly<Record<string, number>> = Object.freeze(computeBfsDistances());

// =============================================================================
// 3. PHYSICAL BOUNDS, SPAWN POINTS, AND AABB DOORS
// =============================================================================

export const WALK_BOUNDS = Object.freeze({
    xMin: 35.0,
    xMax: 930.0,
    yMin: 255.0,
    yMax: 500.0,
});

export const LANDING_SPAWN = Object.freeze({
    Left:   { x: 191.0, y: 351.0 }, // (188+194)/2, (348+354)/2
    Right:  { x: 787.5, y: 351.0 }, // (783+792)/2, (348+354)/2
    Top:    { x: 488.0, y: 430.0 }, // (484+492)/2, (426+434)/2
    Bottom: { x: 488.0, y: 308.0 }, // (484+492)/2, (304+312)/2
});

export const SAFEZONE_SPAWN = Object.freeze({
    Enter0: { x: 322.5, y: 353.0 }, // Blue Safezone
    Enter1: { x: 595.0, y: 353.0 }, // Red Safezone
});

export interface DoorTriggerArea {
    name: "Left" | "Right" | "Top" | "Bottom";
    targetCell: string;
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
    spawnX: number;
    spawnY: number;
}

function createDoor(doorType: "Left" | "Right" | "Top" | "Bottom", targetCell: string): DoorTriggerArea {
    if (doorType === "Right") {
        return {
            name: "Right", targetCell,
            xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0,
            spawnX: LANDING_SPAWN.Left.x, spawnY: LANDING_SPAWN.Left.y,
        };
    } else if (doorType === "Left") {
        return {
            name: "Left", targetCell,
            xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0,
            spawnX: LANDING_SPAWN.Right.x, spawnY: LANDING_SPAWN.Right.y,
        };
    } else if (doorType === "Top") {
        return {
            name: "Top", targetCell,
            xMin: 430.0, xMax: 545.0, yMin: 255.0, yMax: 300.0,
            spawnX: LANDING_SPAWN.Top.x, spawnY: LANDING_SPAWN.Top.y,
        };
    } else if (doorType === "Bottom") {
        return {
            name: "Bottom", targetCell,
            xMin: 35.0, xMax: 930.0, yMin: 465.0, yMax: 500.0,
            spawnX: LANDING_SPAWN.Bottom.x, spawnY: LANDING_SPAWN.Bottom.y,
        };
    }
    throw new Error(`Unknown door type: ${doorType}`);
}

export const ROOM_DOORS: Readonly<Record<string, readonly DoorTriggerArea[]>> = Object.freeze({
    "Enter0": [createDoor("Right", "Morale0C")],
    "Morale0C": [
        createDoor("Left", "Enter0"),
        createDoor("Top", "Captain0"),
        createDoor("Right", "Morale0B")
    ],
    "Captain0": [createDoor("Bottom", "Morale0C")],
    "Morale0B": [
        createDoor("Left", "Morale0C"),
        createDoor("Right", "Morale0A")
    ],
    "Morale0A": [
        createDoor("Left", "Morale0B"),
        createDoor("Right", "Crosslower")
    ],
    "Crosslower": [
        createDoor("Left", "Morale0A"),
        createDoor("Top", "Crossupper"),
        createDoor("Right", "Morale1A")
    ],
    "Crossupper": [
        createDoor("Left", "Resource0A"),
        createDoor("Bottom", "Crosslower"),
        createDoor("Right", "Resource1A")
    ],
    "Resource0A": [
        createDoor("Left", "Resource0B"),
        createDoor("Right", "Crossupper")
    ],
    "Resource0B": [createDoor("Right", "Resource0A")],
    "Resource1A": [
        createDoor("Left", "Crossupper"),
        createDoor("Right", "Resource1B")
    ],
    "Resource1B": [createDoor("Left", "Resource1A")],
    "Morale1A": [
        createDoor("Left", "Crosslower"),
        createDoor("Right", "Morale1B")
    ],
    "Morale1B": [
        createDoor("Left", "Morale1A"),
        createDoor("Right", "Morale1C")
    ],
    "Morale1C": [
        createDoor("Left", "Morale1B"),
        createDoor("Top", "Captain1"),
        createDoor("Right", "Enter1")
    ],
    "Captain1": [createDoor("Bottom", "Morale1C")],
    "Enter1": [createDoor("Left", "Morale1C")]
});

// =============================================================================
// 4. COMBAT & REGENERATION TELEMETRY (GROUND TRUTH)
// =============================================================================

export const AQW_MAX_STAMINA = 100;
export const AQW_SPRINT_STAMINA_COST = 50;
export const AQW_DEFAULT_WALK_SPEED = 8;
export const AQW_SPRINT_WALK_SPEED = 24;

/** Server regeneration tick interval tepat 2.0 detik (terverifikasi dari paket log ct). */
export const AQW_SERVER_REGEN_TICK_MS = 2000;

/** Stamina (+25 SP saat OOC, +5 SP saat in-combat per tick 2.0s). */
export const AQW_OOC_STAMINA_REGEN_PER_TICK = 25;
export const AQW_IN_COMBAT_STAMINA_REGEN_PER_TICK = 5;

/** Out-of-combat Rest HP (+15% MaxHP per tick 2.0s). */
export const AQW_OOC_REST_HP_RATIO_PER_TICK = 0.15;
export const AQW_OOC_REST_MP_REGEN_PER_TICK = 15;

/** Battle Warrior Mana Gains. */
export const AQW_WARRIOR_AA_HIT_MP_GAIN = 6;
export const AQW_WARRIOR_AA_CRIT_MP_GAIN = 12;
export const AQW_WARRIOR_ON_HIT_TAKEN_MP_GAIN = 3;

/** Battle Healer Mana Gains. */
export const AQW_HEALER_AA_HIT_MP_GAIN = 1;
export const AQW_HEALER_AA_CRIT_MP_GAIN = 5;
export const AQW_HEALER_ON_HIT_TAKEN_MP_GAIN = 1;

export const PROTECTED_NPC_KEYWORDS: readonly string[] = [
    "brawler", "restorer", "captain", "healer",
] as const;

export const NPC_MANA_TAP_SAFE_HP_MIN = 2000;
export const NPC_MANA_TAP_SAFE_HP_RATIO = 0.35;