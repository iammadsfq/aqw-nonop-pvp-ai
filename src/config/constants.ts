/**
 * @file constants.ts
 * @description Official Classic (Non-OP) PvP ruleset constants, allowed classes,
 * enhancement IDs, amulet requirements, and BludrutBrawl map metadata.
 * Dikalibrasi 1:1 mengikuti sim/world/bludrut_map.py.
 */

// =============================================================================
// 1. CLASS & RULESET CONSTANTS
// =============================================================================

export const STANDARD_NON_OP_CLASSES: readonly string[] = [
  "Warrior", "Classic Warrior", "Battle Warrior", "Ninja", "Classic Ninja",
  "Dragonlord", "Classic Dragonlord", "Starlord", "Classic Starlord",
  "Paladin", "Classic Paladin", "Battle Bard", "Bard", "Defender",
  "Classic Defender", "Guardian", "Classic Guardian", "Barber",
  "Rustbucket", "UndeadSlayer", "DoomKnight", "Classic DoomKnight",
  "No Class", "Hobo Highlord", "Obsidian No Class", "Clawsuit",
] as const;

export const EXCEPTIONAL_3V3_4V4_CLASSES: readonly string[] = [
  "Healer", "Battle Healer", "PvP Test Healer",
] as const;

export const STATIC_RANGE_CLASSES: readonly string[] = [
  "Defender", "Classic Defender",
] as const;

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
// 2. MAP TOPOLOGY & GRAPH DISTANCES (16 SCREENS - IDENTIK BLUDRUT_MAP.PY)
// =============================================================================

export const SCREEN_NAMES: readonly string[] = [
  "Enter0",       // 0: Blue Safezone
  "Morale0C",     // 1: Blue B3
  "Morale0B",     // 2: Blue B2
  "Morale0A",     // 3: Blue B1
  "Captain0",     // 4: Blue Captain
  "Resource0B",   // 5: Blue Restorer 2
  "Resource0A",   // 6: Blue Restorer 1
  "Crosslower",   // 7: Mid
  "Crossupper",   // 8: Top
  "Resource1A",   // 9: Red Restorer 1
  "Resource1B",   // 10: Red Restorer 2
  "Captain1",     // 11: Red Captain
  "Morale1A",     // 12: Red B1
  "Morale1B",     // 13: Red B2
  "Morale1C",     // 14: Red B3
  "Enter1"        // 15: Red Safezone
] as const;

export const SCREEN_INDICES: Readonly<Record<string, number>> = Object.freeze(
  SCREEN_NAMES.reduce((acc, name, idx) => {
    acc[name] = idx;
    return acc;
  }, {} as Record<string, number>)
);

export const WALK_BOUNDS = Object.freeze({
  xMin: 35.0,
  xMax: 930.0,
  yMin: 255.0,
  yMax: 500.0,
});

export const SPAWN_COORDS = Object.freeze({
  Left: { x: 191.0, y: 351.0 },
  Right: { x: 787.5, y: 351.0 },
  Top: { x: 488.0, y: 430.0 },
  Bottom: { x: 488.0, y: 308.0 },
});

export const SAFEZONE_SPAWN = Object.freeze({
  Enter0: { x: 322.5, y: 353.0 },
  Enter1: { x: 595.0, y: 353.0 },
});

// =============================================================================
// 3. AABB DOORS & GRAPH TOPOLOGY
// =============================================================================

export interface DoorTriggerArea {
  targetCell: string;
  targetPad: string;
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  spawnX: number;
  spawnY: number;
}

export const ROOM_DOORS: Readonly<Record<string, readonly DoorTriggerArea[]>> = Object.freeze({
  // --- BLUE SIDE ---
  "Enter0": [
    { targetCell: "Morale0C", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 }
  ],
  "Morale0C": [
    { targetCell: "Enter0", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 322.5, spawnY: 353.0 },
    { targetCell: "Morale0B", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 },
    { targetCell: "Captain0", targetPad: "Top", xMin: 401.0, xMax: 573.0, yMin: 275.0, yMax: 300.0, spawnX: 488.0, spawnY: 430.0 }
  ],
  "Morale0B": [
    { targetCell: "Morale0C", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 },
    { targetCell: "Morale0A", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 }
  ],
  "Morale0A": [
    { targetCell: "Morale0B", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 },
    { targetCell: "Crosslower", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 }
  ],
  "Captain0": [
    { targetCell: "Morale0C", targetPad: "Bottom", xMin: 35.0, xMax: 930.0, yMin: 455.0, yMax: 500.0, spawnX: 488.0, spawnY: 308.0 }
  ],

  // --- TOP ROW (RESOURCES) ---
  "Resource0B": [
    { targetCell: "Resource0A", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 }
  ],
  "Resource0A": [
    { targetCell: "Resource0B", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 },
    { targetCell: "Crossupper", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 }
  ],

  // --- CENTER CROSS ---
  "Crosslower": [
    { targetCell: "Morale0A", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 },
    { targetCell: "Morale1A", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 },
    { targetCell: "Crossupper", targetPad: "Top", xMin: 430.0, xMax: 545.0, yMin: 255.0, yMax: 300.0, spawnX: 488.0, spawnY: 430.0 }
  ],
  "Crossupper": [
    { targetCell: "Resource0A", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 },
    { targetCell: "Resource1A", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 },
    { targetCell: "Crosslower", targetPad: "Bottom", xMin: 35.0, xMax: 930.0, yMin: 465.0, yMax: 500.0, spawnX: 488.0, spawnY: 308.0 }
  ],

  // --- RED SIDE ---
  "Resource1A": [
    { targetCell: "Crossupper", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 },
    { targetCell: "Resource1B", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 }
  ],
  "Resource1B": [
    { targetCell: "Resource1A", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 }
  ],
  "Morale1A": [
    { targetCell: "Crosslower", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 },
    { targetCell: "Morale1B", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 }
  ],
  "Morale1B": [
    { targetCell: "Morale1A", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 },
    { targetCell: "Morale1C", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 191.0, spawnY: 351.0 }
  ],
  "Morale1C": [
    { targetCell: "Morale1B", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 },
    { targetCell: "Captain1", targetPad: "Top", xMin: 401.0, xMax: 573.0, yMin: 275.0, yMax: 300.0, spawnX: 488.0, spawnY: 430.0 },
    { targetCell: "Enter1", targetPad: "Left", xMin: 765.0, xMax: 865.0, yMin: 255.0, yMax: 300.0, spawnX: 595.0, spawnY: 353.0 }
  ],
  "Captain1": [
    { targetCell: "Morale1C", targetPad: "Bottom", xMin: 35.0, xMax: 930.0, yMin: 455.0, yMax: 500.0, spawnX: 488.0, spawnY: 308.0 }
  ],
  "Enter1": [
    { targetCell: "Morale1C", targetPad: "Right", xMin: 120.0, xMax: 220.0, yMin: 255.0, yMax: 300.0, spawnX: 787.5, spawnY: 351.0 }
  ]
});

function computeBfsDistances(): Record<string, number> {
  const distMap: Record<string, number> = {};
  for (const start of SCREEN_NAMES) {
    const queue: [string, number][] = [[start, 0]];
    const visited = new Set<string>([start]);
    while (queue.length > 0) {
      const [curr, d] = queue.shift()!;
      distMap[`${start}->${curr}`] = d;
      const doors = ROOM_DOORS[curr] || [];
      for (const door of doors) {
        const nxt = door.targetCell;
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
// 4. COMBAT & REGENERATION TELEMETRY
// =============================================================================

export const AQW_MAX_STAMINA = 100;
export const AQW_SPRINT_STAMINA_COST = 50;
export const AQW_DEFAULT_WALK_SPEED = 8;
export const AQW_SPRINT_WALK_SPEED = 24;

export const AQW_SERVER_REGEN_TICK_MS = 2000;
export const AQW_OOC_STAMINA_REGEN_PER_TICK = 25;
export const AQW_IN_COMBAT_STAMINA_REGEN_PER_TICK = 5;
export const AQW_OOC_REST_HP_RATIO_PER_TICK = 0.15;
export const AQW_OOC_REST_MP_REGEN_PER_TICK = 15;

export const AQW_WARRIOR_AA_HIT_MP_GAIN = 6;
export const AQW_WARRIOR_AA_CRIT_MP_GAIN = 12;
export const AQW_WARRIOR_ON_HIT_TAKEN_MP_GAIN = 3;

export const AQW_HEALER_AA_HIT_MP_GAIN = 1;
export const AQW_HEALER_AA_CRIT_MP_GAIN = 5;
export const AQW_HEALER_ON_HIT_TAKEN_MP_GAIN = 1;

export const PROTECTED_NPC_KEYWORDS: readonly string[] = [
  "brawler", "restorer", "captain", "healer",
] as const;