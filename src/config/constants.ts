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
    "Wait",
]);

/**
 * 2D Axis-Aligned Bounding Box (AABB) representing a door activation zone on a screen.
 * In AQW (`/bludrutbrawl`), doors are NOT single points — they are 2D trigger areas
 * (e.g., `Top` -> `Mid` spans almost the entire bottom width `x: 140..900, y: 455..520`).
 * Any walk trajectory from A to B whose line segment intersects this box will trigger
 * the door transition mid-walk!
 */
export interface DoorTriggerArea {
    targetAlias: string;
    targetCell: string;
    targetPad: string;
    /** Door trigger bounding box in the origin cell. */
    xMin: number;
    xMax: number;
    yMin: number;
    yMax: number;
    /** Safe aim point `(x, y)` inside the door trigger when intentionally taking this door. */
    aimX: number;
    aimY: number;
    /** Empirical arrival spawn coordinates `(x, y)` in `targetCell`. */
    spawnX: number;
    spawnY: number;
}

/**
 * Complete 16-screen topology of `/bludrutbrawl` (7 Blue + 7 Red + 2 Center),
 * calibrated directly from live telemetry (`aqw_combat_telemetry.log`).
 *
 * Key topological rules verified from telemetry:
 * 1. `Blue SZ` (`Enter0`) connects to `Blue B3` (`Morale0C`).
 * 2. `Blue B3` (`Morale0C`) connects to `Blue SZ` (`Enter0`), `Blue Capt` (`Captain0`), and `Blue B2` (`Morale0B`).
 * 3. `Blue B2` (`Morale0B`) connects to `Blue B3` (`Morale0C`) and `Blue B1` (`Morale0A`).
 * 4. `Blue B1` (`Morale0A`) connects to `Blue B2` (`Morale0B`) and `Mid` (`Crosslower`).
 * 5. `Mid` (`Crosslower`) connects to `Blue B1` (`Morale0A`), `Red B1` (`Morale1A`), and `Top` (`Crossupper`).
 * 6. `Top` (`Crossupper`) connects to `Mid` (`Crosslower`), `Blue R1` (`Resource0A`), and `Red R1` (`Resource1A`).
 * 7. `Blue R2` (`Resource0B`) ONLY connects to `Blue R1` (`Resource0A`) — NO door to `Captain0`!
 * 8. `Blue Capt` (`Captain0`) ONLY connects to `Blue B3` (`Morale0C`) — NO door to `Resource0B`!
 */
export interface BludrutScreenSpec {
    /** Community PvP alias (e.g., "Blue SZ", "Blue B3", "Mid", "Top"). */
    alias: string;
    /** Internal AQW cell name in `/bludrutbrawl` (e.g., "Enter0", "Morale0C", "Crosslower"). */
    cell: string;
    /** Team side ownership ("blue" | "red" | "neutral"). */
    side: "blue" | "red" | "neutral";
    /** Screen role in BludrutBrawl. */
    role: "safezone" | "brawler" | "restorer" | "captain" | "mid" | "top";
    /** Directed door connections keyed by neighbor alias -> target cell. */
    neighbors: Readonly<Record<string, string>>;
    /** 2D Door Trigger bounding boxes inside this screen. */
    doors: readonly DoorTriggerArea[];
}

export const BLUDRUT_16_SCREENS: readonly BludrutScreenSpec[] = [
    // --- BLUE SIDE (LEFT, 7 SCREENS) ---
    {
        alias: "Blue SZ",
        cell: "Enter0",
        side: "blue",
        role: "safezone",
        neighbors: { "Blue B3": "Morale0C" },
        doors: [
            {
                targetAlias: "Blue B3",
                targetCell: "Morale0C",
                targetPad: "Left",
                xMin: 750,
                xMax: 960,
                yMin: 245,
                yMax: 315,
                aimX: 850,
                aimY: 262,
                spawnX: 173,
                spawnY: 342,
            },
        ],
    },
    {
        alias: "Blue B3",
        cell: "Morale0C",
        side: "blue",
        role: "brawler",
        neighbors: {
            "Blue SZ": "Enter0",
            "Blue B2": "Morale0B",
            "Blue Capt": "Captain0",
        },
        doors: [
            {
                targetAlias: "Blue SZ",
                targetCell: "Enter0",
                targetPad: "Right",
                xMin: 0,
                xMax: 220,
                yMin: 245,
                yMax: 312,
                aimX: 120,
                aimY: 265,
                spawnX: 789,
                spawnY: 354,
            },
            {
                targetAlias: "Blue B2",
                targetCell: "Morale0B",
                targetPad: "Left",
                xMin: 750,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 265,
                spawnX: 191,
                spawnY: 352,
            },
            {
                targetAlias: "Blue Capt",
                targetCell: "Captain0",
                targetPad: "Spawn",
                xMin: 400,
                xMax: 540,
                yMin: 220,
                yMax: 285,
                aimX: 465,
                aimY: 255,
                spawnX: 468,
                spawnY: 272,
            },
        ],
    },
    {
        alias: "Blue B2",
        cell: "Morale0B",
        side: "blue",
        role: "brawler",
        neighbors: {
            "Blue B3": "Morale0C",
            "Blue B1": "Morale0A",
        },
        doors: [
            {
                targetAlias: "Blue B3",
                targetCell: "Morale0C",
                targetPad: "Right",
                xMin: 0,
                xMax: 225,
                yMin: 245,
                yMax: 312,
                aimX: 120,
                aimY: 268,
                spawnX: 787,
                spawnY: 354,
            },
            {
                targetAlias: "Blue B1",
                targetCell: "Morale0A",
                targetPad: "Left",
                xMin: 735,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 268,
                spawnX: 192,
                spawnY: 349,
            },
        ],
    },
    {
        alias: "Blue B1",
        cell: "Morale0A",
        side: "blue",
        role: "brawler",
        neighbors: {
            "Blue B2": "Morale0B",
            Mid: "Crosslower",
        },
        doors: [
            {
                targetAlias: "Blue B2",
                targetCell: "Morale0B",
                targetPad: "Right",
                xMin: 0,
                xMax: 225,
                yMin: 245,
                yMax: 312,
                aimX: 120,
                aimY: 270,
                spawnX: 787,
                spawnY: 350,
            },
            {
                targetAlias: "Mid",
                targetCell: "Crosslower",
                targetPad: "Left",
                xMin: 745,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 266,
                spawnX: 186,
                spawnY: 350,
            },
        ],
    },
    {
        alias: "Blue Capt",
        cell: "Captain0",
        side: "blue",
        role: "captain",
        neighbors: {
            "Blue B3": "Morale0C",
        },
        doors: [
            {
                targetAlias: "Blue B3",
                targetCell: "Morale0C",
                targetPad: "Top",
                xMin: 360,
                xMax: 580,
                yMin: 445,
                yMax: 530,
                aimX: 485,
                aimY: 480,
                spawnX: 483,
                spawnY: 472,
            },
        ],
    },
    {
        alias: "Blue R2",
        cell: "Resource0B",
        side: "blue",
        role: "restorer",
        neighbors: {
            "Blue R1": "Resource0A",
        },
        doors: [
            {
                targetAlias: "Blue R1",
                targetCell: "Resource0A",
                targetPad: "Left",
                xMin: 780,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 268,
                spawnX: 188,
                spawnY: 351,
            },
        ],
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
        doors: [
            {
                targetAlias: "Blue R2",
                targetCell: "Resource0B",
                targetPad: "Right",
                xMin: 0,
                xMax: 235,
                yMin: 245,
                yMax: 312,
                aimX: 120,
                aimY: 268,
                spawnX: 785,
                spawnY: 350,
            },
            {
                targetAlias: "Top",
                targetCell: "Crossupper",
                targetPad: "Left",
                xMin: 755,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 270,
                spawnX: 192,
                spawnY: 351,
            },
        ],
    },

    // --- CENTER (2 SCREENS) ---
    {
        alias: "Mid",
        cell: "Crosslower",
        side: "neutral",
        role: "mid",
        neighbors: {
            "Blue B1": "Morale0A",
            "Red B1": "Morale1A",
            Top: "Crossupper",
        },
        doors: [
            {
                targetAlias: "Blue B1",
                targetCell: "Morale0A",
                targetPad: "Right",
                xMin: 0,
                xMax: 230,
                yMin: 245,
                yMax: 312,
                aimX: 120,
                aimY: 270,
                spawnX: 789,
                spawnY: 349,
            },
            {
                targetAlias: "Red B1",
                targetCell: "Morale1A",
                targetPad: "Left",
                xMin: 745,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 268,
                spawnX: 186,
                spawnY: 354,
            },
            {
                targetAlias: "Top",
                targetCell: "Crossupper",
                targetPad: "Bottom",
                xMin: 400,
                xMax: 530,
                yMin: 240,
                yMax: 305,
                aimX: 465,
                aimY: 258,
                spawnX: 488,
                spawnY: 429,
            },
        ],
    },
    {
        alias: "Top",
        cell: "Crossupper",
        side: "neutral",
        role: "top",
        neighbors: {
            "Blue R1": "Resource0A",
            "Red R1": "Resource1A",
            Mid: "Crosslower",
        },
        doors: [
            {
                targetAlias: "Blue R1",
                targetCell: "Resource0A",
                targetPad: "Right",
                xMin: 0,
                xMax: 230,
                yMin: 245,
                yMax: 315,
                aimX: 120,
                aimY: 275,
                spawnX: 785,
                spawnY: 350,
            },
            {
                targetAlias: "Red R1",
                targetCell: "Resource1A",
                targetPad: "Left",
                xMin: 745,
                xMax: 960,
                yMin: 245,
                yMax: 315,
                aimX: 855,
                aimY: 275,
                spawnX: 185,
                spawnY: 347,
            },
            // Wide horizontal door along the entire bottom edge of Top -> Mid!
            {
                targetAlias: "Mid",
                targetCell: "Crosslower",
                targetPad: "Middle",
                xMin: 120,
                xMax: 900,
                yMin: 450,
                yMax: 530,
                aimX: 485,
                aimY: 482,
                spawnX: 490,
                spawnY: 342,
            },
        ],
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
        doors: [
            {
                targetAlias: "Top",
                targetCell: "Crossupper",
                targetPad: "Right",
                xMin: 0,
                xMax: 185,
                yMin: 245,
                yMax: 315,
                aimX: 115,
                aimY: 275,
                spawnX: 787,
                spawnY: 351,
            },
            {
                targetAlias: "Red R2",
                targetCell: "Resource1B",
                targetPad: "Left",
                xMin: 745,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 268,
                spawnX: 188,
                spawnY: 350,
            },
        ],
    },
    {
        alias: "Red R2",
        cell: "Resource1B",
        side: "red",
        role: "restorer",
        neighbors: {
            "Red R1": "Resource1A",
        },
        doors: [
            {
                targetAlias: "Red R1",
                targetCell: "Resource1A",
                targetPad: "Right",
                xMin: 0,
                xMax: 225,
                yMin: 245,
                yMax: 312,
                aimX: 120,
                aimY: 268,
                spawnX: 785,
                spawnY: 350,
            },
        ],
    },
    {
        alias: "Red Capt",
        cell: "Captain1",
        side: "red",
        role: "captain",
        neighbors: {
            "Red B3": "Morale1C",
        },
        doors: [
            {
                targetAlias: "Red B3",
                targetCell: "Morale1C",
                targetPad: "Top",
                xMin: 360,
                xMax: 580,
                yMin: 445,
                yMax: 530,
                aimX: 410,
                aimY: 480,
                spawnX: 409,
                spawnY: 471,
            },
        ],
    },
    {
        alias: "Red B1",
        cell: "Morale1A",
        side: "red",
        role: "brawler",
        neighbors: {
            Mid: "Crosslower",
            "Red B2": "Morale1B",
        },
        doors: [
            {
                targetAlias: "Mid",
                targetCell: "Crosslower",
                targetPad: "Right",
                xMin: 0,
                xMax: 230,
                yMin: 245,
                yMax: 312,
                aimX: 120,
                aimY: 268,
                spawnX: 787,
                spawnY: 349,
            },
            {
                targetAlias: "Red B2",
                targetCell: "Morale1B",
                targetPad: "Left",
                xMin: 745,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 268,
                spawnX: 190,
                spawnY: 350,
            },
        ],
    },
    {
        alias: "Red B2",
        cell: "Morale1B",
        side: "red",
        role: "brawler",
        neighbors: {
            "Red B1": "Morale1A",
            "Red B3": "Morale1C",
        },
        doors: [
            {
                targetAlias: "Red B1",
                targetCell: "Morale1A",
                targetPad: "Right",
                xMin: 0,
                xMax: 230,
                yMin: 245,
                yMax: 312,
                aimX: 120,
                aimY: 270,
                spawnX: 785,
                spawnY: 350,
            },
            {
                targetAlias: "Red B3",
                targetCell: "Morale1C",
                targetPad: "Left",
                xMin: 745,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 270,
                spawnX: 190,
                spawnY: 350,
            },
        ],
    },
    {
        alias: "Red B3",
        cell: "Morale1C",
        side: "red",
        role: "brawler",
        neighbors: {
            "Red B2": "Morale1B",
            "Red Capt": "Captain1",
            "Red SZ": "Enter1",
        },
        doors: [
            {
                targetAlias: "Red B2",
                targetCell: "Morale1B",
                targetPad: "Right",
                xMin: 0,
                xMax: 235,
                yMin: 245,
                yMax: 312,
                aimX: 120,
                aimY: 270,
                spawnX: 785,
                spawnY: 350,
            },
            {
                targetAlias: "Red Capt",
                targetCell: "Captain1",
                targetPad: "Spawn",
                xMin: 390,
                xMax: 520,
                yMin: 220,
                yMax: 285,
                aimX: 430,
                aimY: 255,
                spawnX: 437,
                spawnY: 271,
            },
            {
                targetAlias: "Red SZ",
                targetCell: "Enter1",
                targetPad: "Left",
                xMin: 750,
                xMax: 960,
                yMin: 245,
                yMax: 312,
                aimX: 855,
                aimY: 268,
                spawnX: 185,
                spawnY: 350,
            },
        ],
    },
    {
        alias: "Red SZ",
        cell: "Enter1",
        side: "red",
        role: "safezone",
        neighbors: { "Red B3": "Morale1C" },
        doors: [
            {
                targetAlias: "Red B3",
                targetCell: "Morale1C",
                targetPad: "Right",
                xMin: 0,
                xMax: 220,
                yMin: 245,
                yMax: 315,
                aimX: 120,
                aimY: 268,
                spawnX: 785,
                spawnY: 350,
            },
        ],
    },
] as const;

/**
 * BFS Shortest-Path Door Resolver across the 16 screens of `/bludrutbrawl`.
 * Given `fromCell` and `goalCell`, returns the immediate `DoorTriggerArea` in `fromCell`
 * that leads along the shortest path to `goalCell` (avoiding Safezones unless `goalCell` is a Safezone).
 */
export function findNextDoorTowardCell(
    fromCell: string,
    goalCell: string,
): DoorTriggerArea | null {
    if (fromCell.toLowerCase() === goalCell.toLowerCase()) return null;
    const startScreen = BLUDRUT_16_SCREENS.find(
        (s) => s.cell.toLowerCase() === fromCell.toLowerCase(),
    );
    if (!startScreen || startScreen.doors.length === 0) return null;

    const queue: { firstDoor: DoorTriggerArea; currentCell: string }[] = [];
    const visited = new Set<string>([startScreen.cell.toLowerCase()]);

    for (const d of startScreen.doors) {
        if (
            (d.targetCell === "Enter0" || d.targetCell === "Enter1") &&
            d.targetCell.toLowerCase() !== goalCell.toLowerCase()
        ) {
            continue;
        }
        queue.push({ firstDoor: d, currentCell: d.targetCell });
    }

    while (queue.length > 0) {
        const item = queue.shift()!;
        if (item.currentCell.toLowerCase() === goalCell.toLowerCase()) {
            return item.firstDoor;
        }
        const key = item.currentCell.toLowerCase();
        if (visited.has(key)) continue;
        visited.add(key);

        const nextScreen = BLUDRUT_16_SCREENS.find((s) => s.cell.toLowerCase() === key);
        if (!nextScreen) continue;
        for (const nd of nextScreen.doors) {
            const nKey = nd.targetCell.toLowerCase();
            if (visited.has(nKey)) continue;
            if (
                (nd.targetCell === "Enter0" || nd.targetCell === "Enter1") &&
                nKey !== goalCell.toLowerCase()
            ) {
                continue;
            }
            queue.push({ firstDoor: item.firstDoor, currentCell: nd.targetCell });
        }
    }

    return startScreen.doors[0] ?? null;
}

/**
 * Calculates the optimal aim coordinates inside a door trigger area.
 * For wide horizontal doors (like the bottom door of Top -> Mid, spanning x: 120..900),
 * it returns the closest point along the door trigger directly matching the character's
 * current X position, avoiding unnecessary diagonal traversal across the screen!
 */
export function getDoorAimCoordinates(
    door: DoorTriggerArea,
    currentX: number,
    _currentY: number,
): { x: number; y: number } {
    const width = door.xMax - door.xMin;
    if (width >= 250) {
        // Wide door trigger area (e.g. Top -> Mid bottom edge): enter through nearest point along its width
        const clampedX = Math.round(Math.max(door.xMin + 30, Math.min(door.xMax - 30, currentX)));
        return { x: clampedX, y: door.aimY };
    }

    // For standard doors, return calibrated aim coordinates squarely inside door hit area
    return { x: door.aimX, y: door.aimY };
}

/**
 * Checks whether a linear walk trajectory from `(x0, y0)` to `(x1, y1)` inside `cell`
 * intersects any 2D door activation zone (`DoorTriggerArea`) along the path.
 * Uses the Liang-Barsky line-segment vs AABB intersection algorithm.
 *
 * Returns the FIRST door hit along `t in [0, 1]` (if any), preventing accidental
 * screen transitions when kiting inside a room (and modeling them 1:1 in the simulator).
 */
export function checkTrajectoryDoorCollision(
    cell: string,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
): { door: DoorTriggerArea; hitT: number; hitX: number; hitY: number } | null {
    const screen = BLUDRUT_16_SCREENS.find((s) => s.cell.toLowerCase() === cell.toLowerCase());
    if (!screen) return null;

    const dx = x1 - x0;
    const dy = y1 - y0;
    let bestHit: { door: DoorTriggerArea; hitT: number; hitX: number; hitY: number } | null = null;

    for (const door of screen.doors) {
        let tMin = 0;
        let tMax = 1;
        const p = [-dx, dx, -dy, dy];
        const q = [x0 - door.xMin, door.xMax - x0, y0 - door.yMin, door.yMax - y0];
        let intersects = true;

        for (let i = 0; i < 4; i++) {
            const pi = p[i]!;
            const qi = q[i]!;
            if (Math.abs(pi) < 1e-8) {
                if (qi < 0) {
                    intersects = false;
                    break;
                }
            } else {
                const t = qi / pi;
                if (pi < 0) {
                    if (t > tMax) {
                        intersects = false;
                        break;
                    }
                    if (t > tMin) tMin = t;
                } else {
                    if (t < tMin) {
                        intersects = false;
                        break;
                    }
                    if (t < tMax) tMax = t;
                }
            }
        }

        if (intersects && tMin <= tMax) {
            if (!bestHit || tMin < bestHit.hitT) {
                bestHit = {
                    door,
                    hitT: tMin,
                    hitX: x0 + dx * tMin,
                    hitY: y0 + dy * tMin,
                };
            }
        }
    }

    return bestHit;
}

/**
 * Movement, Stamina (`intSP`), and Server Regeneration constants in AQW (`Game4000r7.swf` + server `ct` ticks),
 * calibrated from `/bludrutbrawl` telemetry (`aqw_combat_telemetry.log`).
 */
export const AQW_MAX_STAMINA = 100;
export const AQW_SPRINT_STAMINA_COST = 50;
export const AQW_DEFAULT_WALK_SPEED = 8;
export const AQW_SPRINT_WALK_SPEED = 24;

/** Server regeneration tick interval (~6.0 seconds). */
export const AQW_SERVER_REGEN_TICK_MS = 6000;
/** Stamina (`intSP`) restored per 6.0s server tick (+25 SP = 25% of MaxSP). */
export const AQW_STAMINA_REGEN_PER_TICK = 25;
/** Passive out-of-combat MP restored on initial non-rest tick (+3 MP = 3% of MaxMP). */
export const AQW_OOC_PASSIVE_MP_REGEN_PER_TICK = 3;
/** Out-of-combat / Rest HP & MP restored per 6.0s server tick (+15% MaxHP, +15% MaxMP). */
export const AQW_OOC_REST_HP_RATIO_PER_TICK = 0.15;
export const AQW_OOC_REST_MP_REGEN_PER_TICK = 15;
/** Warrior / Melee Mana Gain in combat: +6 MP on normal Auto Attack hit, +9 MP on Crit Auto Attack, +3 MP when hit. */
export const AQW_WARRIOR_AA_HIT_MP_GAIN = 6;
export const AQW_WARRIOR_AA_CRIT_MP_GAIN = 9;
export const AQW_WARRIOR_ON_HIT_TAKEN_MP_GAIN = 3;

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
