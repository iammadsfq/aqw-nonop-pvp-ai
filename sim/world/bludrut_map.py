#!/usr/bin/env python3
"""
sim/world/bludrut_map.py
Topologi 16-Screen Map Bludrut Brawl, AABB Pintu, BFS Hop Distances, dan NPC Definitions.
Dikalibrasi 1:1 dari Client Telemetry & Map SWF bludrutbrawl_r1.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque

# ==============================================================================
# 1. SCREEN CONSTANTS & INDICES (16 SCREENS)
# ==============================================================================

SCREEN_NAMES = [
    "Enter0",       # 0: Blue Safezone
    "Morale0C",     # 1: Blue B3
    "Morale0B",     # 2: Blue B2
    "Morale0A",     # 3: Blue B1
    "Captain0",     # 4: Blue Captain
    "Resource0B",   # 5: Blue Restorer 2
    "Resource0A",   # 6: Blue Restorer 1
    "Crosslower",   # 7: Mid
    "Crossupper",   # 8: Top
    "Resource1A",   # 9: Red Restorer 1
    "Resource1B",   # 10: Red Restorer 2
    "Captain1",     # 11: Red Captain
    "Morale1A",     # 12: Red B1
    "Morale1B",     # 13: Red B2
    "Morale1C",     # 14: Red B3
    "Enter1"        # 15: Red Safezone
]

SCREEN_INDICES: Dict[str, int] = {name: idx for idx, name in enumerate(SCREEN_NAMES)}

# ==============================================================================
# 2. STAGE BOUNDS & SPAWN POINTS
# ==============================================================================

WALK_BOUNDS = {
    "x_min": 35.0,
    "x_max": 930.0,
    "y_min": 255.0,
    "y_max": 500.0
}

SPAWN_COORDS = {
    "Left":   {"x": 191.0, "y": 351.0},  # rentang: [188..194, 348..354]
    "Right":  {"x": 787.5, "y": 351.0},  # rentang: [783..792, 348..354]
    "Top":    {"x": 488.0, "y": 430.0},  # rentang: [484..492, 426..434] (mendarat di bawah saat datang dari atas)
    "Bottom": {"x": 488.0, "y": 308.0},  # rentang: [484..492, 304..312] (mendarat di atas saat datang dari bawah)
}

SAFEZONE_SPAWN = {
    "Enter0": {"x": 322.5, "y": 353.0},
    "Enter1": {"x": 595.0, "y": 353.0}
}

# ==============================================================================
# 3. DOOR TRIGGER & AABB DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class DoorTrigger:
    target_cell: str
    target_pad: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    spawn_x: float
    spawn_y: float

    def contains(self, px: float, py: float) -> bool:
        return (self.x_min <= px <= self.x_max) and (self.y_min <= py <= self.y_max)


ROOM_DOORS: Dict[str, List[DoorTrigger]] = {
    # --- BLUE SIDE ---
    "Enter0": [
        DoorTrigger("Morale0C", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"])
    ],
    "Morale0C": [
        DoorTrigger("Enter0", "Right", 120.0, 220.0, 255.0, 300.0, SAFEZONE_SPAWN["Enter0"]["x"], SAFEZONE_SPAWN["Enter0"]["y"]),
        DoorTrigger("Morale0B", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"]),
        DoorTrigger("Captain0", "Top", 401.0, 573.0, 275.0, 300.0, SPAWN_COORDS["Top"]["x"], SPAWN_COORDS["Top"]["y"])
    ],
    "Morale0B": [
        DoorTrigger("Morale0C", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"]),
        DoorTrigger("Morale0A", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"])
    ],
    "Morale0A": [
        DoorTrigger("Morale0B", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"]),
        DoorTrigger("Crosslower", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"])
    ],
    "Captain0": [
        DoorTrigger("Morale0C", "Bottom", 35.0, 930.0, 455.0, 500.0, SPAWN_COORDS["Bottom"]["x"], SPAWN_COORDS["Bottom"]["y"])
    ],

    # --- TOP ROW (RESOURCES) ---
    "Resource0B": [
        DoorTrigger("Resource0A", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"])
    ],
    "Resource0A": [
        DoorTrigger("Resource0B", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"]),
        DoorTrigger("Crossupper", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"])
    ],

    # --- CENTER CROSS ---
    "Crosslower": [
        DoorTrigger("Morale0A", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"]),
        DoorTrigger("Morale1A", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"]),
        DoorTrigger("Crossupper", "Top", 430.0, 545.0, 255.0, 300.0, SPAWN_COORDS["Top"]["x"], SPAWN_COORDS["Top"]["y"])
    ],
    "Crossupper": [
        DoorTrigger("Resource0A", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"]),
        DoorTrigger("Resource1A", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"]),
        DoorTrigger("Crosslower", "Bottom", 35.0, 930.0, 465.0, 500.0, SPAWN_COORDS["Bottom"]["x"], SPAWN_COORDS["Bottom"]["y"])
    ],

    # --- RED SIDE ---
    "Resource1A": [
        DoorTrigger("Crossupper", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"]),
        DoorTrigger("Resource1B", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"])
    ],
    "Resource1B": [
        DoorTrigger("Resource1A", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"])
    ],
    "Morale1A": [
        DoorTrigger("Crosslower", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"]),
        DoorTrigger("Morale1B", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"])
    ],
    "Morale1B": [
        DoorTrigger("Morale1A", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"]),
        DoorTrigger("Morale1C", "Left", 765.0, 865.0, 255.0, 300.0, SPAWN_COORDS["Left"]["x"], SPAWN_COORDS["Left"]["y"])
    ],
    "Morale1C": [
        DoorTrigger("Morale1B", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"]),
        DoorTrigger("Captain1", "Top", 401.0, 573.0, 275.0, 300.0, SPAWN_COORDS["Top"]["x"], SPAWN_COORDS["Top"]["y"]),
        DoorTrigger("Enter1", "Left", 765.0, 865.0, 255.0, 300.0, SAFEZONE_SPAWN["Enter1"]["x"], SAFEZONE_SPAWN["Enter1"]["y"])
    ],
    "Captain1": [
        DoorTrigger("Morale1C", "Bottom", 35.0, 930.0, 455.0, 500.0, SPAWN_COORDS["Bottom"]["x"], SPAWN_COORDS["Bottom"]["y"])
    ],
    "Enter1": [
        DoorTrigger("Morale1C", "Right", 120.0, 220.0, 255.0, 300.0, SPAWN_COORDS["Right"]["x"], SPAWN_COORDS["Right"]["y"])
    ]
}

# ==============================================================================
# 4. PRECOMPUTED BFS TOPOLOGIC HOP DISTANCES
# ==============================================================================

def _build_hop_matrix() -> Dict[Tuple[str, str], int]:
    dist: Dict[Tuple[str, str], int] = {}
    for start_node in SCREEN_NAMES:
        queue = deque([(start_node, 0)])
        visited = {start_node}
        while queue:
            curr, d = queue.popleft()
            dist[(start_node, curr)] = d
            for door in ROOM_DOORS.get(curr, []):
                nxt = door.target_cell
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, d + 1))
    return dist

CELL_GRAPH_DIST: Dict[Tuple[str, str], int] = _build_hop_matrix()

# ==============================================================================
# 5. BLUDRUT NPCS SPECIFICATION
# ==============================================================================

@dataclass(frozen=True)
class NpcSpec:
    mon_map_id: int
    name: str
    cell: str
    team: int
    max_hp: float

BLUDRUT_NPCS: Tuple[NpcSpec, ...] = (
    # Blue Side (Team 0)
    NpcSpec(1, "Team A Captain", "Captain0", 0, 15000.0),
    NpcSpec(2, "(A) Brawler", "Morale0C", 0, 10000.0),
    NpcSpec(3, "(A) Brawler", "Morale0B", 0, 10000.0),
    NpcSpec(4, "(A) Brawler", "Morale0A", 0, 10000.0),
    NpcSpec(5, "(A) Defensive Restorer", "Resource0B", 0, 5000.0),
    NpcSpec(6, "(A) Defensive Restorer", "Resource0B", 0, 5000.0),
    NpcSpec(7, "(A) Defensive Restorer", "Resource0A", 0, 5000.0),
    NpcSpec(8, "(A) Defensive Restorer", "Resource0A", 0, 5000.0),

    # Red Side (Team 1)
    NpcSpec(9, "(B) Defensive Restorer", "Resource1A", 1, 5000.0),
    NpcSpec(10, "(B) Defensive Restorer", "Resource1A", 1, 5000.0),
    NpcSpec(11, "(B) Defensive Restorer", "Resource1B", 1, 5000.0),
    NpcSpec(12, "(B) Defensive Restorer", "Resource1B", 1, 5000.0),
    NpcSpec(13, "(B) Brawler", "Morale1A", 1, 10000.0),
    NpcSpec(14, "(B) Brawler", "Morale1B", 1, 10000.0),
    NpcSpec(15, "(B) Brawler", "Morale1C", 1, 10000.0),
    NpcSpec(16, "Team B Captain", "Captain1", 1, 15000.0),
    
)