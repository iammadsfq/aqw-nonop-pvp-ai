#!/usr/bin/env python3
"""
sim/world/bludrut_map.py
Topologi Resmi 16 Ruangan Bludrut Brawl AQW, AABB Hitbox Pintu, Landing Spawn, dan NPC.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

# 16 Ruangan Playable Resmi Bludrut Brawl (Topological Order)
SCREEN_NAMES = [
    # Blue Side
    "Enter0", "Morale0C", "Captain0", "Morale0B", "Morale0A",
    # Mid Hub
    "Crosslower", "Crossupper",
    # Upper Resource Corridor (Buff Lane)
    "Resource0B", "Resource0A", "Resource1A", "Resource1B",
    # Red Side
    "Morale1A", "Morale1B", "Morale1C", "Captain1", "Enter1"
]

SCREEN_INDICES: Dict[str, int] = {name: idx for idx, name in enumerate(SCREEN_NAMES)}

# Graf Koneksi Ruangan (Adjacency List Resmi)
ADJACENCY: Dict[str, List[str]] = {
    # Blue Side
    "Enter0": ["Morale0C"],
    "Morale0C": ["Enter0", "Captain0", "Morale0B"],
    "Captain0": ["Morale0C"],
    "Morale0B": ["Morale0C", "Morale0A"],
    "Morale0A": ["Morale0B", "Crosslower"],

    # Mid Junction
    "Crosslower": ["Morale0A", "Morale1A", "Crossupper"],
    "Crossupper": ["Crosslower", "Resource0A", "Resource1A"],

    # Top Resource Wings
    "Resource0B": ["Resource0A"],
    "Resource0A": ["Resource0B", "Crossupper"],
    "Resource1A": ["Crossupper", "Resource1B"],
    "Resource1B": ["Resource1A"],

    # Red Side
    "Morale1A": ["Crosslower", "Morale1B"],
    "Morale1B": ["Morale1A", "Morale1C"],
    "Morale1C": ["Morale1B", "Captain1", "Enter1"],
    "Captain1": ["Morale1C"],
    "Enter1": ["Morale1C"],
}

# BFS Precomputation: Jarak Hop Antar-Ruangan
def _compute_bfs_distances() -> Dict[Tuple[str, str], int]:
    dist_map: Dict[Tuple[str, str], int] = {}
    for start in SCREEN_NAMES:
        queue = [(start, 0)]
        visited = {start}
        while queue:
            curr, d = queue.pop(0)
            dist_map[(start, curr)] = d
            for nxt in ADJACENCY.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, d + 1))
    return dist_map

CELL_GRAPH_DIST = _compute_bfs_distances()

# Batas Maksimum Gerak Pemain (Valid Move Area)
WALK_BOUNDS = {
    "x_min": 35.0,
    "x_max": 930.0,
    "y_min": 255.0,
    "y_max": 500.0,
}

# Landing Zones / Spawn Points (Nilai Tengah dari Interval Resmi)
LANDING_SPAWN = {
    "Left":   {"x": (188.0 + 194.0) / 2.0, "y": (348.0 + 354.0) / 2.0},  # (191.0, 351.0)
    "Right":  {"x": (783.0 + 792.0) / 2.0, "y": (348.0 + 354.0) / 2.0},  # (787.5, 351.0)
    "Top":    {"x": (484.0 + 492.0) / 2.0, "y": (426.0 + 434.0) / 2.0},  # (488.0, 430.0) -> turun dari atas
    "Bottom": {"x": (484.0 + 492.0) / 2.0, "y": (304.0 + 312.0) / 2.0},  # (488.0, 308.0) -> naik dari bawah
}

# Titik Mendarat Khusus Safezone (Centroid Interval)
SAFEZONE_SPAWN = {
    "Enter0": {"x": (319.0 + 326.0) / 2.0, "y": (349.0 + 357.0) / 2.0},  # (322.5, 353.0)
    "Enter1": {"x": (592.0 + 598.0) / 2.0, "y": (349.0 + 357.0) / 2.0},  # (595.0, 353.0)
}

# Model Pintu Axis-Aligned Bounding Box (AABB)
@dataclass
class DoorTrigger:
    name: str
    target_cell: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    spawn_x: float
    spawn_y: float

    def contains(self, x: float, y: float) -> bool:
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)

def _create_door(door_type: str, target_cell: str) -> DoorTrigger:
    if door_type == "Right":
        # Masuk lewat kanan (765-865, 255-300) -> Muncul di sisi kiri room target
        return DoorTrigger("Right", target_cell, 765.0, 865.0, 255.0, 300.0,
                           LANDING_SPAWN["Left"]["x"], LANDING_SPAWN["Left"]["y"])
    elif door_type == "Left":
        # Masuk lewat kiri (120-220, 255-300) -> Muncul di sisi kanan room target
        return DoorTrigger("Left", target_cell, 120.0, 220.0, 255.0, 300.0,
                           LANDING_SPAWN["Right"]["x"], LANDING_SPAWN["Right"]["y"])
    elif door_type == "Top":
        # Masuk lewat atas (430-545, 255-300) -> Muncul di sisi atas room target
        return DoorTrigger("Top", target_cell, 430.0, 545.0, 255.0, 300.0,
                           LANDING_SPAWN["Top"]["x"], LANDING_SPAWN["Top"]["y"])
    elif door_type == "Bottom":
        # Masuk lewat bawah (35-930, 465-500) -> Muncul di sisi bawah room target
        return DoorTrigger("Bottom", target_cell, 35.0, 930.0, 465.0, 500.0,
                           LANDING_SPAWN["Bottom"]["x"], LANDING_SPAWN["Bottom"]["y"])
    raise ValueError(f"Unknown door_type: {door_type}")

# Pemetaan Pintu per Ruangan
ROOM_DOORS: Dict[str, List[DoorTrigger]] = {
    "Enter0": [
        _create_door("Right", "Morale0C")
    ],
    "Morale0C": [
        _create_door("Left", "Enter0"),
        _create_door("Top", "Captain0"),
        _create_door("Right", "Morale0B")
    ],
    "Captain0": [
        _create_door("Bottom", "Morale0C")
    ],
    "Morale0B": [
        _create_door("Left", "Morale0C"),
        _create_door("Right", "Morale0A")
    ],
    "Morale0A": [
        _create_door("Left", "Morale0B"),
        _create_door("Right", "Crosslower")
    ],
    "Crosslower": [
        _create_door("Left", "Morale0A"),
        _create_door("Top", "Crossupper"),
        _create_door("Right", "Morale1A")
    ],
    "Crossupper": [
        _create_door("Left", "Resource0A"),
        _create_door("Bottom", "Crosslower"),
        _create_door("Right", "Resource1A")
    ],
    "Resource0A": [
        _create_door("Left", "Resource0B"),
        _create_door("Right", "Crossupper")
    ],
    "Resource0B": [
        _create_door("Right", "Resource0A")
    ],
    "Resource1A": [
        _create_door("Left", "Crossupper"),
        _create_door("Right", "Resource1B")
    ],
    "Resource1B": [
        _create_door("Left", "Resource1A")
    ],
    "Morale1A": [
        _create_door("Left", "Crosslower"),
        _create_door("Right", "Morale1B")
    ],
    "Morale1B": [
        _create_door("Left", "Morale1A"),
        _create_door("Right", "Morale1C")
    ],
    "Morale1C": [
        _create_door("Left", "Morale1B"),
        _create_door("Top", "Captain1"),
        _create_door("Right", "Enter1")
    ],
    "Captain1": [
        _create_door("Bottom", "Morale1C")
    ],
    "Enter1": [
        _create_door("Left", "Morale1C")
    ]
}

# Definisi NPC Resmi Peta (monmap)
@dataclass
class NpcDefinition:
    mon_map_id: int
    mon_id: int
    name: str
    cell: str
    team: int
    max_hp: float
    dps: float

BLUDRUT_NPCS: List[NpcDefinition] = [
    # Team 0 (Blue)
    NpcDefinition(1,  308, "Team A Captain", "Captain0", 0, 15000.0, 54.0),
    NpcDefinition(2,  310, "(A) Brawler", "Morale0C", 0, 10000.0, 214.0),
    NpcDefinition(3,  310, "(A) Brawler", "Morale0B", 0, 10000.0, 214.0),
    NpcDefinition(4,  310, "(A) Brawler", "Morale0A", 0, 10000.0, 214.0),
    NpcDefinition(5,  312, "(A) Defensive Restorer", "Resource0B", 0, 5000.0, 107.0),
    NpcDefinition(6,  312, "(A) Defensive Restorer", "Resource0B", 0, 5000.0, 107.0),
    NpcDefinition(7,  312, "(A) Defensive Restorer", "Resource0A", 0, 5000.0, 107.0),
    NpcDefinition(8,  312, "(A) Defensive Restorer", "Resource0A", 0, 5000.0, 107.0),

    # Team 1 (Red)
    NpcDefinition(9,  313, "(B) Defensive Restorer", "Resource1A", 1, 5000.0, 107.0),
    NpcDefinition(10, 313, "(B) Defensive Restorer", "Resource1A", 1, 5000.0, 107.0),
    NpcDefinition(11, 313, "(B) Defensive Restorer", "Resource1B", 1, 5000.0, 107.0),
    NpcDefinition(12, 313, "(B) Defensive Restorer", "Resource1B", 1, 5000.0, 107.0),
    NpcDefinition(13, 311, "(B) Brawler", "Morale1A", 1, 10000.0, 214.0),
    NpcDefinition(14, 311, "(B) Brawler", "Morale1B", 1, 10000.0, 214.0),
    NpcDefinition(15, 311, "(B) Brawler", "Morale1C", 1, 10000.0, 214.0),