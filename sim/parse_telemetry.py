#!/usr/bin/env python3
"""
Parses raw AQW server telemetry recorded by `src/telemetry/combat_recorder.ts`
(`~/.local/share/Turu/army/aqw_combat_telemetry.log`) and compiles a 1:1
calibrated class & combat specification database (`sim/data/aqw_classes_spec.json`)
for the Python Reinforcement Learning simulator.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_TELEMETRY_PATH = Path.home() / ".local/share/Turu/army/aqw_combat_telemetry.log"
OUTPUT_SPEC_PATH = Path(__file__).resolve().parent / "data" / "aqw_classes_spec.json"


def parse_telemetry_file(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        raise FileNotFoundError(
            f"Telemetry log not found at {log_path}. "
            "Run `src/telemetry/combat_recorder.ts` in Turu first!"
        )

    classes: dict[str, Any] = {}
    auras_catalog: dict[str, dict[str, Any]] = {}
    hit_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with log_path.open("r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            rec_type = entry.get("type")
            class_name = entry.get("className") or "Unknown"
            payload = entry.get("payload") or {}

            if rec_type == "class_spec":
                classes[class_name] = {
                    "className": class_name,
                    "level": payload.get("level"),
                    "maxHealth": payload.get("maxHealth"),
                    "maxMana": payload.get("maxMana"),
                    "walkSpeed": payload.get("walkSpeed"),
                    "wDPS": payload.get("wDPS"),
                    "computedStats": payload.get("computedStats"),
                    "rawSta": payload.get("rawSta"),
                    "equippedItems": payload.get("equippedItems"),
                    "skills": payload.get("skills"),
                }

            elif rec_type == "combat_trace":
                # Extract exact aura specifications from `ct.a`
                for aura_event in payload.get("a") or []:
                    if not isinstance(aura_event, dict):
                        continue
                    aura_list = aura_event.get("auras") or [aura_event]
                    for aura in aura_list:
                        if not isinstance(aura, dict):
                            continue
                        name = aura.get("nam")
                        if name:
                            auras_catalog[name] = {
                                "name": name,
                                "duration": aura.get("dur"),
                                "category": aura.get("cat"),
                                "type": aura.get("t"),
                                "value": aura.get("val"),
                            }

                # Extract exact hit/crit/dodge damage samples from `ct.sarsa`
                for sarsa_entry in payload.get("sarsa") or []:
                    if not isinstance(sarsa_entry, dict):
                        continue
                    for action_res in sarsa_entry.get("a") or []:
                        if not isinstance(action_res, dict):
                            continue
                        hit_samples[class_name].append(
                            {
                                "hp_damage": action_res.get("hp"),
                                "hit_type": action_res.get("type"),
                                "action_ref": action_res.get("strl"),
                            }
                        )

    return {
        "classes": classes,
        "aurasCatalog": auras_catalog,
        "combatSamplesCount": {k: len(v) for k, v in hit_samples.items()},
        "combatSamplesPreview": {k: v[:50] for k, v in hit_samples.items()},
    }


def main() -> None:
    turu_data_dir = os.environ.get("TURU_DATA_DIR")
    log_path = (
        Path(turu_data_dir) / "army/aqw_combat_telemetry.log"
        if turu_data_dir
        else DEFAULT_TELEMETRY_PATH
    )
    compiled = parse_telemetry_file(log_path)
    OUTPUT_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_SPEC_PATH.open("w", encoding="utf-8") as out_f:
        json.dump(compiled, out_f, indent=2)
    print(f"[OK] Saved calibrated AQW specification to {OUTPUT_SPEC_PATH}")
    print(f"     Classes recorded: {list(compiled['classes'].keys())}")
    print(f"     Unique auras recorded: {len(compiled['aurasCatalog'])}")


if __name__ == "__main__":
    main()
