#!/usr/bin/env bash
# ==============================================================================
# scripts/train-and-deploy.sh
# Pipeline otomatis untuk merakit bundle simulasi terbaru, menjalankan training
# PPO (di Kaggle GPU atau Lokal), mengunduh hasil training, dan memasang bobot
# langsung ke src/ai/weights.ts.
#
# Contoh Penggunaan:
#   ./scripts/train-and-deploy.sh --matches 500 --kills 10
#   ./scripts/train-and-deploy.sh --matches 500 --kills 10 --max-time 1200 --out-name run_500m
#   ./scripts/train-and-deploy.sh --matches 500 --kills 10 --local
#
# Daftar Parameter yang Diterima:
#   --matches <int>        : Target total match selesai untuk menghentikan training
#                            (default: 500; set 0 jika ingin patokan --updates saja)
#   --kills <int>          : Target kill untuk memenangkan 1 match / first-to-N kills
#                            (default: 10)
#   --max-time <float>     : Batas waktu maksimum per match dalam detik sebelum
#                            dinyatakan selesai/tiebreaker (default: 1200.0 = 20 menit)
#   --updates <int>        : Batas maksimum iterasi PPO update sebagai safety cap
#                            (default: 15000)
#   --envs <int>           : Jumlah arena environment paralel (default: 16; minimal 4).
#                            Otomatis dibagi 25% Blue vs Greedy, 25% Red vs Greedy,
#                            dan 50% Self-Play (Blue & Red).
#   --rollout-steps <int>  : Jumlah step per rollout sebelum PPO update (default: 128)
#   --chunk-len <int>      : Panjang sekuens BPTT GRU per chunk (default: 16)
#   --lr <float>           : Learning rate Adam optimizer (default: 3e-4)
#   --gamma <float>        : Discount factor GAE/PPO (default: 0.99)
#   --ent-coef <float>     : Koefisien entropy bonus eksplorasi (default: 0.01)
#   --seed <int>           : Random seed (default: 0)
#   --out-name <string>    : Nama subfolder di sim/weights/<out-name> untuk menyimpan
#                            artefak model (default: out_v1)
#   --local                : Jalankan training secara lokal (CPU/GPU lokal) tanpa
#                            mengunggah ke Kaggle
#   -h, --help             : Tampilkan panduan parameter ini
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

KERNEL_SLUG="iammadsfq/aqw-pvp-training"
STAGE_DIR="sim/.kaggle_stage"

# Default Hyperparameters & Konfigurasi Match
MATCHES=500
KILLS=10
MAX_TIME=1200.0
UPDATES=15000
ENVS=16
ROLLOUT_STEPS=128
CHUNK_LEN=16
LR="3e-4"
GAMMA="0.99"
ENT_COEF="0.01"
SEED=0
OUTPUT_NAME="out_v1"
RUN_LOCAL=0

usage() {
  sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matches)
      MATCHES="$2"; shift 2 ;;
    --kills)
      KILLS="$2"; shift 2 ;;
    --max-time)
      MAX_TIME="$2"; shift 2 ;;
    --updates)
      UPDATES="$2"; shift 2 ;;
    --envs)
      ENVS="$2"; shift 2 ;;
    --rollout-steps)
      ROLLOUT_STEPS="$2"; shift 2 ;;
    --chunk-len)
      CHUNK_LEN="$2"; shift 2 ;;
    --lr)
      LR="$2"; shift 2 ;;
    --gamma)
      GAMMA="$2"; shift 2 ;;
    --ent-coef)
      ENT_COEF="$2"; shift 2 ;;
    --seed)
      SEED="$2"; shift 2 ;;
    --out-name)
      OUTPUT_NAME="$2"; shift 2 ;;
    --local)
      RUN_LOCAL=1; shift 1 ;;
    -h|--help)
      usage ;;
    -*)
      echo "❌ Parameter tidak dikenal: $1"
      usage ;;
    *)
      # Kompatibilitas argumen posisional pertama sebagai OUTPUT_NAME
      OUTPUT_NAME="$1"; shift 1 ;;
  esac
done

if [ -x ".venv/bin/python3" ]; then
  PYTHON_BIN=".venv/bin/python3"
else
  PYTHON_BIN="$(command -v python3)"
fi

OUTPUT_DIR="sim/weights/$OUTPUT_NAME"
mkdir -p "$OUTPUT_DIR"

echo "======================================================"
echo "⚙️  Konfigurasi Training Bludrut Brawl PPO:"
echo "   - Target Matches : $MATCHES matches"
echo "   - Target Kills   : First to $KILLS kills / match"
echo "   - Max Match Time : ${MAX_TIME}s"
echo "   - Safety Updates : $UPDATES updates"
echo "   - Parallel Envs  : $ENVS envs"
echo "   - Rollout / Chunk: $ROLLOUT_STEPS / $CHUNK_LEN"
echo "   - LR / Gamma / Ent: $LR / $GAMMA / $ENT_COEF"
echo "   - Seed           : $SEED"
echo "   - Output Folder  : $OUTPUT_DIR"
echo "   - Execution Mode : $([ "$RUN_LOCAL" -eq 1 ] && echo "LOCAL" || echo "KAGGLE GPU OFFLOAD")"
echo "======================================================"

if [ "$RUN_LOCAL" -eq 1 ]; then
  echo ""
  echo "🚀 [1/2] Menjalankan training secara LOKAL..."
  "$PYTHON_BIN" sim/train_kaggle_bundle.py \
    --matches "$MATCHES" \
    --kills "$KILLS" \
    --max-time "$MAX_TIME" \
    --updates "$UPDATES" \
    --envs "$ENVS" \
    --rollout-steps "$ROLLOUT_STEPS" \
    --chunk-len "$CHUNK_LEN" \
    --lr "$LR" \
    --gamma "$GAMMA" \
    --ent-coef "$ENT_COEF" \
    --seed "$SEED"
  cp -f sim/weights/weights.ts "$OUTPUT_DIR/weights.ts"
  cp -f sim/weights/bludrut_ppo_warrior.pt "$OUTPUT_DIR/bludrut_ppo_warrior.pt"
else
  if [ -x ".venv/bin/kaggle" ]; then
    KAGGLE_BIN=".venv/bin/kaggle"
  elif command -v kaggle >/dev/null 2>&1; then
    KAGGLE_BIN="$(command -v kaggle)"
  else
    echo "❌ Error: CLI kaggle tidak ditemukan!"
    exit 1
  fi

  echo ""
  echo "📦 [1/4] Merakit single-file bundle & mengunggah ke Kaggle GPU..."
  rm -rf "$STAGE_DIR"
  mkdir -p "$STAGE_DIR"
  cp "sim/kernel-metadata.json" "$STAGE_DIR/kernel-metadata.json"

  "$PYTHON_BIN" - \
    "$STAGE_DIR/train_kaggle_bundle.py" \
    "$MATCHES" "$KILLS" "$MAX_TIME" "$UPDATES" "$ENVS" \
    "$ROLLOUT_STEPS" "$CHUNK_LEN" "$LR" "$GAMMA" "$ENT_COEF" "$SEED" << 'PYEOF'
import pathlib
import sys

(
    out_file,
    matches,
    kills,
    max_time,
    updates,
    envs,
    rollout,
    chunk_len,
    lr,
    gamma,
    ent_coef,
    seed,
) = sys.argv[1:]

def strip_future(text: str) -> str:
    return "\n".join(
        line for line in text.splitlines()
        if not line.strip().startswith("from __future__ import annotations")
    )

map_src = strip_future(pathlib.Path("sim/world/bludrut_map.py").read_text())
mrm_src = strip_future(pathlib.Path("sim/combat/mrm.py").read_text())

env_lines = pathlib.Path("sim/bludrut_env.py").read_text().splitlines()
# Ambil mulai dari bagian "# 1. CONSTANTS, BOUNDS & SIMULATION HYPERPARAMETERS"
env_start = next(i for i, l in enumerate(env_lines) if "1. CONSTANTS, BOUNDS" in l) - 1
env_src = "\n".join(env_lines[env_start:])

train_lines = pathlib.Path("sim/train_kaggle_bundle.py").read_text().splitlines()
# Ambil mulai dari bagian "# 1. NEURAL NETWORK ARCHITECTURE"
train_start = next(i for i, l in enumerate(train_lines) if "1. NEURAL NETWORK ARCHITECTURE" in l) - 1
train_src = "\n".join(train_lines[train_start:])

bundle = f'''#!/usr/bin/env python3
"""
AUTO-GENERATED SELF-CONTAINED KAGGLE BUNDLE
Dibuat otomatis oleh scripts/train-and-deploy.sh dari:
  - sim/world/bludrut_map.py
  - sim/combat/mrm.py
  - sim/bludrut_env.py
  - sim/train_kaggle_bundle.py
"""
from __future__ import annotations

import math
import os
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Injected CLI Parameters dari scripts/train-and-deploy.sh
os.environ.setdefault("BB_MATCHES", "{matches}")
os.environ.setdefault("BB_TARGET_KILLS", "{kills}")
os.environ.setdefault("BB_MAX_MATCH_TIME", "{max_time}")
os.environ.setdefault("BB_UPDATES", "{updates}")
os.environ.setdefault("BB_NUM_ENVS", "{envs}")
os.environ.setdefault("BB_ROLLOUT", "{rollout}")
os.environ.setdefault("BB_CHUNK_LEN", "{chunk_len}")
os.environ.setdefault("BB_LR", "{lr}")
os.environ.setdefault("BB_GAMMA", "{gamma}")
os.environ.setdefault("BB_ENT_COEF", "{ent_coef}")
os.environ.setdefault("BB_SEED", "{seed}")

CUR_DIR = Path(__file__).resolve().parent

# ==============================================================================
# MODULE 1: sim/world/bludrut_map.py
# ==============================================================================
{map_src}

# ==============================================================================
# MODULE 2: sim/combat/mrm.py
# ==============================================================================
{mrm_src}

# ==============================================================================
# MODULE 3: sim/bludrut_env.py
# ==============================================================================
{env_src}

# ==============================================================================
# MODULE 4: sim/train_kaggle_bundle.py
# ==============================================================================
{train_src}
'''

pathlib.Path(out_file).write_text(bundle)
print(f"[✓] Self-contained Kaggle bundle generated: {out_file} ({len(bundle.splitlines())} lines)")
PYEOF

  "$KAGGLE_BIN" kernels push -p "$STAGE_DIR"

  echo ""
  echo "======================================================"
  echo "⏳ [2/4] Menunggu training PPO selesai di Kaggle..."
  echo "======================================================"
  sleep 10
  while true; do
    STATUS_OUTPUT=$("$KAGGLE_BIN" kernels status "$KERNEL_SLUG" 2>&1 || true)
    echo "[$(date '+%H:%M:%S')] $STATUS_OUTPUT"

    if echo "$STATUS_OUTPUT" | grep -qi "complete"; then
      echo "✅ Training selesai!"
      break
    elif echo "$STATUS_OUTPUT" | grep -qi "error"; then
      echo "❌ Training GAGAL di Kaggle! Mengambil log error..."
      "$KAGGLE_BIN" kernels output "$KERNEL_SLUG" -p "$OUTPUT_DIR" || true
      rm -rf "$STAGE_DIR"
      exit 1
    fi
    sleep 25
  done

  echo ""
  echo "======================================================"
  echo "📥 [3/4] Mengunduh output artifact ke $OUTPUT_DIR..."
  echo "======================================================"
  "$KAGGLE_BIN" kernels output "$KERNEL_SLUG" -p "$OUTPUT_DIR"
  rm -rf "$STAGE_DIR"
fi

echo ""
echo "======================================================"
echo "🔄 [4/4] Memasang weights.ts ke runtime bot..."
echo "======================================================"
mkdir -p "src/ai"
mkdir -p "sim/data"

if [ -f "$OUTPUT_DIR/weights.ts" ]; then
  cp -v "$OUTPUT_DIR/weights.ts" "src/ai/weights.ts"
elif [ -f "$OUTPUT_DIR/weights/weights.ts" ]; then
  cp -v "$OUTPUT_DIR/weights/weights.ts" "src/ai/weights.ts"
else
  echo "⚠️ File weights.ts tidak ditemukan di output!"
fi

if [ -f "$OUTPUT_DIR/bludrut_ppo_warrior.pt" ]; then
  cp -v "$OUTPUT_DIR/bludrut_ppo_warrior.pt" "sim/data/bludrut_ppo_warrior.pt"
elif [ -f "$OUTPUT_DIR/weights/bludrut_ppo_warrior.pt" ]; then
  cp -v "$OUTPUT_DIR/weights/bludrut_ppo_warrior.pt" "sim/data/bludrut_ppo_warrior.pt"
fi

if [ -f "src/ai/weights.ts" ]; then
  FILE_SIZE=$(ls -lh "src/ai/weights.ts" | awk '{print $5}')
  echo ""
  echo "✨ SUKSES! Model terpasang ($FILE_SIZE): src/ai/weights.ts"
  echo "🎮 Siap dijalankan tanpa dependensi eksternal."
else
  echo "❌ Gagal: src/ai/weights.ts tidak tersedia."
  exit 1
fi
