#!/usr/bin/env bash
# Contoh menjalankan Turu dalam mode Headless (0% GPU, tanpa window)
# dilengkapi Web Monitor (http://127.0.0.1:8790) dan Control HTTP API (http://127.0.0.1:8791)
set -euo pipefail

export RUST_LOG="${RUST_LOG:-warn,turu=info}"
export TURU_API_TOKEN="${TURU_API_TOKEN:-aqw_pvp_secret_token_12345}"

cd "/home/saf/saf/projects/AQW/turu"
exec ./turu \
  --headless \
  --monitor 127.0.0.1:8790 \
  --api 127.0.0.1:8791 \
  "$@"
