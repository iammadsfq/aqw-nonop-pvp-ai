#!/usr/bin/env bash
set -euo pipefail
exec flatpak run --command=bottles-cli com.usebottles.bottles run -b Skua__156 -e "C:\Program Files\Skua\Skua.exe" "$@"
