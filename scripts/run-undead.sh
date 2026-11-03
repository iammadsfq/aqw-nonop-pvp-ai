#!/usr/bin/env bash
set -euo pipefail
exec flatpak run --command=bottles-cli com.usebottles.bottles run -b Skua__156 -e "C:\undead\undead-v2.4.0-win.exe" "$@"
