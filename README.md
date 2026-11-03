# AQW Classic (Non-OP) PvP AI

A rule-compliant, multi-agent AI PvP bot system for **AdventureQuest Worlds (AQW) Classic / Non-OP PvP** (`1v1` to `3v3`/`4v4` in `/bludrutbrawl`), powered by [**Turu**](https://github.com/muktiarafi/turu)'s headless Ruffle AVM2 engine.

---

## Key Features

- **100% Non-OP Ruleset Enforcement (`RuleShield` & `PreFlight`):**
  - **Pre-Flight Equipment Audit:** Verifies legal Standard/Exceptional Non-OP classes, enforces *Platinum (+5000)* or *Diamond (+5500) PvP Amulet*, rejects *Awe/Forge* enhancements (`EnhPatternID 23..32`), and disables `infiniteRange` for static weapon range compliance.
  - **Protected NPC Mana-Tap Guard:** Allows basic-attacking `/bludrutbrawl` NPCs (*Brawlers/Restorers*) when low on Mana **only** while NPC HP remains above a safe threshold (`>= 2000 HP` and `>= 35%`), automatically canceling attacks and blocking offensive skills so an NPC is never killed.
  - **Mid-Wave Safezone Lock:** Prevents retreating to the Safezone (`Enter0`/`Enter1`) mid-wave unless the format's wave condition is satisfied (`1v1`: opponent killed; `2v2+`: a teammate or opponent died).
- **Zero-GPU Headless Execution (`--headless`):**
  - Runs directly on Linux/macOS/Windows using Turu's `NoOp` Ruffle renderer (`0% GPU`, minimal RAM), enabling multi-account **1v1 and 3v3 Self-Play** on a single machine.
- **Pilot 1v1 Micro-Controller (`Classic Warrior`) -> 3v3 Team Architecture:**
  - Modular state extraction (`100ms` tick) reading real-time HP, Mana, coordinates, skill cooldowns, and active buff/debuff `AuraSnapshot` stacks for all allies and opponents in the room.

---

## Repository Structure

```text
.
├── docs/
│   ├── ARCHITECTURE.md        # System architecture, 3v3 Blackboard, and RL pipeline
│   ├── RULES_NON_OP.md        # Official Classic (Non-OP) ruleset & code enforcement spec
│   └── ROADMAP.md             # 5-Phase development roadmap (1v1 -> 3v3 MAPPO)
├── src/
│   ├── sdk/
│   │   └── turu.ts            # Turu TypeScript SDK & AVM2 bridge definitions
│   ├── config/
│   │   ├── constants.ts       # Allowed classes, amulets, enhancements, BludrutBrawl cells
│   │   └── match_config.ts    # Format configs (1v1: 10 kills, 2v2: 20, 3v3: 30, 4v4: 40)
│   ├── rules/
│   │   ├── preflight.ts       # Pre-match class, enhancement, boost & amulet validator
│   │   └── shield.ts          # Runtime Hard Action Masking (NPC guard & Safezone lock)
│   ├── state/
│   │   ├── types.ts           # Observation space, Action space & WaveState interfaces
│   │   └── extractor.ts       # Real-time AVM2 state & aura snapshot extractor
│   ├── classes/
│   │   └── classic_warrior.ts # Pilot 1v1 combat controller for Classic Warrior
│   └── main_1v1.ts            # 1v1 BludrutBrawl entry-point script for Turu
├── scripts/
│   ├── run-turu.sh            # Launch Turu with GUI (native Linux)
│   ├── run-turu-headless.sh   # Launch Turu in Headless mode + Web Monitor + Control API
│   ├── run-undead.sh          # Launch Undead client via Bottles
│   └── run-skua.sh            # Launch Skua client via Bottles
└── sim/
    └── README.md              # Phase 3-4 Python Gymnasium fast offline simulator spec
```

---

## Quick Start

### 1. Prerequisites
- **Linux (x86_64):** Place the extracted `turu` binary inside `./turu/turu` (git-ignored).
- **Node.js 18+:** Used only for static TypeScript validation (`tsc --noEmit`); Turu executes `.ts` scripts directly at runtime without a build step.

### 2. Type-Check All Scripts
```bash
npm run typecheck
```

### 3. Run 1v1 Pilot Bot (`Classic Warrior`) in Headless Mode
```bash
cp .env.example .env
# Edit .env or pass credentials via environment variables:
TURU_PASSWORD="your_password" ./scripts/run-turu-headless.sh \
  --user "YourUsername" \
  --server "Artix" \
  --run-script "/home/saf/saf/projects/AQW/src/main_1v1.ts"
```
- **Live Web Monitor:** Open `http://127.0.0.1:8790` in your browser to watch logs, state, and bot status.
- **Control HTTP API:** Available at `http://127.0.0.1:8791` (authenticated via `TURU_API_TOKEN`).

---

## Documentation

- [**System Architecture (`docs/ARCHITECTURE.md`)**](docs/ARCHITECTURE.md)
- [**Classic (Non-OP) Rules Specification (`docs/RULES_NON_OP.md`)**](docs/RULES_NON_OP.md)
- [**Development Roadmap (`docs/ROADMAP.md`)**](docs/ROADMAP.md)
