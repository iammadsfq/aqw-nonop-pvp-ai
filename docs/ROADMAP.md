# Development Roadmap

## Phase 1: 1v1 Foundation & Rule Shield (Current Milestone)
- [x] Extract full Turu TypeScript SDK (`src/sdk/turu.ts`) and configure strict static type-checking (`npm run typecheck`).
- [x] Implement **Non-OP Pre-Flight Validator** (`src/rules/preflight.ts`) for classes, amulets (`+5000`/`+5500`), standard enhancements, and static weapon range.
- [x] Implement **Runtime Rule Shield** (`src/rules/shield.ts`) for Safe NPC Mana-Tapping and Mid-Wave Safezone Locking.
- [x] Implement **1v1 Pilot Controller for Classic Warrior** (`src/classes/classic_warrior.ts` & `src/main_1v1.ts`) in `/bludrutbrawl` (10-kill target).
- [ ] Live 1v1 validation & telemetry logging in `/bludrutbrawl` using 2 `turu --headless` instances.

## Phase 2: Complete Non-OP Class Roster & 3v3 Blackboard
- [ ] Add heuristic micro-controllers for remaining core 3v3 classes:
  - [ ] **Battle Healer** (Triage healing, HoT uptime, kiting, safe NPC mana tap)
  - [ ] **Battle Bard** (Party buff uptime, stun/debuff peeling)
  - [ ] **Classic Ninja** (Shadowblade blind timing, stun lock, DoT management)
  - [ ] **Classic Dragonlord**, **Classic Guardian**, and **Classic Defender**
- [ ] Implement **3v3 Team Coordinator (Shared Blackboard)** so all 3 headless instances synchronize focus targets and wave death state in real time.

## Phase 3: Fast Offline Gymnasium Simulator (`sim/`)
- [ ] Record combat telemetry (damage distributions, crit/dodge rates, aura durations, mana regen curves) from headless 1v1 and 3v3 matches.
- [ ] Build `AQWNonOp1v1Env` and `AQWNonOp3v3Env` in Python (`Gymnasium`) capable of simulating `>1,000` full matches per minute with identical Non-OP action masks.

## Phase 4: Multi-Agent Reinforcement Learning (MAPPO + Self-Play)
- [ ] Train 1v1 duel policies and 3v3 team policies (**Battle Healer + Battle Bard + Non-OP DPS**) using **Action-Masked MAPPO**.
- [ ] Run League Self-Play across all 5 Non-OP DPS classes so the trained team adapts dynamically to any opponent composition.
- [ ] Export trained policy networks to lightweight **ONNX / JSON weights**.

## Phase 5: Live 3v3 Headless Deployment
- [ ] Connect trained model inference to 3 concurrent `turu --headless` instances (`30-kill` target in `/bludrutbrawl`).
- [ ] Add automated match reporting and win-rate analytics via Turu's Web Monitor and Control API.
