# Offline Combat Simulator (`sim/`)

This directory is reserved for **Phase 3 & Phase 4** of the roadmap:
1. **Combat Telemetry Parser:** Converts packet and aura logs recorded by `turu --headless` into calibrated class damage/cooldown/mana parameters.
2. **Gymnasium Environments (`AQWNonOp1v1Env` & `AQWNonOp3v3Env`):** High-speed Python simulation of `/bludrutbrawl` Non-OP combat with identical action masks (`RuleShield`) for training PPO / MAPPO agents via Self-Play.
3. **Model Export:** Exports trained policy weights for real-time inference inside `src/`.
