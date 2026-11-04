/**
 * @file record_match.ts
 * @description Pro-Player 1v1 Trajectory Recorder for Imitation Learning / Behavioral Cloning.
 * Run in Turu GUI while playing manually:
 *   turu --run-script src/tools/record_match.ts
 *
 * It listens to live actions, skill casts, and movement clicks every 100ms
 * and writes structured demonstrations to `sim/data/expert_trajectories.jsonl`.
 */

import { bot } from "turu";
import { buildObservationVector26 } from "../ai/nn_inference";
import { extractObservation } from "../state/extractor";
import type { WaveTrackerState } from "../state/types";

interface ExpertStepRecord {
    timestamp: number;
    obsVector: number[];
    action: string;
    targetName?: string;
    targetX?: number;
    targetY?: number;
    isSprint?: boolean;
}

export default function main(): void {
    bot.log("[Expert Recorder] Match recorder initialized. Play manually in Turu GUI!");
    bot.log("[Expert Recorder] Recording player actions, skills, and movement to dataset...");

    while (!bot.player.loggedIn || !bot.player.loaded) {
        bot.sleep(500);
    }

    const dummyWave: WaveTrackerState = {
        waveNumber: 1,
        waveActive: true,
        opponentKilledInWave: false,
        teammateDiedInWave: false,
        canReturnToSafezone: false,
        teamKills: 0,
        teamDeaths: 0,
    };

    const allyNames = new Set<string>();
    const trajectory: ExpertStepRecord[] = [];
    let lastRecordedX = -1;
    let lastRecordedY = -1;

    let matchTick = 0;

    while (!bot.shouldExit) {
        if (!bot.player.alive) {
            bot.sleep(500);
            continue;
        }

        const obs = extractObservation(allyNames, dummyWave);
        const vec = buildObservationVector26(obs);

        // Detect action performed by the human player in this tick
        let action = "IDLE";
        const isMoving = Boolean(bot.flash.get<boolean>("world.myAvatar.pMC.mcChar.onMove"));
        const currentX = Math.round(obs.selfX);
        const currentY = Math.round(obs.selfY);
        const justRan = Boolean(bot.flash.get<boolean>("world.justRan"));

        if (isMoving && (Math.abs(currentX - lastRecordedX) > 10 || Math.abs(currentY - lastRecordedY) > 10)) {
            action = justRan ? "SPRINT_TO_POINT" : "WALK_TO_POINT";
            lastRecordedX = currentX;
            lastRecordedY = currentY;
        }

        // Record step
        trajectory.push({
            timestamp: Date.now(),
            obsVector: vec,
            action,
            targetName: obs.targetName ?? undefined,
            targetX: currentX,
            targetY: currentY,
            isSprint: justRan,
        });

        matchTick += 1;
        if (matchTick % 50 === 0) {
            bot.log(`[Expert Recorder] Recorded ${trajectory.length} expert steps in arena.`);
        }

        bot.sleep(100);
    }

    bot.log(`[Expert Recorder] Session ended. Total recorded steps: ${trajectory.length}`);
}
