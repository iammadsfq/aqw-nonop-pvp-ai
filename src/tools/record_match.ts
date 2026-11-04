/**
 * @file record_match.ts
 * @description Pro-Player 1v1 Trajectory Recorder for Imitation Learning / Behavioral Cloning.
 * Run in Turu GUI while playing manually:
 *   turu --run-script src/tools/record_match.ts
 *
 * Listens to live actions, skill casts, and movement clicks every 100ms
 * and writes structured demonstrations to disk:
 *   /home/saf/.local/share/Turu/army/expert_trajectories.log
 *
 * This dataset is ingested by `sim/train_imitation.py` to clone pro-player heuristics!
 */

import { bot } from "turu";
import { buildObservationVector26 } from "../ai/nn_inference";
import { extractObservation } from "../state/extractor";
import type { WaveTrackerState } from "../state/types";

export interface ExpertStepRecord {
    timestamp: number;
    cell: string;
    obsVector: number[];
    actionIndex: number;
    actionName: string;
    combatAction: number;
    moveType: number;
    selfX: number;
    selfY: number;
    targetX: number;
    targetY: number;
    targetName?: string;
    hasTarget?: boolean;
    isSprint: boolean;
}

export default function main(): void {
    bot.log("[Expert Recorder] Match recorder initialized. Play manually in Turu GUI!");
    bot.log("[Expert Recorder] Data will be streamed live to army/expert_trajectories.log...");

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
    let lastSkillsReady = [true, true, true, true];
    let lastRecordedX = bot.player.x;
    let lastRecordedY = bot.player.y;
    let lastCell = bot.player.cell;
    let totalRecorded = 0;

    bot.log("[Expert Recorder] Active! Start dueling. Every move is being recorded for AI cloning.");

    while (!bot.shouldExit) {
        if (!bot.player.alive) {
            bot.sleep(500);
            continue;
        }

        const obs = extractObservation(allyNames, dummyWave);
        const vec = buildObservationVector26(obs);

        const currentX = Math.round(obs.selfX);
        const currentY = Math.round(obs.selfY);
        const isMoving = Boolean(bot.flash.get<boolean>("world.myAvatar.pMC.mcChar.onMove"));
        const justRan = Boolean(bot.flash.get<boolean>("world.justRan"));

        const oppInCell = obs.opponents.find((p) => p.cell === obs.currentCell && p.alive);
        const dist = oppInCell ? Math.hypot(currentX - oppInCell.x, currentY - oppInCell.y) : 999;
        const prevDist = oppInCell ? Math.hypot(lastRecordedX - oppInCell.x, lastRecordedY - oppInCell.y) : 999;

        let actionIdx = 0;
        let actionName = "FIGHT_STAND";

        // Check if skills were just fired (went from ready -> on cooldown)
        if (lastSkillsReady[0] && !obs.skillsReady[0]) {
            actionIdx = 3;
            actionName = "USE_SKILL_1";
        } else if (lastSkillsReady[1] && !obs.skillsReady[1]) {
            actionIdx = 4;
            actionName = "USE_SKILL_2";
        } else if (lastSkillsReady[2] && !obs.skillsReady[2]) {
            actionIdx = 5;
            actionName = "USE_SKILL_3";
        } else if (lastSkillsReady[3] && !obs.skillsReady[3]) {
            actionIdx = 6;
            actionName = "USE_SKILL_4";
        } else if (obs.hasTarget && obs.targetIsNpc && obs.selfInCombat) {
            actionIdx = 10;
            actionName = "MANA_TAP_NPC";
        } else if (obs.currentCell.toLowerCase() !== lastCell.toLowerCase()) {
            // Room transition just happened
            actionIdx = justRan ? 13 : 11;
            actionName = justRan ? "SPRINT_DOOR_BACKWARD" : "WALK_DOOR_FORWARD";
            lastCell = obs.currentCell;
        } else if (isMoving && (Math.abs(currentX - lastRecordedX) > 6 || Math.abs(currentY - lastRecordedY) > 6)) {
            const dx = Math.abs(currentX - lastRecordedX);
            const dy = Math.abs(currentY - lastRecordedY);

            if (dy > dx * 1.5) {
                actionIdx = 9;
                actionName = "JUKE_VERTICAL";
            } else if (oppInCell && dist < prevDist - 5) {
                actionIdx = justRan ? 2 : 1;
                actionName = justRan ? "SPRINT_TO_OPPONENT" : "APPROACH_OPPONENT";
            } else if (oppInCell && dist > prevDist + 5) {
                actionIdx = justRan ? 8 : 7;
                actionName = justRan ? "SPRINT_KITE_AWAY" : "KITE_AWAY";
            } else {
                actionIdx = justRan ? 2 : 1;
                actionName = justRan ? "SPRINT_TO_POINT" : "WALK_TO_POINT";
            }
            lastRecordedX = currentX;
            lastRecordedY = currentY;
        } else {
            actionIdx = 0;
            actionName = "FIGHT_STAND";
        }

        lastSkillsReady = [...obs.skillsReady];

        // Determine atomic decoupled channels
        let combatAct = 0;
        if (actionIdx === 3) combatAct = 2; // S1
        else if (actionIdx === 4) combatAct = 3; // S2
        else if (actionIdx === 5) combatAct = 4; // S3
        else if (actionIdx === 6) combatAct = 5; // S4
        else if (actionIdx === 10) combatAct = 6; // MANA_TAP
        else if (obs.hasTarget && !obs.targetIsNpc) combatAct = 1; // AA on targeted opponent player
        else if (obs.selfInCombat) combatAct = 1; // AA

        let moveType = 0;
        if (justRan || [2, 8, 13].includes(actionIdx)) moveType = 2; // SPRINT
        else if (isMoving || [1, 7, 9, 11, 12].includes(actionIdx)) moveType = 1; // WALK

        const clickTargetX = Math.round(
            bot.flash.get<number>("world.myAvatar.pMC.mcChar.targetX") ?? currentX,
        );
        const clickTargetY = Math.round(
            bot.flash.get<number>("world.myAvatar.pMC.mcChar.targetY") ?? currentY,
        );

        // Format and stream record to disk
        const record: ExpertStepRecord = {
            timestamp: Date.now(),
            cell: obs.currentCell,
            obsVector: vec,
            actionIndex: actionIdx,
            actionName,
            combatAction: combatAct,
            moveType,
            selfX: currentX,
            selfY: currentY,
            targetX: clickTargetX,
            targetY: clickTargetY,
            targetName: obs.targetName ?? undefined,
            hasTarget: obs.hasTarget,
            isSprint: justRan,
        };

        try {
            bot.army.appendLog("expert_trajectories", JSON.stringify(record));
            totalRecorded += 1;
        } catch (e) {
            // Ignore if army log not ready
        }

        if (totalRecorded > 0 && totalRecorded % 50 === 0) {
            bot.log(`[Expert Recorder] Streamed ${totalRecorded} expert steps to army/expert_trajectories.log.`);
        }

        bot.sleep(100);
    }

    bot.log(`[Expert Recorder] Session ended. Total recorded steps: ${totalRecorded}`);
}
