/**
 * @file main_1v1.ts
 * @description Entry-point script for 1v1 Non-OP BludrutBrawl testing in Turu.
 * Run with:
 *   ./scripts/run-turu-headless.sh --user <user> --server <server> --run-script src/main_1v1.ts
 */

import { bot } from "turu";
import { inferBattleWarriorDecision } from "./ai/nn_inference";
import {
    AQW_DEFAULT_WALK_SPEED,
    AQW_SPRINT_STAMINA_COST,
    AQW_SPRINT_WALK_SPEED,
} from "./config/constants";
import { createMatchConfig } from "./config/match_config";
import { runNonOpPreflightCheck } from "./rules/preflight";
import { applyRuleShield, enforceNpcProtectionGuard } from "./rules/shield";
import { extractObservation } from "./state/extractor";
import type { CombatDecision, CombatObservation, WaveTrackerState } from "./state/types";

let lastWalkCell = "";
let lastWalkX = -1;
let lastWalkY = -1;
let lastWalkTime = 0;
let lastSprintTime = 0;
let currentCellTracker = "";
let cellEnteredAt = 0;

function maintainStaminaRegen(now: number): void {
    // Regenerate +25 stamina every 4 seconds when not sprinting (up to 100 max)
    if (now - lastSprintTime >= 4000) {
        const currentSp = bot.flash.get<number>("world.myAvatar.dataLeaf.intSP") ?? 100;
        if (currentSp < 100) {
            const regenedSp = Math.min(100, currentSp + 25);
            bot.flash.set("world.myAvatar.dataLeaf.intSP", regenedSp);
            bot.flash.set("world.myAvatar.objData.intSP", regenedSp);
            bot.flash.call("world.updatePortrait", "world.myAvatar");
            bot.flash.call("world.updateActBar");
            lastSprintTime = now;
        }
    }
}

function triggerSprintWalk(obs: CombatObservation, x: number, y: number, isSprint: boolean): boolean {
    // If stunned, immediately freeze character movement on client to match server state!
    if (obs.selfStunned) {
        bot.flash.call("world.myAvatar.pMC.stopWalking");
        return false;
    }

    const now = Date.now();
    const isMoving = Boolean(bot.flash.get<boolean>("world.myAvatar.pMC.mcChar.onMove"));
    const sameTarget =
        obs.currentCell === lastWalkCell && Math.hypot(x - lastWalkX, y - lastWalkY) <= 25;

    // Throttle walkTo calls so Flash's mvTimer and onEnterFrameWalk are not reset every 100ms!
    if (sameTarget && isMoving && now - lastWalkTime < 650) {
        return false;
    }

    lastWalkCell = obs.currentCell;
    lastWalkX = x;
    lastWalkY = y;
    lastWalkTime = now;

    const currentSp = bot.flash.get<number>("world.myAvatar.dataLeaf.intSP") ?? 100;
    const canReallySprint = isSprint && currentSp >= AQW_SPRINT_STAMINA_COST;

    if (canReallySprint) {
        lastSprintTime = now;
        const newSp = Math.max(0, currentSp - AQW_SPRINT_STAMINA_COST);
        bot.flash.set("world.myAvatar.dataLeaf.intSP", newSp);
        bot.flash.set("world.myAvatar.objData.intSP", newSp);
        bot.flash.set("world.justRan", true);
        bot.flash.set("world.rootClass.pDash", true);
        bot.flash.call("world.updatePortrait", "world.myAvatar");
        bot.flash.call("world.updateActBar");

        bot.player.walkTo(x, y, AQW_SPRINT_WALK_SPEED);
    } else {
        bot.flash.set("world.justRan", false);
        bot.flash.set("world.rootClass.pDash", false);
        bot.player.walkTo(x, y, AQW_DEFAULT_WALK_SPEED);
    }
    return true;
}

function executeDecision(decision: CombatDecision, obs: CombatObservation, combatCell: string, safezoneCell: string): void {
    let shouldLog = decision.action !== "IDLE";

    switch (decision.action) {
        case "IDLE":
            if (decision.targetName && (!obs.hasTarget || obs.targetIsNpc)) {
                bot.combat.attackPlayer(decision.targetName);
            }
            break;
        case "CANCEL_AUTO_ATTACK":
            bot.combat.cancelAutoAttack();
            bot.combat.cancelTarget();
            break;
        case "REST_OUT_OF_COMBAT":
            if (!obs.selfInCombat) {
                bot.player.rest(false);
            }
            break;
        case "TARGET_OPPONENT":
            if (decision.targetName) {
                bot.combat.attackPlayer(decision.targetName);
            }
            break;
        case "MANA_TAP_NPC":
            if (decision.npcMapId !== undefined) {
                bot.combat.attack(decision.npcMapId);
            }
            break;
        case "USE_SKILL_1":
            if (decision.targetName) {
                bot.combat.attackPlayer(decision.targetName);
            }
            bot.skills.useSkill(1);
            break;
        case "USE_SKILL_2":
            if (decision.targetName) {
                bot.combat.attackPlayer(decision.targetName);
            }
            bot.skills.useSkill(2);
            break;
        case "USE_SKILL_3":
            if (decision.targetName) {
                bot.combat.attackPlayer(decision.targetName);
            }
            bot.skills.useSkill(3);
            break;
        case "USE_SKILL_4":
            if (decision.targetName) {
                bot.combat.attackPlayer(decision.targetName);
            }
            bot.skills.useSkill(4);
            break;
        case "WALK_TO_POINT":
            if (decision.targetName && (!obs.hasTarget || obs.targetIsNpc)) {
                bot.combat.attackPlayer(decision.targetName);
            }
            if (typeof decision.targetX === "number" && typeof decision.targetY === "number") {
                shouldLog = triggerSprintWalk(obs, decision.targetX, decision.targetY, false);
            }
            break;
        case "SPRINT_TO_POINT":
            if (decision.targetName && (!obs.hasTarget || obs.targetIsNpc)) {
                bot.combat.attackPlayer(decision.targetName);
            }
            if (typeof decision.targetX === "number" && typeof decision.targetY === "number") {
                shouldLog = triggerSprintWalk(obs, decision.targetX, decision.targetY, obs.canSprint);
            }
            break;
        case "WALK_TO_DOOR":
        case "SPRINT_TO_DOOR":
            if (typeof decision.targetX === "number" && typeof decision.targetY === "number") {
                // Room traversal ALWAYS uses normal walk speed (8 px/s) to ensure 100% server sync!
                // Never sprint into doors. Normal walking ensures zero speed mismatch across room boundaries.
                shouldLog = triggerSprintWalk(obs, decision.targetX, decision.targetY, false);
            }
            break;
        case "RETREAT_TO_SAFEZONE":
            bot.combat.cancelAutoAttack();
            bot.combat.cancelTarget();
            bot.map.jump(safezoneCell, "Spawn");
            break;
        case "ADVANCE_TO_COMBAT_CELL":
            bot.map.jump(combatCell, "Left");
            break;
    }
    if (shouldLog) {
        bot.log(`[1v1 AI] ${decision.action} -> ${decision.reason}`);
    }
}

export default function main(): void {
    const config = createMatchConfig("1v1", {
        mapName: "bludrutbrawl",
        combatCell: "Morale0C",
        safezoneCell: "Enter0",
    });

    bot.log(`[Non-OP PvP AI] Initializing ${config.format} match (Target: ${config.killTarget} kills)...`);

    while (!bot.player.loggedIn || !bot.player.loaded) {
        bot.sleep(500);
    }

    // Automatically join private /bludrutbrawl-99999 room if not already in bludrutbrawl
    if (bot.map.name.toLowerCase() !== "bludrutbrawl") {
        bot.log("[Non-OP PvP AI] Joining /bludrutbrawl-99999 arena...");
        bot.map.join("bludrutbrawl-99999", "Enter0", "Spawn");
        bot.wait.forMapLoad("bludrutbrawl", 30);
        bot.sleep(1000);
    }

    const preflight = runNonOpPreflightCheck(config);
    bot.log(`[PreFlight] Class="${preflight.equippedClass}", Amulet="${preflight.equippedAmulet ?? "None"}"`);
    if (!preflight.passed) {
        for (const v of preflight.violations) {
            bot.log(`[PreFlight WARNING] ${v}`);
        }
        bot.log("[PreFlight] Continuing in 1v1 test mode despite preflight warnings.");
    }

    const wave: WaveTrackerState = {
        waveNumber: 1,
        waveActive: false,
        opponentKilledInWave: false,
        teammateDiedInWave: false,
        canReturnToSafezone: false,
        teamKills: 0,
        teamDeaths: 0,
    };

    const allyNames = new Set<string>();
    let engagedOpponentName: string | null = null;

    while (!bot.shouldExit && wave.teamKills < config.killTarget) {
        if (!bot.player.alive) {
            wave.waveActive = false;
            wave.opponentKilledInWave = false;
            wave.canReturnToSafezone = false;
            engagedOpponentName = null;
            bot.sleep(500);
            continue;
        }

        const obs = extractObservation(allyNames, wave);
        if (obs.currentCell !== currentCellTracker) {
            currentCellTracker = obs.currentCell;
            cellEnteredAt = Date.now();
        }

        // Immediately enforce NPC protection so a Brawler/Restorer is never killed
        enforceNpcProtectionGuard(obs);

        // Update 1v1 wave & kill tracking
        const opponentInCell = obs.opponents.find((p) => p.cell === obs.currentCell && p.alive);
        if (opponentInCell) {
            engagedOpponentName = opponentInCell.username;
            wave.waveActive = true;
        } else if (engagedOpponentName && !obs.inSafezone) {
            const trackedOpp = obs.opponents.find(
                (p) => p.username.toLowerCase() === engagedOpponentName!.toLowerCase(),
            );
            if (
                !trackedOpp ||
                !trackedOpp.alive ||
                trackedOpp.cell === "Wait" ||
                trackedOpp.cell === "Enter0" ||
                trackedOpp.cell === "Enter1"
            ) {
                engagedOpponentName = null;
                wave.opponentKilledInWave = true;
                wave.canReturnToSafezone = true;
                wave.teamKills += 1;
                bot.log(`[WaveTracker] Opponent eliminated! Score: ${wave.teamKills}/${config.killTarget}. Safezone unlocked.`);
            }
        }

        if (obs.inSafezone && wave.canReturnToSafezone) {
            bot.player.rest(true);
            wave.waveNumber += 1;
            wave.waveActive = false;
            wave.opponentKilledInWave = false;
            wave.canReturnToSafezone = false;
        }

        maintainStaminaRegen(Date.now());

        const rawDecision = inferBattleWarriorDecision(obs);
        const safeDecision = applyRuleShield(rawDecision, obs, config);
        executeDecision(safeDecision, obs, config.combatCell, config.safezoneCell);

        bot.sleep(100);
    }

    bot.log(`[Non-OP PvP AI] Match finished! Final Kills: ${wave.teamKills}/${config.killTarget}.`);
}
