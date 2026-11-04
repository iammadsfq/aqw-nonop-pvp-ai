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

declare const process: { argv?: string[]; env?: Record<string, string> } | undefined;

let lastWalkCell = "";
let lastWalkX = -1;
let lastWalkY = -1;
let lastWalkTime = 0;
let lastSprintTime = 0;
let lastRegenTickTime = 0;
let currentCellTracker = "";
let cellEnteredAt = 0;

function maintainStaminaRegen(now: number, inCombat: boolean): void {
    // In AQW:
    // Out-of-Combat: +25 SP every 4.0 seconds
    // In-Combat: stamina recovery is much slower and smaller: +10 SP every 8.0 seconds
    const regenInterval = inCombat ? 8000 : 4000;
    const regenAmount = inCombat ? 10 : 25;

    // Must wait at least 3.0s after sprinting before stamina begins recovering
    if (now - lastSprintTime >= 3000 && now - lastRegenTickTime >= regenInterval) {
        const currentSp = bot.flash.get<number>("world.myAvatar.dataLeaf.intSP") ?? 100;
        if (currentSp < 100) {
            const regenedSp = Math.min(100, currentSp + regenAmount);
            bot.flash.set("world.myAvatar.dataLeaf.intSP", regenedSp);
            bot.flash.set("world.myAvatar.objData.intSP", regenedSp);
            bot.flash.call("world.updatePortrait", "world.myAvatar");
            bot.flash.call("world.updateActBar");
            lastRegenTickTime = now;
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
    const rawArgs = typeof process !== "undefined" && process.argv ? process.argv : [];
    const teamArg =
        rawArgs.find((a) => a.startsWith("--team="))?.split("=")[1]?.toLowerCase() ??
        (rawArgs.includes("--red") || rawArgs.includes("red") ? "red" : undefined) ??
        (rawArgs.includes("--blue") || rawArgs.includes("blue") ? "blue" : undefined) ??
        (typeof process !== "undefined" && process.env?.AQW_TEAM ? process.env.AQW_TEAM.toLowerCase() : undefined);

    let isRedTeam = teamArg === "red";
    let safezoneCell = isRedTeam ? "Enter1" : "Enter0";
    let combatCell = isRedTeam ? "Morale1C" : "Morale0C";

    const config = createMatchConfig("1v1", {
        mapName: "bludrutbrawl",
        combatCell,
        safezoneCell,
    });

    bot.log(`[Non-OP PvP AI] Initializing ${config.format} match for Team ${isRedTeam ? "RED" : "BLUE"} (Target: ${config.killTarget} kills)...`);

    while (!bot.player.loggedIn || !bot.player.loaded) {
        bot.sleep(500);
    }

    // Automatically join private /bludrutbrawl-99999 room if not already in bludrutbrawl
    if (bot.map.name.toLowerCase() !== "bludrutbrawl") {
        const joinCell = isRedTeam ? "Enter1" : "Enter0";
        bot.log(`[Non-OP PvP AI] Joining /bludrutbrawl-99999 arena (${joinCell})...`);
        bot.map.join("bludrutbrawl-99999", joinCell, "Spawn");
        bot.wait.forMapLoad("bludrutbrawl", 30);
        bot.sleep(1000);
    }

    // Inspect server-assigned PvP Faction in Flash (0 = Blue, 1 = Red)
    // to detect if server overrode team or if player was placed in opposite safezone
    const serverPvpTeam =
        bot.flash.get<number>("world.myAvatar.objData.pvpTeam") ??
        bot.flash.get<number>("world.myAvatar.pvpTeam");

    if (serverPvpTeam === 1 || bot.player.cell.toLowerCase() === "enter1" || bot.player.cell.includes("1")) {
        isRedTeam = true;
        safezoneCell = "Enter1";
        combatCell = "Morale1C";
        config.safezoneCell = safezoneCell;
        config.combatCell = combatCell;

        if (bot.player.cell.toLowerCase() === "enter0") {
            bot.log("[TeamSync] Assigned to Red Team (1) but currently in Blue Safezone (Enter0). Relocating to Enter1...");
            bot.map.jump("Enter1", "Spawn");
            bot.sleep(1000);
        }
    } else {
        isRedTeam = false;
        safezoneCell = "Enter0";
        combatCell = "Morale0C";
        config.safezoneCell = safezoneCell;
        config.combatCell = combatCell;

        if (bot.player.cell.toLowerCase() === "enter1") {
            bot.log("[TeamSync] Assigned to Blue Team (0) but currently in Red Safezone (Enter1). Relocating to Enter0...");
            bot.map.jump("Enter0", "Spawn");
            bot.sleep(1000);
        }
    }

    bot.log(`[TeamSync] Confirmed active team: ${isRedTeam ? "RED (Enter1 / Morale1C)" : "BLUE (Enter0 / Morale0C)"}.`);

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

        maintainStaminaRegen(Date.now(), obs.selfInCombat);

        const rawDecision = inferBattleWarriorDecision(obs);
        const safeDecision = applyRuleShield(rawDecision, obs, config);
        executeDecision(safeDecision, obs, config.combatCell, config.safezoneCell);

        bot.sleep(100);
    }

    bot.log(`[Non-OP PvP AI] Match finished! Final Kills: ${wave.teamKills}/${config.killTarget}.`);
}
