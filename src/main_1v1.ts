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
    AQW_MAX_STAMINA,
    AQW_SPRINT_STAMINA_COST,
    AQW_SPRINT_WALK_SPEED,
    BLUDRUT_16_SCREENS,
} from "./config/constants";
import { createMatchConfig } from "./config/match_config";
import { runNonOpPreflightCheck } from "./rules/preflight";
import { applyRuleShield, enforceNpcProtectionGuard } from "./rules/shield";
import { extractObservation } from "./state/extractor";
import { attachCombatTelemetryListeners } from "./telemetry/combat_recorder";
import type { CombatDecision, CombatObservation, WaveTrackerState } from "./state/types";

declare const process: { argv?: string[]; env?: Record<string, string> } | undefined;

let lastSentWalkCell = "";
let lastSentWalkX = -1;
let lastSentWalkY = -1;
let lastSentWalkSpeed = 0;
let lastSentWalkTime = 0;
let lastSprintTime = 0;
let lastLogTime = 0;
let lastLoggedReason = "";

let currentCellTracker = "";
let cellEnteredAt = 0;

// Door sync re-entry state ("turun dikit lalu masuk lagi")
let activeDoorCell = "";
let doorArrivedAt = 0;
let isSteppingDown = false;
let stepDownUntil = 0;
let stepDownTargetX = 0;
let stepDownTargetY = 0;
let stepDownAttempts = 0;

function triggerSprintWalk(obs: CombatObservation, x: number, y: number, isSprint: boolean): boolean {
    // If stunned, immediately freeze character movement on client to match server state!
    if (obs.selfStunned) {
        lastSprintTime = 0;
        try {
            bot.flash.callGameFunction("world.myAvatar.pMC.stopWalking");
        } catch (_) {}
        return false;
    }

    const now = Date.now();
    const isMoving = Boolean(bot.flash.get<boolean>("world.myAvatar.pMC.mcChar.onMove"));

    // Rely on Flash/Server native intSP without artificially overriding it
    const currentSp = bot.flash.get<number>("world.myAvatar.dataLeaf.intSP") ?? 100;
    const canReallySprint = isSprint && currentSp >= AQW_SPRINT_STAMINA_COST;
    const targetSpeed = canReallySprint ? AQW_SPRINT_WALK_SPEED : AQW_DEFAULT_WALK_SPEED;

    // 1. Sprint Commitment Lock (250ms):
    // If an active sprint was launched less than 250ms ago and avatar is moving,
    // do NOT interrupt it with repeated walkTo calls, allowing the server to validate the dash!
    if (isMoving && now - lastSprintTime < 250) {
        return false;
    }

    // 2. Deadzone & Anti-Flooding Packet Throttler:
    const targetDist = Math.hypot(x - lastSentWalkX, y - lastSentWalkY);
    const sameDestination = obs.currentCell === lastSentWalkCell && targetDist < 40;
    const sameSpeed = targetSpeed === lastSentWalkSpeed;

    if (sameDestination && sameSpeed) {
        // If avatar is currently walking towards this target, do NOT resend packets.
        // Flash client is smoothly interpolating along the tween.
        if (isMoving) {
            return false;
        }
        // If avatar is stationary, wait at least 350ms before retrying to avoid spamming stopped state
        if (now - lastSentWalkTime < 350) {
            return false;
        }
    }

    // 3. Absolute Rate Limiter: Max 7-8 movement packets per second (~120ms interval)
    // unless destination changed significantly (> 60px)
    if (now - lastSentWalkTime < 120 && targetDist < 60) {
        return false;
    }

    lastSentWalkCell = obs.currentCell;
    lastSentWalkX = x;
    lastSentWalkY = y;
    lastSentWalkSpeed = targetSpeed;
    lastSentWalkTime = now;

    // Send the authentic AQW SmartFox movement packet to the server:
    // pushMove format: %xt%zm%mv%<curRoom>%<x>%<y>%<speed>%
    // When targetSpeed is 24, the server validates the dash request, registers sprint,
    // decrements SP on the server, and syncs intsp via uotls back to the client!
    const curRoom = bot.flash.get<number>("world.curRoom") ?? 1;
    try {
        bot.send.packet(`%xt%zm%mv%${curRoom}%${x}%${y}%${targetSpeed}%`, "String");
    } catch (_) {}

    if (canReallySprint) {
        lastSprintTime = now;
        bot.flash.set("world.justRan", true);
        bot.flash.set("world.rootClass.pDash", false);
        bot.player.walkTo(x, y, AQW_SPRINT_WALK_SPEED);
    } else {
        bot.flash.set("world.justRan", false);
        bot.flash.set("world.rootClass.pDash", false);
        bot.player.walkTo(x, y, AQW_DEFAULT_WALK_SPEED);
    }
    return true;
}

function executeDecision(decision: CombatDecision, obs: CombatObservation, combatCell: string, safezoneCell: string): void {
    // 1. Target Selection: Attacks the chosen target cleanly through Turu combat API
    if (
        decision.targetName &&
        (!obs.hasTarget || obs.targetIsNpc || obs.targetName?.toLowerCase() !== decision.targetName.toLowerCase())
    ) {
        bot.combat.attackPlayer(decision.targetName);
    } else if (decision.npcMapId !== undefined && (!obs.hasTarget || !obs.targetIsNpc)) {
        bot.combat.attack(decision.npcMapId);
    }

    const movement = decision.movement ?? {
        type:
            decision.action === "SPRINT_TO_POINT" || decision.action === "SPRINT_TO_DOOR"
                ? "SPRINT"
                : decision.action === "WALK_TO_POINT" || decision.action === "WALK_TO_DOOR"
                  ? "WALK"
                  : "STAND",
        targetX: decision.targetX ?? Math.round(obs.selfX),
        targetY: decision.targetY ?? Math.round(obs.selfY),
        targetCell: decision.targetCell,
        targetPad: decision.targetPad,
    };

    const combatAction =
        decision.combatAction ??
        (decision.action === "USE_SKILL_1"
            ? "USE_SKILL_1"
            : decision.action === "USE_SKILL_2"
              ? "USE_SKILL_2"
              : decision.action === "USE_SKILL_3"
                ? "USE_SKILL_3"
                : decision.action === "USE_SKILL_4"
                  ? "USE_SKILL_4"
                  : decision.action === "TARGET_OPPONENT"
                    ? "AUTO_ATTACK"
                    : "NONE");

    // 2. Simultaneous Movement Execution & Door Re-sync ("turun dikit lalu masuk lagi"):
    const now = Date.now();
    const currentScreen = BLUDRUT_16_SCREENS.find((s) => s.cell.toLowerCase() === obs.currentCell.toLowerCase());

    // Find target door: only engage DoorSync when an intentional screen transition is commanded (movement.targetCell)
    // and avatar is NOT actively engaged in combat with an opponent in the same cell!
    const inCombatWithOpp = Boolean(
        obs.opponents.some((p) => p.cell.toLowerCase() === obs.currentCell.toLowerCase() && p.alive),
    );
    const targetDoor = (movement.targetCell && !inCombatWithOpp)
        ? currentScreen?.doors.find(
              (d) => d.targetCell.toLowerCase() === movement.targetCell!.toLowerCase(),
          )
        : null;

    if (targetDoor) {
        if (targetDoor.targetCell !== activeDoorCell) {
            activeDoorCell = targetDoor.targetCell;
            doorArrivedAt = 0;
            isSteppingDown = false;
            stepDownAttempts = 0;
        }

        const distToDoorAim = Math.hypot(obs.selfX - targetDoor.aimX, obs.selfY - targetDoor.aimY);
        const inDoorBox =
            obs.selfX >= targetDoor.xMin - 15 &&
            obs.selfX <= targetDoor.xMax + 15 &&
            obs.selfY >= targetDoor.yMin - 15 &&
            obs.selfY <= targetDoor.yMax + 15;
        const atDoor = inDoorBox || distToDoorAim < 60;

        // Jika sampai di pintu dan tertahan >= 3000ms tanpa pindah room (desync server)
        if (atDoor && !isSteppingDown) {
            if (doorArrivedAt === 0) {
                doorArrivedAt = now;
            } else if (now - doorArrivedAt > 3000) {
                stepDownAttempts++;
                isSteppingDown = true;
                stepDownUntil = now + 600;

                const stepX =
                    targetDoor.aimX > 500
                        ? Math.max(120, Math.round(obs.selfX - 120))
                        : Math.min(840, Math.round(obs.selfX + 120));

                const stepY =
                    targetDoor.aimY < 380
                        ? Math.min(430, Math.round(obs.selfY + 70))
                        : Math.max(280, Math.round(obs.selfY - 70));

                stepDownTargetX = stepX;
                stepDownTargetY = stepY;
                bot.log(
                    `[DoorSync] Stuck di pintu ${targetDoor.targetAlias} (${targetDoor.targetCell}) selama ${now - doorArrivedAt}ms. Turun dikit ke (${stepX}, ${stepY}) lalu masuk lagi... (attempt #${stepDownAttempts})`,
                );
            }
        }

        if (isSteppingDown) {
            if (now < stepDownUntil) {
                movement.targetX = stepDownTargetX;
                movement.targetY = stepDownTargetY;
                movement.targetCell = undefined; // Jangan minta pindah room saat sedang turun
                movement.type = "WALK";
            } else {
                // Selesai turun: masuk lagi ke pintu!
                isSteppingDown = false;
                doorArrivedAt = now;
                movement.targetX = targetDoor.aimX;
                movement.targetY = targetDoor.aimY;
                movement.targetCell = targetDoor.targetCell;
                movement.targetPad = targetDoor.targetPad;
                movement.type = "WALK";
            }
        }
    } else {
        activeDoorCell = "";
        doorArrivedAt = 0;
        isSteppingDown = false;
        stepDownAttempts = 0;
    }

    // Walks or Sprints to target ground coordinates (X, Y)
    if (movement.type !== "STAND") {
        const isSprint = movement.type === "SPRINT";
        // Room transitions always use standard walk speed for clean server sync
        const willTransition = Boolean(movement.targetCell);
        triggerSprintWalk(
            obs,
            movement.targetX,
            movement.targetY,
            willTransition ? false : isSprint,
        );
    } else {
        // If STAND was commanded and avatar is currently walking, stop cleanly (unless sprint-committed)
        if (now - lastSprintTime >= 250 && bot.flash.get<boolean>("world.myAvatar.pMC.mcChar.onMove")) {
            try {
                bot.flash.callGameFunction("world.myAvatar.pMC.stopWalking");
            } catch (_) {}
        }
    }

    // 3. Simultaneous Combat Execution:
    // Casts skills 1-4, auto-attacks, or mana-taps on the EXACT SAME FRAME without stopping movement!
    switch (combatAction) {
        case "NONE":
            break;
        case "AUTO_ATTACK":
            if (decision.targetName) {
                bot.combat.attackPlayer(decision.targetName);
            }
            break;
        case "USE_SKILL_1":
            bot.skills.useSkill(1);
            break;
        case "USE_SKILL_2":
            bot.skills.useSkill(2);
            break;
        case "USE_SKILL_3":
            bot.skills.useSkill(3);
            break;
        case "USE_SKILL_4":
            bot.skills.useSkill(4);
            break;
    }

    // Log throttling: only log when semantic state changes or at most once every 1000ms
    if (combatAction !== "NONE" || movement.type !== "STAND") {
        const logKey = `${combatAction}|${movement.type}|${movement.targetCell ?? ""}|${decision.targetName ?? ""}`;
        if (logKey !== lastLoggedReason || now - lastLogTime >= 1000) {
            lastLoggedReason = logKey;
            lastLogTime = now;
            bot.log(`[1v1 AI] ${decision.reason}`);
        }
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

    // Attach live telemetry listeners to capture pro-player demonstration data
    attachCombatTelemetryListeners();

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

    // Diagnostic Game Warning Monitor: tracks client Flash range errors and server warning packets
    let lastRangeWarningTs = 0;
    try {
        lastRangeWarningTs = bot.flash.get<number>("world.actionRangeSpamTS") ?? 0;
    } catch (_) {}

    // Tournament 1-Minute Mid Rule Tracker:
    // Players may tactically enter Captain, Restorer, or Brawler rooms to kite/stall cooldowns,
    // but must touch Mid (Crosslower) at least once every 60 seconds if not engaged in combat!
    let awayFromMidMs = 0;
    let lastMidTimerTick = Date.now();

    bot.events.on("extensionPacket", (pkt: any) => {
        try {
            const dataObj = pkt?.params?.dataObj;
            if (Array.isArray(dataObj) && dataObj[0] === "warning") {
                const msg = dataObj[2] ?? dataObj[1] ?? JSON.stringify(dataObj);
                bot.log(`[GameWarning] [Server] ${msg}`);
            } else if (pkt?.params?.type === "str") {
                const str = String(pkt.params.dataObj);
                if (str.includes("warning") || str.includes("range")) {
                    bot.log(`[GameWarning] [Server] ${str}`);
                }
            }
        } catch (_) {}
    });

    while (!bot.shouldExit && wave.teamKills < config.killTarget) {
        // Poll Flash client-side range check warnings (World.as:9086)
        try {
            const spamTs = bot.flash.get<number>("world.actionRangeSpamTS");
            if (typeof spamTs === "number" && spamTs > lastRangeWarningTs) {
                lastRangeWarningTs = spamTs;
                bot.log("[GameWarning] You are out of range! Move closer to your target.");
            }
        } catch (_) {}

        if (!bot.player.alive) {
            wave.waveActive = false;
            wave.opponentKilledInWave = false;
            wave.canReturnToSafezone = false;
            engagedOpponentName = null;
            awayFromMidMs = 0;
            lastMidTimerTick = Date.now();
            bot.sleep(500);
            continue;
        }

        const now = Date.now();
        const dt = Math.max(0, now - lastMidTimerTick);
        lastMidTimerTick = now;

        const currentCellLower = bot.player.cell?.toLowerCase() ?? "";
        const isInMid = currentCellLower === "crosslower";
        const isInSz = currentCellLower === "enter0" || currentCellLower === "enter1";
        const isInCombat = bot.player.inCombat;

        if (isInMid) {
            if (awayFromMidMs > 0) {
                bot.log("[TournamentRule] Mid (Crosslower) touched! 60s stall timer reset.");
            }
            awayFromMidMs = 0;
        } else if (isInCombat) {
            // Active combat anywhere in the arena suspends/resets the stall timer
            awayFromMidMs = 0;
        } else if (isInSz) {
            // Respawn / safezone recovery does not count toward illegal stall
            awayFromMidMs = 0;
        } else {
            awayFromMidMs += dt;
            // Diagnostic warnings at 45s and 60s
            if (awayFromMidMs >= 45000 && awayFromMidMs - dt < 45000) {
                bot.log(`[TournamentRule] ⚠️ 45s away from Mid without combat! Returning to Mid to prevent stall violation.`);
            } else if (awayFromMidMs >= 60000 && awayFromMidMs - dt < 60000) {
                bot.log(`[TournamentRule] ⚠️ 60s Mid rule limit reached! Prioritizing immediate Mid contest.`);
            }
        }

        const obs = extractObservation(
            allyNames,
            wave,
            isRedTeam ? "red" : "blue",
            awayFromMidMs / 1000.0,
        );

        // Anti-Roofing Safeguard: If avatar is caught on ceiling rafters (y < 180) or off-screen bottom (y > 530)
        if (bot.player.y < 180 || bot.player.y > 530) {
            try {
                bot.flash.callGameFunction("world.myAvatar.pMC.stopWalking");
            } catch (_) {}
            const recoverX = Math.min(800, Math.max(160, bot.player.x));
            bot.player.walkTo(recoverX, 350, AQW_DEFAULT_WALK_SPEED);
            bot.log(`[AntiRoofing] Avatar off-bounds (Y=${bot.player.y})! Walking down to floor (${recoverX}, 350).`);
            bot.sleep(250);
            continue;
        }

        if (obs.currentCell !== currentCellTracker) {
            const screenSpec = BLUDRUT_16_SCREENS.find((s) => s.cell.toLowerCase() === obs.currentCell.toLowerCase());
            bot.log(`[ScreenTransition] Entered ${obs.currentCell} (${screenSpec?.alias ?? ""}).`);
            // Stop residual walking immediately upon changing rooms so the avatar
            // does NOT overshoot across the new screen to the previous click point!
            try {
                bot.flash.callGameFunction("world.myAvatar.pMC.stopWalking");
            } catch (_) {}
            lastSentWalkCell = "";
            lastSentWalkX = -1;
            lastSentWalkY = -1;
            lastSentWalkSpeed = 0;
            lastSentWalkTime = 0;
            activeDoorCell = "";
            doorArrivedAt = 0;
            isSteppingDown = false;
            stepDownAttempts = 0;
            currentCellTracker = obs.currentCell;
            cellEnteredAt = Date.now();
            bot.sleep(150);
        }

        // Immediately enforce NPC protection so a Brawler/Restorer is never killed
        enforceNpcProtectionGuard(obs);

        // Update 1v1 wave & kill tracking
        const opponentInCell = obs.opponents.find(
            (p) => p.cell.toLowerCase() === obs.currentCell.toLowerCase() && p.alive,
        );
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
                trackedOpp.cell.toLowerCase() === "wait" ||
                trackedOpp.cell.toLowerCase() === "enter0" ||
                trackedOpp.cell.toLowerCase() === "enter1"
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

        const rawDecision = inferBattleWarriorDecision(obs);
        const safeDecision = applyRuleShield(rawDecision, obs, config);
        executeDecision(safeDecision, obs, config.combatCell, config.safezoneCell);

        bot.sleep(50); // 20 Hz tick rate (down from 100ms) for snappy, human-grade responsiveness!
    }

    bot.log(`[Non-OP PvP AI] Match finished! Final Kills: ${wave.teamKills}/${config.killTarget}.`);
}
