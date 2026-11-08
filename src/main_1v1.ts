/**
 * @file main_1v1.ts
 * @description Entry-point script for 1v1 Non-OP BludrutBrawl in Turu with Pure RL Multi-Head Policy.
 * Run with:
 *   ./scripts/run-turu-headless.sh --user <user> --server <server> --run-script src/main_1v1.ts
 */

import { bot } from "turu";
import {
  initInferenceSession,
  inferBattleWarriorDecision,
  resetPolicyState,
} from "./ai/nn_inference";
import {
  AQW_DEFAULT_WALK_SPEED,
  AQW_SPRINT_STAMINA_COST,
  AQW_SPRINT_WALK_SPEED,
  ROOM_DOORS,
  SCREEN_NAMES,
  WALK_BOUNDS,
  type DoorTriggerArea,
} from "./config/constants";
import { createMatchConfig } from "./config/match_config";
import { EntityTracker } from "./state/entity_tracker";
import { attachCombatTelemetryListeners } from "./telemetry/combat_recorder";
import type { CombatDecision, WaveTrackerState } from "./state/types";

declare const process: { argv?: string[]; env?: Record<string, string> } | undefined;

// Telemetry & Rate-limit Tracking
let lastSentWalkCell = "";
let lastSentWalkX = -1;
let lastSentWalkY = -1;
let lastSentWalkSpeed = 0;
let lastSentWalkTime = 0;
let lastSprintTime = 0;
let lastLogTime = 0;
let lastLoggedReason = "";
let currentCellTracker = "";

// Door Collision & Fail-Safe State
let doorStuckTimer = 0;
let isUnstucking = false;
let unstuckUntil = 0;
let unstuckTargetX = 0;
let unstuckTargetY = 0;

function getCurrentTargetPlayerName(): string | null {
  if (!bot.player.hasTarget) return null;
  const rawTargetName =
    bot.flash.get<string>("world.myAvatar.target.objData.strUsername") ??
    bot.flash.get<string>("world.myAvatar.target.pnm");
  return rawTargetName ? rawTargetName.toLowerCase() : null;
}
/**
 * Mengambil daftar pintu untuk cell saat ini (case-insensitive)
 */
function getCurrentDoors(cell: string): readonly DoorTriggerArea[] {
  if (!cell) return [];
  if (ROOM_DOORS[cell]) return ROOM_DOORS[cell]!;
  const canonical = SCREEN_NAMES.find((s) => s.toLowerCase() === cell.toLowerCase());
  return canonical ? ROOM_DOORS[canonical] || [] : [];
}

/**
 * Eksekusi pergerakan continuous vector ke client Flash & server socket AQW
 */
function triggerMovement(x: number, y: number, isSprint: boolean, isStunned: boolean): boolean {
  if (isStunned) {
    lastSprintTime = 0;
    try {
      bot.flash.callGameFunction("world.myAvatar.pMC.stopWalking");
    } catch (_) {}
    return false;
  }

  const now = Date.now();
  const isMoving = Boolean(bot.flash.get<boolean>("world.myAvatar.pMC.mcChar.onMove"));
  const currentSp = bot.flash.get<number>("world.myAvatar.dataLeaf.intSP") ?? 100;

  // Penegakan biaya sprint 50 SP
  const canReallySprint = isSprint && currentSp >= AQW_SPRINT_STAMINA_COST;
  const targetSpeed = canReallySprint ? AQW_SPRINT_WALK_SPEED : AQW_DEFAULT_WALK_SPEED;

  const targetDist = Math.hypot(x - lastSentWalkX, y - lastSentWalkY);
  const currentCell = bot.player.cell;
  const sameDestination = currentCell === lastSentWalkCell && targetDist < 30;
  const sameSpeed = targetSpeed === lastSentWalkSpeed;

  if (sameDestination && sameSpeed) {
    if (isMoving) return false;
    if (now - lastSentWalkTime < 250) return false;
  }

  if (now - lastSentWalkTime < 100 && targetDist < 35) {
    return false;
  }

  const intX = Math.round(x);
  const intY = Math.round(y);

  lastSentWalkCell = currentCell;
  lastSentWalkX = intX;
  lastSentWalkY = intY;
  lastSentWalkSpeed = targetSpeed;
  lastSentWalkTime = now;

  const curRoom = bot.flash.get<number>("world.curRoom") ?? 1;
  try {
    bot.send.packet(`%xt%zm%mv%${curRoom}%${intX}%${intY}%${targetSpeed}%`, "String");
  } catch (_) {}

  if (canReallySprint) {
    lastSprintTime = now;
    bot.flash.set("world.justRan", true);
    bot.flash.set("world.rootClass.pDash", false);
    bot.player.walkTo(intX, intY, AQW_SPRINT_WALK_SPEED);
  } else {
    bot.flash.set("world.justRan", false);
    bot.flash.set("world.rootClass.pDash", false);
    bot.player.walkTo(intX, intY, AQW_DEFAULT_WALK_SPEED);
  }

  return true;
}

/**
 * Menjalankan keputusan simultan model (Movement & Combat) murni tanpa override koordinat
 */
function executeDecision(
  decision: CombatDecision,
  isStunned: boolean
): void {
  const now = Date.now();
  const movement = decision.movement;
  const combatAction = decision.combatAction;

  // 1. Target Synchronization
  const currentTargetName = getCurrentTargetPlayerName();
  const desiredTargetName = decision.targetName?.toLowerCase() ?? "";

  if (desiredTargetName && (!bot.player.hasTarget || currentTargetName !== desiredTargetName)) {
    bot.combat.attackPlayer(decision.targetName!);
  }

  let execX = movement.targetX;
  let execY = movement.targetY;
  const doors = getCurrentDoors(bot.player.cell);

  // 2. Hardware / Flash Door Un-Stuck Fail-Safe
  const insideAnyDoor = doors.some(
    (d) => bot.player.x >= d.xMin && bot.player.x <= d.xMax && bot.player.y >= d.yMin && bot.player.y <= d.yMax
  );

  if (insideAnyDoor && movement.type !== "STAND") {
    if (doorStuckTimer === 0) {
      doorStuckTimer = now;
    } else if (now - doorStuckTimer >= 1200 && !isUnstucking) {
      isUnstucking = true;
      unstuckUntil = now + 350;
      unstuckTargetX = bot.player.x > 500 ? Math.max(120, bot.player.x - 120) : Math.min(840, bot.player.x + 120);
      unstuckTargetY = bot.player.y;
      bot.log(`[DoorSync] Re-sync di border pintu. Mundur ke (${unstuckTargetX}, ${unstuckTargetY})...`);
    }
  } else {
    doorStuckTimer = 0;
  }

  if (isUnstucking) {
    if (now < unstuckUntil) {
      execX = unstuckTargetX;
      execY = unstuckTargetY;
    } else {
      isUnstucking = false;
      doorStuckTimer = 0;
    }
  }

  // 4. Movement Execution (Continuous point pilihan model)
  if (movement.type !== "STAND") {
    triggerMovement(execX, execY, movement.type === "SPRINT", isStunned);
  } else {
    if (now - lastSprintTime >= 200 && bot.flash.get<boolean>("world.myAvatar.pMC.mcChar.onMove")) {
      try {
        bot.flash.callGameFunction("world.myAvatar.pMC.stopWalking");
      } catch (_) {}
    }
  }

  // 5. Combat Execution (Discrete action head)
  switch (combatAction) {
    case "NONE":
      break;
    case "AUTO_ATTACK":
      const currentTarget = getCurrentTargetPlayerName();
      const desiredTarget = decision.targetName?.toLowerCase() ?? "";

      if (desiredTarget && (!bot.player.hasTarget || currentTarget !== desiredTarget)) {
        bot.combat.attackPlayer(decision.targetName!);
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

  // 6. Telemetry Logger
  if (combatAction !== "NONE" || movement.type !== "STAND") {
    const logKey = `${combatAction}|${movement.type}|${decision.targetName ?? ""}`;
    if (logKey !== lastLoggedReason || now - lastLogTime >= 1000) {
      lastLoggedReason = logKey;
      lastLogTime = now;
      bot.log(`[1v1 Pure RL] ${decision.reason}`);
    }
  }
}

export default async function main(): Promise<void> {
  try {
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

    bot.log(`[Pure RL AI] Menginisialisasi match 1v1 Bludrut Brawl untuk Tim ${isRedTeam ? "RED" : "BLUE"}...`);

    while (!bot.player.loggedIn || !bot.player.loaded) {
      bot.sleep(500);
    }

    // 1. Matikan cheat bawaan Turu agar fisika jarak 160px/38px bekerja valid
    bot.options.infiniteRange = false;
    bot.options.aggroMonsters = false;
    bot.options.aggroAllMonsters = false;
    bot.options.attackWithoutTarget = false;
    bot.log(`1`)
    // 2. Inisialisasi ONNX Runtime Inference Session
    const session = initInferenceSession();
    bot.log(`2`)
    // 3. Inisialisasi Telemetry Observer (EntityTracker)
    const tracker = new EntityTracker();
    bot.log(`3`)
    // 4. Join arena jika belum di dalam
    if (bot.map.name.toLowerCase() !== "bludrutbrawl") {
      const joinCell = isRedTeam ? "Enter1" : "Enter0";
      bot.log(`[Pure RL AI] Bergabung ke room arena (${joinCell})...`);
      bot.map.join("bludrutbrawl-99999", joinCell, "Spawn");
      bot.wait.forMapLoad("bludrutbrawl", 30);
      bot.sleep(1000);
    }
    bot.log(`4`)

    attachCombatTelemetryListeners();

    // 5. Verifikasi Tim Server
    const serverPvpTeam =
      bot.flash.get<number>("world.myAvatar.objData.pvpTeam") ??
      bot.flash.get<number>("world.myAvatar.pvpTeam");

    if (serverPvpTeam === 1 || bot.player.cell.toLowerCase() === "enter1" || bot.player.cell.includes("1")) {
      isRedTeam = true;
      safezoneCell = "Enter1";
      combatCell = "Morale1C";
    } else {
      isRedTeam = false;
      safezoneCell = "Enter0";
      combatCell = "Morale0C";
    }
    config.safezoneCell = safezoneCell;
    config.combatCell = combatCell;
    bot.log(`[TeamSync] Tim terverifikasi: ${isRedTeam ? "RED" : "BLUE"}. Safezone: ${safezoneCell}`);

    const wave: WaveTrackerState = {
      waveNumber: 1,
      waveActive: false,
      opponentKilledInWave: false,
      teammateDiedInWave: false,
      canReturnToSafezone: false,
      teamKills: 0,
      teamDeaths: 0,
    };

    let engagedOpponentName: string | null = null;
    let oocTimer = 0.0;
    let lastLoopTs = Date.now();

    // 6. Main 10 Hz Control Loop
    while (!bot.shouldExit && wave.teamKills < config.killTarget) {
      const now = Date.now();
      const dt = Math.max(0.001, (now - lastLoopTs) / 1000.0);
      lastLoopTs = now;

      // Boundary Kematian: Reset Hidden State GRU
      if (!bot.player.alive) {
        wave.waveActive = false;
        wave.opponentKilledInWave = false;
        wave.canReturnToSafezone = false;
        engagedOpponentName = null;
        oocTimer = 0.0;
        resetPolicyState();
        bot.sleep(500);
        continue;
      }

      // A. Tick Sensor State Tracker
      tracker.tick(dt);

      if (bot.player.inCombat) {
        oocTimer = 0.0;
      } else {
        oocTimer = Math.min(12.0, oocTimer + dt);
      }

      // B. Listener Transisi Ruangan
      const curCell = bot.player.cell;
      if (curCell.toLowerCase() !== currentCellTracker.toLowerCase()) {
        bot.log(`[ScreenTransition] Memasuki room: ${curCell}`);
        try {
          bot.flash.callGameFunction("world.myAvatar.pMC.stopWalking");
        } catch (_) {}
        lastSentWalkCell = "";
        lastSentWalkX = -1;
        lastSentWalkY = -1;
        currentCellTracker = curCell;
        doorStuckTimer = 0;
        isUnstucking = false;
        bot.sleep(150);
      }

      // C. Evaluasi Eliminasi Lawan & Transisi Mode 2
      const enemies = tracker.getEnemies();
      const oppInRoom = enemies.find(
        (e) => e.cell.toLowerCase() === curCell.toLowerCase() && e.alive
      );
      const oppInMid = enemies.some(
        (e) => e.cell.toLowerCase() === "crosslower" && e.alive
      );

      if (oppInRoom) {
        engagedOpponentName = oppInRoom.username;
        wave.waveActive = true;
        wave.canReturnToSafezone = false;
      } else if (oppInMid) {
        wave.canReturnToSafezone = false;
      } else if (engagedOpponentName && curCell.toLowerCase() !== safezoneCell.toLowerCase()) {
        const trackedOpp = enemies.find(
          (e) => e.username.toLowerCase() === engagedOpponentName!.toLowerCase()
        );
        if (
          !trackedOpp ||
          !trackedOpp.alive ||
          trackedOpp.cell.toLowerCase() === "enter0" ||
          trackedOpp.cell.toLowerCase() === "enter1"
        ) {
          engagedOpponentName = null;
          wave.opponentKilledInWave = true;
          wave.canReturnToSafezone = true;
          wave.teamKills += 1;
          bot.log(`[WaveTracker] Lawan tereliminasi! Skor: ${wave.teamKills}/${config.killTarget}`);
        }
      }

      // D. Safezone Rest Handler (Hanya jika wave telah dimenangkan)
      const curCellLower = curCell.toLowerCase();
      if ((curCellLower === "enter0" || curCellLower === "enter1") && wave.canReturnToSafezone) {
        bot.player.rest(true);
        wave.waveNumber += 1;
        wave.waveActive = false;
        wave.opponentKilledInWave = false;
        wave.canReturnToSafezone = false;
        resetPolicyState();
      }

      // E. Status Kesiapan Skill [S1..S4]
      const skillsReady = [
        bot.skills.canUseSkill(1) ? 1.0 : 0.0,
        bot.skills.canUseSkill(2) ? 1.0 : 0.0,
        bot.skills.canUseSkill(3) ? 1.0 : 0.0,
        bot.skills.canUseSkill(4) ? 1.0 : 0.0,
      ];

      // F. Pure RL Forward Inference
      const decision = inferBattleWarriorDecision(
        tracker,
        session,
        skillsReady,
        bot.player.inCombat,
        oocTimer
      );

      // G. Eksekusi Fisik Murni Tanpa Override Koordinat
      const selfEntity = tracker.getSelf();
      executeDecision(decision, selfEntity.isStunned);

      bot.sleep(100); // 10 Hz cycle tick
    }

    bot.log(`[Pure RL AI] Match selesai! Skor Akhir: ${wave.teamKills}/${config.killTarget}`);
  } catch (e) {
    bot.log(`[FATAL] ${e instanceof Error ? e.stack : String(e)}`);
    throw e;
  }
}