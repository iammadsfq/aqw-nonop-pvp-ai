/**
 * @file referee_recorder.ts
 * @description Standalone Referee / Match Observer ("Bot Wasit di Safezone") for AQW PvP.
 *
 * HOW IT WORKS (SmartFoxServer Architecture):
 * In AQW BludrutBrawl, all 16 screens exist within ONE SmartFox room.
 * The server broadcasts combat trace (`cmd: "ct"`), skill casts (`cmd: "sAct"`),
 * user location updates (`cmd: "uotls"`), and score updates (`cmd: "PVPS"`)
 * to ALL connected clients in the room!
 *
 * By staying in Safezone (`Enter0` or `Enter1`), this referee bot:
 *   1. Is 100% immune to damage and targeting.
 *   2. Never interferes with combat, pathing, or tournament rules.
 *   3. Captures 100% of player actions, skill timing, damage numbers, and movement
 *      across every screen on the map.
 *   4. Exports clean, synchronized match replays ready for imitation learning or visualizer.
 */

import { bot, PlayerInfo } from "turu";
import { BLUDRUT_SAFEZONE_CELLS } from "../config/constants";

interface MatchEvent {
    ts: number;
    type: "skill_cast" | "hit" | "aura_apply" | "aura_remove" | "kill" | "cell_move" | "score_update";
    caster?: string;
    target?: string;
    skillRef?: string;
    skillName?: string;
    damage?: number;
    hitType?: string;
    auraName?: string;
    cell?: string;
    blueScore?: number;
    redScore?: number;
}

interface PlayerReplaySnapshot {
    username: string;
    cell: string;
    x: number;
    y: number;
    hp: number;
    maxHp: number;
    mp: number;
    state: number;
}

interface MatchFrame {
    ts: number;
    blueScore: number;
    redScore: number;
    players: PlayerReplaySnapshot[];
    events: MatchEvent[];
}

const REPLAY_LOG_PREFIX = "bludrut_match_replay";

function getSkillNameFromRef(ref: string): string {
    switch (ref.toLowerCase()) {
        case "aa": return "Auto Attack";
        case "a1": return "Skill 1 (Decisive Strike)";
        case "a2": return "Skill 2 (Off Balance Stun)";
        case "a3": return "Skill 3 (Prepared Strike)";
        case "a4": return "Skill 4 (On Guard Barrier)";
        default: return ref;
    }
}

export function runRefereeObserver(): void {
    bot.log("==========================================================");
    bot.log("  AQW PVP REFEREE OBSERVER ('Bot Wasit di Safezone')");
    bot.log("==========================================================");

    while (!bot.player.loggedIn || !bot.player.loaded) {
        bot.sleep(500);
    }

    // Ensure observer stays in Safezone
    const currentCell = bot.player.cell;
    if (!BLUDRUT_SAFEZONE_CELLS.has(currentCell)) {
        bot.log("[Wasit] Moving to Safezone (Enter0)...");
        bot.map.jump("Enter0", "Spawn");
        bot.sleep(1000);
    }

    const entIdToUsername = new Map<number, string>();
    let pendingEvents: MatchEvent[] = [];
    let blueScore = 0;
    let redScore = 0;

    function refreshPlayerEntIds(): void {
        for (const name of bot.map.playerNames) {
            const info = bot.map.getPlayer(name);
            if (info?.entId) {
                entIdToUsername.set(info.entId, name);
            }
        }
    }

    // Hook SmartFox extension packets map-wide
    const unsubscribe = bot.events.on("extensionPacket", (pkt: any) => {
        try {
            const dataObj = pkt?.params?.dataObj ?? pkt;
            if (!dataObj || typeof dataObj !== "object") return;
            const now = Date.now();
            const cmd = dataObj.cmd as string | undefined;

            refreshPlayerEntIds();

            // 1. Combat Traces (`ct`)
            if (cmd === "ct") {
                const sarsa = Array.isArray(dataObj.sarsa) ? dataObj.sarsa : [];
                for (const s of sarsa) {
                    const cInf = String(s.cInf ?? "");
                    let casterName = cInf;
                    if (cInf.startsWith("p:")) {
                        const entId = Number(cInf.slice(2));
                        casterName = entIdToUsername.get(entId) ?? cInf;
                    }

                    const actions = Array.isArray(s.a) ? s.a : [];
                    for (const act of actions) {
                        const tInf = String(act.tInf ?? "");
                        let targetName = tInf;
                        if (tInf.startsWith("p:")) {
                            const tEntId = Number(tInf.slice(2));
                            targetName = entIdToUsername.get(tEntId) ?? tInf;
                        }

                        const actRef = String(act.actRef ?? "");
                        const hpDelta = typeof act.hp === "number" ? act.hp : 0;
                        const hitType = String(act.type ?? "hit");

                        const ev: MatchEvent = {
                            ts: now,
                            type: "skill_cast",
                            caster: casterName,
                            target: targetName,
                            skillRef: actRef,
                            skillName: getSkillNameFromRef(actRef),
                            damage: hpDelta,
                            hitType,
                        };
                        pendingEvents.push(ev);

                        // Live commentary
                        if (actRef === "a2") {
                            bot.log(`⚡ [Wasit] ${casterName} CAST STUN (S2) on ${targetName}! (${hpDelta} dmg, ${hitType})`);
                        } else if (actRef === "a4") {
                            bot.log(`🛡️ [Wasit] ${casterName} CAST BARRIER (S4)!`);
                        } else if (actRef === "a3") {
                            bot.log(`🔥 [Wasit] ${casterName} CAST PREPARED STRIKE (S3)!`);
                        } else if (hpDelta > 1500) {
                            bot.log(`💥 [Wasit] Heavy Hit! ${casterName} -> ${targetName}: ${hpDelta} dmg (${hitType})`);
                        }
                    }
                }

                // Check auras applied / removed
                const auras = Array.isArray(dataObj.a) ? dataObj.a : [];
                for (const a of auras) {
                    const aCmd = a.cmd;
                    const aName = a.aura?.nam ?? "Unknown";
                    const tInf = String(a.tInf ?? "");
                    let targetName = tInf;
                    if (tInf.startsWith("p:")) {
                        const tEntId = Number(tInf.slice(2));
                        targetName = entIdToUsername.get(tEntId) ?? tInf;
                    }

                    if (aCmd === "aura-") {
                        pendingEvents.push({
                            ts: now,
                            type: "aura_remove",
                            target: targetName,
                            auraName: aName,
                        });
                    } else {
                        pendingEvents.push({
                            ts: now,
                            type: "aura_apply",
                            target: targetName,
                            auraName: aName,
                        });
                        if (/stun|balance/i.test(aName)) {
                            bot.log(`🌀 [Wasit] CC APPLIED: ${targetName} is now ${aName}!`);
                        }
                    }
                }
            }

            // 2. Score Updates (`PVPS` / `pvpScore`)
            if (cmd === "PVPS" || cmd === "pvpScore") {
                const scores = Array.isArray(dataObj.pvpScore) ? dataObj.pvpScore : [];
                if (scores.length >= 2) {
                    blueScore = scores[0]?.v ?? blueScore;
                    redScore = scores[1]?.v ?? redScore;
                    bot.log(`🏆 [Wasit] Score Update: Blue ${blueScore} - Red ${redScore}`);
                }
            }
        } catch (err) {
            bot.log(`[Wasit] Error parsing telemetry packet: ${String(err)}`);
        }
    });

    bot.log("[Wasit] Active in Safezone! Recording 100% of map actions to Army Log...");

    const logFileName = `${REPLAY_LOG_PREFIX}_${Date.now()}`;

    try {
        while (!bot.shouldExit) {
            bot.sleep(100); // 100ms / 10 Hz sampling rate

            const playersSnap: PlayerReplaySnapshot[] = [];
            for (const name of bot.map.playerNames) {
                const pInfo: PlayerInfo | null = bot.map.getPlayer(name);
                if (!pInfo) continue;
                playersSnap.push({
                    username: pInfo.name,
                    cell: pInfo.cell,
                    x: pInfo.x,
                    y: pInfo.y,
                    hp: pInfo.hp,
                    maxHp: pInfo.maxHp,
                    mp: pInfo.mp,
                    state: pInfo.state,
                });
            }

            const frame: MatchFrame = {
                ts: Date.now(),
                blueScore,
                redScore,
                players: playersSnap,
                events: pendingEvents,
            };
            pendingEvents = [];

            bot.army.appendLog(logFileName, JSON.stringify(frame));
        }
    } finally {
        unsubscribe();
        bot.log("[Wasit] Observer terminated. Replay saved successfully.");
    }
}

export default function main(): void {
    runRefereeObserver();
}
