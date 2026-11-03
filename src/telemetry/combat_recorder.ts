/**
 * @file combat_recorder.ts
 * @description Built-in AQW Server Telemetry & Combat Trace (`ct`) Recorder for Turu.
 *
 * No external packet sniffer (like Wireshark) is needed! This script hooks directly
 * into Turu's AVM2 memory (`world.actions.active`, `world.myAvatar.dataLeaf`) and
 * SmartFoxServer packet stream (`bot.events.on("extensionPacket")`) to record:
 *   1. Exact class skill specifications (`mp`, `cd`, `range`, `damage`, `tgt`, `typ`)
 *   2. Exact server-computed character stats (`$ap`, `$sp`, `$tcr`, `$scm`, `$tdo`, `$tha`, `wDPS`)
 *   3. Every real-time `ct` (Combat Trace) packet (`sarsa` damage hits/crits/dodges,
 *      `a` aura applications & durations, `p`/`m` HP & Mana deltas)
 *
 * Output is saved as JSON Lines (JSONL) in:
 *   ~/.local/share/Turu/army/aqw_combat_telemetry.log
 */

import { bot } from "turu";

const TELEMETRY_LOG_NAME = "aqw_combat_telemetry";

interface TelemetryRecord {
    ts: number;
    type:
        | "class_spec"
        | "combat_trace"
        | "player_sent_action"
        | "movement_packet"
        | "door_transition"
        | "regen_tick"
        | "map_snapshot";
    className: string | null;
    map: string;
    cell: string;
    payload: unknown;
}

function getStamina(): number {
    const rawSp = bot.flash.get<number>("world.myAvatar.dataLeaf.intSP");
    return typeof rawSp === "number" && !Number.isNaN(rawSp) ? rawSp : 100;
}

function writeTelemetry(type: TelemetryRecord["type"], payload: unknown): void {
    const record: TelemetryRecord = {
        ts: Date.now(),
        type,
        className: bot.player.currentClass,
        map: bot.map.name,
        cell: bot.player.cell,
        payload,
    };
    bot.army.appendLog(TELEMETRY_LOG_NAME, JSON.stringify(record));
}

/**
 * Extracts and records the complete class specification and stat sheet directly
 * from the game's AVM2 memory (`world.actions.active` & `world.myAvatar.dataLeaf`).
 */
export function recordCurrentClassSpecification(): void {
    const className = bot.player.currentClass ?? "Unknown";
    const classRank = bot.player.currentClassRank;
    const stats = bot.player.stats;
    const dataLeaf = bot.flash.get<Record<string, unknown> | null>("world.myAvatar.dataLeaf");
    const activeActions = bot.flash.get<unknown[] | null>("world.actions.active");
    const equippedItems = bot.inventory.items
        .filter((i) => i.equipped)
        .map((i) => ({
            id: i.id,
            name: i.name,
            category: i.category,
            equipSlot: i.equipSlot,
            enhancementLevel: i.enhancementLevel,
            enhancementPatternId: i.enhancementPatternId,
        }));

    const specPayload = {
        username: bot.player.username,
        level: bot.player.level,
        className,
        classRank,
        maxHealth: bot.player.maxHealth,
        maxMana: bot.player.maxMana,
        walkSpeed: bot.player.walkSpeed,
        computedStats: stats,
        rawSta: dataLeaf?.sta ?? null,
        wDPS: dataLeaf?.wDPS ?? null,
        equippedItems,
        skills: activeActions ?? bot.player.skills,
    };

    writeTelemetry("class_spec", specPayload);
    bot.log(
        `[Telemetry] Recorded class spec for "${className}" (Rank ${classRank}, MaxHP=${bot.player.maxHealth}, Skills=${Array.isArray(activeActions) ? activeActions.length : 0}).`,
    );
}

export default function main(): void {
    bot.log("==========================================================");
    bot.log("[AQW Telemetry Recorder] Waiting for player login & load...");
    bot.log("==========================================================");

    while (!bot.player.loggedIn || !bot.player.loaded) {
        bot.sleep(500);
    }

    let lastRecordedClass = "";

    // 1. Hook into incoming server packets (specifically `ct` combat trace & `sAct` skill packets)
    const unsubscribeExt = bot.events.on("extensionPacket", (packet: any) => {
        try {
            const dataObj = packet?.params?.dataObj ?? packet;
            if (!dataObj || typeof dataObj !== "object") return;

            const cmd = dataObj.cmd as string | undefined;
            // Record `ct` (Combat Trace), `sAct` (Skill Definitions), `uotls` (Stat/HP/MP updates), and PvP score packets
            if (
                cmd === "ct" ||
                cmd === "sAct" ||
                cmd === "uotls" ||
                cmd === "pvpScore" ||
                cmd === "PVPS" ||
                dataObj.sarsa !== undefined ||
                dataObj.a !== undefined
            ) {
                writeTelemetry("combat_trace", dataObj);
            }
        } catch (err) {
            bot.log(`[Telemetry] Packet parse error: ${String(err)}`);
        }
    });

    // 2. Hook into outgoing client packets (skill activations, rest requests, target switches, and movement/sprint)
    const unsubscribeSent = bot.events.on("packetSent", (rawPacket: string) => {
        if (
            rawPacket.includes("%gar%") ||
            rawPacket.includes("%restRequest%") ||
            rawPacket.includes("%aggroMon%")
        ) {
            writeTelemetry("player_sent_action", {
                packet: rawPacket,
                selfHp: bot.player.health,
                selfMp: bot.player.mana,
                selfSp: getStamina(),
                selfX: bot.player.x,
                selfY: bot.player.y,
                selfAuras: bot.self.snapshots,
                targetAuras: bot.target.snapshots,
            });
        } else if (rawPacket.includes("%mv%")) {
            // Format: %xt%zm%mv%roomId%x%y%speed%
            const parts = rawPacket.split("%").filter((s) => s.length > 0);
            const mvIdx = parts.indexOf("mv");
            const targetX = mvIdx >= 0 ? Number(parts[mvIdx + 2]) : null;
            const targetY = mvIdx >= 0 ? Number(parts[mvIdx + 3]) : null;
            const speed = mvIdx >= 0 ? Number(parts[mvIdx + 4]) : null;
            writeTelemetry("movement_packet", {
                packet: rawPacket,
                fromX: bot.player.x,
                fromY: bot.player.y,
                targetX,
                targetY,
                speed,
                isSprint: speed !== null && speed > 8,
                selfSp: getStamina(),
                inCombat: bot.player.inCombat,
            });
        }
    });

    bot.log("[Telemetry] Packet hooks active! Recording to ~/.local/share/Turu/army/aqw_combat_telemetry.log");

    // Track state for Door Transitions and HP/MP/Stamina Regeneration rates
    let prevCell = bot.player.cell;
    let prevPad = bot.player.pad;
    let prevX = bot.player.x;
    let prevY = bot.player.y;

    let prevHp = bot.player.health;
    let prevMp = bot.player.mana;
    let prevSp = getStamina();
    let prevInCombat = bot.player.inCombat;
    let lastTickTs = Date.now();
    let lastCombatExitTs = bot.player.inCombat ? 0 : Date.now();

    try {
        while (!bot.shouldExit) {
            if (bot.player.loggedIn && bot.player.loaded) {
                const now = Date.now();
                const currentClass = bot.player.currentClass ?? "";
                if (currentClass && currentClass !== lastRecordedClass) {
                    recordCurrentClassSpecification();
                    lastRecordedClass = currentClass;
                }

                const curCell = bot.player.cell;
                const curPad = bot.player.pad;
                const curX = bot.player.x;
                const curY = bot.player.y;

                // Detect screen/door transition across the 16 BludrutBrawl screens
                if (curCell && prevCell && curCell !== prevCell) {
                    writeTelemetry("door_transition", {
                        fromCell: prevCell,
                        fromPad: prevPad,
                        exitX: prevX,
                        exitY: prevY,
                        toCell: curCell,
                        toPad: curPad,
                        spawnX: curX,
                        spawnY: curY,
                    });
                    bot.log(
                        `[Telemetry] Door transition: ${prevCell} (${Math.round(prevX)}, ${Math.round(prevY)}) -> ${curCell} [pad=${curPad}] (${Math.round(curX)}, ${Math.round(curY)})`,
                    );
                }
                prevCell = curCell;
                prevPad = curPad;
                prevX = curX;
                prevY = curY;

                // Track combat exit timestamp and HP/MP/SP regeneration ticks
                const curInCombat = bot.player.inCombat;
                if (prevInCombat && !curInCombat) {
                    lastCombatExitTs = now;
                }
                prevInCombat = curInCombat;

                const curHp = bot.player.health;
                const curMp = bot.player.mana;
                const curSp = getStamina();
                const deltaHp = curHp - prevHp;
                const deltaMp = curMp - prevMp;
                const deltaSp = curSp - prevSp;

                if (deltaHp !== 0 || deltaMp !== 0 || deltaSp !== 0) {
                    const dtMs = Math.max(1, now - lastTickTs);
                    writeTelemetry("regen_tick", {
                        dtMs,
                        inCombat: curInCombat,
                        state: bot.player.state,
                        timeSinceCombatExitMs: curInCombat ? 0 : now - lastCombatExitTs,
                        hp: curHp,
                        maxHp: bot.player.maxHealth,
                        deltaHp,
                        mp: curMp,
                        maxMp: bot.player.maxMana,
                        deltaMp,
                        sp: curSp,
                        maxSp: 100,
                        deltaSp,
                    });
                    prevHp = curHp;
                    prevMp = curMp;
                    prevSp = curSp;
                    lastTickTs = now;
                }
            }
            bot.sleep(100);
        }
    } finally {
        unsubscribeExt();
        unsubscribeSent();
        bot.log("[Telemetry] Stopped recording.");
    }
}
