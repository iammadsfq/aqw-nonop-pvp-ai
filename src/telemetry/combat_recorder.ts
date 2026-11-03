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
    type: "class_spec" | "combat_trace" | "player_sent_action" | "map_snapshot";
    className: string | null;
    map: string;
    cell: string;
    payload: unknown;
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

    // 2. Hook into outgoing client packets (skill activations, rest requests, target switches)
    const unsubscribeSent = bot.events.on("packetSent", (rawPacket: string) => {
        if (rawPacket.includes("%gar%") || rawPacket.includes("%restRequest%") || rawPacket.includes("%aggroMon%")) {
            writeTelemetry("player_sent_action", {
                packet: rawPacket,
                selfHp: bot.player.health,
                selfMp: bot.player.mana,
                selfAuras: bot.self.snapshots,
                targetAuras: bot.target.snapshots,
            });
        }
    });

    bot.log("[Telemetry] Packet hooks active! Recording to ~/.local/share/Turu/army/aqw_combat_telemetry.log");

    try {
        while (!bot.shouldExit) {
            if (bot.player.loggedIn && bot.player.loaded) {
                const currentClass = bot.player.currentClass ?? "";
                if (currentClass && currentClass !== lastRecordedClass) {
                    recordCurrentClassSpecification();
                    lastRecordedClass = currentClass;
                }
            }
            bot.sleep(250);
        }
    } finally {
        unsubscribeExt();
        unsubscribeSent();
        bot.log("[Telemetry] Stopped recording.");
    }
}
