/**
 * @file extractor.ts
 * @description Real-time state extractor bridging Turu's AVM2 memory API into a
 * structured CombatObservation vector for 1v1 and 3v3 PvP agents.
 */

import { bot, PlayerInfo } from "turu";
import {
    AQW_MAX_STAMINA,
    AQW_SPRINT_STAMINA_COST,
    BLUDRUT_16_SCREENS,
    BLUDRUT_SAFEZONE_CELLS,
    PROTECTED_NPC_KEYWORDS,
} from "../config/constants";
import type { CombatObservation, PlayerSnapshot, WaveTrackerState } from "./types";

function toPlayerSnapshot(info: PlayerInfo, currentCellCanonical: string, cellPlayersSet: Set<string>): PlayerSnapshot {
    const maxHp = Math.max(1, info.maxHp);
    const maxMp = 100;
    const auras = bot.self.getPlayerAura(info.name) ?? [];

    const canonicalScreen = BLUDRUT_16_SCREENS.find(
        (s) => s.cell.toLowerCase() === info.cell?.toLowerCase(),
    );
    let cell = canonicalScreen ? canonicalScreen.cell : (info.cell ?? "");

    // Ground truth from Flash stage: if player is in cellPlayers, they are in our current cell!
    if (cellPlayersSet.has(info.name.toLowerCase())) {
        cell = currentCellCanonical;
    }

    return {
        username: info.name,
        cell,
        pad: info.pad,
        x: info.x,
        y: info.y,
        hp: info.hp,
        maxHp,
        hpRatio: info.hp / maxHp,
        mp: info.mp,
        maxMp,
        mpRatio: info.mp / maxMp,
        alive: info.hp > 0 && info.state > 0,
        auras,
    };
}

export function isProtectedNpcName(name: string | null | undefined): boolean {
    if (!name) return false;
    const lower = name.toLowerCase();
    return PROTECTED_NPC_KEYWORDS.some((kw) => lower.includes(kw));
}

function checkSelfStunned(): boolean {
    // 1. Direct ActionScript game function
    try {
        if (bot.flash.callGameFunction<boolean>("world.auraMoveBlocked")) {
            return true;
        }
    } catch (_) {}

    // 2. Avatar MC / dataLeaf stun flags
    try {
        if (
            bot.flash.get<boolean>("world.myAvatar.pMC.isStunned") ||
            bot.flash.get<boolean>("world.myAvatar.dataLeaf.isStunned")
        ) {
            return true;
        }
    } catch (_) {}

    // 3. Check bot.self.auras (rich Aura objects with category and type)
    try {
        const activeAuras = bot.self.auras;
        for (const a of activeAuras) {
            const cat = (a.category ?? "").toLowerCase();
            const typ = (a.type ?? "").toLowerCase();
            const nm = (a.name ?? "").toLowerCase();
            if (
                cat === "stun" ||
                cat === "disabled" ||
                cat === "lock" ||
                typ === "stun" ||
                typ === "disabled" ||
                /off balance|stun|stone|paralyze|disabled|petrif|frozen|sleep|root/i.test(nm)
            ) {
                return true;
            }
        }
    } catch (_) {}

    // 4. Check HUD aura snapshots
    try {
        const snapshots = bot.self.snapshots;
        for (const s of snapshots) {
            if (/off balance|stun|stone|paralyze|disabled|petrif|frozen|sleep|root/i.test(s.name)) {
                return true;
            }
        }
    } catch (_) {}

    return false;
}

export function extractObservation(
    allyNames: ReadonlySet<string>,
    wave: WaveTrackerState,
    selfTeam?: "blue" | "red",
): CombatObservation {
    const selfName = bot.player.username;
    const selfMaxHp = Math.max(1, bot.player.maxHealth);
    const selfMaxMp = Math.max(1, bot.player.maxMana);
    const selfMaxSp = AQW_MAX_STAMINA;
    const selfHp = bot.player.health;
    const selfMp = bot.player.mana;
    const rawSp = bot.flash.get<number>("world.myAvatar.dataLeaf.intSP");
    const selfSp = typeof rawSp === "number" && !Number.isNaN(rawSp) ? rawSp : AQW_MAX_STAMINA;
    const canonicalCurrent = BLUDRUT_16_SCREENS.find(
        (s) => s.cell.toLowerCase() === bot.player.cell?.toLowerCase(),
    );
    const currentCell = canonicalCurrent ? canonicalCurrent.cell : bot.player.cell;
    const currentPad = bot.player.pad;

    const cellPlayersSet = new Set(
        (bot.map.cellPlayers ?? []).map((n) => n.toLowerCase()),
    );

    const allies: PlayerSnapshot[] = [];
    const opponents: PlayerSnapshot[] = [];

    const roomPlayerNames = bot.map.playerNames;
    for (const name of roomPlayerNames) {
        if (name.toLowerCase() === selfName.toLowerCase()) continue;
        const info = bot.map.getPlayer(name);
        if (!info) continue;
        const snap = toPlayerSnapshot(info, currentCell, cellPlayersSet);
        if (allyNames.has(name.toLowerCase())) {
            allies.push(snap);
        } else {
            opponents.push(snap);
        }
    }

    const inSafezone = BLUDRUT_SAFEZONE_CELLS.has(currentCell);
    const oppInCell = opponents.find(
        (p) =>
            (p.cell.toLowerCase() === currentCell.toLowerCase() ||
                cellPlayersSet.has(p.username.toLowerCase())) &&
            p.alive,
    );

    const rawTargetType = bot.flash.get<string>("world.myAvatar.target.npcType");
    const rawTargetPlayerName =
        bot.flash.get<string>("world.myAvatar.target.objData.strUsername") ??
        bot.flash.get<string>("world.myAvatar.target.pnm");

    const rawPvpTeam =
        bot.flash.get<number>("world.myAvatar.objData.pvpTeam") ??
        bot.flash.get<number>("world.myAvatar.pvpTeam");
    const detectedTeam: "blue" | "red" =
        selfTeam ??
        (rawPvpTeam === 1 ? "red" : (rawPvpTeam === 0 ? "blue" : (currentCell.includes("1") ? "red" : "blue")));

    const isPlayerTarget = rawTargetType === "player" && Boolean(rawTargetPlayerName);
    const monsterTarget = bot.player.target;
    const hasTarget = isPlayerTarget || bot.player.hasTarget;
    const targetName = isPlayerTarget ? rawTargetPlayerName! : (monsterTarget?.name ?? null);
    const targetIsNpc = !isPlayerTarget && isProtectedNpcName(targetName);

    return {
        timestamp: Date.now(),
        mapName: bot.map.name,
        currentCell,
        currentPad,
        inSafezone,
        selfTeam: detectedTeam,

        selfName,
        selfClass: bot.player.currentClass ?? "Unknown",
        selfHp,
        selfMaxHp,
        selfHpRatio: selfHp / selfMaxHp,
        selfMp,
        selfMaxMp,
        selfMpRatio: selfMp / selfMaxMp,
        selfSp,
        selfMaxSp,
        selfSpRatio: selfSp / selfMaxSp,
        canSprint: selfSp >= AQW_SPRINT_STAMINA_COST,
        selfX: bot.player.x,
        selfY: bot.player.y,
        selfWalkSpeed: bot.player.walkSpeed,
        selfAlive: bot.player.alive,
        selfInCombat: bot.player.inCombat,
        selfStunned: checkSelfStunned(),
        selfStats: bot.player.stats,
        selfAuras: bot.self.snapshots,
        skillsReady: [
            bot.skills.canUseSkill(1),
            bot.skills.canUseSkill(2),
            bot.skills.canUseSkill(3),
            bot.skills.canUseSkill(4),
        ],

        hasTarget,
        targetIsNpc,
        targetName,
        targetAuras: bot.target.snapshots,
        allies,
        opponents,
        cellMonsters: bot.monsters.currentMonsters,

        wave: { ...wave },
    };
}
