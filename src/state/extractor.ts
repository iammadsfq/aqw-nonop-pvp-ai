/**
 * @file extractor.ts
 * @description Real-time state extractor bridging Turu's AVM2 memory API into a
 * structured CombatObservation vector for 1v1 and 3v3 PvP agents.
 */

import { bot, PlayerInfo } from "turu";
import {
    AQW_MAX_STAMINA,
    AQW_SPRINT_STAMINA_COST,
    BLUDRUT_SAFEZONE_CELLS,
    PROTECTED_NPC_KEYWORDS,
} from "../config/constants";
import type { CombatObservation, PlayerSnapshot, WaveTrackerState } from "./types";

function toPlayerSnapshot(info: PlayerInfo): PlayerSnapshot {
    const maxHp = Math.max(1, info.maxHp);
    const maxMp = 100;
    const auras = bot.self.getPlayerAura(info.name) ?? [];
    return {
        username: info.name,
        cell: info.cell,
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

export function extractObservation(
    allyNames: ReadonlySet<string>,
    wave: WaveTrackerState,
): CombatObservation {
    const selfName = bot.player.username;
    const selfMaxHp = Math.max(1, bot.player.maxHealth);
    const selfMaxMp = Math.max(1, bot.player.maxMana);
    const selfMaxSp = AQW_MAX_STAMINA;
    const selfHp = bot.player.health;
    const selfMp = bot.player.mana;
    const rawSp = bot.flash.get<number>("world.myAvatar.dataLeaf.intSP");
    const selfSp = typeof rawSp === "number" && !Number.isNaN(rawSp) ? rawSp : AQW_MAX_STAMINA;
    const currentCell = bot.player.cell;
    const currentPad = bot.player.pad;

    const allies: PlayerSnapshot[] = [];
    const opponents: PlayerSnapshot[] = [];

    const roomPlayerNames = bot.map.playerNames;
    for (const name of roomPlayerNames) {
        if (name.toLowerCase() === selfName.toLowerCase()) continue;
        const info = bot.map.getPlayer(name);
        if (!info) continue;
        const snap = toPlayerSnapshot(info);
        if (allyNames.has(name.toLowerCase())) {
            allies.push(snap);
        } else {
            opponents.push(snap);
        }
    }

    // If an alive opponent is in the same cell and we do not currently have them targeted
    // (Wait until we are outside Safezone), lock target onto the opponent player first so
    // `world.myAvatar.target` is populated and `bot.skills.canUseSkill(1..4)` evaluates accurately.
    const inSafezone = BLUDRUT_SAFEZONE_CELLS.has(currentCell);
    const oppInCell = opponents.find((p) => p.cell === currentCell && p.alive);

    let rawTargetType = bot.flash.get<string>("world.myAvatar.target.npcType");
    let rawTargetPlayerName =
        bot.flash.get<string>("world.myAvatar.target.objData.strUsername") ??
        bot.flash.get<string>("world.myAvatar.target.pnm");

    if (
        !inSafezone &&
        oppInCell &&
        (rawTargetType !== "player" ||
            !rawTargetPlayerName ||
            rawTargetPlayerName.toLowerCase() !== oppInCell.username.toLowerCase())
    ) {
        bot.combat.attackPlayer(oppInCell.username);
        rawTargetType = bot.flash.get<string>("world.myAvatar.target.npcType");
        rawTargetPlayerName =
            bot.flash.get<string>("world.myAvatar.target.objData.strUsername") ??
            bot.flash.get<string>("world.myAvatar.target.pnm");
    }

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
