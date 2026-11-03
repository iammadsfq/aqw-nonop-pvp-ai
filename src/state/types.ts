/**
 * @file types.ts
 * @description State, Observation Space, and Action Space interfaces for the
 * Non-OP PvP AI agent (supporting both 1v1 duel and 3v3 multi-agent modes).
 */

import type { Aura, AuraSnapshot, Monster, PlayerStats } from "turu";

export interface PlayerSnapshot {
    username: string;
    cell: string;
    pad: string;
    x: number;
    y: number;
    hp: number;
    maxHp: number;
    hpRatio: number;
    mp: number;
    maxMp: number;
    mpRatio: number;
    alive: boolean;
    auras: Aura[];
}

export interface WaveTrackerState {
    waveNumber: number;
    waveActive: boolean;
    opponentKilledInWave: boolean;
    teammateDiedInWave: boolean;
    canReturnToSafezone: boolean;
    teamKills: number;
    teamDeaths: number;
}

export interface CombatObservation {
    timestamp: number;
    mapName: string;
    currentCell: string;
    currentPad: string;
    inSafezone: boolean;

    // Self state
    selfName: string;
    selfClass: string;
    selfHp: number;
    selfMaxHp: number;
    selfHpRatio: number;
    selfMp: number;
    selfMaxMp: number;
    selfMpRatio: number;
    selfX: number;
    selfY: number;
    selfAlive: boolean;
    selfInCombat: boolean;
    selfStats: PlayerStats | null;
    selfAuras: AuraSnapshot[];
    skillsReady: [boolean, boolean, boolean, boolean];

    // Current target & nearby entities
    hasTarget: boolean;
    targetIsNpc: boolean;
    targetName: string | null;
    targetAuras: AuraSnapshot[];
    allies: PlayerSnapshot[];
    opponents: PlayerSnapshot[];
    cellMonsters: Monster[];

    // Wave & Rule state
    wave: WaveTrackerState;
}

export type CombatActionType =
    | "IDLE"
    | "CANCEL_AUTO_ATTACK"
    | "USE_SKILL_1"
    | "USE_SKILL_2"
    | "USE_SKILL_3"
    | "USE_SKILL_4"
    | "TARGET_OPPONENT"
    | "MANA_TAP_NPC"
    | "RETREAT_TO_SAFEZONE"
    | "ADVANCE_TO_COMBAT_CELL";

export interface CombatDecision {
    action: CombatActionType;
    targetName?: string;
    npcMapId?: number;
    reason: string;
}
