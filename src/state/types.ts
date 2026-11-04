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
    /** Stamina / Green Bar (`world.myAvatar.dataLeaf.intSP`, 0..100). */
    selfSp: number;
    selfMaxSp: number;
    selfSpRatio: number;
    /** True when `selfSp >= 50` (able to activate Space + Click Sprint/Dash). */
    canSprint: boolean;
    selfX: number;
    selfY: number;
    selfWalkSpeed: number;
    selfAlive: boolean;
    selfInCombat: boolean;
    /** True when under Stun, Off Balance, or movement-blocking CC (server locks position). */
    selfStunned: boolean;
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

export type AtomicMovementType = "STAND" | "WALK" | "SPRINT";

export type AtomicCombatAction =
    | "NONE"
    | "AUTO_ATTACK"
    | "USE_SKILL_1"
    | "USE_SKILL_2"
    | "USE_SKILL_3"
    | "USE_SKILL_4";

export interface AtomicMovementDecision {
    type: AtomicMovementType;
    /** Target 2D coordinates `(x, y)` in the current screen. */
    targetX: number;
    targetY: number;
    /** If transitioning screens, the destination cell and pad. */
    targetCell?: string;
    targetPad?: string;
}

export type CombatActionType =
    | "IDLE"
    | "CANCEL_AUTO_ATTACK"
    | "REST_OUT_OF_COMBAT"
    | "USE_SKILL_1"
    | "USE_SKILL_2"
    | "USE_SKILL_3"
    | "USE_SKILL_4"
    | "TARGET_OPPONENT"
    | "MANA_TAP_NPC"
    | "WALK_TO_POINT"
    | "SPRINT_TO_POINT"
    | "WALK_TO_DOOR"
    | "SPRINT_TO_DOOR"
    | "RETREAT_TO_SAFEZONE"
    | "ADVANCE_TO_COMBAT_CELL";

export interface CombatDecision {
    /** Primary combat action executed this tick (skills 1-4, AA, or None). */
    combatAction?: AtomicCombatAction;
    /** Primary movement decision executed this tick (STAND, WALK, SPRINT to target point). */
    movement?: AtomicMovementDecision;
    /** Legacy action for backward compatibility. */
    action?: CombatActionType;
    targetName?: string;
    npcMapId?: number;
    /** Target 2D coordinates `(x, y)` for in-screen movement or door approach. */
    targetX?: number;
    targetY?: number;
    /** Target neighboring screen/cell when walking or sprinting to a door. */
    targetCell?: string;
    /** Target pad in neighboring cell when transitioning through a door. */
    targetPad?: string;
    reason: string;
}
