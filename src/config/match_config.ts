/**
 * @file match_config.ts
 * @description Match format definitions (1v1, 2v2, 3v3, 4v4) and kill targets
 * according to the Classic (Non-OP) PvP ruleset.
 */

export type MatchFormat = "1v1" | "2v2" | "3v3" | "4v4";

export interface MatchRulesConfig {
    format: MatchFormat;
    teamSize: number;
    killTarget: number;
    allowBattleHealer: boolean;
    /**
     * In 1v1, returning to Safezone is only allowed after killing your opponent.
     * In 2v2+, returning to Safezone is allowed once any teammate or opponent dies in the wave.
     */
    szReturnPolicy: "on_opponent_kill_only" | "on_any_wave_death";
    mapName: string;
    roomNumber?: number;
    combatCell: string;
    safezoneCell: string;
}

export const KILL_TARGETS_BY_FORMAT: Readonly<Record<MatchFormat, number>> = {
    "1v1": 10,
    "2v2": 20,
    "3v3": 30,
    "4v4": 40,
};

export function createMatchConfig(
    format: MatchFormat,
    options?: Partial<Pick<MatchRulesConfig, "mapName" | "roomNumber" | "combatCell" | "safezoneCell">>,
): MatchRulesConfig {
    const teamSize = Number(format.charAt(0));
    return {
        format,
        teamSize,
        killTarget: KILL_TARGETS_BY_FORMAT[format],
        allowBattleHealer: format === "3v3" || format === "4v4",
        szReturnPolicy: format === "1v1" ? "on_opponent_kill_only" : "on_any_wave_death",
        mapName: options?.mapName ?? "bludrutbrawl",
        roomNumber: options?.roomNumber,
        combatCell: options?.combatCell ?? "Morale0C",
        safezoneCell: options?.safezoneCell ?? "Enter0",
    };
}
