/**
 * @file preflight.ts
 * @description Pre-match validator that verifies full compliance with the
 * Classic (Non-OP) PvP ruleset before the bot engages in combat.
 */

import { bot, InventoryItem } from "turu";
import {
    ALLOWED_PVP_AMULETS,
    EXCEPTIONAL_3V3_4V4_CLASSES,
    FORBIDDEN_ENHANCEMENT_PATTERN_IDS,
    STANDARD_NON_OP_CLASSES,
} from "../config/constants";
import type { MatchRulesConfig } from "../config/match_config";

export interface PreflightReport {
    passed: boolean;
    violations: string[];
    equippedClass: string;
    equippedAmulet: string | null;
}

export function runNonOpPreflightCheck(config: MatchRulesConfig): PreflightReport {
    const violations: string[] = [];

    // 1. Enforce Static Weapon Range (disable infiniteRange & illegal combat cheats)
    bot.options.infiniteRange = false;
    bot.options.aggroMonsters = false;
    bot.options.aggroAllMonsters = false;
    bot.options.attackWithoutTarget = false;

    // 2. Validate Equipped Class
    const currentClass = (bot.player.currentClass ?? "").trim();
    const isStandard = STANDARD_NON_OP_CLASSES.some(
        (c) => c.toLowerCase() === currentClass.toLowerCase(),
    );
    const isExceptional = EXCEPTIONAL_3V3_4V4_CLASSES.some(
        (c) => c.toLowerCase() === currentClass.toLowerCase(),
    );

    if (!isStandard && !(config.allowBattleHealer && isExceptional)) {
        violations.push(
            `Illegal class "${currentClass}" for format ${config.format}. Allowed: Standard Non-OP classes${config.allowBattleHealer ? " + Battle Healer" : ""}.`,
        );
    }

    // 3. Inspect Equipped Inventory Items (Amulet, Enhancements, Potions/Scrolls)
    const equippedItems: InventoryItem[] = bot.inventory.items.filter((i) => i.equipped);
    let equippedAmulet: string | null = null;

    for (const item of equippedItems) {
        // Check for forbidden Awe / Forge enhancements
        if (FORBIDDEN_ENHANCEMENT_PATTERN_IDS.has(item.enhancementPatternId)) {
            violations.push(
                `Forbidden special enhancement (Awe/Forge patternId=${item.enhancementPatternId}) on equipped item "${item.name}".`,
            );
        }

        // Check for PvP Amulet (+5000 or +5500)
        if (item.category.toLowerCase() === "amulet" || item.name.toLowerCase().includes("amulet")) {
            equippedAmulet = item.name;
            const validAmulet = ALLOWED_PVP_AMULETS.some((allowed) =>
                item.name.toLowerCase().includes(allowed.toLowerCase()),
            );
            if (!validAmulet) {
                violations.push(
                    `Invalid PvP Amulet "${item.name}". Required: Platinum PvP Amulet +5000 or Diamond PvP Amulet +5500.`,
                );
            }
        }
    }

    if (!equippedAmulet) {
        violations.push(
            "No PvP Amulet equipped. Required: Platinum PvP Amulet +5000 or Diamond PvP Amulet +5500.",
        );
    }

    return {
        passed: violations.length === 0,
        violations,
        equippedClass: currentClass,
        equippedAmulet,
    };
}
