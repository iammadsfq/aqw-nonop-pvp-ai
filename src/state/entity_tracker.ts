// src/state/entity_tracker.ts
import { bot, PlayerInfo } from "turu";

export interface CanonicalStats {
    inDmgMult: number;     // Multiplier damage masuk (0.4 s/d 0.5 = shield, 1.0 = normal)
    outDmgMult: number;    // Multiplier damage keluar (1.15 = buff, 0.7 = debuff, 1.0 = normal)
    manaCostMult: number;  // Multiplier konsumsi mana (0.5 = hemat, 1.0 = normal)
    critRate: number;      // Peluang crit
    dodgeRate: number;     // Peluang dodge
    haste: number;         // Cooldown reduction
}

export interface TrackedEntity {
    username: string;
    entId: number;
    team: "blue" | "red";
    isAlly: boolean;
    cell: string;
    x: number;
    y: number;

    currentHp: number;
    maxHp: number;

    hp_norm: number;       // currentHp / maxHp [0.0, 1.0]
    mp_norm: number;       // mp / 100.0 [0.0, 1.0]
    sp_norm: number;       // sp / 100.0 [0.0, 1.0]
    alive: boolean;

    stats: CanonicalStats;
    hardCcRem: number;     // Sisa durasi stun (detik)
    isStunned: boolean;
    roleCategory: number;  // 0 = DPS/Warrior, 1 = Healer, 2 = Bard/Support

    timeSinceLastAction: number;
    lastSeenTs: number;
}

interface AuraMeta {
    cat?: string;
    inDmgMult?: number;
    outDmgMult?: number;
}

const STATIC_AURA_CATALOG: Record<string, AuraMeta> = {
    "on guard": { inDmgMult: 0.50 },
    "inhibition": { outDmgMult: 0.70 },
    "off balance": { cat: "stun" },
};

export class EntityTracker {
    private entities = new Map<string, TrackedEntity>();
    private entIdToName = new Map<number, string>();
    private myUsername: string = "";

    private selfLocalStats: CanonicalStats = {
        inDmgMult: 1.0,
        outDmgMult: 1.0,
        manaCostMult: 1.0,
        critRate: 0.20,
        dodgeRate: 0.20,
        haste: 0.10,
    };

    constructor() {
        this.initListeners();
    }

    private initListeners(): void {
        // Cukup gunakan "extensionPacket" untuk menangkap paket server (baik JSON maupun string xt)
        bot.events.on("extensionPacket", (pkt: any) => {
            try {
                // Handle jika paket berupa string raw xt (contoh: %xt%uotls%...)
                if (typeof pkt === "string") {
                    if (pkt.includes("%uotls%")) {
                        this.parseRawUotls(pkt);
                    }
                    return;
                }

                const params = pkt?.params;
                if (params?.type === "str" && typeof params?.dataObj === "string") {
                    if (params.dataObj.includes("%uotls%")) {
                        this.parseRawUotls(params.dataObj);
                    }
                    return;
                }

                // Handle dataObj JSON
                const dataObj = params?.dataObj ?? pkt;
                if (!dataObj || typeof dataObj !== "object") return;
                const cmd = dataObj.cmd as string | undefined;

                if (!this.myUsername && bot.player.loggedIn) {
                    this.myUsername = bot.player.username.toLowerCase();
                }

                // Update mapping entId -> username
                for (const name of bot.map.playerNames) {
                    const p: PlayerInfo | null = bot.map.getPlayer(name);
                    if (p?.entId) {
                        this.entIdToName.set(p.entId, name.toLowerCase());
                    }
                }

                // 1. Channel 1: Stat diri sendiri dari paket stu
                if (cmd === "stu" && dataObj.sta) {
                    const s = dataObj.sta;
                    if (typeof s["$cai"] === "number") this.selfLocalStats.inDmgMult = s["$cai"];
                    if (typeof s["$cdi"] === "number") this.selfLocalStats.inDmgMult = Math.min(this.selfLocalStats.inDmgMult, s["$cdi"]);
                    if (typeof s["$cao"] === "number") this.selfLocalStats.outDmgMult = s["$cao"];
                    if (typeof s["$cmo"] === "number") this.selfLocalStats.outDmgMult = Math.max(this.selfLocalStats.outDmgMult, s["$cmo"]);
                    if (typeof s["$cmc"] === "number") this.selfLocalStats.manaCostMult = s["$cmc"];
                    if (typeof s["$tcr"] === "number") this.selfLocalStats.critRate = s["$tcr"];
                    if (typeof s["$tdo"] === "number") this.selfLocalStats.dodgeRate = s["$tdo"];
                    if (typeof s["$tha"] === "number") this.selfLocalStats.haste = s["$tha"];
                }

                // 2. Channel 2: Combat trace global (ct)
                if (cmd === "ct") {
                    // Update status HP / MP / SP entitas dari payload p
                    if (dataObj.p && typeof dataObj.p === "object") {
                        for (const [uName, vObj] of Object.entries<any>(dataObj.p)) {
                            const ent = this.getOrCreateEntity(uName.toLowerCase());
                            if (typeof vObj.intHPMax === "number" && vObj.intHPMax > 0) {
                                ent.maxHp = vObj.intHPMax;
                            }
                            if (typeof vObj.intHP === "number") {
                                ent.currentHp = vObj.intHP;
                                ent.hp_norm = Math.min(1.0, Math.max(0.0, ent.currentHp / ent.maxHp));
                                ent.alive = ent.currentHp > 0;
                            }
                            if (typeof vObj.intMP === "number") {
                                ent.mp_norm = Math.min(1.0, Math.max(0.0, vObj.intMP / 100.0));
                            }
                            if (typeof vObj.intSP === "number") {
                                ent.sp_norm = Math.min(1.0, Math.max(0.0, vObj.intSP / 100.0));
                            }
                            if (typeof vObj.intState === "number") {
                                ent.alive = vObj.intState !== 0;
                            }
                        }
                    }

                    // Catat timestamp aksi tempur terakhir
                    const sarsa = Array.isArray(dataObj.sarsa) ? dataObj.sarsa : [];
                    for (const s of sarsa) {
                        const caster = this.resolveTargetName(String(s.cInf ?? ""));
                        if (caster) {
                            const ent = this.getOrCreateEntity(caster);
                            ent.timeSinceLastAction = 0.0;
                        }
                    }

                    // Parse buff / debuff
                    const auras = Array.isArray(dataObj.a) ? dataObj.a : [];
                    for (const aEntry of auras) {
                        const isRemove = aEntry.cmd === "aura-";
                        const target = this.resolveTargetName(String(aEntry.tInf ?? ""));
                        if (!target) continue;

                        const ent = this.getOrCreateEntity(target);
                        const auraList = Array.isArray(aEntry.auras) ? aEntry.auras : (aEntry.aura ? [aEntry.aura] : []);

                        for (const a of auraList) {
                            const cat = String(a.cat ?? "").toLowerCase();
                            const nam = String(a.nam ?? "").toLowerCase();
                            const val = typeof a.val === "number" ? a.val : null;
                            const dur = typeof a.dur === "number" ? a.dur : 4.0;
                            const meta = STATIC_AURA_CATALOG[nam];

                            if (cat === "stun" || meta?.cat === "stun") {
                                ent.hardCcRem = isRemove ? 0.0 : dur;
                                ent.isStunned = !isRemove;
                            }

                            if (val !== null) {
                                ent.stats.outDmgMult = isRemove ? 1.0 : val;
                            } else if (meta?.outDmgMult !== undefined) {
                                ent.stats.outDmgMult = isRemove ? 1.0 : meta.outDmgMult;
                            }

                            if (meta?.inDmgMult !== undefined) {
                                ent.stats.inDmgMult = isRemove ? 1.0 : meta.inDmgMult;
                            }
                        }
                    }
                }

                // 3. Deteksi kematian
                if (cmd === "playerDeath" || (cmd === "uotls" && dataObj.o?.intHP === 0)) {
                    const uName = dataObj.unm ? String(dataObj.unm).toLowerCase() : null;
                    if (uName) {
                        const ent = this.getOrCreateEntity(uName);
                        ent.currentHp = 0;
                        ent.hp_norm = 0.0;
                        ent.alive = false;
                        ent.hardCcRem = 0.0;
                        ent.isStunned = false;
                    }
                }

                // 4. Update respawn / navigasi JSON
                if (cmd === "uotls" && dataObj.unm && dataObj.o) {
                    const ent = this.getOrCreateEntity(String(dataObj.unm).toLowerCase());
                    const o = dataObj.o;
                    if (o.strFrame) ent.cell = o.strFrame;
                    if (typeof o.intHPMax === "number" && o.intHPMax > 0) ent.maxHp = o.intHPMax;
                    if (typeof o.intHP === "number") {
                        ent.currentHp = o.intHP;
                        ent.hp_norm = Math.min(1.0, Math.max(0.0, ent.currentHp / ent.maxHp));
                        ent.alive = ent.currentHp > 0;
                    }
                    if (typeof o.intState === "number") ent.alive = o.intState !== 0;
                }
            } catch (err) {
                // Ignore parse errors
            }
        });
    }

    private parseRawUotls(raw: string): void {
        const parts = raw.split("%");
        if (parts.length < 5) return;
        const uName = parts[3].toLowerCase();
        const payload = parts[4];
        const ent = this.getOrCreateEntity(uName);

        const kvPairs = payload.split(",");
        for (const pair of kvPairs) {
            const [k, v] = pair.split(":");
            if (!k || !v) continue;
            if (k === "strFrame") ent.cell = v;
            if (k === "px") ent.x = Number(v);
            if (k === "py") ent.y = Number(v);
            if (k === "intHP") {
                ent.currentHp = Number(v);
                ent.hp_norm = Math.min(1.0, Math.max(0.0, ent.currentHp / ent.maxHp));
                ent.alive = ent.currentHp > 0;
            }
            if (k === "intSP") {
                ent.sp_norm = Math.min(1.0, Math.max(0.0, Number(v) / 100.0));
            }
        }
    }

    private resolveTargetName(tInf: string): string | null {
        if (!tInf) return null;
        if (tInf.startsWith("p:")) {
            const entId = Number(tInf.slice(2));
            return this.entIdToName.get(entId) ?? null;
        }
        return tInf.toLowerCase();
    }

    private getOrCreateEntity(uName: string): TrackedEntity {
        let ent = this.entities.get(uName);
        if (!ent) {
            ent = {
                username: uName,
                entId: 0,
                team: "red",
                isAlly: false,
                cell: "Enter1",
                x: 0,
                y: 0,
                currentHp: 8455,
                maxHp: 8455,
                hp_norm: 1.0,
                mp_norm: 1.0,
                sp_norm: 1.0,
                alive: true,
                stats: {
                    inDmgMult: 1.0,
                    outDmgMult: 1.0,
                    manaCostMult: 1.0,
                    critRate: 0.25,
                    dodgeRate: 0.20,
                    haste: 0.13,
                },
                hardCcRem: 0.0,
                isStunned: false,
                roleCategory: 0,
                timeSinceLastAction: 0.0,
                lastSeenTs: Date.now(),
            };
            this.entities.set(uName, ent);
        }
        return ent;
    }

    /**
     * Memperbarui status setiap tick frame (100ms)
     */
    public tick(dtSec: number = 0.10): void {
        const myCell = bot.player.cell;
        const myTeam: "blue" | "red" = myCell.includes("0") ? "blue" : "red";

        for (const [uName, ent] of this.entities.entries()) {
            if (uName === this.myUsername) {
                ent.stats = { ...this.selfLocalStats };
                ent.cell = myCell;
                ent.x = bot.player.x;
                ent.y = bot.player.y;

                // Menggunakan getter resmi Turu: health dan maxHealth
                ent.currentHp = bot.player.health;
                ent.maxHp = bot.player.maxHealth > 0 ? bot.player.maxHealth : 8455;
                ent.hp_norm = Math.min(1.0, Math.max(0.0, ent.currentHp / ent.maxHp));
                ent.mp_norm = Math.min(1.0, Math.max(0.0, bot.player.mana / 100.0));
                ent.alive = bot.player.alive;
                ent.isAlly = true;
                ent.team = myTeam;
            } else {
                const targetTeam: "blue" | "red" = ent.cell.includes("0") ? "blue" : "red";
                ent.team = targetTeam;
                ent.isAlly = targetTeam === myTeam;

                if (ent.hardCcRem > 0.0) {
                    ent.hardCcRem = Math.max(0.0, ent.hardCcRem - dtSec);
                    ent.isStunned = ent.hardCcRem > 0.0;
                }
                ent.timeSinceLastAction += dtSec;
            }
        }
    }

    public getSelf(): TrackedEntity {
        return this.getOrCreateEntity(this.myUsername || bot.player.username.toLowerCase());
    }

    public getEnemies(): TrackedEntity[] {
        return Array.from(this.entities.values()).filter((e) => !e.isAlly && e.username !== this.myUsername);
    }

    public getAllies(): TrackedEntity[] {
        return Array.from(this.entities.values()).filter((e) => e.isAlly && e.username !== this.myUsername);
    }

    public getPrimaryOpponent(): TrackedEntity | null {
        const enemies = this.getEnemies();
        if (enemies.length === 0) return null;
        const myCell = bot.player.cell;
        const inRoom = enemies.find((e) => e.cell === myCell && e.alive);
        return inRoom ?? enemies[0];
    }
}