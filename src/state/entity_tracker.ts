// src/state/entity_tracker.ts
import { bot, PlayerInfo } from "turu";

export interface CanonicalStats {
  inDmgMult: number;     // Multiplier damage masuk (<1.0 mitigasi/shield, >1.0 vulnerability)
  outDmgMult: number;    // Multiplier damage keluar (>1.0 buff/boost, <1.0 debuff)
  manaCostMult: number;  // Multiplier konsumsi mana (1.0 = normal)
  critRate: number;      // Peluang crit
  dodgeRate: number;     // Peluang dodge
  haste: number;         // Cooldown reduction
}

export interface ActiveAura {
  name: string;
  cat?: string;
  durationRem: number;
  inDmgMult?: number;
  outDmgMult?: number;
}

export interface TrackedEntity {
  username: string;
  entId: number;
  team: "blue" | "red" | "unknown";
  pvpTeam: number;       // -1: Unknown, 0: Blue (Team A), 1: Red (Team B)
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
  activeAuras: Map<string, ActiveAura>;
  hardCcRem: number;     // Sisa durasi hard CC / stun (detik)
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
  private myPvpTeam: number = -1; // -1: Unknown, 0: Blue, 1: Red
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

  // ==========================================
  // DISPATCHER & LISTENER SETUP
  // ==========================================

  private initListeners(): void {
    bot.events.on("extensionPacket", (pkt: any) => {
      try {
        if (typeof pkt === "string") {
          this.dispatchRawPacket(pkt);
          return;
        }

        const params = pkt?.params;
        if (params?.type === "str" && typeof params?.dataObj === "string") {
          this.dispatchRawPacket(params.dataObj);
          return;
        }

        const dataObj = params?.dataObj ?? pkt;
        if (!dataObj || typeof dataObj !== "object") return;

        this.syncPlayerIdentity();

        const cmd = dataObj.cmd as string | undefined;
        if (!cmd) return;

        switch (cmd) {
          case "moveToArea":
            this.handleMoveToArea(dataObj);
            break;
          case "initUserData":
            this.handleInitUserData(dataObj);
            break;
          case "stu":
            this.handleStu(dataObj);
            break;
          case "ct":
            this.handleCt(dataObj);
            break;
          case "uotls":
            this.handleUotlsObj(dataObj);
            break;
        }
      } catch (_) {}
    });
  }

  private dispatchRawPacket(raw: string): void {
    if (raw.startsWith("%xt%exitArea%")) {
      this.handleExitAreaRaw(raw);
    } else if (raw.includes("%uotls%")) {
      this.parseRawUotls(raw);
    }
  }

  private syncPlayerIdentity(): void {
    if (!this.myUsername && bot.player?.loggedIn) {
      this.myUsername = bot.player.username.toLowerCase();
    }

    if (bot.map?.playerNames) {
      for (const name of bot.map.playerNames) {
        const p: PlayerInfo | null = bot.map.getPlayer(name);
        if (p?.entId) {
          this.entIdToName.set(p.entId, name.toLowerCase());
        }
      }
    }
  }

  // ==========================================
  // COMMAND HANDLERS
  // ==========================================

  private handleExitAreaRaw(raw: string): void {
    const parts = raw.split("%");
    const entId = Number(parts[4]);
    const uName = parts[5]?.toLowerCase();

    if (uName) this.entities.delete(uName);
    if (!isNaN(entId)) this.entIdToName.delete(entId);
  }

  private handleMoveToArea(dataObj: any): void {
    this.entities.clear();
    this.entIdToName.clear();

    if (typeof dataObj.pvpTeam === "number") {
      this.myPvpTeam = dataObj.pvpTeam;
      const selfEnt = this.getOrCreateEntity(this.myUsername || bot.player.username.toLowerCase());
      this.setEntityPvpTeam(selfEnt, this.myPvpTeam);
    }

    if (Array.isArray(dataObj.uoBranch)) {
      for (const uo of dataObj.uoBranch) {
        const uName = (uo.uoName || uo.strUsername || "").toLowerCase();
        if (!uName) continue;

        const ent = this.getOrCreateEntity(uName);
        ent.lastSeenTs = Date.now();

        if (uo.entID) {
          ent.entId = uo.entID;
          this.entIdToName.set(uo.entID, uName);
        }
        if (typeof uo.pvpTeam === "number") {
          this.setEntityPvpTeam(ent, uo.pvpTeam);
        }
        if (uo.strFrame) ent.cell = uo.strFrame;
        if (uo.strClassName) ent.roleCategory = this.resolveRoleCategory(uo.strClassName);
        if (typeof uo.intHPMax === "number" && uo.intHPMax > 0) ent.maxHp = uo.intHPMax;
        if (typeof uo.intHP === "number") {
          ent.currentHp = uo.intHP;
          ent.hp_norm = ent.maxHp > 0 ? Math.min(1.0, Math.max(0.0, ent.currentHp / ent.maxHp)) : 0.0;
          ent.alive = ent.currentHp > 0;
        }
        if (typeof uo.intMP === "number") {
          ent.mp_norm = Math.min(1.0, Math.max(0.0, uo.intMP / 100.0));
        }
        if (typeof uo.intState === "number") {
          ent.alive = uo.intState !== 0;
        }
      }
    }

    this.refreshAllies();
  }

  private handleInitUserData(dataObj: any): void {
    const data = dataObj.data;
    if (!data) return;

    const uName = (data.strUsername || "").toLowerCase();
    if (!uName) return;

    const ent = this.getOrCreateEntity(uName);
    if (dataObj.uid) {
      ent.entId = Number(dataObj.uid);
      this.entIdToName.set(ent.entId, uName);
    }
    if (data.strClassName) {
      ent.roleCategory = this.resolveRoleCategory(data.strClassName);
    }
  }

  private handleStu(dataObj: any): void {
    if (!dataObj.sta) return;
    const s = dataObj.sta;
    if (typeof s["$cai"] === "number") this.selfLocalStats.inDmgMult = s["$cai"];
    if (typeof s["$cdi"] === "number") this.selfLocalStats.inDmgMult = this.selfLocalStats.inDmgMult, s["$cdi"];
    if (typeof s["$cao"] === "number") this.selfLocalStats.outDmgMult = s["$cao"];
    if (typeof s["$cmo"] === "number") this.selfLocalStats.outDmgMult = Math.max(this.selfLocalStats.outDmgMult, s["$cmo"]);
    if (typeof s["$cmc"] === "number") this.selfLocalStats.manaCostMult = s["$cmc"];
    if (typeof s["$tcr"] === "number") this.selfLocalStats.critRate = s["$tcr"];
    if (typeof s["$tdo"] === "number") this.selfLocalStats.dodgeRate = s["$tdo"];
    if (typeof s["$tha"] === "number") this.selfLocalStats.haste = s["$tha"];

    // Update langsung ke self entity
    const selfEnt = this.entities.get(this.myUsername);
    if (selfEnt) {
      selfEnt.stats.inDmgMult = this.selfLocalStats.inDmgMult;
      selfEnt.stats.outDmgMult = this.selfLocalStats.outDmgMult;
      selfEnt.stats.manaCostMult = this.selfLocalStats.manaCostMult;
      selfEnt.stats.critRate = this.selfLocalStats.critRate;
      selfEnt.stats.dodgeRate = this.selfLocalStats.dodgeRate;
      selfEnt.stats.haste = this.selfLocalStats.haste;
    }
  }

  private handleCt(dataObj: any): void {
    // 1. Resource & HP update
    if (dataObj.p && typeof dataObj.p === "object") {
      for (const [uName, vObj] of Object.entries<any>(dataObj.p)) {
        if (this.isMonsterIdentifier(uName)) continue;

        const ent = this.getOrCreateEntity(uName.toLowerCase());
        ent.lastSeenTs = Date.now();

        if (typeof vObj.intHPMax === "number" && vObj.intHPMax > 0) {
          ent.maxHp = vObj.intHPMax;
        }
        if (typeof vObj.intHP === "number") {
          ent.currentHp = Math.max(0, vObj.intHP);
          ent.hp_norm = ent.maxHp > 0 ? Math.min(1.0, Math.max(0.0, ent.currentHp / ent.maxHp)) : 0.0;
          ent.alive = ent.currentHp > 0;
        }

        if (typeof vObj.intState === "number") {
          ent.alive = vObj.intState !== 0;
        }

        if (!ent.alive) {
          ent.currentHp = 0;
          ent.hp_norm = 0.0;
          ent.hardCcRem = 0.0;
          ent.isStunned = false;
          ent.activeAuras.clear();
          this.recalculateAuras(ent);
        }

        if (typeof vObj.intMP === "number") {
          ent.mp_norm = Math.min(1.0, Math.max(0.0, vObj.intMP / 100.0));
        }
        if (typeof vObj.intSP === "number") {
          ent.sp_norm = Math.min(1.0, Math.max(0.0, vObj.intSP / 100.0));
        }
      }
    }

    // 2. Action reset
    const sarsa = Array.isArray(dataObj.sarsa) ? dataObj.sarsa : [];
    for (const s of sarsa) {
      const caster = this.resolveTargetName(String(s.cInf ?? ""));
      if (caster) {
        const ent = this.getOrCreateEntity(caster);
        ent.timeSinceLastAction = 0.0;
        ent.lastSeenTs = Date.now();
      }
    }

    const anims = Array.isArray(dataObj.anims) ? dataObj.anims : [];
    for (const an of anims) {
      const caster = this.resolveTargetName(String(an.cInf ?? ""));
      if (caster) {
        const ent = this.getOrCreateEntity(caster);
        ent.timeSinceLastAction = 0.0;
        ent.lastSeenTs = Date.now();
      }
    }

    // 3. Aura management
    const auras = Array.isArray(dataObj.a) ? dataObj.a : [];
    for (const aEntry of auras) {
      const isRemove = aEntry.cmd === "aura-";
      const target = this.resolveTargetName(String(aEntry.tInf ?? ""));
      if (!target) continue;

      const ent = this.getOrCreateEntity(target);
      ent.lastSeenTs = Date.now();

      // Mendukung array `auras` maupun objek tunggal `aura`
      const auraList = Array.isArray(aEntry.auras) 
        ? aEntry.auras 
        : (aEntry.aura ? [aEntry.aura] : []);

      for (const a of auraList) {
        const nam = String(a.nam ?? "").toLowerCase();
        if (isRemove) {
          ent.activeAuras.delete(nam);
        } else {
          const meta = STATIC_AURA_CATALOG[nam];
          const cat = String(a.cat ?? meta?.cat ?? "").toLowerCase();
          const dur = typeof a.dur === "number" ? a.dur : 4.0;
          
          let outDmg: number | undefined = meta?.outDmgMult;
          if (typeof a.val === "number") {
            outDmg = a.val;
          }
          const inDmg = meta?.inDmgMult;

          ent.activeAuras.set(nam, {
            name: nam,
            cat,
            durationRem: dur,
            outDmgMult: outDmg,
            inDmgMult: inDmg,
          });
        }
      }
      this.recalculateAuras(ent);
    }
  }

  private handleUotlsObj(dataObj: any): void {
    if (!dataObj.o) return;
    const uName = (dataObj.unm || dataObj.o.uoName || dataObj.o.strUsername || "").toLowerCase();
    if (!uName || this.isMonsterIdentifier(uName)) return;

    const ent = this.getOrCreateEntity(uName);
    ent.lastSeenTs = Date.now();
    const o = dataObj.o;

    if (o.entID) {
      ent.entId = o.entID;
      this.entIdToName.set(o.entID, uName);
    }

    if (typeof o.pvpTeam === "number") {
      this.setEntityPvpTeam(ent, o.pvpTeam);
      if (uName === this.myUsername && this.myPvpTeam !== o.pvpTeam) {
        this.myPvpTeam = o.pvpTeam;
        this.refreshAllies();
      }
    }

    if (o.strFrame) ent.cell = o.strFrame;
    if (o.strClassName) ent.roleCategory = this.resolveRoleCategory(o.strClassName);
    if (typeof o.intHPMax === "number" && o.intHPMax > 0) ent.maxHp = o.intHPMax;
    if (typeof o.intHP === "number") {
      ent.currentHp = Math.max(0, o.intHP);
      ent.hp_norm = ent.maxHp > 0 ? Math.min(1.0, Math.max(0.0, ent.currentHp / ent.maxHp)) : 0.0;
      ent.alive = ent.currentHp > 0;
    }
    if (typeof o.intMP === "number") {
      ent.mp_norm = Math.min(1.0, Math.max(0.0, o.intMP / 100.0));
    }
    if (typeof o.intSP === "number") {
      ent.sp_norm = Math.min(1.0, Math.max(0.0, o.intSP / 100.0));
    }
    if (typeof o.intState === "number") ent.alive = o.intState !== 0;
  }

  private parseRawUotls(raw: string): void {
    const parts = raw.split("%");
    if (parts.length < 6) return;

    const uName = parts[4]?.toLowerCase();
    const payload = parts[5];
    if (!uName || !payload || this.isMonsterIdentifier(uName)) return;

    const ent = this.getOrCreateEntity(uName);
    ent.lastSeenTs = Date.now();

    const kvPairs = payload.split(",");
    for (const pair of kvPairs) {
      const [k, v] = pair.split(":");
      if (!k || v === undefined) continue;

      if (k === "strFrame") ent.cell = v;
      if (k === "px") ent.x = Number(v) || ent.x;
      if (k === "py") ent.y = Number(v) || ent.y;
      if (k === "intHP") {
        ent.currentHp = Number(v);
        ent.hp_norm = ent.maxHp > 0 ? Math.min(1.0, Math.max(0.0, ent.currentHp / ent.maxHp)) : 0.0;
        ent.alive = ent.currentHp > 0;
      }
      if (k === "intSP") {
        ent.sp_norm = Math.min(1.0, Math.max(0.0, Number(v) / 100.0));
      }
    }
  }

  // ==========================================
  // HELPERS & STATE COMPUTATIONS
  // ==========================================

  private isMonsterIdentifier(id: string): boolean {
    const lower = id.toLowerCase();
    return lower.startsWith("m:") || lower.startsWith("mon:") || lower.includes("captain") || lower.includes("brawler");
  }

  private resolveTargetName(tInf: string): string | null {
    if (!tInf || this.isMonsterIdentifier(tInf)) return null;

    if (tInf.startsWith("p:")) {
      const entId = Number(tInf.slice(2));
      return this.entIdToName.get(entId) ?? null;
    }
    return tInf.toLowerCase();
  }

  private resolveRoleCategory(className: string): number {
    const c = className.toLowerCase();
    if (c.includes("healer") || c.includes("cleric") || c.includes("paladin")) return 1;
    if (c.includes("bard") || c.includes("lord of order") || c.includes("support")) return 2;
    return 0; // Default DPS / Warrior
  }

  private setEntityPvpTeam(ent: TrackedEntity, pvpTeam: number): void {
    ent.pvpTeam = pvpTeam;
    if (pvpTeam === 0) ent.team = "blue";
    else if (pvpTeam === 1) ent.team = "red";
    else ent.team = "unknown";

    if (this.myPvpTeam !== -1 && ent.pvpTeam !== -1) {
      ent.isAlly = ent.pvpTeam === this.myPvpTeam;
    }
  }

  private refreshAllies(): void {
    if (this.myPvpTeam === -1) return;
    for (const ent of this.entities.values()) {
      if (ent.pvpTeam !== -1) {
        ent.isAlly = ent.pvpTeam === this.myPvpTeam;
      }
    }
  }

  private recalculateAuras(ent: TrackedEntity): void {
    let outMult = 1.0;
    let inMult = 1.0;
    let isStunned = false;
    let maxCc = 0.0;

    for (const aura of ent.activeAuras.values()) {
      if (aura.cat === "stun") {
        isStunned = true;
        maxCc = Math.max(maxCc, aura.durationRem);
      }
      if (aura.outDmgMult !== undefined) outMult *= aura.outDmgMult;
      if (aura.inDmgMult !== undefined) inMult *= aura.inDmgMult;
    }

    // Diri sendiri: Nilai kontinu murni dari stu (tidak dikalikan lagi agar terhindar dari double-counting)
    if (ent.username === this.myUsername) {
      ent.stats.outDmgMult = this.selfLocalStats.outDmgMult;
      ent.stats.inDmgMult = this.selfLocalStats.inDmgMult;
    } else {
      ent.stats.outDmgMult = outMult;
      ent.stats.inDmgMult = inMult;
    }

    ent.isStunned = isStunned;
    ent.hardCcRem = maxCc;
  }

  private getOrCreateEntity(uName: string): TrackedEntity {
    let ent = this.entities.get(uName);
    if (!ent) {
      ent = {
        username: uName,
        entId: 0,
        team: "unknown",
        pvpTeam: -1,
        isAlly: false,
        cell: "",
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
          critRate: 0.20,
          dodgeRate: 0.20,
          haste: 0.10,
        },
        activeAuras: new Map(),
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

  // ==========================================
  // PUBLIC INTERFACE (CYCLE & QUERIES)
  // ==========================================

  public tick(dtSec: number = 0.10): void {
    if (!this.myUsername && bot.player?.loggedIn) {
      this.myUsername = bot.player.username.toLowerCase();
    }

    for (const [uName, ent] of this.entities.entries()) {
      if (uName === this.myUsername) {
        // Sinkronisasi data lokal bot dari Turu API & stu
        ent.stats.inDmgMult = this.selfLocalStats.inDmgMult;
        ent.stats.outDmgMult = this.selfLocalStats.outDmgMult;
        ent.stats.manaCostMult = this.selfLocalStats.manaCostMult;
        ent.stats.critRate = this.selfLocalStats.critRate;
        ent.stats.dodgeRate = this.selfLocalStats.dodgeRate;
        ent.stats.haste = this.selfLocalStats.haste;

        ent.cell = bot.player.cell;
        ent.x = bot.player.x;
        ent.y = bot.player.y;
        ent.currentHp = bot.player.health;
        ent.maxHp = bot.player.maxHealth > 0 ? bot.player.maxHealth : 8455;
        ent.hp_norm = Math.min(1.0, Math.max(0.0, ent.currentHp / ent.maxHp));
        ent.mp_norm = Math.min(1.0, Math.max(0.0, bot.player.mana / 100.0));

        const rawSp = bot.flash?.get<number>("world.myAvatar.dataLeaf.intSP") ?? 100;
        ent.sp_norm = Math.min(1.0, Math.max(0.0, rawSp / 100.0));
        ent.alive = bot.player.alive;
        ent.isAlly = true;
        ent.lastSeenTs = Date.now();
      }

      // Hitung mundur durasi aura aktif
      let auraChanged = false;
      for (const [nam, aura] of ent.activeAuras.entries()) {
        aura.durationRem -= dtSec;
        if (aura.durationRem <= 0) {
          ent.activeAuras.delete(nam);
          auraChanged = true;
        }
      }
      if (auraChanged) {
        this.recalculateAuras(ent);
      }

      if (ent.hardCcRem > 0.0) {
        ent.hardCcRem = Math.max(0.0, ent.hardCcRem - dtSec);
        ent.isStunned = ent.hardCcRem > 0.0;
      }

      if (uName !== this.myUsername) {
        ent.timeSinceLastAction += dtSec;
      }
    }
  }

  public getSelf(): TrackedEntity {
    return this.getOrCreateEntity(this.myUsername || bot.player.username.toLowerCase());
  }

  public getEnemies(): TrackedEntity[] {
    return Array.from(this.entities.values()).filter(
      (e) => !e.isAlly && e.pvpTeam !== -1 && e.username !== this.myUsername
    );
  }

  public getAllies(): TrackedEntity[] {
    return Array.from(this.entities.values()).filter(
      (e) => e.isAlly && e.username !== this.myUsername
    );
  }

  /**
   * Target operasional serang (wajib berada di cell yang sama dan hidup).
   */
  public getPrimaryOpponent(): TrackedEntity | null {
    const enemies = this.getEnemies();
    if (enemies.length === 0) return null;
    const myCell = bot.player.cell;
    return enemies.find((e) => e.cell === myCell && e.alive) ?? null;
  }

  /**
   * Target strategis global (untuk keperluan navigasi antar-cell jika tidak ada musuh di cell saat ini).
   */
  public getGlobalTarget(): TrackedEntity | null {
    const enemies = this.getEnemies();
    return enemies.find((e) => e.alive) ?? enemies[0] ?? null;
  }
}