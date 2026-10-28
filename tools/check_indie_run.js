// Fairness check for INDIE RUN (app/game/indie_run.html). Plain Node, no packages:
//   node tools/check_indie_run.js [seeds=500] [firstSeed=1] [--jobs=8] [--game=path/to/indie_run.html] [--rules-only]
// (--rules-only skips the input search: the rule checks alone, in minutes)
//   node tools/check_indie_run.js --chains [--jobs=10]
// (every combo chain type, built by the game, at the slowest and fastest speed of every level
// it can appear in and at endless top speed, each 10 times: full input search + human slack)
// For every seed it runs the game's own world step (level intros in slow motion included)
// through all year levels, the Tuwaiq level and endless mode (at top speed, plus some seeds
// with the normal speed ramp, some with COZY bought in every shop and some with reduced
// motion, where intros do not slow down), then searches
// every input sequence (slide held or not, jump / slide hop pressed or not, decided every
// 2 frames; in the air slide can be pressed but not released) through the game's own physics
// and collision code. A course is impossible when no sequence survives. No shield, extra life
// or intro safety is used, so they never help. The shop only pauses the game (the world is
// frozen) or changes the speed of time (COZY), which the "cozy" courses cover.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const gameArg = process.argv.find(a => a.startsWith("--game="));
const RULES_ONLY = process.argv.includes("--rules-only");
const CHAINS_MODE = process.argv.includes("--chains");
const GAME = gameArg ? path.resolve(gameArg.slice(7)) : path.join(__dirname, "..", "app", "game", "indie_run.html");
const html = fs.readFileSync(GAME, "utf8");
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);

function rng(seed) {                               // mulberry32
  return function () {
    seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Minimal browser stand-ins: drawing calls do nothing, no audio, no animation loop
function fakeCtx() {
  return new Proxy({}, {
    get(t, k) {
      if (k in t) return t[k];
      if (k === "createImageData") return (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) });
      return () => {};
    },
    set(t, k, v) { t[k] = v; return true; },
  });
}
function fakeEl() {
  return { style: {}, width: 0, height: 0, getContext: fakeCtx, addEventListener() {}, focus() {},
           getBoundingClientRect: () => ({ left: 0, top: 0, width: 870, height: 492 }),
           querySelector: () => fakeEl() };
}
function loadGame(random, reduced) {
  const M = Object.create(Math);
  M.random = random;
  const win = {
    INDIE_RUN_LAB: {}, Math: M, innerWidth: 1010, innerHeight: 572, console,
    Image: function () { return { addEventListener() {} }; },
    addEventListener() {}, matchMedia: () => ({ matches: !!reduced }), requestAnimationFrame() {},
    performance: { now: () => 0 }, localStorage: { getItem: () => null, setItem() {} },
    document: { getElementById: fakeEl, createElement: fakeEl, addEventListener() {}, hidden: false },
  };
  win.window = win;
  const ctx = vm.createContext(win);
  for (const s of scripts) vm.runInContext(s, ctx);
  return win.INDIE_RUN_LAB;
}

// 1) The course: the world does not depend on the player, so record it once per seed
function makeCourse(lab, endlessMode, cozy) {
  const { STEP, TU } = lab.consts;
  lab.start();
  const g = lab.g, all = new Set(), items = new Set(), frames = [];
  let endlessFrames = endlessMode === "max" ? 40 * 60 : 130 * 60;
  while (true) {
    const dt = lab.stepWorld(STEP);
    if (g.shopNow) {                                // the shop: time stands still, maybe buy COZY
      g.shopNow = false;
      if (cozy) g.cozyNext = 10;
    }
    for (const o of g.obs) all.add(o);
    for (const it of g.items) items.add(it);
    g.obs = g.obs.filter(o => o.x + o.w > g.dist - 400);
    frames.push({ dist: g.dist, speed: g.speed, L: g.L, endless: g.endless, t: g.simT, dt });
    if (!g.endless && g.L >= TU && g.levelT >= 40) {      // same as KEEP GROWING on the win screen
      g.endless = true;
      g.endlessT = endlessMode === "max" ? 1e6 : 0;
      if (endlessMode === "max") {
        // The real game reaches top speed slowly; this test jumps there at once, so the
        // generator starts afresh just off screen (obstacles it made for the old speed go)
        const cut = g.dist + lab.consts.W + 40;
        // (stars go from 40 px before the first obstacle that goes: a star row starts up to
        // 9 px before its obstacle, and a row must never outlive the obstacle it belongs to)
        let B = g.dist + lab.consts.PX - 20;
        for (const o of [...all]) if (hi(o) > B) { all.delete(o); B = Math.min(B, lo(o)); }
        for (const it of [...items]) if (it.x + 5 > B - 40) items.delete(it);
        g.obs = g.obs.filter(o => all.has(o));
        g.items = g.items.filter(it => items.has(it));
        g.nextX = g.segStart = cut; g.lastEnd = cut - 1000;
      }
    }
    if (g.endless && --endlessFrames <= 0) break;
  }
  return { obs: [...all].sort((a, b) => lo(a) - lo(b)), items: [...items], frames };
}

// Moving bugs walk back and forth: lo..hi is the whole stretch they can be on
const lo = o => (o.lo !== undefined ? o.lo : o.x), hi = o => (o.hi !== undefined ? o.hi : o.x + o.w);

// 2) Search all input sequences through the real physics
const key = (p, slide) => [p.y.toFixed(3), p.vy.toFixed(2), p.onGround, p.jumps, p.hop, p.hopT.toFixed(3), slide,
  Math.max(0, p.coyote).toFixed(3), Math.max(0, p.buffer).toFixed(3),
  Math.max(0, p.slideMin).toFixed(3), Math.max(0, p.fastFall).toFixed(3)].join();
function solve(lab, course) {
  const { STEP, GROUND, PX } = lab.consts;
  const g = lab.g, obs = course.obs;
  let front = [{ p: Object.assign({}, lab.p), slide: false }];
  let a = 0, maxFront = 1;
  for (let i = 0; i < course.frames.length; i++) {
    const f = course.frames[i];
    g.dist = f.dist; g.speed = f.speed; g.L = f.L; g.endless = f.endless; g.simT = f.t; g.parts = []; g.floats = [];
    while (a < obs.length && hi(obs[a]) < f.dist - 30) a++;
    let b = a;
    while (b < obs.length && lo(obs[b]) < f.dist + PX + 60) b++;
    g.obs = obs.slice(a, b);
    for (const o of g.obs) if (o.type === "mover") o.x = lab.moverX(o, f.t);   // where it is now
    // Running on the ground, ready for anything, with nothing close: this state can do
    // whatever any other state can, so keep only it
    let next = obs[a];
    for (let j = a; j < obs.length && hi(obs[j]) < f.dist + PX; j++) next = obs[j + 1];
    const far = !next || lo(next) > f.dist + PX + 12 + 0.9 * f.speed;
    if (far) {
      const ready = front.find(s => s.p.onGround && !s.p.hop && !s.slide && s.p.jumps === 2 &&
        s.p.y === GROUND && s.p.buffer <= 0 && s.p.slideMin <= 0);
      if (ready) front = [ready];
    }
    const decide = i % 2 === 0, seen = new Map();
    for (const s of front) {
      // nothing near: no new moves (let go of slide when on the ground), so the states settle
      // into the ready one; the same assumption as above (that state can do anything later)
      const options = far ? [[s.p.onGround ? false : s.slide, false]]
        : decide ? [[s.slide, false], [!s.slide, false], [s.slide, true], [!s.slide, true]]
        : [[s.slide, false]];
      for (const [slide, jump] of options) {
        if (jump && !(s.p.onGround || s.p.jumps > 0)) continue;
        // In the air, slide may be pressed (fast fall) but not released again: this keeps
        // the search small and only drops moves nobody needs
        if (!s.p.onGround && s.slide && !slide) continue;
        const p = Object.assign({}, s.p);
        lab.p = p;
        if (slide && !s.slide) {                   // same as refreshSlide() on a new press
          p.slideMin = 0.15;
          if (!p.onGround && !p.hop) p.fastFall = 0.5;
        }
        lab.input.slideHeld = slide;
        if (jump) p.buffer = 0.12;                 // same as a jump keydown (held, no short hop)
        const pits = g.pits;
        lab.updatePlayer(f.dt);
        if (g.pits !== pits || lab.firstHit()) continue;
        const k = key(p, slide);
        if (!seen.has(k)) seen.set(k, { p: p, slide: slide });
      }
    }
    front = [...seen.values()];
    maxFront = Math.max(maxFront, front.length);
    if (!front.length) return { ok: false, frame: i, maxFront };
  }
  return { ok: true, maxFront };
}

// 3) Rule checks on the generated course, timed at the speed the player has when passing
const RULES = { before: 1.2, after: 0.9, bugIn: 0.6, bugOut: 0.6, bugGap: 1.0 };
function checkRules(lab, course) {
  const { PX } = lab.consts, fr = course.frames, bad = [];
  const speedAt = x => {                           // speed when the player reaches world x
    let lo = 0, hi = fr.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (fr[m].dist + PX < x) lo = m + 1; else hi = m; }
    return fr[lo].speed;
  };
  // Level intros (slow motion under the banner): nothing reaches the player from 0.5 s before
  // a level starts until INTRO_CALM after its intro
  const { LEVEL_TIME, TU, INTRO_SIM, INTRO_CALM } = lab.consts;
  const timeAt = x => {
    let lo = 0, hi = fr.length - 1;
    while (lo < hi) { const m = (lo + hi) >> 1; if (fr[m].dist + PX + 12 < x) lo = m + 1; else hi = m; }
    return fr[lo].t;
  };
  for (const o of course.obs) {
    const t0 = timeAt(lo(o)), t1 = timeAt(hi(o) + 12);
    for (let L = 1; L <= TU; L++) {
      const a = L * LEVEL_TIME - 0.5 + 0.05, b = L * LEVEL_TIME + INTRO_SIM + INTRO_CALM - 0.05;
      if (t1 > a && t0 < b) bad.push({ why: o.type + " during a level intro", x: Math.round(o.x), level: L });
    }
  }
  const main = course.obs.filter(o => o.type !== "spike");
  main.forEach((t, i) => {
    if (t.type !== "tunnel") return;
    const v = speedAt(t.x), end = t.x + t.w, fail = why => bad.push({ why, x: Math.round(t.x), speed: +v.toFixed(1) });
    // a twin tunnel counts as one: its partner may sit in the zone, nothing else
    const prev = main[i - 1], next = main[i + 1];
    // (inside a combo chain the steps around a tunnel follow the slack rule below instead)
    const twinBefore = (t.twinOf && prev && prev.type === "tunnel" && prev.twinNext) || (t.chain && prev && prev.chain === t.chain);
    const twinAfter = (t.twinNext && next && next.type === "tunnel" && next.twinOf) || (t.chain && next && next.chain === t.chain);
    if (t.twinOf && !(prev && prev.type === "tunnel" && prev.twinNext)) fail("twin tunnel without its partner");
    if (!twinBefore && prev && (t.x - hi(prev)) / v < RULES.before) fail("obstacle in the safe zone before the entrance");
    if (!twinAfter && next && (lo(next) - end) / v < RULES.after) fail("obstacle in the safe zone after the exit");
    const tunnels = main.filter(o => o.type === "tunnel");
    const inAnyTunnel = (x, w) => tunnels.some(u => x >= u.x && x + w <= u.x + u.w);
    for (const it of course.items) {
      const ix = it.x, w = it.kind === "star" ? 5 : 13;
      const inBefore = !twinBefore && ix + w > t.x - RULES.before * v && ix < t.x;
      const inAfter = !twinAfter && ix + w > end && ix < end + RULES.after * v;
      // (stars inside a tunnel are the low rows to collect while sliding)
      const ownChain = t.chain && it.chain === t.chain;   // the stars that show a chain's path
      if ((inBefore || inAfter) && !ownChain && !(it.kind === "star" && inAnyTunnel(ix, w))) fail(it.kind + " in a tunnel safe zone");
    }
    const inside = course.obs.filter(o => o !== t && lo(o) < end && hi(o) > t.x);
    if (inside.some(o => o.type !== "spike")) fail("a non-floor-bug obstacle inside the tunnel");
    const bugs = inside.filter(o => o.type === "spike").sort((a, b) => a.x - b.x);
    bugs.forEach((b, j) => {
      if ((b.x - t.x) / v < RULES.bugIn) fail("floor bug too close to the entrance");
      if ((end - b.x - b.w) / v < RULES.bugOut) fail("floor bug too close to the exit");
      if (j && (b.x - bugs[j - 1].x) / v < RULES.bugGap) fail("two floor bugs too close");
    });
  });
  return bad;
}

// 4) Human slack on combo chains. A chain is played with one input per action: jump, second
// jump (double), slide (held through a run of slide steps, let go at the end of the run or
// at the next jump) and slide hop. For each action every press time is tried, with the other
// actions at their best times; the press times that still clear the whole chain must span at
// least SLACK seconds of game time at the chain's real speed (no frame-perfect inputs).
const SLACK = 0.25, SCAN = 45;
function slackCheck(lab, course) {
  lab.start();
  const { PX } = lab.consts, g = lab.g, fr = course.frames, out = [];
  const front = x => {                             // first frame where the player's front reaches x
    let a = 0, b = fr.length - 1;
    while (a < b) { const m = (a + b) >> 1; if (fr[m].dist + PX + 12 < x) a = m + 1; else b = m; }
    return a;
  };
  const back = x => {                              // first frame where the player's back has passed x
    let a = 0, b = fr.length - 1;
    while (a < b) { const m = (a + b) >> 1; if (fr[m].dist + PX < x) a = m + 1; else b = m; }
    return a;
  };
  const P0 = Object.assign({}, lab.p);
  const chains = new Map();
  for (const o of course.obs) if (o.chain) { if (!chains.has(o.chain)) chains.set(o.chain, []); chains.get(o.chain).push(o); }
  for (const [id, list] of chains) {
    list.sort((a, b) => lo(a) - lo(b));
    const steps = list.filter(o => o.act), acts = [];
    // (a chain cut short by the end of the recorded course, or by the endless cut, is skipped)
    if (steps.length !== steps[0].chainSteps || back(hi(list[list.length - 1]) + 4) + 30 >= fr.length - 1) continue;
    let run = null;
    for (const o of steps) {
      if ("SCH".includes(o.act)) {
        if (!run) { run = { last: o }; acts.push({ kind: "slide", o, run }); }
        run.last = o;
        if (o.act === "H") for (const sp of o.spikes) acts.push({ kind: "hop", o: sp });
      } else {
        run = null;
        acts.push({ kind: "jump", o });
        if (o.act === "D") acts.push({ kind: "jump2", o });
      }
    }
    const v = fr[front(lo(steps[0]))].speed;
    const s0 = front(lo(steps[0]) - 1.2 * v), last = list[list.length - 1];
    const e = Math.min(fr.length - 1, back(hi(last) + 4) + 6);
    for (const a of acts) {
      if (a.kind === "jump") a.t = front(lo(a.o) - (0.17 * v + 4));
      if (a.kind === "slide") { a.t = front(lo(a.o) - 0.35 * v); a.end = back(hi(a.run.last) + 2); }
      if (a.kind === "hop") a.t = front(a.o.x - (0.1 * v + 3));
    }
    acts.forEach((a, k) => { if (a.kind === "jump2") a.t = acts[k - 1].t + 17; });
    // play the chain with the actions at their times: true when it is cleared
    const play = () => {
      const p = Object.assign({}, P0); lab.p = p;
      let held = false;
      const pits = g.pits;
      g.obs = list;
      for (let i = s0; i <= e; i++) {
        const f = fr[i];
        g.dist = f.dist; g.speed = f.speed; g.L = f.L; g.endless = f.endless; g.simT = f.t;
        let press = false, slide = false;
        for (const a of acts) {
          if (a.t === i && a.kind !== "slide") press = true;
          if (a.kind === "slide" && i >= a.t && i < a.end) {
            // let go of slide for a jump that comes after this run
            const nextJump = acts.find(b => (b.kind === "jump" || b.kind === "jump2") && b.t > a.t && b.t <= i);
            if (!nextJump) slide = true;
          }
        }
        if (slide && !held) { p.slideMin = 0.15; if (!p.onGround && !p.hop) p.fastFall = 0.5; }
        held = slide; lab.input.slideHeld = slide;
        if (press) p.buffer = 0.12;
        lab.updatePlayer(f.dt);
        if (g.pits !== pits || lab.firstHit()) return false;
      }
      return true;
    };
    let worst = Infinity, worstKind = "";
    for (let pass = 0; pass < 2; pass++) {
      acts.forEach((a, k) => {
        const t0 = a.t, ok = [];
        const lo2 = Math.max(s0, t0 - SCAN), hi2 = Math.min(e, t0 + SCAN);
        for (let t = lo2; t <= hi2; t++) { a.t = t; ok.push(play()); }
        // the runs of good press times; keep the one around the current time, or the longest
        const runs = [];
        for (let j = 0; j < ok.length; j++) {
          if (!ok[j]) continue;
          if (runs.length && runs[runs.length - 1].b === j - 1) runs[runs.length - 1].b = j;
          else runs.push({ a: j, b: j });
        }
        const at = t0 - lo2;
        const best = runs.find(r => r.a <= at && at <= r.b) ||
          runs.reduce((m, r) => (!m || r.b - r.a > m.b - m.a ? r : m), null);
        if (!best) { a.t = t0; a.width = 0; return; }
        let w = 0;
        for (let j = best.a; j <= best.b; j++) w += fr[lo2 + j].dt;
        a.width = w;
        a.t = lo2 + Math.round((best.a + best.b) / 2);   // play it at the middle of its window
      });
    }
    for (const a of acts) if (a.width < worst) { worst = a.width; worstKind = a.kind; }
    out.push({ id, steps: steps.map(o => o.act).join(""), speed: +v.toFixed(1), minSlack: +worst.toFixed(3), tight: worstKind, windows: acts.map(a => a.kind[0] + (a.kind === "jump2" ? "2" : "") + ":" + a.width.toFixed(2)).join(" "),
               x: Math.round(lo(steps[0])) });
  }
  lab.input.slideHeld = false;
  return out;
}

function describe(o, g0, speed) {
  return { type: o.type, x: Math.round(lo(o) - g0), w: Math.round(hi(o) - lo(o)), s: +((hi(o) - lo(o)) / speed).toFixed(2) };
}
function report(course, fail, lab) {
  const f = course.frames[fail.frame], d = f.dist;
  const near = course.obs.filter(o => hi(o) > d - 40 && lo(o) < d + 400)
    .map(o => describe(o, d + lab.consts.PX, f.speed));
  return { level: f.L === lab.consts.TU ? (f.endless ? "ENDLESS" : "TUWAIQ") : 2020 + f.L,
           speed: +f.speed.toFixed(1), obstaclesFromPlayer: near };
}

function runSeeds(first, n) {
  const totals = { seeds: 0, courses: 0, cozyCourses: 0, reducedMotionCourses: 0, impossible: 0, obstacles: 0, tunnels: 0, floorBugs: 0,
                   tunnelsWithTwoBugs: 0, tunnelsWithThreeBugs: 0, twinTunnels: 0, walls: 0, drones: 0, movingBugs: 0,
                   ruleViolations: 0, chains: 0, slackViolations: 0, minChainSlack: Infinity, maxSpeed: 0, maxFront: 0 };
  const fails = [];
  for (let seed = first; seed < first + n; seed++) {
    const modes = seed % 10 === 0 ? ["max", "ramp"] : ["max"];
    const cozy = seed % 4 === 1, reduced = seed % 4 === 2;
    for (const mode of modes) {
      const lab = loadGame(rng(seed * 7919 + (mode === "ramp" ? 1 : 0)), reduced);
      const course = makeCourse(lab, mode, cozy);
      if (cozy) totals.cozyCourses++;
      if (reduced) totals.reducedMotionCourses++;
      lab.start();                                 // fresh player, same course
      const res = RULES_ONLY ? { ok: true, maxFront: 0 } : solve(lab, course);
      const bad = checkRules(lab, course);
      for (const c of slackCheck(lab, course)) {
        totals.chains++;
        totals.minChainSlack = Math.min(totals.minChainSlack, c.minSlack);
        if (c.minSlack < SLACK) { totals.slackViolations++; if (fails.length < 40) fails.push({ seed, mode, slack: c }); }
      }
      totals.ruleViolations += bad.length;
      if (bad.length) fails.push({ seed, mode, rules: bad.slice(0, 3) });
      totals.courses++;
      totals.obstacles += course.obs.length;
      for (const o of course.obs) {
        if (o.type !== "tunnel") continue;
        const k = course.obs.filter(b => b.type === "spike" && b.x >= o.x && b.x < o.x + o.w).length;
        totals.tunnels++;
        totals.floorBugs += k;
        if (k === 2) totals.tunnelsWithTwoBugs++;
        if (k === 3) totals.tunnelsWithThreeBugs++;
        if (o.twinNext) totals.twinTunnels++;
      }
      for (const o of course.obs) {
        if (o.type === "wall") totals.walls++;
        if (o.type === "drone") totals.drones++;
        if (o.type === "mover") totals.movingBugs++;
      }
      totals.maxSpeed = Math.max(totals.maxSpeed, ...course.frames.map(f => f.speed));
      totals.maxFront = Math.max(totals.maxFront, res.maxFront);
      if (!res.ok) { totals.impossible++; fails.push(Object.assign({ seed, mode }, report(course, res, lab))); }
    }
    totals.seeds++;
  }
  return { totals, fails };
}

// 5) Combo chains alone. A case = one chain design at one level, played at a speed v while
// sized (by the game's own builder) for a speed spd: the slowest and fastest speed of that
// level, the chain sized for the fastest one but met at the slowest, and endless top speed.
function chainCases() {
  const lab = loadGame(Math.random);
  lab.start();
  const { TU, LEVEL_TIME } = lab.consts, cases = [];
  lab.CHAINS.forEach((c, d) => {
    for (let L = c.from; L <= TU; L++) {
      if (L >= 4 && c.from < 3) continue;          // (the game's own pool rule)
      const lo = lab.speedFor(L, 0), hi = lab.speedFor(L, LEVEL_TIME);
      for (const [v, size] of [[lo, lo], [lo, hi], [hi, hi]]) cases.push({ d, L, v, spd: 1.05 * size });
    }
    if (c.from >= 3) {                             // endless at top speed
      lab.g.endless = true; lab.g.endlessT = 1e6;
      const top = lab.speedFor(TU, LEVEL_TIME);
      cases.push({ d, L: TU, v: top, spd: 1.05 * top, endless: true });
      lab.g.endless = false;
    }
  });
  return cases;
}
function chainCourse(lab, cs) {
  const { STEP, PX } = lab.consts;
  lab.start();
  const P = [];
  lab.buildChain(P, cs.L, cs.spd, lab.CHAINS[cs.d]);
  const x0 = PX + 12 + 1.6 * cs.v;                 // a clear stretch before it
  for (const o of P) o.x += x0;
  const end = Math.max.apply(null, P.map(hi)), frames = [];
  for (let i = 0, d = 0; d < end + 2.5 * cs.v; i++) {
    d += cs.v * STEP;
    frames.push({ dist: d, speed: cs.v, L: cs.L, endless: !!cs.endless, t: i * STEP, dt: STEP });
  }
  return { obs: P.sort((a, b) => lo(a) - lo(b)), items: [], frames };
}
function runChains(cases, first) {
  const totals = { cases: 0, courses: 0, impossible: 0, slackViolations: 0, minSlack: Infinity }, fails = [], per = {};
  cases.forEach((cs, k) => {
    for (let r = 0; r < 10; r++) {
      const lab = loadGame(rng((first + k) * 977 + r * 31 + 5));
      const course = chainCourse(lab, cs);
      lab.start();
      const res = solve(lab, course);
      const sl = slackCheck(lab, course)[0];
      const steps = lab.CHAINS[cs.d].steps, key = steps;
      per[key] = per[key] || { courses: 0, impossible: 0, minSlack: Infinity, minSpeed: Infinity, maxSpeed: 0 };
      per[key].courses++; per[key].minSpeed = Math.min(per[key].minSpeed, cs.v); per[key].maxSpeed = Math.max(per[key].maxSpeed, cs.v);
      totals.courses++;
      if (!res.ok) { totals.impossible++; per[key].impossible++; fails.push({ steps, L: cs.L, v: +cs.v.toFixed(1), spd: +cs.spd.toFixed(1), endless: !!cs.endless }); }
      const w = sl ? sl.minSlack : 0;
      totals.minSlack = Math.min(totals.minSlack, w); per[key].minSlack = Math.min(per[key].minSlack, w);
      if (w < SLACK) { totals.slackViolations++; fails.push({ steps, slack: sl, L: cs.L, v: +cs.v.toFixed(1) }); }
    }
    totals.cases++;
  });
  return { totals, fails, per };
}

const { Worker, isMainThread, parentPort, workerData } = require("worker_threads");
if (!isMainThread) {
  parentPort.postMessage(workerData.cases ? runChains(workerData.cases, workerData.first)
    : runSeeds(workerData.first, workerData.n));
} else if (CHAINS_MODE) {
  const jobsArg = process.argv.find(a => a.startsWith("--jobs="));
  const jobs = Math.max(1, jobsArg ? parseInt(jobsArg.slice(7), 10) : 1);
  const cases = chainCases(), t0 = Date.now(), parts = [];
  for (let j = 0; j < jobs; j++) {
    const mine = cases.filter((c, i) => i % jobs === j);
    parts.push(new Promise((res, rej) => {
      const w = new Worker(__filename, { workerData: { cases: mine, first: j * 1000 }, argv: process.argv.slice(2) });
      w.on("message", res); w.on("error", rej);
    }));
  }
  Promise.all(parts).then(results => {
    const totals = { cases: 0, courses: 0, impossible: 0, slackViolations: 0, minSlack: Infinity }, per = {}, fails = [];
    for (const r of results) {
      for (const k of ["cases", "courses", "impossible", "slackViolations"]) totals[k] += r.totals[k];
      totals.minSlack = Math.min(totals.minSlack, r.totals.minSlack);
      fails.push(...r.fails);
      for (const k in r.per) {
        const a = per[k] || (per[k] = { courses: 0, impossible: 0, minSlack: Infinity, minSpeed: Infinity, maxSpeed: 0 }), b = r.per[k];
        a.courses += b.courses; a.impossible += b.impossible; a.minSlack = Math.min(a.minSlack, b.minSlack);
        a.minSpeed = Math.min(a.minSpeed, b.minSpeed); a.maxSpeed = Math.max(a.maxSpeed, b.maxSpeed);
      }
    }
    totals.minSlack = +totals.minSlack.toFixed(3);
    console.log(JSON.stringify(totals));
    for (const k in per) console.log(k.padEnd(7), JSON.stringify({ courses: per[k].courses, impossible: per[k].impossible,
      minSlack: +per[k].minSlack.toFixed(3), speeds: per[k].minSpeed.toFixed(1) + "-" + per[k].maxSpeed.toFixed(1) }));
    for (const f of fails.slice(0, 15)) console.log(f.slack !== undefined ? "TOO LITTLE SLACK" : "IMPOSSIBLE", JSON.stringify(f));
    console.log("seconds:", ((Date.now() - t0) / 1000).toFixed(1));
    process.exitCode = totals.impossible || totals.slackViolations ? 1 : 0;
  });
} else {
  const args = process.argv.slice(2).filter(a => !a.startsWith("--"));
  const jobsArg = process.argv.find(a => a.startsWith("--jobs="));
  const N = parseInt(args[0] || "500", 10), first = parseInt(args[1] || "1", 10);
  const jobs = Math.max(1, Math.min(N, jobsArg ? parseInt(jobsArg.slice(7), 10) : 1));
  const t0 = Date.now();
  const parts = [];
  for (let j = 0; j < jobs; j++) {
    const a = first + Math.floor(N * j / jobs), b = first + Math.floor(N * (j + 1) / jobs);
    parts.push(new Promise((res, rej) => {
      const w = new Worker(__filename, { workerData: { first: a, n: b - a }, argv: process.argv.slice(2) });
      w.on("message", res); w.on("error", rej);
    }));
  }
  Promise.all(parts).then(results => {
    const totals = {}, fails = [];
    for (const r of results) {
      for (const k in r.totals) {
        totals[k] = k.startsWith("max") ? Math.max(totals[k] || 0, r.totals[k])
          : k.startsWith("min") ? Math.min(totals[k] === undefined ? Infinity : totals[k], r.totals[k])
          : (totals[k] || 0) + r.totals[k];
      }
      fails.push(...r.fails);
    }
    totals.maxSpeed = +totals.maxSpeed.toFixed(1);
    console.log(JSON.stringify(totals));
    totals.minChainSlack = totals.chains ? +totals.minChainSlack.toFixed(3) : null;
    for (const f of fails.slice(0, 15)) console.log(f.rules ? "RULE BROKEN" : f.slack ? "TOO LITTLE SLACK" : "IMPOSSIBLE", JSON.stringify(f));
    console.log("seconds:", ((Date.now() - t0) / 1000).toFixed(1));
    process.exitCode = totals.impossible || totals.ruleViolations || totals.slackViolations ? 1 : 0;
  });
}
