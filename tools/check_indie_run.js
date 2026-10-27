// Fairness check for INDIE RUN (app/game/indie_run.html). Plain Node, no packages:
//   node tools/check_indie_run.js [seeds=500] [firstSeed=1] [--jobs=8] [--game=path/to/indie_run.html]
// For every seed it runs the game's own generator through all year levels, the Tuwaiq level
// and endless mode (at top speed, plus some seeds with the normal speed ramp), then searches
// every input sequence (slide held or not, jump / slide hop pressed or not, decided every
// 2 frames; in the air slide can be pressed but not released) through the game's own physics and collision code. A course is impossible when
// no sequence survives. No power-ups are used, so the shield never helps.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const gameArg = process.argv.find(a => a.startsWith("--game="));
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
function loadGame(random) {
  const M = Object.create(Math);
  M.random = random;
  const win = {
    INDIE_RUN_LAB: {}, Math: M, innerWidth: 1010, innerHeight: 572, console,
    addEventListener() {}, matchMedia: () => ({ matches: false }), requestAnimationFrame() {},
    performance: { now: () => 0 }, localStorage: { getItem: () => null, setItem() {} },
    document: { getElementById: fakeEl, createElement: fakeEl, addEventListener() {}, hidden: false },
  };
  win.window = win;
  const ctx = vm.createContext(win);
  for (const s of scripts) vm.runInContext(s, ctx);
  return win.INDIE_RUN_LAB;
}

// 1) The course: the world does not depend on the player, so record it once per seed
function makeCourse(lab, endlessMode) {
  const { STEP, TU } = lab.consts;
  lab.start();
  const g = lab.g, all = new Set(), items = new Set(), frames = [];
  let endlessFrames = endlessMode === "max" ? 40 * 60 : 130 * 60;
  while (true) {
    lab.advanceWorld(STEP);
    for (const o of g.obs) all.add(o);
    for (const it of g.items) items.add(it);
    g.obs = g.obs.filter(o => o.x + o.w > g.dist - 400);
    frames.push({ dist: g.dist, speed: g.speed, L: g.L, endless: g.endless });
    if (!g.endless && g.L >= TU && g.levelT >= 10) {      // same as CONTINUE on the goal screen
      g.endless = true;
      g.endlessT = endlessMode === "max" ? 1e6 : 0;
      if (endlessMode === "max") {
        // The real game reaches top speed slowly; this test jumps there at once, so the
        // generator starts afresh just off screen (obstacles it made for the old speed go)
        const cut = g.dist + lab.consts.W + 40;
        for (const o of [...all]) if (o.x + o.w > g.dist + lab.consts.PX - 20) all.delete(o);
        for (const it of [...items]) if (it.x > g.dist + lab.consts.PX - 20) items.delete(it);
        g.obs = g.obs.filter(o => all.has(o));
        g.items = g.items.filter(it => items.has(it));
        g.nextX = g.segStart = cut; g.lastEnd = cut - 1000;
      }
    }
    if (g.endless && --endlessFrames <= 0) break;
  }
  return { obs: [...all].sort((a, b) => a.x - b.x), items: [...items], frames };
}

// 2) Search all input sequences through the real physics
const key = (p, slide) => [p.y.toFixed(3), p.vy.toFixed(2), p.onGround, p.jumps, p.hop, p.hopT.toFixed(3), slide,
  Math.max(0, p.coyote).toFixed(3), Math.max(0, p.buffer).toFixed(3),
  Math.max(0, p.slideMin).toFixed(3), Math.max(0, p.fastFall).toFixed(3)].join();
function solve(lab, course) {
  const { STEP, GROUND, PX } = lab.consts;
  const g = lab.g, obs = course.obs;
  let front = [{ p: Object.assign({}, lab.p), slide: false }];
  let lo = 0, maxFront = 1;
  for (let i = 0; i < course.frames.length; i++) {
    const f = course.frames[i];
    g.dist = f.dist; g.speed = f.speed; g.L = f.L; g.endless = f.endless; g.parts = []; g.floats = [];
    while (lo < obs.length && obs[lo].x + obs[lo].w < f.dist - 30) lo++;
    let hi = lo;
    while (hi < obs.length && obs[hi].x < f.dist + PX + 60) hi++;
    g.obs = obs.slice(lo, hi);
    // Running on the ground, ready for anything, with nothing close: this state can do
    // whatever any other state can, so keep only it
    let next = obs[lo];
    for (let j = lo; j < obs.length && obs[j].x + obs[j].w < f.dist + PX; j++) next = obs[j + 1];
    const far = !next || next.x > f.dist + PX + 12 + 0.9 * f.speed;
    if (far) {
      const ready = front.find(s => s.p.onGround && !s.p.hop && !s.slide && s.p.jumps === 2 &&
        s.p.y === GROUND && s.p.buffer <= 0 && s.p.slideMin <= 0);
      if (ready) front = [ready];
    }
    const decide = i % 2 === 0, seen = new Map();
    for (const s of front) {
      const options = decide
        ? [[s.slide, false], [!s.slide, false], [s.slide, true], [!s.slide, true]]
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
        lab.updatePlayer(STEP);
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
  const main = course.obs.filter(o => o.type !== "spike");
  main.forEach((t, i) => {
    if (t.type !== "tunnel") return;
    const v = speedAt(t.x), end = t.x + t.w, fail = why => bad.push({ why, x: Math.round(t.x), speed: +v.toFixed(1) });
    const prev = main[i - 1], next = main[i + 1];
    if (prev && (t.x - (prev.x + prev.w)) / v < RULES.before) fail("obstacle in the safe zone before the entrance");
    if (next && (next.x - end) / v < RULES.after) fail("obstacle in the safe zone after the exit");
    for (const it of course.items) {
      const ix = it.x, w = it.kind === "star" ? 5 : 13;
      const inBefore = ix + w > t.x - RULES.before * v && ix < t.x;
      const inAfter = ix + w > end && ix < end + RULES.after * v;
      if ((inBefore || inAfter) && !(it.kind === "star" && (ix >= t.x && ix < end))) fail(it.kind + " in a tunnel safe zone");
    }
    const inside = course.obs.filter(o => o !== t && o.x < end && o.x + o.w > t.x);
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

function describe(o, g0, speed) {
  return { type: o.type, x: Math.round(o.x - g0), w: o.w, s: +(o.w / speed).toFixed(2) };
}
function report(course, fail, lab) {
  const f = course.frames[fail.frame], d = f.dist;
  const near = course.obs.filter(o => o.x + o.w > d - 40 && o.x < d + 400)
    .map(o => describe(o, d + lab.consts.PX, f.speed));
  return { level: f.L === lab.consts.TU ? (f.endless ? "ENDLESS" : "TUWAIQ") : 2020 + f.L,
           speed: +f.speed.toFixed(1), obstaclesFromPlayer: near };
}

function runSeeds(first, n) {
  const totals = { seeds: 0, courses: 0, impossible: 0, obstacles: 0, tunnels: 0, floorBugs: 0,
                   tunnelsWithTwoBugs: 0, ruleViolations: 0, maxSpeed: 0, maxFront: 0 };
  const fails = [];
  for (let seed = first; seed < first + n; seed++) {
    const modes = seed % 10 === 0 ? ["max", "ramp"] : ["max"];
    for (const mode of modes) {
      const lab = loadGame(rng(seed * 7919 + (mode === "ramp" ? 1 : 0)));
      const course = makeCourse(lab, mode);
      lab.start();                                 // fresh player, same course
      const res = solve(lab, course);
      const bad = checkRules(lab, course);
      totals.ruleViolations += bad.length;
      if (bad.length) fails.push({ seed, mode, rules: bad.slice(0, 3) });
      totals.courses++;
      totals.obstacles += course.obs.length;
      for (const o of course.obs) {
        if (o.type !== "tunnel") continue;
        const k = course.obs.filter(b => b.type === "spike" && b.x >= o.x && b.x < o.x + o.w).length;
        totals.tunnels++;
        totals.floorBugs += k;
        if (k >= 2) totals.tunnelsWithTwoBugs++;
      }
      totals.maxSpeed = Math.max(totals.maxSpeed, ...course.frames.map(f => f.speed));
      totals.maxFront = Math.max(totals.maxFront, res.maxFront);
      if (!res.ok) { totals.impossible++; fails.push(Object.assign({ seed, mode }, report(course, res, lab))); }
    }
    totals.seeds++;
  }
  return { totals, fails };
}

const { Worker, isMainThread, parentPort, workerData } = require("worker_threads");
if (!isMainThread) {
  parentPort.postMessage(runSeeds(workerData.first, workerData.n));
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
        totals[k] = k.startsWith("max") ? Math.max(totals[k] || 0, r.totals[k]) : (totals[k] || 0) + r.totals[k];
      }
      fails.push(...r.fails);
    }
    totals.maxSpeed = +totals.maxSpeed.toFixed(1);
    console.log(JSON.stringify(totals));
    for (const f of fails.slice(0, 15)) console.log(f.rules ? "RULE BROKEN" : "IMPOSSIBLE", JSON.stringify(f));
    console.log("seconds:", ((Date.now() - t0) / 1000).toFixed(1));
    process.exitCode = totals.impossible || totals.ruleViolations ? 1 : 0;
  });
}
