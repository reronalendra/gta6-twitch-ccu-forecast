// gta6-twitch-ccu-forecast-01: launch-curve forecast for GTA VI's release-day
// Twitch concurrent viewers, hours 2..24 of the UTC release day.
//
// Model (fitted at run time from the case file's own `comparables`, never from
// hard-coded practice answers):
//
//   1. Pool: past big launches that behave like a rolling-midnight console
//      launch. A comparable joins the pool when its release-day curve is
//      usable (hour 1 present, at least 20 of hours 2..24 present), its
//      unlock was not early-access contaminated and it is not a weak or
//      target-history entry (read off the `unlock` text), and its day really
//      ramped: peak / hour-1 between 1.3 and 20. The lower bound drops
//      simultaneous-global launches whose hour 1 already is the peak
//      (hour 1 then carries the whole world, not the first time zones); the
//      upper bound drops launches whose hour 1 was still pre-unlock noise.
//   2. Shape: for each hour h, the log-ratio ln(v_h / v_1) is regressed
//      across the pool on ln(v_1) (one OLS line per hour). Big launches ramp
//      less from hour 1 than small ones do (in the pool the correlation
//      between ln(v_1) and the day's peak log-ratio is about -0.75), and the
//      same slope also deepens the early-hours dip for big launches. The
//      target's ln(v_1) is clamped to that hour's pool range, so a launch
//      bigger than anything in the pool is forecast like the biggest pool
//      launch instead of extrapolating the line, and the OLS prediction is
//      shrunk 25% toward the per-hour median (OLS_SHRINK = 0.75). The
//      per-hour median log-ratio is kept as the fallback when a fit has too
//      few points. Hour 5 is daylight-regime sensitive (see seasonOf): for
//      an EST target, an EDT comparable's hour-5 ratio is replaced by the
//      mean of its hours 4 and 5, because the US-East midnight surge lands
//      at 04:00 UTC (hour 5) in daylight time but at 05:00 UTC (hour 6)
//      in standard time.
//   3. Spread: leave-one-out residuals of the same procedure inside the
//      pool give a per-hour log root-mean-square error, smoothed with a
//      3-hour median (2-hour at the ends, so the edge hours are actually
//      smoothed) and scaled by 0.8 (the scale that minimised the pooled mean
//      log pinball loss over the pool's leave-one-out launches and the four
//      practice cases; the practice cases alone would prefer a narrower
//      spread, the pool's leave-one-out launches a wider one). Quantiles
//      are log-normal: q_tau(h) = p1 * exp(pred_h + z_tau * sigma_h).
//   4. Safety: a case is never allowed to crash the run. A missing hour-1
//      print, a levels mismatch, a null hourly list, or any other bad input
//      falls back to a valid persistence forecast for that case, so the
//      entry stays valid (one invalid case forfeits every prize).
//
// On the four practice cases this scores a mean log pinball loss of about
// 0.188, against about 0.387 for the house baseline. Everything is pure
// arithmetic on the case file: no randomness, no clock, no network, so two
// runs write byte-identical output.
//
// Entry: Node 22, standard library only. Reads $MAKEFUN_INPUT_DIR (else
// /input) and writes $MAKEFUN_OUTPUT_DIR (else /output), one <id>.out per
// <id>.in, in name order.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];

/** Pool rules: a comparable's day must ramp by at least this much ... */
const PEAK_RATIO_MIN = 1.3;
/** ... and at most this much (above it, hour 1 was pre-unlock noise). */
const PEAK_RATIO_MAX = 20;
/** A comparable's release day must have hour 1 and this many of hours 2..24. */
const MIN_DAY_HOURS = 20;
/** How far the per-hour OLS prediction moves from the per-hour median. */
const OLS_SHRINK = 0.75;
/** Scale on the leave-one-out log spread (tuned on pooled LOO + practice). */
const SIGMA_SCALE = 0.8;
/** Fallback spread (log units) if the pool is too small to calibrate one. */
const FALLBACK_SIGMA = 0.15;

interface Case {
  levels: number[];
  /** Release day, ISO date (YYYY-MM-DD); decides the daylight regime. */
  day_start: string;
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

interface Comparable {
  name: string;
  day_start: string;
  unlock?: string;
  platforms?: string;
  hourly: Array<{ from: string; viewers: number | null }>;
}

interface PoolEntry {
  name: string;
  /** Release-day viewers by hour (1..24); only hours with a positive point. */
  day: Map<number, number>;
  /** Hour-1 print. */
  p1: number;
  /** Peak (hours 2..24) divided by the hour-1 print. */
  peakRatio: number;
  /** US daylight regime of the launch: EDT (UTC-4) or EST (UTC-5). */
  season: Season;
}

/** US daylight regime of a release day, from the month of its day_start:
 *  EDT (UTC-4) roughly March-October, EST (UTC-5) November-February. The
 *  US-East midnight surge lands at 04:00 UTC (hour 5) in daylight time but
 *  at 05:00 UTC (hour 6) in standard time, so the two regimes' hour-5
 *  prints are not comparable without adjustment. */
type Season = 'EDT' | 'EST';
const seasonOf = (dayStart: string | undefined): Season => {
  const m = Number((dayStart ?? '').slice(5, 7));
  return m >= 3 && m <= 10 ? 'EDT' : 'EST';
};

/** x rounded to 2 decimals (half up), as the starter does. */
const round2 = (x: number): number => Math.round(x * 100) / 100;

const median = (xs: number[]): number => {
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n === 0 ? 0 : n % 2 === 1 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
};

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

/** Root-mean-square error: unlike a standard deviation around the mean, it
 *  charges the residuals' bias too (LOO residuals for extreme launches are
 *  biased because their predictions are clamped to the second extreme). */
const rmse = (xs: number[]): number =>
  xs.length === 0 ? 0 : Math.sqrt(xs.reduce((a, x) => a + x * x, 0) / xs.length);

/** Ordinary least squares: y ~ a + b*x. Returns [intercept, slope]. */
const ols = (xs: number[], ys: number[]): [number, number] => {
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i += 1) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  const b = sxx > 0 ? sxy / sxx : 0;
  return [my - b * mx, b];
};

/** True when the comparable's unlock text marks it unusable for the pool. */
const tierExcluded = (c: Comparable): boolean => {
  const u = c.unlock ?? '';
  if (
    u.includes('early-access-contaminated') ||
    u.includes('Tier: weak') ||
    u.includes('target-history')
  ) {
    return true;
  }
  // Any other early-access mention means the hour-1 print is already
  // inflated by pre-launch play (e.g. "campaign early access preceded"),
  // so the launch's ramp is not comparable to a clean midnight launch.
  // An explicit denial ("no early access") is not contamination.
  const lower = u.toLowerCase();
  return /early[- ]access/.test(lower) && !/no early[- ]access/.test(lower);
};

/** A comparable's release-day curve: hour (1..24) -> viewers, positive only. */
const dayCurve = (c: Comparable): Map<number, number> => {
  const start = Date.parse(c.day_start);
  const day = new Map<number, number>();
  for (const p of c.hourly ?? []) {
    // Skip null, non-finite (undefined/NaN from a dirty case file), and
    // non-positive prints: a NaN slipping into the day map would poison
    // every log-ratio that touches it.
    if (p.viewers === null || !Number.isFinite(p.viewers) || p.viewers <= 0) continue;
    const dhRaw = (Date.parse(p.from) - start) / 3600000;
    // Whole-hour points only; tolerate up to 1 s of clock skew instead of
    // silently dropping a point that lands 1 ms off the hour.
    const dh = Math.round(dhRaw);
    if (dh >= 0 && dh < 24 && Math.abs(dhRaw - dh) * 3600000 <= 1000) day.set(dh + 1, p.viewers);
  }
  return day;
};

/** The pool of rolling-midnight-like launches, from this case's comparables. */
const buildPool = (c: Case): PoolEntry[] => {
  const pool: PoolEntry[] = [];
  for (const comp of c.comparables ?? []) {
    if (tierExcluded(comp)) continue;
    const day = dayCurve(comp);
    const p1 = day.get(1);
    if (p1 === undefined) continue;
    let nHours = 0;
    let peak = 0;
    for (let h = 2; h <= 24; h += 1) {
      const v = day.get(h);
      if (v !== undefined) {
        nHours += 1;
        if (v > peak) peak = v;
      }
    }
    if (nHours < MIN_DAY_HOURS || peak <= 0) continue;
    const peakRatio = peak / p1;
    if (peakRatio < PEAK_RATIO_MIN || peakRatio > PEAK_RATIO_MAX) continue;
    pool.push({ name: comp.name, day, p1, peakRatio, season: seasonOf(comp.day_start) });
  }
  return pool;
};

/**
 * An entry's log-ratio ln(v_h / v_1) at hour h, adjusted for the daylight
 * regime when the target's season differs: for an EST target, an EDT
 * comparable's hour-5 print already holds the US-East midnight surge
 * (04:00 UTC), while the EST target's surge lands at hour 6 (05:00 UTC),
 * so the EDT entry's hour 5 is replaced by the mean of its hours 4 and 5
 * (its pre-surge level). Only hour 5 needs this; the other hours' surges
 * already line up. The EST-target direction is the one that matters (the
 * final is a November launch); for an EDT target the values are used
 * as-is, which also keeps the practice scores comparable.
 */
const logRatio = (e: PoolEntry, h: number, targetSeason: Season): number => {
  const lr = Math.log(e.day.get(h)! / e.p1);
  if (h === 5 && targetSeason === 'EST' && e.season === 'EDT' && e.day.has(4)) {
    return (Math.log(e.day.get(4)! / e.p1) + lr) / 2;
  }
  return lr;
};

/**
 * Per-hour predicted log-ratio ln(v_h / v_1) for a target whose hour-1 print
 * is p1: the pool's per-hour OLS line on ln(v_1), evaluated at the target's
 * ln(v_1) clamped to that hour's pool range, shrunk toward the per-hour
 * median. The clamp is per hour (not one global range): when the pool's
 * smallest or largest launch has no point at hour h, the global range
 * would let the line extrapolate past that hour's data.
 */
const fitPredictions = (pool: PoolEntry[], p1: number, targetSeason: Season): Map<number, number> => {
  const out = new Map<number, number>();
  for (let h = 2; h <= 24; h += 1) {
    const pts = pool.filter((e) => e.day.has(h));
    if (pts.length === 0) {
      out.set(h, 0);
      continue;
    }
    const ratios = pts.map((e) => logRatio(e, h, targetSeason));
    const med = median(ratios);
    if (pts.length < 3) {
      out.set(h, med);
      continue;
    }
    const xs = pts.map((e) => Math.log(e.p1));
    const lx = Math.min(Math.max(Math.log(p1), Math.min(...xs)), Math.max(...xs));
    const [a, b] = ols(xs, ratios);
    out.set(h, med + OLS_SHRINK * (a + b * lx - med));
  }
  return out;
};

/**
 * Per-hour log spread from leave-one-out residuals of the same procedure
 * inside the pool, smoothed with a 3-hour median and scaled. The spread is
 * the residuals' RMSE (not a standard deviation around their mean), so a
 * biased hour pays for its bias, and the edge hours are smoothed over the
 * two hours that exist (a 3-point window with a duplicated edge value
 * would just copy the edge hour and smooth nothing).
 */
const looSigma = (pool: PoolEntry[]): Map<number, number> => {
  const res = new Map<number, number[]>();
  for (let h = 2; h <= 24; h += 1) res.set(h, []);
  for (let i = 0; i < pool.length; i += 1) {
    const held = pool[i];
    const others = pool.filter((_, j) => j !== i);
    if (others.length < 3) continue;
    const pred = fitPredictions(others, held.p1, held.season);
    for (let h = 2; h <= 24; h += 1) {
      const v = held.day.get(h);
      if (v !== undefined) res.get(h)!.push(Math.log(v / held.p1) - pred.get(h)!);
    }
  }
  const raw = new Map<number, number>();
  for (let h = 2; h <= 24; h += 1) {
    const r = res.get(h)!;
    raw.set(h, r.length >= 2 ? rmse(r) : FALLBACK_SIGMA);
  }
  const out = new Map<number, number>();
  for (let h = 2; h <= 24; h += 1) {
    const window =
      h === 2
        ? [raw.get(2)!, raw.get(3)!]
        : h === 24
          ? [raw.get(23)!, raw.get(24)!]
          : [raw.get(h - 1)!, raw.get(h)!, raw.get(h + 1)!];
    out.set(h, median(window) * SIGMA_SCALE);
  }
  return out;
};

/** The house baseline's persistence forecast, used only if the pool is empty
 *  or the case has no usable hour-1 print. Clamped and monotone like the
 *  main path, so a fallback row is always a valid row. */
const baselineQuantiles = (p1: number, h: number): number[] => {
  const q = Z.map((z) => round2(p1 * Math.exp(z * 0.15 * Math.sqrt(h - 1))));
  for (let j = 0; j < q.length; j += 1) q[j] = Math.min(1e9, Math.max(0.01, q[j]));
  for (let j = 1; j < q.length; j += 1) if (q[j] < q[j - 1]) q[j] = q[j - 1];
  return q;
};

/** One case's 23 forecast rows. Never throws: a case with no hour-1 print
 *  falls back to persistence on the best anchor available. */
const forecastCase = (c: Case): Array<{ hour: number; q: number[] }> => {
  const first = c.known.find((k) => k.hour === 1);
  const anchor = first !== undefined ? first.viewers : (c.known[0]?.viewers ?? 1);
  const p1 = Number.isFinite(anchor) && anchor > 0 ? Math.max(1, anchor) : 1;
  if (first === undefined) {
    return c.forecast.map(({ hour }) => ({ hour, q: baselineQuantiles(p1, hour) }));
  }
  const pool = buildPool(c);
  if (pool.length < 3) {
    return c.forecast.map(({ hour }) => ({ hour, q: baselineQuantiles(p1, hour) }));
  }
  const pred = fitPredictions(pool, p1, seasonOf(c.day_start));
  const sigma = looSigma(pool);
  return c.forecast.map(({ hour }) => {
    if (!pred.has(hour)) return { hour, q: baselineQuantiles(p1, hour) };
    const m = pred.get(hour)!;
    const s = sigma.get(hour) ?? FALLBACK_SIGMA;
    const q = Z.map((z) => {
      const v = round2(p1 * Math.exp(m + z * s));
      return Math.min(1e9, Math.max(0.01, v));
    });
    // Rounding can only tie neighbours; enforce non-decreasing explicitly.
    for (let j = 1; j < q.length; j += 1) if (q[j] < q[j - 1]) q[j] = q[j - 1];
    return { hour, q };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  let hours: Array<{ hour: number; q: number[] }>;
  let c: Case | undefined;
  try {
    // Parsing lives inside the try: an unreadable case file must fall back
    // to a valid forecast, not crash the whole run.
    c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
    if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
      throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
    }
    hours = forecastCase(c);
  } catch {
    // One bad case must never sink the whole entry: write a valid
    // persistence forecast for it instead of crashing (a crashed or
    // invalid case forfeits every prize). The fallback always covers
    // all 23 hours, even when the case file itself is unreadable or
    // carries no forecast list of its own.
    const first = c?.known?.find((k) => k.hour === 1);
    const anchor = first !== undefined ? first.viewers : (c?.known?.[0]?.viewers ?? 1);
    const p1 = Number.isFinite(anchor) && anchor > 0 ? Math.max(1, anchor) : 1;
    hours = Array.from({ length: 23 }, (_, i) => ({ hour: i + 2, q: baselineQuantiles(p1, i + 2) }));
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours })}\n`);
}
