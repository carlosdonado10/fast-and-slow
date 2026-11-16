import type { BandPoint, CostRow, Curve, CurvePoint, EfficiencyByArm, Num, Report } from '../types';
import { sortArms } from './arms';
import { isNum } from './format';

/** Learners with a curve at this horizon, in the fixed arm order. */
export function curveArms(report: Pick<Report, 'curves'>, horizon: string): string[] {
  return sortArms(Object.keys(report.curves[horizon] ?? {}));
}

/** Mean-curve points that have a score, in decision order. */
export function cleanPoints(points: CurvePoint[] | undefined): CurvePoint[] {
  return (points ?? [])
    .filter((p) => isNum(p.decisions) && isNum(p.score))
    .sort((a, b) => a.decisions - b.decisions);
}

/**
 * A log axis can't show 0 decisions, but "untrained" is the most important point on the chart (H1).
 * So the zero point gets its own slot left of a broken axis; everything else sits on log x.
 */
export function splitZero<T extends CurvePoint>(points: T[]): { zero: T | null; rest: T[] } {
  const zero = points.find((p) => p.decisions <= 0) ?? null;
  return { zero, rest: points.filter((p) => p.decisions > 0) };
}

/** An arm that doesn't consume experience: one point, so its score holds at any x. */
export function isStatic(curve: Curve | undefined): boolean {
  return cleanPoints(curve?.mean).length === 1;
}

/** Log domain covering every positive decision count, snapped out to powers of ten. */
export function logDomain(curves: (Curve | undefined)[]): [number, number] {
  const xs = curves
    .flatMap((c) => cleanPoints(c?.mean).map((p) => p.decisions))
    .filter((d) => d > 0);
  if (!xs.length) return [1, 10];
  const lo = 10 ** Math.floor(Math.log10(Math.min(...xs)));
  let hi = 10 ** Math.ceil(Math.log10(Math.max(...xs)));
  if (hi <= lo) hi = lo * 10;
  return [lo, hi];
}

export function logTicks([lo, hi]: [number, number]): number[] {
  const out: number[] = [];
  for (let e = Math.round(Math.log10(lo)); 10 ** e <= hi * 1.0001; e++) out.push(10 ** e);
  return out;
}

/** y domain in % of optimal: always shows 0 (random) and 100 (Bayes), snapped to 20s. */
export function yDomain(values: Num[], step = 20): [number, number] {
  const v = values.filter(isNum);
  const lo = Math.min(0, ...v);
  const hi = Math.max(100, ...v);
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

/**
 * Score at x decisions, interpolated linearly in log-decisions.
 * Static arms hold their score everywhere; a learner returns null past the end of its run,
 * because we don't know what more training would have done.
 */
export function valueAt(curve: Curve | undefined, x: number): Num {
  const pts = cleanPoints(curve?.mean);
  if (!pts.length) return null;
  if (pts.length === 1) return pts[0]!.score;
  return interp(pts, x, (p) => p.score);
}

export function secondsAt(curve: Curve | undefined, x: number): Num {
  const pts = cleanPoints(curve?.mean).filter((p) => isNum(p.seconds));
  if (!pts.length) return null;
  if (pts.length === 1) return pts[0]!.seconds ?? null;
  return interp(pts, x, (p) => p.seconds ?? null);
}

function interp(pts: CurvePoint[], x: number, get: (p: CurvePoint) => Num): Num {
  const first = pts[0]!;
  const last = pts[pts.length - 1]!;
  if (x <= first.decisions) return get(first);
  if (x > last.decisions) return null;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1]!;
    const b = pts[i]!;
    if (x <= b.decisions) {
      const ya = get(a);
      const yb = get(b);
      if (!isNum(ya) || !isNum(yb)) return null;
      if (x === b.decisions) return yb;
      const la = Math.log10(Math.max(a.decisions, 1));
      const lb = Math.log10(Math.max(b.decisions, 1));
      const t = lb === la ? 1 : (Math.log10(Math.max(x, 1)) - la) / (lb - la);
      return ya + t * (yb - ya);
    }
  }
  return get(last);
}

/** Every distinct decision count across curves: the stops a keyboard user steps through. */
export function decisionStops(curves: (Curve | undefined)[]): number[] {
  const s = new Set<number>();
  curves.forEach((c) => cleanPoints(c?.mean).forEach((p) => s.add(p.decisions)));
  return [...s].sort((a, b) => a - b);
}

export function bandPoints(
  curve: Curve | undefined,
): { decisions: number; lo: number; hi: number }[] {
  return (curve?.band ?? [])
    .filter((b: BandPoint) => b.decisions > 0 && isNum(b.lo) && isNum(b.hi))
    .map((b) => ({ decisions: b.decisions, lo: b.lo as number, hi: b.hi as number }));
}

export interface ThresholdMark {
  threshold: number;
  decisions: Num;
}

export function thresholdMarks(
  eff: EfficiencyByArm | undefined,
  arm: string,
  thresholds: number[],
): ThresholdMark[] {
  const t = eff?.[arm]?.to_threshold ?? {};
  return thresholds.map((th) => ({ threshold: th, decisions: lookupThreshold(t, th) }));
}

/** Threshold keys are written as "80" or "80.0" depending on the exporter; accept both. */
export function lookupThreshold(t: Record<string, Num>, th: number): Num {
  const keys = [String(th), th.toFixed(1), String(Math.round(th))];
  for (const k of keys) if (k in t) return t[k] ?? null;
  return null;
}

export interface Crossover {
  slow: string;
  fast: string;
  decisions: number;
}

/** "_crossover" keys read "ppo_sig>kev_rl": the PPO curve passes the Kev curve at that decision count. */
export function crossovers(eff: EfficiencyByArm | undefined): Crossover[] {
  const raw = eff?._crossover ?? {};
  return Object.entries(raw)
    .map(([k, v]) => {
      const [slow, fast] = k.split('>');
      return slow && fast && isNum(v) ? { slow, fast, decisions: v } : null;
    })
    .filter((c): c is Crossover => c !== null)
    .sort((a, b) => a.decisions - b.decisions);
}

/** Dollar rate per training hour implied by the cost table (GPU arms only). */
export function usdPerHour(cost: CostRow | undefined): Num {
  if (!cost || !isNum(cost.train_hours) || !isNum(cost.train_usd)) return null;
  if (cost.train_hours <= 0 || cost.train_usd <= 0) return null;
  return cost.train_usd / cost.train_hours;
}

/** "Cost so far" at a point on the curve: compute time, and dollars when the arm trains on a GPU. */
export function costSoFar(
  curve: Curve | undefined,
  cost: CostRow | undefined,
  x: number,
): { seconds: Num; usd: Num } {
  const seconds = secondsAt(curve, x);
  const rate = usdPerHour(cost);
  return { seconds, usd: isNum(seconds) && isNum(rate) ? (seconds / 3600) * rate : null };
}
