import type { Num, ShiftResult, ShiftTimeline, Strategy } from '../types';
import { isNum } from './format';

/** Drawing order: quiet references first, the bridge (the thing H6 is about) last, on top. */
export const STRATEGIES: readonly Strategy[] = ['stale', 'retrain', 'kev', 'bridge'];

export const STRATEGY_LABEL: Record<Strategy, string> = {
  stale: 'Keep the old RL policy',
  retrain: 'Retrain RL, nothing in between',
  kev: 'Hand everything to Kev',
  bridge: 'Bridge: Kev until retrained RL wins',
};

/** Colours reuse the arm tokens, so strategies read in the families used everywhere else in the report. */
export const STRATEGY_STYLE: Record<Strategy, { color: string; dash?: string; width: number }> = {
  stale: { color: 'var(--arm-basestock)', dash: '6 4', width: 1.5 },
  retrain: { color: 'var(--arm-ppo_emb)', width: 1.75 },
  kev: { color: 'var(--arm-kev_zero)', dash: '2 3', width: 1.75 },
  bridge: { color: 'var(--arm-kev_rl)', width: 3 },
};

export const TRIGGER_LABEL = {
  declared: 'A planner flags it',
  detected: 'Kev’s alarm flags it',
} as const;
export const MODE_LABEL = { sim: 'Rebuild the simulator', live: 'Learn from live weeks' } as const;

export function timelineKey(trigger: string, mode: string, variant: string): string {
  return `${trigger}|${mode}|${variant}`;
}

export function variantsOf(shift: ShiftResult): string[] {
  return Object.keys(shift.ppo);
}

export function getTimeline(
  shift: ShiftResult,
  trigger: string,
  mode: string,
  variant: string,
): ShiftTimeline | undefined {
  return shift.timelines[timelineKey(trigger, mode, variant)];
}

/** Weeks after the shift, or a plain-language fallback. */
export function fmtWeeksAfter(w: number | null | undefined, never = 'not within the year'): string {
  if (w === null || w === undefined) return never;
  if (w === 0) return 'the same week';
  return `${w} week${w === 1 ? '' : 's'}`;
}

/** The strategy that lost the least, with how much the bridge saved against the better RL-only choice. */
export function summarise(t: ShiftTimeline): {
  best: Strategy;
  saving: Num;
  rlOnly: Num;
  bridge: Num;
} {
  const finite = STRATEGIES.filter((s) => isNum(t.regret_usd[s]));
  const best = finite.reduce<Strategy>(
    (b, s) => ((t.regret_usd[s] as number) < (t.regret_usd[b] as number) ? s : b),
    finite[0] ?? 'bridge',
  );
  const rl = [t.regret_usd.stale, t.regret_usd.retrain].filter(isNum);
  return {
    best,
    saving: t.bridge_saving,
    rlOnly: rl.length ? Math.min(...rl) : null,
    bridge: t.regret_usd.bridge,
  };
}

/** SVG path for a weekly series, drawn as steps: a week's decision holds for the whole week, so nothing slants
 *  between weeks. Week w spans x(w) to x(w + 1); gaps (null) break the line. */
export function weeklyPath(
  values: Num[],
  x: (w: number) => number,
  y: (v: number) => number,
): string {
  let d = '';
  let pen = false;
  values.forEach((v, w) => {
    if (!isNum(v)) {
      pen = false;
      return;
    }
    d += pen ? `V ${y(v)} H ${x(w + 1)} ` : `M ${x(w)} ${y(v)} H ${x(w + 1)} `;
    pen = true;
  });
  return d.trim();
}

/** Area between a strategy's weekly score and 100% (optimal), from the shift on: the regret, drawn. */
export function regretArea(
  values: Num[],
  from: number,
  x: (w: number) => number,
  y: (v: number) => number,
): string {
  const pts = values
    .map((v, w) => [w, v] as const)
    .filter((p): p is readonly [number, number] => p[0] >= from && isNum(p[1]));
  if (!pts.length) return '';
  const first = pts[0]![0];
  const last = pts[pts.length - 1]![0] + 1;
  const steps = [...pts]
    .reverse()
    .map(([w, v]) => `${x(w + 1)} ${y(Math.min(v, 100))} L ${x(w)} ${y(Math.min(v, 100))}`)
    .join(' L ');
  return `M ${x(first)} ${y(100)} L ${x(last)} ${y(100)} L ${steps} Z`;
}
