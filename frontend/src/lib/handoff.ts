import type { ArmResult, HandoffSplit, Num, ReliabilityBin } from '../types';
import { isNum } from './format';

/** Budgets are exported as keys like "0.05" and "0.10" (two decimals). */
export function budgetKey(b: number): string {
  return b.toFixed(2);
}

export function automationAt(split: HandoffSplit | undefined, budget: number): Num {
  const a = split?.automation;
  if (!a) return null;
  for (const k of [budgetKey(budget), String(budget), budget.toFixed(1)]) {
    if (k in a) return a[k] ?? null;
  }
  return null;
}

export type SplitName = 'decidable' | 'judgement' | 'all';

export interface HandoffRow {
  split: SplitName;
  label: string;
  n: number;
  automation: Num;
  accuracy: Num;
}

const SPLIT_LABEL: Record<SplitName, string> = {
  decidable: 'Decidable calls',
  judgement: 'Judgement calls',
  all: 'All calls',
};

export function handoffRows(result: ArmResult | undefined, budget: number): HandoffRow[] {
  return (['decidable', 'judgement', 'all'] as const).map((split) => {
    const s = result?.handoff?.[split];
    return {
      split,
      label: SPLIT_LABEL[split],
      n: s?.n ?? 0,
      automation: automationAt(s, budget),
      accuracy: s?.accuracy ?? null,
    };
  });
}

/** Points for a reliability diagram; bins without data are dropped, not drawn at zero. */
export function reliabilityPoints(
  bins: ReliabilityBin[] | undefined,
): { confidence: number; accuracy: number; count: number; lo: number; hi: number }[] {
  return (bins ?? [])
    .filter((b) => isNum(b.confidence) && isNum(b.accuracy) && b.count > 0)
    .map((b) => ({
      confidence: b.confidence as number,
      accuracy: b.accuracy as number,
      count: b.count,
      lo: b.lo,
      hi: b.hi,
    }))
    .sort((a, b) => a.confidence - b.confidence);
}

/** Decidable minus judgement automation, in points: the H4 gap a planner cares about. */
export function handoffGap(result: ArmResult | undefined, budget: number): Num {
  const d = automationAt(result?.handoff?.decidable, budget);
  const j = automationAt(result?.handoff?.judgement, budget);
  return isNum(d) && isNum(j) ? 100 * (d - j) : null;
}

export function handoffSentence(
  label: string,
  result: ArmResult | undefined,
  budget: number,
): string {
  const [d, j] = handoffRows(result, budget);
  const pct = (x: Num) => (isNum(x) ? `${Math.round(x * 100)}%` : 'an unknown share');
  return `At a ${Math.round(budget * 100)}% error budget, ${label} could take ${pct(
    d?.automation ?? null,
  )} of the decidable calls and ${pct(j?.automation ?? null)} of the judgement calls on its own.`;
}
