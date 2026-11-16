import type { DeskQuestion, Num } from '../types';
import { isNum } from './format';

export interface PastWeek {
  week: number;
  ordered: number;
  demand: number;
  sold: number;
  missed: number;
}

export interface InboxNote {
  when: string;
  current: boolean;
  text: string;
}

export interface DeskView {
  brief: string;
  weekLine: string | null;
  stock: number | null;
  past: PastWeek[];
  inbox: InboxNote[];
}

/**
 * The desk state is the exact text the agents read. The first paragraph is the standing brief;
 * the rest is this week's view. Parse the week's view into parts so it can be laid out, and keep
 * anything unrecognised out rather than guessing.
 */
export function parseDeskState(state: string): DeskView {
  const [brief = '', ...rest] = state.split(/\n\s*\n/);
  const lines = rest
    .join('\n')
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean);
  const view: DeskView = { brief: brief.trim(), weekLine: null, stock: null, past: [], inbox: [] };
  let mode: 'none' | 'past' | 'inbox' = 'none';
  for (const line of lines) {
    if (/^week \d+ of \d+/i.test(line)) {
      view.weekLine = line;
      mode = 'none';
      continue;
    }
    const stock = line.match(/^stock on hand now:\s*(-?\d+)/i);
    if (stock) {
      view.stock = Number(stock[1]);
      mode = 'none';
      continue;
    }
    if (/^past weeks/i.test(line)) {
      mode = 'past';
      continue;
    }
    if (/^inbox/i.test(line)) {
      mode = 'inbox';
      continue;
    }
    if (mode === 'past') {
      const m = line.match(/^-\s*week\s*(\d+):\s*(\d+)\s*\/\s*(\d+)\s*\/\s*(\d+)\s*\/\s*(\d+)/i);
      if (m) {
        view.past.push({
          week: Number(m[1]),
          ordered: Number(m[2]),
          demand: Number(m[3]),
          sold: Number(m[4]),
          missed: Number(m[5]),
        });
      }
    } else if (mode === 'inbox') {
      const m = line.match(/^-\s*\(([^)]+)\)\s*(.*)$/);
      if (m) {
        const when = m[1]!.trim();
        view.inbox.push({ when, current: /this week/i.test(when), text: m[2]!.trim() });
      } else if (line.startsWith('-')) {
        view.inbox.push({ when: '', current: false, text: line.replace(/^-\s*/, '') });
      }
    }
  }
  return view;
}

export interface OrderOption {
  value: number;
  label: string;
}

/** Order options come from the question the agents answer; fall back to the world's order menu. */
export function orderOptions(
  questions: Record<string, DeskQuestion> | undefined,
  menu: number[],
): OrderOption[] {
  const crit = questions?.order?.criteria;
  if (crit && Object.keys(crit).length) {
    return Object.entries(crit)
      .map(([k, label]) => ({ value: Number(k.replace(/\D+/g, '')), label }))
      .filter((o) => Number.isFinite(o.value))
      .sort((a, b) => a.value - b.value);
  }
  return menu.map((v) => ({ value: v, label: v === 0 ? 'Order nothing' : `Order ${v} units` }));
}

export interface PosteriorRow {
  regime: string;
  p: number;
}

export function posteriorRows(
  posterior: Record<string, Num> | Num[] | undefined,
  regimes: string[],
): PosteriorRow[] {
  if (!posterior) return [];
  if (Array.isArray(posterior)) {
    return regimes.map((r, i) => ({ regime: r, p: isNum(posterior[i]) ? posterior[i] : 0 }));
  }
  return regimes.map((r) => ({ regime: r, p: isNum(posterior[r]) ? (posterior[r] as number) : 0 }));
}

export function mostLikely(rows: PosteriorRow[]): PosteriorRow | null {
  return rows.reduce<PosteriorRow | null>((best, r) => (!best || r.p > best.p ? r : best), null);
}

/** How the planner's own pick compares with the Bayes-optimal order. */
export function compareOrder(mine: number | null, bayes: number): string {
  if (mine === null) return 'You skipped this one.';
  if (mine === bayes) return 'Same call as the Bayes-optimal planner.';
  const d = mine - bayes;
  return `You ordered ${Math.abs(d)} ${d > 0 ? 'more' : 'fewer'} than the Bayes-optimal planner.`;
}

export const REGIME_LABEL: Record<string, string> = {
  low: 'Slow spell',
  normal: 'Usual demand',
  surge: 'Surge',
};
