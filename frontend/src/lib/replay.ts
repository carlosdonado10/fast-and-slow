import type { DialPoint, Num, Replay, Report } from '../types';
import { sortArms } from './arms';
import { capitalize, humanize, isNum } from './format';

export function replayArms(replay: Replay | undefined): string[] {
  const s = new Set<string>();
  replay?.weeks.forEach((w) => Object.keys(w.decisions).forEach((a) => s.add(a)));
  return sortArms(s);
}

const KEV_PREF = ['kev_rl', 'kev_icl', 'kev_zero'];
const PPO_PREF = ['ppo_emb', 'ppo_sig', 'ppo_num'];

export function defaultPair(arms: string[]): { kev: string | null; ppo: string | null } {
  return {
    kev: KEV_PREF.find((a) => arms.includes(a)) ?? null,
    ppo: PPO_PREF.find((a) => arms.includes(a)) ?? null,
  };
}

/** Weeks where the two agents ordered different amounts. */
export function disagreements(replay: Replay | undefined, a: string, b: string): number[] {
  return (replay?.weeks ?? [])
    .filter((w) => {
      const x = w.decisions[a];
      const y = w.decisions[b];
      return x && y && x.order !== y.order;
    })
    .map((w) => w.week);
}

export function maxOrder(report: Pick<Report, 'meta'>): number {
  return Math.max(1, ...report.meta.world.order_menu);
}

/** Arms worth a small multiple: learners and the two informative references. */
export function dialArms(dial: Record<string, DialPoint[]>): string[] {
  return sortArms(Object.keys(dial)).filter(
    (a) => a !== 'random' && a !== 'bayes' && (dial[a]?.length ?? 0) > 0,
  );
}

/** Change from the shortest to the longest horizon, in points of % optimal. */
export function dialSlope(points: DialPoint[] | undefined): Num {
  const pts = (points ?? [])
    .filter((p) => isNum(p.pct_optimal))
    .sort((a, b) => a.horizon - b.horizon);
  if (pts.length < 2) return null;
  return (pts[pts.length - 1]!.pct_optimal as number) - (pts[0]!.pct_optimal as number);
}

/** Signal notes carry their source; other weeks carry the note kind ("none", "distractor"). */
export function noteSourceLabel(source: string | null | undefined): string {
  if (!source || source === 'none') return 'No note';
  if (source === 'distractor') return 'Irrelevant note (a distractor)';
  return capitalize(humanize(source));
}
