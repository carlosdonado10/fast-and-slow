import type { Hypothesis, Verdict } from '../types';

export function normalizeVerdict(v: string | null | undefined): Verdict {
  const s = (v ?? '').trim().toLowerCase();
  if (s === 'supported' || s === 'not supported' || s === 'inconclusive') return s;
  return 'pending';
}

export interface VerdictStyle {
  label: string;
  /** Drives the glyph and the CSS modifier; never colour alone. */
  tone: 'held' | 'failed' | 'mixed' | 'open';
}

export const VERDICT_STYLE: Record<Verdict, VerdictStyle> = {
  supported: { label: 'Supported', tone: 'held' },
  'not supported': { label: 'Not supported', tone: 'failed' },
  inconclusive: { label: 'Inconclusive', tone: 'mixed' },
  pending: { label: 'Pending', tone: 'open' },
};

export function verdictStyle(v: string | null | undefined): VerdictStyle {
  return VERDICT_STYLE[normalizeVerdict(v)];
}

/** Plain-language clause per hypothesis and verdict, used to write the one-line answer. */
const CLAUSES: Record<string, Partial<Record<Verdict, string>>> = {
  H1: {
    supported: 'it starts strong where RL starts near random',
    'not supported': "it doesn't start ahead of RL",
    inconclusive: 'its head start over RL is unclear',
  },
  H2: {
    supported: 'it improves more slowly than RL, which catches up',
    'not supported': 'it learns at least as fast as RL',
    inconclusive: 'whether RL catches up is unclear',
  },
  H3: {
    supported: 'it loses ground as plans get longer',
    'not supported': 'it holds up on longer plans',
    inconclusive: 'longer plans give a mixed picture',
  },
  H4: {
    supported: 'its confidence is honest enough to hand off the decidable calls',
    'not supported': "its confidence isn't yet good enough to hand off safely",
    inconclusive: "it's unclear whether its confidence is good enough to hand off",
  },
  H5: {
    supported: 'worked examples in the prompt recover part of the training gain for free',
    'not supported': "worked examples in the prompt don't help",
    inconclusive: 'worked examples in the prompt help a little, if at all',
  },
};

export function clauseFor(h: Pick<Hypothesis, 'id' | 'verdict'>): string | null {
  return CLAUSES[h.id]?.[normalizeVerdict(h.verdict)] ?? null;
}

export function tally(hypotheses: Pick<Hypothesis, 'verdict'>[]): Record<Verdict, number> {
  const t: Record<Verdict, number> = {
    supported: 0,
    'not supported': 0,
    inconclusive: 0,
    pending: 0,
  };
  hypotheses.forEach((h) => (t[normalizeVerdict(h.verdict)] += 1));
  return t;
}

/**
 * The hero's one-line answer. The thesis is about hand-off (H4), so it leads; the
 * learning story (H1, H2) follows. Built only from verdicts the export step computed.
 */
export function heroAnswer(hypotheses: Hypothesis[]): { lead: string; detail: string } {
  const byId = new Map(hypotheses.map((h) => [h.id, h]));
  const h4 = byId.get('H4');
  const v4 = normalizeVerdict(h4?.verdict);
  const lead =
    v4 === 'supported'
      ? 'Yes, for the decidable calls.'
      : v4 === 'not supported'
        ? 'Not yet.'
        : v4 === 'inconclusive'
          ? 'Not clearly.'
          : 'Waiting on the runs.';
  const parts = ['H4', 'H1', 'H2']
    .map((id) => byId.get(id))
    .filter((h): h is Hypothesis => !!h && normalizeVerdict(h.verdict) !== 'pending')
    .map(clauseFor)
    .filter((c): c is string => !!c);
  if (!parts.length) {
    return { lead, detail: 'No hypothesis has enough runs behind it to call yet.' };
  }
  const sentence =
    parts.length === 1
      ? parts[0]!
      : `${parts.slice(0, -1).join('; ')}; and ${parts[parts.length - 1]!}`;
  return { lead, detail: `${sentence.charAt(0).toUpperCase()}${sentence.slice(1)}.` };
}
