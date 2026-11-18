import { describe, expect, it, vi } from 'vitest';
import type { ArmResult, Hypothesis, Replay } from '../types';
import { armColor, armDash, armLabel, armMarker, familyOf, isLearner, sortArms } from './arms';
import { clauseFor, heroAnswer, normalizeVerdict, tally, verdictStyle } from './verdict';
import {
  automationAt,
  budgetKey,
  handoffGap,
  handoffRows,
  handoffSentence,
  reliabilityPoints,
} from './handoff';
import { compareOrder, mostLikely, orderOptions, parseDeskState, posteriorRows } from './desk';
import {
  defaultPair,
  dialArms,
  dialSlope,
  disagreements,
  maxOrder,
  noteSourceLabel,
  replayArms,
} from './replay';
import { loadRun, NoRunsYet, runFromUrl } from './load';
import { spreadLabels, xLayout } from './layout';

describe('arms', () => {
  it('assigns families from the report, falling back to the id', () => {
    expect(familyOf('kev_rl')).toBe('system-one');
    expect(familyOf('ppo_num')).toBe('classic-rl');
    expect(familyOf('bayes')).toBe('reference');
    expect(
      familyOf('custom', {
        arms: { custom: { label: 'C', group: 'classic-rl', sees: '', role: '' } },
      }),
    ).toBe('classic-rl');
    expect(isLearner('kev_zero')).toBe(true);
    expect(isLearner('oracle')).toBe(false);
  });

  it('sorts in a fixed order so colour follows the arm', () => {
    expect(sortArms(['random', 'ppo_sig', 'zzz', 'kev_zero', 'bayes'])).toEqual([
      'kev_zero',
      'ppo_sig',
      'bayes',
      'random',
      'zzz',
    ]);
  });

  it('encodes references as dashes and families as shapes', () => {
    expect(armColor('kev_rl')).toBe('var(--arm-kev_rl)');
    expect(armColor('unknown')).toBe('var(--ink-3)');
    expect(armDash('kev_rl')).toBeUndefined();
    for (const a of ['bayes', 'oracle', 'basestock', 'random']) expect(armDash(a)).toBeTruthy();
    expect(armMarker('kev_icl')).toBe('circle');
    expect(armMarker('ppo_emb')).toBe('square');
    expect(armMarker('bayes')).toBe('none');
    expect(armLabel({ arms: {} }, 'x')).toBe('x');
  });
});

describe('verdicts', () => {
  const hyp = (id: string, verdict: string): Hypothesis => ({
    id,
    verdict,
    statement: '',
    rule: '',
    evidence: '',
  });

  it('normalizes unknown verdicts to pending', () => {
    expect(normalizeVerdict(' Supported ')).toBe('supported');
    expect(normalizeVerdict('NOT SUPPORTED')).toBe('not supported');
    expect(normalizeVerdict('maybe')).toBe('pending');
    expect(normalizeVerdict(null)).toBe('pending');
  });

  it('styles each verdict with a label and a tone (never colour alone)', () => {
    expect(verdictStyle('supported')).toEqual({ label: 'Supported', tone: 'held' });
    expect(verdictStyle('not supported').tone).toBe('failed');
    expect(verdictStyle('inconclusive').tone).toBe('mixed');
    expect(verdictStyle(undefined).tone).toBe('open');
  });

  it('tallies verdicts', () => {
    expect(tally([hyp('H1', 'supported'), hyp('H2', 'pending'), hyp('H3', 'supported')])).toEqual({
      supported: 2,
      'not supported': 0,
      inconclusive: 0,
      pending: 1,
    });
  });

  it('writes the one-line answer from H4 first, then H1 and H2', () => {
    const a = heroAnswer([
      hyp('H1', 'supported'),
      hyp('H2', 'supported'),
      hyp('H4', 'not supported'),
    ]);
    expect(a.lead).toBe('Not yet.');
    expect(a.detail).toMatch(/^Its confidence isn't yet good enough/);
    expect(a.detail).toMatch(/; and it improves more slowly than RL, which catches up\.$/);
    expect(heroAnswer([hyp('H4', 'supported')]).lead).toBe('Yes, for the decidable calls.');
    expect(heroAnswer([hyp('H4', 'inconclusive')]).detail).toMatch(/^It's unclear/);
  });

  it('says plainly when nothing can be called yet', () => {
    const a = heroAnswer([hyp('H1', 'pending')]);
    expect(a.lead).toBe('Waiting on the runs.');
    expect(a.detail).toMatch(/No hypothesis/);
    expect(clauseFor(hyp('H9', 'supported'))).toBeNull();
  });
});

const result = {
  handoff: {
    all: { n: 10, accuracy: 0.5, automation: { '0.05': 0.3 } },
    decidable: { n: 4, accuracy: 0.9, automation: { '0.05': 0.75, '0.10': 0.8 } },
    judgement: { n: 6, accuracy: 0.2, automation: { '0.05': 0.1 } },
  },
  calibration: {
    ece: 0.05,
    brier: 0.2,
    bins: [
      { lo: 0.8, hi: 0.9, confidence: 0.85, accuracy: 0.8, count: 5 },
      { lo: 0.1, hi: 0.2, confidence: 0.15, accuracy: 0.2, count: 3 },
      { lo: 0.5, hi: 0.6, confidence: null, accuracy: null, count: 0 },
    ],
  },
} as unknown as ArmResult;

describe('hand-off', () => {
  it('reads budgets with two-decimal keys', () => {
    expect(budgetKey(0.1)).toBe('0.10');
    expect(automationAt(result.handoff.decidable, 0.1)).toBe(0.8);
    expect(automationAt(result.handoff.decidable, 0.02)).toBeNull();
    expect(automationAt(undefined, 0.05)).toBeNull();
  });

  it('builds the decidable / judgement / all rows', () => {
    const rows = handoffRows(result, 0.05);
    expect(rows.map((r) => [r.split, r.automation])).toEqual([
      ['decidable', 0.75],
      ['judgement', 0.1],
      ['all', 0.3],
    ]);
    expect(handoffRows(undefined, 0.05)[0]).toMatchObject({ n: 0, automation: null });
  });

  it('computes the H4 gap and the planner sentence', () => {
    expect(handoffGap(result, 0.05)).toBeCloseTo(65);
    expect(handoffGap(result, 0.02)).toBeNull();
    expect(handoffSentence('Kev', result, 0.05)).toBe(
      'At a 5% error budget, Kev could take 75% of the decidable calls and 10% of the judgement calls on its own.',
    );
    expect(handoffSentence('Kev', undefined, 0.05)).toMatch(/an unknown share/);
  });

  it('drops empty reliability bins and sorts by confidence', () => {
    expect(reliabilityPoints(result.calibration.bins).map((p) => p.confidence)).toEqual([
      0.15, 0.85,
    ]);
    expect(reliabilityPoints(undefined)).toEqual([]);
  });
});

describe('desk', () => {
  const state =
    'You are the planner. Usual demand is 10.\n\nWeek 2 of 4 (3 including this one left in the plan).\nStock on hand now: 6 units.\nPast weeks (ordered / demand / sold / missed):\n- Week 1: 15 / 9 / 9 / 0\nInbox:\n- (week 1) Rep says flat.\n- (this week) Customer confirmed steady weeks.\n- stray line';

  it('parses the week view out of the agent prompt', () => {
    const v = parseDeskState(state);
    expect(v.brief).toBe('You are the planner. Usual demand is 10.');
    expect(v.weekLine).toMatch(/^Week 2 of 4/);
    expect(v.stock).toBe(6);
    expect(v.past).toEqual([{ week: 1, ordered: 15, demand: 9, sold: 9, missed: 0 }]);
    expect(v.inbox).toEqual([
      { when: 'week 1', current: false, text: 'Rep says flat.' },
      { when: 'this week', current: true, text: 'Customer confirmed steady weeks.' },
      { when: '', current: false, text: 'stray line' },
    ]);
  });

  it('copes with a brief-only state', () => {
    const v = parseDeskState('Just a brief.');
    expect(v).toMatchObject({ brief: 'Just a brief.', stock: null, past: [], inbox: [] });
  });

  it('reads order options from the question, or the order menu', () => {
    const q = {
      order: {
        type: 'choice',
        instructions: '',
        criteria: { order_10: 'Order 10', order_0: 'Nothing' },
      },
    };
    expect(orderOptions(q, [])).toEqual([
      { value: 0, label: 'Nothing' },
      { value: 10, label: 'Order 10' },
    ]);
    expect(orderOptions(undefined, [0, 5])).toEqual([
      { value: 0, label: 'Order nothing' },
      { value: 5, label: 'Order 5 units' },
    ]);
  });

  it('orders the posterior by regime, from an object or an array', () => {
    const regimes = ['low', 'normal', 'surge'];
    expect(posteriorRows({ surge: 0.5, low: 0.2, normal: null }, regimes)).toEqual([
      { regime: 'low', p: 0.2 },
      { regime: 'normal', p: 0 },
      { regime: 'surge', p: 0.5 },
    ]);
    expect(posteriorRows([0.1, 0.8, 0.1], regimes)[1]).toEqual({ regime: 'normal', p: 0.8 });
    expect(posteriorRows(undefined, regimes)).toEqual([]);
    expect(mostLikely(posteriorRows([0.1, 0.8, 0.1], regimes))?.regime).toBe('normal');
    expect(mostLikely([])).toBeNull();
  });

  it('compares the reader’s pick with Bayes', () => {
    expect(compareOrder(null, 10)).toMatch(/skipped/);
    expect(compareOrder(10, 10)).toMatch(/Same call/);
    expect(compareOrder(20, 10)).toBe('You ordered 10 more than the Bayes-optimal planner.');
    expect(compareOrder(5, 10)).toBe('You ordered 5 fewer than the Bayes-optimal planner.');
  });
});

describe('replays and dial', () => {
  const d = (order: number) => ({
    order,
    confidence: 0.5,
    acceptable: true,
    inventory: 0,
    decidable: true,
    bayes_order: order,
    posterior: [0.2, 0.6, 0.2],
    reward: 1,
  });
  const replay: Replay = {
    horizon: 2,
    seed: 1,
    totals: {},
    weeks: [
      {
        week: 1,
        note: null,
        note_source: 'none',
        note_claim: null,
        true_regime: 'low',
        demand: 3,
        decisions: { ppo_sig: d(10), kev_rl: d(10), bayes: d(5) },
      },
      {
        week: 2,
        note: 'x',
        note_source: 'sales_rep',
        note_claim: 'up',
        true_regime: 'surge',
        demand: 20,
        decisions: { ppo_sig: d(20), kev_rl: d(10) },
      },
    ],
  };

  it('finds arms, a default pair and the weeks they disagree', () => {
    expect(replayArms(replay)).toEqual(['kev_rl', 'ppo_sig', 'bayes']);
    expect(replayArms(undefined)).toEqual([]);
    expect(defaultPair(['kev_zero', 'ppo_num'])).toEqual({ kev: 'kev_zero', ppo: 'ppo_num' });
    expect(defaultPair([])).toEqual({ kev: null, ppo: null });
    expect(disagreements(replay, 'kev_rl', 'ppo_sig')).toEqual([2]);
    expect(disagreements(replay, 'kev_rl', 'missing')).toEqual([]);
  });

  it('labels note sources', () => {
    expect(noteSourceLabel('none')).toBe('No note');
    expect(noteSourceLabel(null)).toBe('No note');
    expect(noteSourceLabel('distractor')).toMatch(/distractor/);
    expect(noteSourceLabel('customer_order')).toBe('Customer order');
  });

  it('picks dial panels and their slope', () => {
    const pt = (horizon: number, v: number | null) => ({
      horizon,
      pct_optimal: v,
      ci: [],
      inbox_value: null,
    });
    expect(
      dialArms({
        random: [pt(1, 0)],
        bayes: [pt(1, 100)],
        kev_zero: [pt(1, 90)],
        ppo_sig: [],
        oracle: [pt(1, 120)],
      }),
    ).toEqual(['kev_zero', 'oracle']);
    expect(dialSlope([pt(4, 60), pt(1, 95)])).toBe(-35);
    expect(dialSlope([pt(1, 95), pt(4, null)])).toBeNull();
    expect(dialSlope(undefined)).toBeNull();
    expect(maxOrder({ meta: { world: { order_menu: [0, 5, 40] } } } as never)).toBe(40);
  });
});

describe('layout', () => {
  it('gives zero its own slot and inverts back to it', () => {
    const L = xLayout(800, [10, 10000], { top: 0, right: 100, bottom: 0, left: 40 });
    expect(L.x(0)).toBe(L.zeroX);
    expect(L.x(10)).toBe(L.logStart);
    expect(L.x(10000)).toBe(700);
    expect(L.invert(L.zeroX)).toBe(0);
    expect(L.invert(700)).toBeCloseTo(10000);
  });

  it('spreads labels so none overlap, inside the bounds', () => {
    const out = spreadLabels([{ y: 10 }, { y: 12 }, { y: 100 }], 14, 0, 105);
    expect(out.map((o) => o.ly)).toEqual([10, 24, 100]);
    const clamped = spreadLabels([{ y: 98 }, { y: 100 }], 14, 0, 100);
    expect(clamped.map((o) => o.ly)).toEqual([86, 100]);
    const low = spreadLabels([{ y: -5 }], 14, 0, 100);
    expect(low[0]!.ly).toBe(0);
  });
});

describe('loadRun', () => {
  const report = { meta: {}, results: {}, hypotheses: [] };
  const run = { id: 'r1', label: 'A100 run', report };
  const res = (status: number, body?: unknown) =>
    Promise.resolve({
      ok: status < 400,
      status,
      json: () =>
        body === undefined ? Promise.reject(new SyntaxError('<')) : Promise.resolve(body),
    } as Response);
  const routes =
    (map: Record<string, () => Promise<Response>>) =>
    (url: string): Promise<Response> =>
      (map[url] ?? (() => res(500, {})))();

  it('opens the newest run and lists every run', async () => {
    const f = vi.fn(
      routes({ 'api/runs/latest': () => res(200, run), 'api/runs': () => res(200, [run]) }),
    );
    const out = await loadRun(null, f as unknown as typeof fetch);
    expect(out.run.id).toBe('r1');
    expect(out.runs).toHaveLength(1);
  });

  it('opens a named run, escaping its id', async () => {
    const f = vi.fn(
      routes({ 'api/runs/a%2Fb': () => res(200, run), 'api/runs': () => res(200, {}) }),
    );
    const out = await loadRun('a/b', f as unknown as typeof fetch);
    expect(out.run.label).toBe('A100 run');
    expect(out.runs).toEqual([]);
  });

  it('tells an empty store apart from a missing run', async () => {
    const none = vi.fn(
      routes({ 'api/runs/latest': () => res(404, {}), 'api/runs': () => res(200, []) }),
    );
    await expect(loadRun(null, none as unknown as typeof fetch)).rejects.toBeInstanceOf(NoRunsYet);
    const gone = vi.fn(
      routes({ 'api/runs/x': () => res(404, {}), 'api/runs': () => res(200, []) }),
    );
    await expect(loadRun('x', gone as unknown as typeof fetch)).rejects.toThrow(/no run "x"/);
  });

  it('explains API failures in plain words', async () => {
    const down = vi.fn(() => Promise.reject(new TypeError('Failed to fetch')));
    await expect(loadRun(null, down as unknown as typeof fetch)).rejects.toThrow(
      /Couldn't reach the runs API.*Failed to fetch/,
    );
    const err = vi.fn(
      routes({ 'api/runs/latest': () => res(503, {}), 'api/runs': () => res(200, []) }),
    );
    await expect(loadRun(null, err as unknown as typeof fetch)).rejects.toThrow(/HTTP 503/);
    const html = vi.fn(
      routes({ 'api/runs/latest': () => res(200), 'api/runs': () => res(200, []) }),
    );
    await expect(loadRun(null, html as unknown as typeof fetch)).rejects.toThrow(
      /didn't return JSON/,
    );
    const odd = vi.fn(
      routes({ 'api/runs/latest': () => res(200, { id: 'r' }), 'api/runs': () => res(200, []) }),
    );
    await expect(loadRun(null, odd as unknown as typeof fetch)).rejects.toThrow(
      /didn't return a report/,
    );
  });

  it('reads the run from the address bar', () => {
    expect(runFromUrl('?run=2026-11-18T0200-ab12')).toBe('2026-11-18T0200-ab12');
    expect(runFromUrl('')).toBeNull();
  });
});
