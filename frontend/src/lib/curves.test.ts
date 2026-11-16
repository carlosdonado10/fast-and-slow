import { describe, expect, it } from 'vitest';
import type { Curve, EfficiencyByArm } from '../types';
import {
  bandPoints,
  cleanPoints,
  costSoFar,
  crossovers,
  curveArms,
  decisionStops,
  isStatic,
  logDomain,
  logTicks,
  lookupThreshold,
  secondsAt,
  splitZero,
  thresholdMarks,
  usdPerHour,
  valueAt,
  yDomain,
} from './curves';

const ppo: Curve = {
  mean: [
    { decisions: 0, score: -88, seconds: 0 },
    { decisions: 100, score: -88, seconds: 0.01 },
    { decisions: 1000, score: 40, seconds: 0.1 },
    { decisions: 10000, score: 96, seconds: 1 },
  ],
  seeds: [],
  band: [
    { decisions: 0, lo: -90, hi: -86 },
    { decisions: 100, lo: -90, hi: -86 },
    { decisions: 1000, lo: null, hi: 44 },
    { decisions: 10000, lo: 90, hi: 99 },
  ],
};
const kevZero: Curve = { mean: [{ decisions: 0, score: 65 }], seeds: [], band: [] };

describe('curve points', () => {
  it('drops points without a score and sorts by decisions', () => {
    const pts = cleanPoints([
      { decisions: 10, score: 1 },
      { decisions: 5, score: null },
      { decisions: 1, score: 2 },
    ]);
    expect(pts.map((p) => p.decisions)).toEqual([1, 10]);
    expect(cleanPoints(undefined)).toEqual([]);
  });

  it('splits the untrained point off the log axis', () => {
    const { zero, rest } = splitZero(cleanPoints(ppo.mean));
    expect(zero?.score).toBe(-88);
    expect(rest).toHaveLength(3);
    expect(splitZero([{ decisions: 4, score: 1 }]).zero).toBeNull();
  });

  it('knows which arms do not train', () => {
    expect(isStatic(kevZero)).toBe(true);
    expect(isStatic(ppo)).toBe(false);
    expect(isStatic(undefined)).toBe(false);
  });

  it('keeps only complete, positive band points', () => {
    expect(bandPoints(ppo).map((b) => b.decisions)).toEqual([100, 10000]);
  });

  it('lists curve arms in the fixed order', () => {
    expect(curveArms({ curves: { '4': { ppo_sig: ppo, kev_zero: kevZero } } }, '4')).toEqual([
      'kev_zero',
      'ppo_sig',
    ]);
    expect(curveArms({ curves: {} }, '4')).toEqual([]);
  });
});

describe('axes', () => {
  it('snaps the log domain to powers of ten', () => {
    expect(logDomain([ppo, kevZero])).toEqual([100, 10000]);
    expect(
      logDomain([
        {
          mean: [
            { decisions: 16, score: 1 },
            { decisions: 6144, score: 2 },
          ],
          seeds: [],
          band: [],
        },
      ]),
    ).toEqual([10, 10000]);
    expect(logDomain([kevZero])).toEqual([1, 10]);
    expect(logDomain([{ mean: [{ decisions: 100, score: 1 }], seeds: [], band: [] }])).toEqual([
      100, 1000,
    ]);
  });

  it('ticks every power of ten', () => {
    expect(logTicks([10, 10000])).toEqual([10, 100, 1000, 10000]);
  });

  it('always shows random (0) and Bayes (100)', () => {
    expect(yDomain([50, 70])).toEqual([0, 100]);
    expect(yDomain([-88, 126, null])).toEqual([-100, 140]);
  });

  it('collects every measured decision count', () => {
    expect(decisionStops([ppo, kevZero])).toEqual([0, 100, 1000, 10000]);
  });
});

describe('interpolation', () => {
  it('interpolates in log-decisions', () => {
    expect(valueAt(ppo, 1000)).toBe(40);
    // halfway between 1k and 10k in log space
    expect(valueAt(ppo, Math.sqrt(1000 * 10000))).toBeCloseTo(68, 5);
  });

  it('holds the first value before the curve starts and stops after the run ends', () => {
    expect(valueAt(ppo, 0)).toBe(-88);
    expect(valueAt(ppo, 50000)).toBeNull();
  });

  it('holds a static arm everywhere', () => {
    expect(valueAt(kevZero, 1e6)).toBe(65);
    expect(valueAt(undefined, 10)).toBeNull();
  });

  it('interpolates over points that have no score', () => {
    const gap: Curve = {
      mean: [
        { decisions: 10, score: 1 },
        { decisions: 100, score: null },
        { decisions: 1000, score: 3 },
      ],
      seeds: [],
      band: [],
    };
    expect(valueAt(gap, 500)).toBeCloseTo(2.699, 2);
  });

  it('reads compute time the same way', () => {
    expect(secondsAt(ppo, 1000)).toBeCloseTo(0.1);
    expect(secondsAt(kevZero, 10)).toBeNull();
    expect(
      secondsAt({ mean: [{ decisions: 0, score: 1, seconds: 2 }], seeds: [], band: [] }, 9),
    ).toBe(2);
  });
});

describe('efficiency', () => {
  const eff = {
    ppo_sig: {
      start: -88,
      final: 96,
      to_threshold: { '80': 5942, '90.0': null },
      gain_per_decade: 111,
      aulc: 0.18,
    },
    _crossover: {
      'ppo_sig>kev_rl': 6000,
      'ppo_num>kev_rl': null,
      'ppo_num>kev_zero': 1536,
      bogus: 3,
    },
  } as unknown as EfficiencyByArm;

  it('reads thresholds whatever the key spelling', () => {
    expect(thresholdMarks(eff, 'ppo_sig', [80, 90, 95])).toEqual([
      { threshold: 80, decisions: 5942 },
      { threshold: 90, decisions: null },
      { threshold: 95, decisions: null },
    ]);
    expect(thresholdMarks(undefined, 'x', [80])).toEqual([{ threshold: 80, decisions: null }]);
    expect(lookupThreshold({ '80.0': 12 }, 80)).toBe(12);
  });

  it('parses crossovers, drops nulls and sorts by decisions', () => {
    expect(crossovers(eff)).toEqual([
      { slow: 'ppo_num', fast: 'kev_zero', decisions: 1536 },
      { slow: 'ppo_sig', fast: 'kev_rl', decisions: 6000 },
    ]);
    expect(crossovers(undefined)).toEqual([]);
  });
});

describe('cost so far', () => {
  const gpu = {
    hardware: 'H100',
    train_hours: 2,
    train_usd: 6,
    ms_per_decision_p50: 18,
    ms_per_decision_p95: 25,
    usd_per_1k_decisions: 0.01,
  };

  it('derives the hourly rate only for paid training', () => {
    expect(usdPerHour(gpu)).toBe(3);
    expect(usdPerHour({ ...gpu, train_usd: 0 })).toBeNull();
    expect(usdPerHour(undefined)).toBeNull();
  });

  it('prices compute so far on a GPU, and leaves CPU arms in seconds', () => {
    const c = costSoFar(ppo, gpu, 10000);
    expect(c.seconds).toBe(1);
    expect(c.usd).toBeCloseTo(3 / 3600);
    expect(costSoFar(ppo, undefined, 10000).usd).toBeNull();
  });
});
