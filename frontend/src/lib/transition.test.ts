import { describe, expect, it } from 'vitest';
import type { ShiftTimeline } from '../types';
import { fmtWeeksAfter, regretArea, summarise, timelineKey, weeklyPath } from './transition';

const t = (over: Partial<ShiftTimeline> = {}): ShiftTimeline => ({
  trigger: 'declared',
  mode: 'sim',
  variant: 'num',
  trigger_week: 2,
  weekly: {
    stale: [90, 90, 40, 40],
    retrain: [90, 90, 40, 95],
    kev: [90, 90, 80, 80],
    bridge: [90, 90, 80, 95],
  },
  regret_usd: { stale: 120, retrain: 65, kev: 40, bridge: 25 },
  recover_week: { stale: null, retrain: 1, kev: null, bridge: 1 },
  handback_week: 1,
  bridge_saving: 0.6,
  ...over,
});

describe('transition helpers (F11)', () => {
  it('keys timelines the way the export does', () => {
    expect(timelineKey('declared', 'live', 'emb')).toBe('declared|live|emb');
  });

  it('says weeks in plain language', () => {
    expect(fmtWeeksAfter(0)).toBe('the same week');
    expect(fmtWeeksAfter(1)).toBe('1 week');
    expect(fmtWeeksAfter(6)).toBe('6 weeks');
    expect(fmtWeeksAfter(null)).toBe('not within the year');
    expect(fmtWeeksAfter(undefined, 'never')).toBe('never');
  });

  it('finds the cheapest strategy and the better RL-only cost', () => {
    const s = summarise(t());
    expect(s.best).toBe('bridge');
    expect(s.rlOnly).toBe(65);
    expect(s.saving).toBe(0.6);
    const missing = summarise(
      t({ regret_usd: { stale: null, retrain: null, kev: 10, bridge: null } }),
    );
    expect(missing.best).toBe('kev');
    expect(missing.rlOnly).toBeNull();
  });

  it('draws weekly lines with gaps and the regret area from the shift on', () => {
    const id = (v: number) => v;
    expect(weeklyPath([1, 2, null, 4], id, id)).toBe('M 0 1 H 1 V 2 H 2 M 3 4 H 4');
    expect(regretArea([90, 80, 60], 1, id, id)).toBe(
      'M 1 100 L 3 100 L 3 60 L 2 60 L 2 80 L 1 80 Z',
    );
    expect(regretArea([90, null], 1, id, id)).toBe('');
  });
});
