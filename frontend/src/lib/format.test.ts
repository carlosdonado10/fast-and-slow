import { describe, expect, it } from 'vitest';
import {
  DASH,
  capitalize,
  fmtDate,
  fmtDecisions,
  fmtDuration,
  fmtHours,
  fmtMoney,
  fmtMs,
  fmtNumber,
  fmtPct,
  fmtPoints,
  fmtProb,
  fmtShare,
  fmtSigned,
  fmtUsd,
  horizonLabel,
  humanize,
  isNum,
} from './format';

describe('null handling', () => {
  it('treats null, undefined, NaN and infinities as missing', () => {
    expect(isNum(null)).toBe(false);
    expect(isNum(undefined)).toBe(false);
    expect(isNum(NaN)).toBe(false);
    expect(isNum(Infinity)).toBe(false);
    expect(isNum(0)).toBe(true);
  });

  it('prints a dash for missing values in every formatter', () => {
    for (const f of [
      fmtNumber,
      fmtPct,
      fmtShare,
      fmtSigned,
      fmtPoints,
      fmtDuration,
      fmtHours,
      fmtUsd,
      fmtMs,
      fmtProb,
      fmtMoney,
    ]) {
      expect(f(null)).toBe(DASH);
    }
  });

  it('says "never" for a threshold that was never reached', () => {
    expect(fmtDecisions(null)).toBe('never');
    expect(fmtDecisions(null, 'not reached')).toBe('not reached');
  });
});

describe('numbers', () => {
  it('never prints negative zero', () => {
    expect(fmtNumber(-8.5e-15, 1)).toBe('0.0');
    expect(fmtPct(-2.9e-15)).toBe('0%');
  });

  it('formats percentages and shares', () => {
    expect(fmtPct(65.09)).toBe('65%');
    expect(fmtPct(65.09, 1)).toBe('65.1%');
    expect(fmtShare(0.333)).toBe('33%');
  });

  it('uses a real minus sign and a plus sign for deltas', () => {
    expect(fmtSigned(3.25)).toBe('+3.3');
    expect(fmtSigned(-3.25)).toBe('−3.3');
    expect(fmtSigned(0)).toBe('±0.0');
    expect(fmtPoints(38.4)).toBe('+38.4 pts');
  });

  it('compacts decision counts', () => {
    expect(fmtDecisions(64)).toBe('64');
    expect(fmtDecisions(6144)).toBe('6.1k');
    expect(fmtDecisions(150000)).toBe('150k');
    expect(fmtDecisions(1_250_000)).toBe('1.3M');
  });

  it('formats durations at a sensible unit', () => {
    expect(fmtDuration(0)).toBe('0 s');
    expect(fmtDuration(0.29)).toBe('290 ms');
    expect(fmtDuration(12.34)).toBe('12.3 s');
    expect(fmtDuration(600)).toBe('10 min');
    expect(fmtHours(18)).toBe('18 h');
  });

  it('formats money and latency', () => {
    expect(fmtUsd(0)).toBe('$0');
    expect(fmtUsd(0.0006)).toBe('<$0.01');
    expect(fmtUsd(4.5)).toBe('$4.50');
    expect(fmtUsd(1234)).toBe('$1,234');
    expect(fmtMoney(143)).toBe('$143');
    expect(fmtMoney(-30.4)).toBe('−$30');
    expect(fmtMs(1.64)).toBe('1.6 ms');
    expect(fmtMs(0.00893)).toBe('0.0089 ms');
    expect(fmtMs(250)).toBe('250 ms');
    expect(fmtProb(0.0712)).toBe('0.07');
  });
});

describe('text', () => {
  it('humanizes ids and labels horizons', () => {
    expect(humanize('sales_rep')).toBe('sales rep');
    expect(humanize(null)).toBe('');
    expect(capitalize('rumour')).toBe('Rumour');
    expect(capitalize('')).toBe('');
    expect(horizonLabel(1)).toBe('1 week');
    expect(horizonLabel('12')).toBe('12 weeks');
  });

  it('formats dates in UTC and passes through junk', () => {
    expect(fmtDate('2026-11-15T16:02:56+00:00')).toBe('15 November 2026');
    expect(fmtDate('not a date')).toBe('not a date');
  });
});
