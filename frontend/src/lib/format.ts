import type { Num } from '../types';

export const DASH = '–';

/** A finite number, or false for null / undefined / NaN / ±Infinity. */
export function isNum(x: Num | undefined): x is number {
  return typeof x === 'number' && Number.isFinite(x);
}

/** Round, and turn -0 into 0 so "-0" never prints. */
function clean(x: number, digits: number): number {
  const r = Number(x.toFixed(digits));
  return Object.is(r, -0) ? 0 : r;
}

export function fmtNumber(x: Num | undefined, digits = 0): string {
  if (!isNum(x)) return DASH;
  return clean(x, digits).toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** A value already on a 0–100 scale. */
export function fmtPct(x: Num | undefined, digits = 0): string {
  return isNum(x) ? `${fmtNumber(x, digits)}%` : DASH;
}

/** A 0–1 share shown as a percentage. */
export function fmtShare(x: Num | undefined, digits = 0): string {
  return isNum(x) ? fmtPct(x * 100, digits) : DASH;
}

export function fmtSigned(x: Num | undefined, digits = 1, unit = ''): string {
  if (!isNum(x)) return DASH;
  const v = clean(x, digits);
  const sign = v > 0 ? '+' : v < 0 ? '−' : '±';
  return `${sign}${fmtNumber(Math.abs(v), digits)}${unit}`;
}

export function fmtPoints(x: Num | undefined, digits = 1): string {
  return isNum(x) ? fmtSigned(x, digits, ' pts') : DASH;
}

function trim(x: number): string {
  const d = Math.abs(x) >= 100 ? 0 : 1;
  return String(clean(x, d));
}

/** Decisions consumed: 64, 6.1k, 1.2M. null means the threshold was never reached. */
export function fmtDecisions(x: Num | undefined, never = 'never'): string {
  if (!isNum(x)) return never;
  const a = Math.abs(x);
  if (a >= 1e6) return `${trim(x / 1e6)}M`;
  if (a >= 1e3) return `${trim(x / 1e3)}k`;
  return fmtNumber(x, 0);
}

export function fmtDuration(seconds: Num | undefined): string {
  if (!isNum(seconds)) return DASH;
  if (seconds === 0) return '0 s';
  if (seconds < 1) return `${trim(seconds * 1000)} ms`;
  if (seconds < 60) return `${trim(seconds)} s`;
  if (seconds < 3600) return `${trim(seconds / 60)} min`;
  return `${trim(seconds / 3600)} h`;
}

export function fmtHours(hours: Num | undefined): string {
  return isNum(hours) ? fmtDuration(hours * 3600) : DASH;
}

export function fmtUsd(x: Num | undefined): string {
  if (!isNum(x)) return DASH;
  if (x === 0) return '$0';
  if (Math.abs(x) < 0.01) return '<$0.01';
  if (Math.abs(x) < 100) return `$${fmtNumber(x, 2)}`;
  return `$${fmtNumber(x, 0)}`;
}

/** Whole-dollar amounts with a real minus sign: $143, −$30. */
export function fmtMoney(x: Num | undefined): string {
  if (!isNum(x)) return DASH;
  const v = Math.round(x);
  return `${v < 0 ? '−' : ''}$${Math.abs(v).toLocaleString('en-US')}`;
}

export function fmtMs(x: Num | undefined): string {
  if (!isNum(x)) return DASH;
  if (x >= 100) return `${fmtNumber(x, 0)} ms`;
  if (x >= 1) return `${fmtNumber(x, 1)} ms`;
  return `${Number(x.toPrecision(2))} ms`;
}

export function fmtProb(x: Num | undefined, digits = 2): string {
  return isNum(x) ? x.toFixed(digits) : DASH;
}

/** "sales_rep" -> "sales rep". */
export function humanize(s: string | null | undefined): string {
  return (s ?? '').replace(/_/g, ' ').trim();
}

export function capitalize(s: string): string {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

export function horizonLabel(h: number | string): string {
  const n = Number(h);
  return n === 1 ? '1 week' : `${n} weeks`;
}

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString('en-GB', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
    timeZone: 'UTC',
  });
}
