import { scaleLog } from 'd3-scale';

export interface Margins {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

/**
 * x layout shared by the learning curves and the efficiency strip below them, so a threshold
 * marker sits exactly under the point on the curve. 0 decisions gets its own slot left of a
 * broken log axis.
 */
export interface XLayout {
  width: number;
  m: Margins;
  zeroX: number;
  logStart: number;
  plotRight: number;
  domain: [number, number];
  x: (decisions: number) => number;
  invert: (px: number) => number;
}

export function xLayout(width: number, domain: [number, number], m: Margins): XLayout {
  const zeroX = m.left + 12;
  const logStart = m.left + 48;
  const plotRight = Math.max(logStart + 40, width - m.right);
  const s = scaleLog().domain(domain).range([logStart, plotRight]).clamp(true);
  return {
    width,
    m,
    zeroX,
    logStart,
    plotRight,
    domain,
    x: (d) => (d <= 0 ? zeroX : s(d)),
    invert: (px) => (px < (zeroX + logStart) / 2 ? 0 : s.invert(Math.max(px, logStart))),
  };
}

/** Spread labels vertically so none overlap; keeps order, minimal movement. */
export function spreadLabels<T extends { y: number }>(
  items: T[],
  gap: number,
  min: number,
  max: number,
): (T & { ly: number })[] {
  const sorted = [...items].sort((a, b) => a.y - b.y).map((it) => ({ ...it, ly: it.y }));
  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1]!;
    const cur = sorted[i]!;
    if (cur.ly - prev.ly < gap) cur.ly = prev.ly + gap;
  }
  const last = sorted[sorted.length - 1];
  if (last && last.ly > max) {
    const shift = last.ly - max;
    sorted.forEach((s) => (s.ly -= shift));
  }
  const first = sorted[0];
  if (first && first.ly < min) {
    const shift = min - first.ly;
    sorted.forEach((s) => (s.ly += shift));
  }
  return sorted;
}
