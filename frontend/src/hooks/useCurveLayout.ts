import type { Report } from '../types';
import { logDomain } from '../lib/curves';
import { xLayout, type XLayout } from '../lib/layout';

/** The x layout of the learning curves, shared with the efficiency strip so both align. */
export function useCurveLayout(report: Report, horizon: string, width: number): XLayout {
  const curves = report.curves[horizon] ?? {};
  const domain = logDomain(Object.values(curves));
  const narrow = width < 560;
  return xLayout(width, domain, {
    top: 18,
    right: narrow ? 12 : 150,
    bottom: 46,
    left: 44,
  });
}
