import { useState, type KeyboardEvent, type PointerEvent } from 'react';
import { scaleLinear } from 'd3-scale';
import { area, curveLinear, line } from 'd3-shape';
import type { Curve, Report } from '../types';
import { armColor, armDash, armLabel, armMarker, sortArms } from '../lib/arms';
import {
  bandPoints,
  cleanPoints,
  costSoFar,
  crossovers,
  decisionStops,
  isStatic,
  logTicks,
  splitZero,
  valueAt,
  yDomain,
} from '../lib/curves';
import { fmtDecisions, fmtDuration, fmtPct, fmtUsd, isNum } from '../lib/format';
import { spreadLabels } from '../lib/layout';
import { useCurveLayout } from '../hooks/useCurveLayout';
import { useWidth } from '../hooks/useWidth';

interface Props {
  report: Report;
  horizon: string;
  visible: string[];
}

type Pt = { decisions: number; score: number };

const REF_ARMS = ['oracle', 'bayes', 'basestock', 'random'] as const;

/** Mean points with a finite score, typed for the line generator. */
function scored(curve: Curve | undefined): Pt[] {
  return cleanPoints(curve?.mean).filter((p): p is Pt => isNum(p.score));
}

function Marker({ arm, cx, cy, r = 4.5 }: { arm: string; cx: number; cy: number; r?: number }) {
  const m = armMarker(arm);
  const c = armColor(arm);
  if (m === 'square') {
    return (
      <rect
        className="mark"
        x={cx - r}
        y={cy - r}
        width={r * 2}
        height={r * 2}
        rx={1}
        fill={c}
        stroke="var(--page)"
        strokeWidth={2}
      />
    );
  }
  return (
    <circle className="mark" cx={cx} cy={cy} r={r} fill={c} stroke="var(--page)" strokeWidth={2} />
  );
}

/** F4: % of optimal vs decisions consumed, log x, seed band, references, crossover, hover readout. */
export function LearningChart({ report, horizon, visible }: Props) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const curves: Record<string, Curve> = report.curves[horizon] ?? {};
  const results = report.results[horizon] ?? {};
  const L = useCurveLayout(report, horizon, width);
  const narrow = width < 560;
  const height = narrow ? 300 : 400;
  const plotBottom = height - L.m.bottom;
  const shown = sortArms(Object.keys(curves)).filter((a) => visible.includes(a));

  const refs = REF_ARMS.map((a) => ({
    arm: a,
    value: a === 'bayes' ? 100 : a === 'random' ? 0 : (results[a]?.pct_optimal.mean ?? null),
  })).filter((r) => isNum(r.value) && (r.arm !== 'oracle' || !!results.oracle));

  const allScores = Object.values(curves).flatMap((c) => [
    ...cleanPoints(c.mean).map((p) => p.score),
    ...bandPoints(c).flatMap((b) => [b.lo, b.hi]),
  ]);
  const [y0, y1] = yDomain([...allScores, ...refs.map((r) => r.value)]);
  const y = scaleLinear().domain([y0, y1]).range([plotBottom, L.m.top]);
  const yTicks = y.ticks(narrow ? 5 : 7);
  const xTicks = logTicks(L.domain);

  const stops = decisionStops(Object.values(curves));
  const [hoverX, setHoverX] = useState<number | null>(null);

  const lineGen = line<{ decisions: number; score: number }>()
    .x((p) => L.x(p.decisions))
    .y((p) => y(p.score))
    .curve(curveLinear);
  const areaGen = area<{ decisions: number; lo: number; hi: number }>()
    .x((p) => L.x(p.decisions))
    .y0((p) => y(p.lo))
    .y1((p) => y(p.hi))
    .curve(curveLinear);

  const cross = (narrow ? [] : crossovers(report.efficiency[horizon])).filter(
    (c) => c.fast === 'kev_rl' && shown.includes(c.slow) && shown.includes(c.fast),
  );

  // Right-margin labels: references always, learner line-ends when there's room and few enough
  // lines to label cleanly. One spread pass, so no label lands on another.
  const learnerEnds = !narrow
    ? shown
        .map((a) => {
          const pts = cleanPoints(curves[a]?.mean);
          const last = pts[pts.length - 1];
          return last && isNum(last.score) ? { arm: a, y: y(last.score), kind: 'line' } : null;
        })
        .filter((v): v is { arm: string; y: number; kind: string } => v !== null)
    : [];
  const marginLabels = spreadLabels(
    [...refs.map((r) => ({ arm: r.arm, y: y(r.value as number), kind: 'ref' })), ...learnerEnds],
    15,
    L.m.top,
    plotBottom,
  );

  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const box = e.currentTarget.ownerSVGElement?.getBoundingClientRect();
    if (!box) return;
    setHoverX(L.invert(e.clientX - box.left));
  };
  const onKey = (e: KeyboardEvent<SVGRectElement>) => {
    if (!stops.length) return;
    const i = hoverX === null ? -1 : stops.findIndex((s) => s >= hoverX - 1e-9);
    if (e.key === 'ArrowRight') {
      e.preventDefault();
      setHoverX(stops[Math.min(stops.length - 1, i + 1)]!);
    } else if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setHoverX(stops[Math.max(0, i <= 0 ? 0 : i - 1)]!);
    } else if (e.key === 'Escape') {
      setHoverX(null);
    }
  };

  const hx = hoverX === null ? null : L.x(hoverX);
  const titleId = `lc-title-${horizon}`;
  const descId = `lc-desc-${horizon}`;

  return (
    <div className="chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-labelledby={`${titleId} ${descId}`}>
        <title id={titleId}>Learning curves, {horizon}-week horizon</title>
        <desc id={descId}>
          Percent of optimal on the vertical axis against training decisions consumed on a log
          horizontal axis. Untrained scores sit in a separate slot at zero. Bayes-optimal is 100,
          random is 0. Use the data table below for exact values; with the chart focused, the left
          and right arrow keys step through the measured points.
        </desc>

        {/* grid + y axis */}
        <g className="grid">
          {yTicks.map((t) => (
            <line key={t} x1={L.m.left} x2={L.plotRight} y1={y(t)} y2={y(t)} />
          ))}
        </g>
        {yTicks.map((t) => (
          <text key={t} x={L.m.left - 8} y={y(t)} dy="0.32em" textAnchor="end">
            {t}
          </text>
        ))}
        <text className="axis-title" x={L.m.left - 8} y={L.m.top - 6} textAnchor="end">
          %
        </text>

        {/* x axis: zero slot, break, log ticks */}
        <line className="baseline" x1={L.m.left} x2={L.plotRight} y1={plotBottom} y2={plotBottom} />
        <g aria-hidden="true">
          <rect x={L.zeroX + 16} y={plotBottom - 6} width={14} height={12} fill="var(--page)" />
          <line
            x1={L.zeroX + 17}
            x2={L.zeroX + 22}
            y1={plotBottom + 5}
            y2={plotBottom - 5}
            stroke="var(--ink-3)"
          />
          <line
            x1={L.zeroX + 23}
            x2={L.zeroX + 28}
            y1={plotBottom + 5}
            y2={plotBottom - 5}
            stroke="var(--ink-3)"
          />
        </g>
        <text x={L.zeroX} y={plotBottom + 18} textAnchor="middle">
          0
        </text>
        {xTicks.map((t) => (
          <g key={t}>
            <line
              x1={L.x(t)}
              x2={L.x(t)}
              y1={plotBottom}
              y2={plotBottom + 4}
              stroke="var(--rule-strong)"
            />
            <text x={L.x(t)} y={plotBottom + 18} textAnchor="middle">
              {fmtDecisions(t)}
            </text>
          </g>
        ))}
        <text className="axis-title" x={L.plotRight} y={plotBottom + 38} textAnchor="end">
          Training decisions consumed (log scale)
        </text>
        <text className="axis-title" x={L.zeroX - 10} y={plotBottom + 38} textAnchor="start">
          untrained
        </text>

        {/* reference lines */}
        {refs.map((r) => (
          <line
            key={r.arm}
            x1={L.m.left}
            x2={L.plotRight}
            y1={y(r.value as number)}
            y2={y(r.value as number)}
            stroke={armColor(r.arm)}
            strokeWidth={1.25}
            strokeDasharray={armDash(r.arm)}
            className="series"
          />
        ))}
        {marginLabels.map((r) =>
          r.kind === 'line' ? (
            <g key={`line-${r.arm}`}>
              <Marker arm={r.arm} cx={L.plotRight + 13} cy={r.ly} r={3.5} />
              <text className="direct-label" x={L.plotRight + 22} y={r.ly} dy="0.32em">
                {armLabel(report, r.arm)}
              </text>
            </g>
          ) : (
            <text
              key={`${r.kind}-${r.arm}`}
              className="ref-label"
              x={narrow ? L.plotRight : L.plotRight + 8}
              y={narrow ? r.ly - 5 : r.ly}
              dy={narrow ? 0 : '0.32em'}
              textAnchor={narrow ? 'end' : 'start'}
            >
              {armLabel(report, r.arm)}
              {r.arm !== 'bayes' && r.arm !== 'random'
                ? ` ${fmtPct(refs.find((x) => x.arm === r.arm)?.value)}`
                : ''}
            </text>
          ),
        )}

        {/* crossover markers (H2): where a PPO curve passes Kev + RL */}
        {cross.map((c) => (
          <g key={`${c.slow}>${c.fast}`} aria-hidden="true">
            <line
              x1={L.x(c.decisions)}
              x2={L.x(c.decisions)}
              y1={L.m.top}
              y2={plotBottom}
              stroke="var(--ink-3)"
              strokeOpacity={0.5}
            />
            <rect
              x={L.x(c.decisions) - 4}
              y={L.m.top - 4}
              width={8}
              height={8}
              transform={`rotate(45 ${L.x(c.decisions)} ${L.m.top})`}
              fill="var(--page)"
              stroke="var(--ink)"
              strokeWidth={1.5}
            />
          </g>
        ))}

        {/* bands, then lines, then markers */}
        {shown.map((a) => {
          const b = bandPoints(curves[a]);
          return b.length > 1 ? (
            <path
              key={`band-${a}`}
              className="band"
              d={areaGen(b) ?? undefined}
              fill={armColor(a)}
              opacity={0.16}
            />
          ) : null;
        })}
        {shown.map((a) => {
          const pts = scored(curves[a]);
          const { zero, rest } = splitZero(pts);
          const c = armColor(a);
          if (isStatic(curves[a])) {
            const p = pts[0]!;
            return (
              <g key={a}>
                <line
                  className="series"
                  x1={L.x(p.decisions)}
                  x2={L.plotRight}
                  y1={y(p.score)}
                  y2={y(p.score)}
                  stroke={c}
                  strokeWidth={1.5}
                  strokeDasharray="1 4"
                  opacity={0.75}
                />
                <Marker arm={a} cx={L.x(p.decisions)} cy={y(p.score)} r={5} />
              </g>
            );
          }
          const first = rest[0];
          return (
            <g key={a}>
              {zero && first && isNum(zero.score) && (
                <line
                  className="series"
                  x1={L.x(0)}
                  y1={y(zero.score)}
                  x2={L.x(first.decisions)}
                  y2={y(first.score)}
                  stroke={c}
                  strokeWidth={1.25}
                  opacity={0.4}
                />
              )}
              {rest.length > 1 && (
                <path className="series" d={lineGen(rest) ?? undefined} stroke={c} />
              )}
              {zero && isNum(zero.score) && <Marker arm={a} cx={L.x(0)} cy={y(zero.score)} />}
              {rest.map((p, i) => (
                <Marker
                  key={p.decisions}
                  arm={a}
                  cx={L.x(p.decisions)}
                  cy={y(p.score)}
                  r={i === rest.length - 1 ? 4.5 : 3}
                />
              ))}
            </g>
          );
        })}

        {/* crosshair */}
        {hx !== null && hoverX !== null && (
          <g aria-hidden="true">
            <line className="crosshair" x1={hx} x2={hx} y1={L.m.top} y2={plotBottom} />
            {shown.map((a) => {
              const v = valueAt(curves[a], hoverX);
              return isNum(v) ? <Marker key={a} arm={a} cx={hx} cy={y(v)} r={4} /> : null;
            })}
          </g>
        )}

        <rect
          className="hit"
          x={L.m.left}
          y={L.m.top}
          width={Math.max(0, L.plotRight - L.m.left)}
          height={Math.max(0, plotBottom - L.m.top)}
          tabIndex={0}
          role="slider"
          aria-label="Inspect scores at a decision count"
          aria-valuemin={0}
          aria-valuemax={L.domain[1]}
          aria-valuenow={hoverX ?? 0}
          aria-valuetext={
            hoverX === null ? 'No point selected' : `${fmtDecisions(hoverX)} decisions`
          }
          onPointerMove={onMove}
          onPointerLeave={() => setHoverX(null)}
          onKeyDown={onKey}
          onBlur={() => setHoverX(null)}
        />
      </svg>

      {hx !== null && hoverX !== null && (
        <Tooltip
          report={report}
          horizon={horizon}
          arms={shown}
          x={hoverX}
          left={Math.min(hx + 14, width - 230)}
          top={L.m.top}
        />
      )}

      <p className="readout" aria-live="polite">
        {hoverX === null ? (
          <span>
            Hover or focus the chart and use the arrow keys to read scores and the cost so far.
          </span>
        ) : (
          <span>
            At <b>{fmtDecisions(hoverX)}</b> decisions
          </span>
        )}
      </p>
    </div>
  );
}

function Tooltip({
  report,
  horizon,
  arms,
  x,
  left,
  top,
}: {
  report: Report;
  horizon: string;
  arms: string[];
  x: number;
  left: number;
  top: number;
}) {
  const curves = report.curves[horizon] ?? {};
  return (
    <div className="tip" style={{ left: Math.max(0, left), top }}>
      <h4>{x <= 0 ? 'Untrained' : `${fmtDecisions(x)} decisions`}</h4>
      <table>
        <tbody>
          {arms.map((a) => {
            const v = valueAt(curves[a], x);
            const cost = costSoFar(curves[a], report.costs[a], x);
            return (
              <tr key={a}>
                <td>
                  <span className="arm-name">
                    <svg width={8} height={8} aria-hidden="true">
                      {armMarker(a) === 'square' ? (
                        <rect width={8} height={8} fill={armColor(a)} />
                      ) : (
                        <circle cx={4} cy={4} r={4} fill={armColor(a)} />
                      )}
                    </svg>
                    {armLabel(report, a)}
                  </span>
                </td>
                <td className="v">{isNum(v) ? fmtPct(v, 1) : 'run ended'}</td>
                <td className="v" style={{ color: 'var(--ink-3)' }}>
                  {isStatic(curves[a])
                    ? 'no training'
                    : isNum(cost.usd)
                      ? fmtUsd(cost.usd)
                      : fmtDuration(cost.seconds)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="cost">Right column: training cost so far (compute time, or dollars on GPU).</p>
    </div>
  );
}
