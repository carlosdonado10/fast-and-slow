import { scaleLinear, scalePoint } from 'd3-scale';
import type { DialPoint, Report } from '../types';
import { armColor, armDash, armLabel, armMarker, isLearner } from '../lib/arms';
import { yDomain } from '../lib/curves';
import { fmtPct, fmtPoints, horizonLabel, isNum } from '../lib/format';
import { dialArms, dialSlope } from '../lib/replay';
import { ArmName, DataDetails, Section, Swatch } from './ui';

const W = 220;
const H = 150;
const M = { top: 12, right: 14, bottom: 24, left: 30 };

/** F5: each arm's final score as the horizon grows. One small multiple per arm; the slope is the finding. */
export function HorizonDial({ report }: { report: Report }) {
  const feature = report.features.find((f) => f.id === 'F5');
  const dial = report.horizon_dial ?? {};
  const arms = dialArms(dial);
  const horizons = report.meta.horizons;
  const values = arms.flatMap((a) =>
    (dial[a] ?? []).flatMap((p) => [p.pct_optimal, ...(p.ci ?? [])]),
  );
  const [y0, y1] = yDomain(values, 25);
  const x = scalePoint<number>()
    .domain(horizons)
    .range([M.left + 10, W - M.right - 10]);
  const y = scaleLinear()
    .domain([y0, y1])
    .range([H - M.bottom, M.top]);

  return (
    <Section
      id="horizon"
      fid="F5"
      feature={feature}
      title="Does one-shot strength survive a longer plan?"
      intro={
        <p>
          For planners and practitioners alike. Same world, same notes; only the number of weeks in
          the plan changes ({horizons.map(horizonLabel).join(', then ')}). Longer plans add batching
          and carry-over, which is where H3 expects a decision model to slip. Each panel shares one
          scale; whiskers are bootstrap 95% intervals.
        </p>
      }
    >
      {arms.length ? (
        <div className="multiples">
          {arms.map((a) => {
            const pts = (dial[a] ?? []).filter((p) => isNum(p.pct_optimal));
            const slope = dialSlope(dial[a]);
            return (
              <figure className="multiple" key={a}>
                <h3>
                  <Swatch arm={a} />
                  {armLabel(report, a)}
                </h3>
                <p className="slope">
                  {slope === null ? (
                    'one horizon only'
                  ) : (
                    <>
                      <b>{fmtPoints(slope)}</b> from {horizons[0]} to{' '}
                      {horizons[horizons.length - 1]} weeks
                    </>
                  )}
                </p>
                <Panel arm={a} points={pts} x={x} y={y} report={report} />
              </figure>
            );
          })}
        </div>
      ) : (
        <p className="empty">No horizon results yet.</p>
      )}
      <p className="chart-caption">
        The dashed line in each panel is Bayes-optimal (100). Random (0) and Bayes are left out as
        panels because they sit at 0 and 100 by definition.
      </p>
      <DataDetails summary="Show horizon results as a table">
        <table className="tbl">
          <caption>% of optimal by horizon, with bootstrap 95% interval</caption>
          <thead>
            <tr>
              <th scope="col">Arm</th>
              {horizons.map((h) => (
                <th scope="col" key={h}>
                  {horizonLabel(h)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {arms.map((a) => (
              <tr key={a}>
                <th scope="row">
                  <ArmName arm={a} label={armLabel(report, a)} />
                </th>
                {horizons.map((h) => {
                  const p = dial[a]?.find((d) => d.horizon === h);
                  return (
                    <td key={h}>
                      {p
                        ? `${fmtPct(p.pct_optimal, 1)} (${fmtPct(p.ci?.[0], 0)} to ${fmtPct(p.ci?.[1], 0)})`
                        : '–'}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </DataDetails>
    </Section>
  );
}

function Panel({
  arm,
  points,
  x,
  y,
  report,
}: {
  arm: string;
  points: DialPoint[];
  x: ReturnType<typeof scalePoint<number>>;
  y: ReturnType<typeof scaleLinear<number, number>>;
  report: Report;
}) {
  const c = armColor(arm);
  const learner = isLearner(arm);
  const path = points
    .map((p, i) => `${i ? 'L' : 'M'} ${x(p.horizon) ?? 0} ${y(p.pct_optimal as number)}`)
    .join(' ');
  const ticks = y.ticks(4);
  const square = armMarker(arm) === 'square';
  return (
    <div className="chart">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label={`${armLabel(report, arm)}: ${points
          .map((p) => `${horizonLabel(p.horizon)} ${fmtPct(p.pct_optimal)}`)
          .join(', ')}`}
      >
        <g className="grid">
          {ticks.map((t) => (
            <line key={t} x1={M.left} x2={W - M.right} y1={y(t)} y2={y(t)} />
          ))}
        </g>
        {ticks.map((t) => (
          <text
            key={t}
            x={M.left - 6}
            y={y(t)}
            dy="0.32em"
            textAnchor="end"
            style={{ fontSize: 10.5 }}
          >
            {t}
          </text>
        ))}
        <line
          x1={M.left}
          x2={W - M.right}
          y1={y(100)}
          y2={y(100)}
          stroke="var(--arm-bayes)"
          strokeDasharray={armDash('bayes')}
          strokeWidth={1}
        />
        {x.domain().map((h) => (
          <text key={h} x={x(h)} y={H - 6} textAnchor="middle" style={{ fontSize: 10.5 }}>
            {h}w
          </text>
        ))}
        {points.map((p) =>
          isNum(p.ci?.[0]) && isNum(p.ci?.[1]) ? (
            <line
              key={`ci-${p.horizon}`}
              x1={x(p.horizon)}
              x2={x(p.horizon)}
              y1={y(p.ci[0] as number)}
              y2={y(p.ci[1] as number)}
              stroke={c}
              strokeOpacity={0.45}
              strokeWidth={2}
              strokeLinecap="round"
            />
          ) : null,
        )}
        <path
          d={path}
          fill="none"
          stroke={c}
          strokeWidth={2}
          strokeDasharray={learner ? undefined : armDash(arm)}
        />
        {points.map((p) => {
          const cx = x(p.horizon) ?? 0;
          const cy = y(p.pct_optimal as number);
          return square ? (
            <rect
              key={p.horizon}
              x={cx - 4}
              y={cy - 4}
              width={8}
              height={8}
              rx={1}
              fill={c}
              stroke="var(--page)"
              strokeWidth={2}
            />
          ) : (
            <circle
              key={p.horizon}
              cx={cx}
              cy={cy}
              r={4}
              fill={c}
              stroke="var(--page)"
              strokeWidth={2}
            />
          );
        })}
      </svg>
    </div>
  );
}
