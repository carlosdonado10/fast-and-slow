import type { Report } from '../types';
import { armColor, armLabel, armMarker, sortArms } from '../lib/arms';
import { crossovers, isStatic, logTicks, thresholdMarks } from '../lib/curves';
import { fmtDecisions, fmtNumber, fmtPct, fmtPoints, horizonLabel, isNum } from '../lib/format';
import { useWidth } from '../hooks/useWidth';
import { useCurveLayout } from '../hooks/useCurveLayout';
import { ArmName, DataDetails, Section } from './ui';
import { ArmToggles, HorizonControl } from './Learning';

interface Props {
  report: Report;
  horizon: string;
  onHorizon: (h: string) => void;
  visible: string[];
  onToggle: (arm: string) => void;
}

const ROW = 44;

/** F4b: decisions-to-threshold markers on the same log x as the curves, gain per 10x, crossovers. */
export function Efficiency({ report, horizon, onHorizon, visible, onToggle }: Props) {
  const feature = report.features.find((f) => f.id === 'F4b');
  const eff = report.efficiency[horizon];
  const curves = report.curves[horizon] ?? {};
  const thresholds = report.meta.thresholds;
  const arms = sortArms(Object.keys(eff ?? {}).filter((k) => !k.startsWith('_')));
  const shown = arms.filter((a) => visible.includes(a));
  const cross = crossovers(eff);

  return (
    <Section
      id="efficiency"
      fid="F4b"
      feature={feature}
      title="How much experience before it's useful?"
      intro={
        <p>
          For practitioners and buyers. In planning, experience is the scarce thing: you can't
          replay 300,000 weeks of a real business. Decisions-to-threshold turns the curves into the
          number a budget is built on: how many simulated decisions each arm needed to first reach{' '}
          {thresholds.map((t) => `${t}%`).join(', ')} of optimal.
        </p>
      }
    >
      <div className="controls">
        <HorizonControl report={report} horizon={horizon} onChange={onHorizon} />
      </div>
      <div className="controls">
        <ArmToggles report={report} arms={arms} visible={visible} onToggle={onToggle} />
      </div>
      {shown.length ? (
        <ThresholdStrip report={report} horizon={horizon} arms={shown} />
      ) : (
        <p className="empty">Turn on at least one arm to see its thresholds.</p>
      )}

      <div className="mini-bars">
        <MiniBars
          title="Gain per 10× more decisions"
          sub="Slope of the curve in log space, points of % optimal. Arms that don't train have none."
          arms={shown}
          report={report}
          value={(a) => (isStatic(curves[a]) ? null : (eff?.[a]?.gain_per_decade ?? null))}
          format={(v) => fmtPoints(v)}
          symmetric
        />
        <MiniBars
          title="Area under the learning curve"
          sub="0 to 1 over log decisions: one number for good early and improving."
          arms={shown}
          report={report}
          value={(a) => eff?.[a]?.aulc ?? null}
          format={(v) => fmtNumber(v, 2)}
          max={1}
        />
      </div>

      {cross.length > 0 ? (
        <ul className="crossings" aria-label="Crossovers">
          {cross.map((c) => (
            <li key={`${c.slow}>${c.fast}`}>
              {armLabel(report, c.slow)} passes {armLabel(report, c.fast)} at{' '}
              <b className="num">{fmtDecisions(c.decisions)}</b> decisions.
            </li>
          ))}
        </ul>
      ) : (
        <p className="empty">No PPO curve passes a Kev curve at this horizon.</p>
      )}

      <DataDetails summary="Show sample-efficiency numbers as a table">
        <table className="tbl">
          <caption>
            Sample efficiency, {horizonLabel(horizon)}. “Never” means not reached in this run.
          </caption>
          <thead>
            <tr>
              <th scope="col">Arm</th>
              <th scope="col">Start</th>
              <th scope="col">Final</th>
              {thresholds.map((t) => (
                <th scope="col" key={t}>
                  To {t}%
                </th>
              ))}
              <th scope="col">Gain per 10×</th>
              <th scope="col">AULC</th>
            </tr>
          </thead>
          <tbody>
            {arms.map((a) => (
              <tr key={a}>
                <th scope="row">
                  <ArmName arm={a} label={armLabel(report, a)} />
                </th>
                <td>{fmtPct(eff?.[a]?.start, 1)}</td>
                <td>{fmtPct(eff?.[a]?.final, 1)}</td>
                {thresholdMarks(eff, a, thresholds).map((m) => (
                  <td key={m.threshold}>{fmtDecisions(m.decisions)}</td>
                ))}
                <td>{fmtPoints(eff?.[a]?.gain_per_decade)}</td>
                <td>{fmtNumber(eff?.[a]?.aulc, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </DataDetails>
    </Section>
  );
}

function ThresholdStrip({
  report,
  horizon,
  arms,
}: {
  report: Report;
  horizon: string;
  arms: string[];
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const L = useCurveLayout(report, horizon, width);
  const narrow = width < 560;
  const eff = report.efficiency[horizon];
  const thresholds = report.meta.thresholds;
  const top = 26;
  const height = top + arms.length * ROW + 30;
  const ticks = logTicks(L.domain);
  // threshold glyph size grows with the bar it marks: small ring, mid, filled
  const glyph = (i: number) => (i === 0 ? 'ring' : i === thresholds.length - 1 ? 'fill' : 'half');

  return (
    <div className="chart" ref={ref}>
      <svg
        width={width}
        height={height}
        role="img"
        aria-label={`Decisions needed to reach ${thresholds.join(', ')} percent of optimal, per arm, on the same log scale as the learning curves. Exact values are in the table below.`}
      >
        <g className="grid">
          {ticks.map((t) => (
            <line key={t} x1={L.x(t)} x2={L.x(t)} y1={top - 6} y2={height - 26} />
          ))}
          <line x1={L.zeroX} x2={L.zeroX} y1={top - 6} y2={height - 26} />
        </g>
        {/* legend for the three threshold glyphs */}
        <g transform={`translate(${L.m.left}, 8)`}>
          {thresholds.map((t, i) => (
            <g key={t} transform={`translate(${i * 64}, 0)`}>
              <ThresholdGlyph kind={glyph(i)} x={5} y={4} color="var(--ink)" />
              <text x={14} y={4} dy="0.32em">
                {t}%
              </text>
            </g>
          ))}
        </g>
        {arms.map((a, i) => {
          const yRow = top + i * ROW;
          const yTrack = yRow + 28;
          const marks = thresholdMarks(eff, a, thresholds);
          const missed = marks.filter((m) => !isNum(m.decisions));
          return (
            <g key={a}>
              <text x={L.m.left} y={yRow + 12} className="direct-label" style={{ fontWeight: 500 }}>
                {armLabel(report, a)}
              </text>
              <line
                x1={L.zeroX}
                x2={L.plotRight}
                y1={yTrack}
                y2={yTrack}
                stroke="var(--rule-strong)"
              />
              {marks.map((m, k) =>
                isNum(m.decisions) ? (
                  <ThresholdGlyph
                    key={m.threshold}
                    kind={glyph(k)}
                    x={L.x(m.decisions)}
                    y={yTrack}
                    color={armColor(a)}
                    square={armMarker(a) === 'square'}
                  />
                ) : null,
              )}
              {missed.length > 0 && (
                <text
                  x={narrow ? L.plotRight : L.plotRight + 10}
                  y={narrow ? yRow + 12 : yTrack}
                  dy={narrow ? 0 : '0.32em'}
                  textAnchor={narrow ? 'end' : 'start'}
                >
                  {missed.length === marks.length
                    ? 'reached none'
                    : `never ${missed.map((m) => `${m.threshold}%`).join(', ')}`}
                </text>
              )}
            </g>
          );
        })}
        <text x={L.zeroX} y={height - 8} textAnchor="middle">
          0
        </text>
        {ticks.map((t) => (
          <text key={t} x={L.x(t)} y={height - 8} textAnchor="middle">
            {fmtDecisions(t)}
          </text>
        ))}
      </svg>
    </div>
  );
}

function ThresholdGlyph({
  kind,
  x,
  y,
  color,
  square,
}: {
  kind: 'ring' | 'half' | 'fill';
  x: number;
  y: number;
  color: string;
  square?: boolean;
}) {
  const r = 5;
  const shape = (fill: string, stroke: string) =>
    square ? (
      <rect
        className="mark"
        x={x - r}
        y={y - r}
        width={r * 2}
        height={r * 2}
        rx={1}
        fill={fill}
        stroke={stroke}
        strokeWidth={1.75}
      />
    ) : (
      <circle className="mark" cx={x} cy={y} r={r} fill={fill} stroke={stroke} strokeWidth={1.75} />
    );
  if (kind === 'fill') return shape(color, color);
  if (kind === 'ring') return shape('var(--page)', color);
  return (
    <g>
      {shape('var(--page)', color)}
      <line x1={x - r + 1} x2={x + r - 1} y1={y} y2={y} stroke={color} strokeWidth={1.75} />
    </g>
  );
}

function MiniBars({
  title,
  sub,
  arms,
  report,
  value,
  format,
  max,
  symmetric,
}: {
  title: string;
  sub: string;
  arms: string[];
  report: Report;
  value: (a: string) => number | null;
  format: (v: number | null) => string;
  max?: number;
  symmetric?: boolean;
}) {
  const vals = arms.map((a) => value(a));
  const finite = vals.filter(isNum);
  const extent = max ?? Math.max(1, ...finite.map(Math.abs));
  const hasNeg = symmetric && finite.some((v) => v < -1e-6);
  return (
    <div>
      <h4>{title}</h4>
      <p className="sub">{sub}</p>
      <div className="auto-rows" style={{ gap: 10 }}>
        {arms.map((a, i) => {
          const v = vals[i] ?? null;
          const frac = isNum(v) ? Math.min(1, Math.abs(v) / extent) : 0;
          const left = hasNeg ? (isNum(v) && v < 0 ? 50 - frac * 50 : 50) : 0;
          const w = hasNeg ? frac * 50 : frac * 100;
          return (
            <div className="auto-row" key={a}>
              <header>
                <span style={{ fontWeight: 500 }}>
                  <ArmName arm={a} label={armLabel(report, a)} />
                </span>
                <span className="num">{isNum(v) ? format(v) : 'none'}</span>
              </header>
              <div className="auto-track" style={{ height: 8 }}>
                <div
                  className="auto-fill"
                  style={{ left: `${left}%`, width: `${w}%`, background: armColor(a) }}
                />
                {hasNeg && (
                  <span
                    style={{
                      position: 'absolute',
                      left: '50%',
                      top: 0,
                      bottom: 0,
                      width: 1,
                      background: 'var(--ink-3)',
                    }}
                  />
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
