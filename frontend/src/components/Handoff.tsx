import { useState } from 'react';
import { scaleLinear, scaleSqrt } from 'd3-scale';
import type { Report } from '../types';
import { armColor, armLabel, armMarker, isLearner, sortArms } from '../lib/arms';
import { handoffRows, handoffSentence, reliabilityPoints } from '../lib/handoff';
import { fmtNumber, fmtProb, fmtShare, horizonLabel, isNum } from '../lib/format';
import { ArmName, DataDetails, Section, Segmented } from './ui';
import { HorizonControl } from './Learning';

interface Props {
  report: Report;
  horizon: string;
  onHorizon: (h: string) => void;
}

/** F6: automation at an error budget, split decidable vs judgement, beside a reliability diagram. */
export function Handoff({ report, horizon, onHorizon }: Props) {
  const feature = report.features.find((f) => f.id === 'F6');
  const results = report.results[horizon] ?? {};
  const arms = sortArms(Object.keys(results)).filter((a) => isLearner(a) || a === 'basestock');
  const budgets = report.handoff_budgets?.length
    ? report.handoff_budgets
    : [report.meta.error_budget];
  const defaultArm =
    ['kev_icl', 'kev_rl', 'kev_zero'].find((a) => arms.includes(a)) ?? arms[0] ?? '';
  const [arm, setArm] = useState(defaultArm);
  const [budget, setBudget] = useState(
    budgets.includes(report.meta.error_budget) ? report.meta.error_budget : budgets[0]!,
  );
  const current = arms.includes(arm) ? arm : defaultArm;
  const res = results[current];
  const rows = handoffRows(res, budget);
  const label = armLabel(report, current);

  return (
    <Section
      id="handoff"
      fid="F6"
      feature={feature}
      title="What could I safely hand off?"
      intro={
        <p>
          For planners. Sort a week's decisions by the model's confidence and let it take the most
          confident ones, stopping before more than the error budget would be wrong. That share is
          what you could automate. If the confidence is honest, the model should take most of the
          decidable calls (where knowing the truth wouldn't change the answer) and leave the
          judgement calls to you.
        </p>
      }
    >
      <div className="controls">
        <HorizonControl report={report} horizon={horizon} onChange={onHorizon} />
        <Segmented
          label="Error budget"
          value={budget}
          options={budgets.map((b) => ({ value: b, label: `${Math.round(b * 100)}%` }))}
          onChange={setBudget}
        />
      </div>
      <div className="controls">
        <div className="chip-group">
          <span className="chip-group-label" id="handoff-arm">
            Arm
          </span>
          <div className="chips" role="radiogroup" aria-labelledby="handoff-arm">
            {arms.map((a) => (
              <button
                key={a}
                type="button"
                className="chip"
                role="radio"
                aria-checked={a === current}
                onClick={() => setArm(a)}
              >
                <ArmName arm={a} label={armLabel(report, a)} />
              </button>
            ))}
          </div>
        </div>
      </div>

      {res ? (
        <div className="handoff">
          <div>
            <p className="handoff-lede" aria-live="polite">
              {handoffSentence(label, res, budget)}
            </p>
            <div className="auto-rows">
              {rows.map((r) => (
                <div className={`auto-row ${r.split}`} key={r.split}>
                  <header>
                    <span>
                      {r.label}
                      <span className="n">
                        {fmtNumber(r.n)} decisions, {fmtShare(r.accuracy)} acceptable overall
                      </span>
                    </span>
                    <span className="val num">{fmtShare(r.automation)}</span>
                  </header>
                  <div
                    className="auto-track"
                    role="img"
                    aria-label={`${r.label}: ${fmtShare(r.automation)} automated`}
                  >
                    <div
                      className="auto-fill"
                      style={{
                        width: `${isNum(r.automation) ? Math.max(0, r.automation * 100) : 0}%`,
                        background: armColor(current),
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
          <Reliability report={report} arm={current} horizon={horizon} />
        </div>
      ) : (
        <p className="empty">No hand-off results for this arm and horizon.</p>
      )}

      <DataDetails summary="Show automation for every arm and budget">
        <table className="tbl">
          <caption>
            Share of decisions automated within each error budget, {horizonLabel(horizon)}
          </caption>
          <thead>
            <tr>
              <th scope="col">Arm</th>
              <th scope="col">Split</th>
              {budgets.map((b) => (
                <th scope="col" key={b}>
                  {Math.round(b * 100)}% budget
                </th>
              ))}
              <th scope="col">ECE</th>
            </tr>
          </thead>
          <tbody>
            {arms.flatMap((a) =>
              (['decidable', 'judgement'] as const).map((s, i) => (
                <tr key={`${a}-${s}`}>
                  <th scope="row">
                    {i === 0 ? <ArmName arm={a} label={armLabel(report, a)} /> : ''}
                  </th>
                  <td style={{ textAlign: 'left' }}>{s}</td>
                  {budgets.map((b) => (
                    <td key={b}>
                      {fmtShare(handoffRows(results[a], b).find((r) => r.split === s)?.automation)}
                    </td>
                  ))}
                  <td>{i === 0 ? fmtProb(results[a]?.calibration.ece) : ''}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      </DataDetails>
    </Section>
  );
}

const RS = 300;
const RM = { top: 12, right: 12, bottom: 40, left: 56 };

function Reliability({ report, arm, horizon }: { report: Report; arm: string; horizon: string }) {
  const res = report.results[horizon]?.[arm];
  const pts = reliabilityPoints(res?.calibration.bins);
  const x = scaleLinear()
    .domain([0, 1])
    .range([RM.left, RS - RM.right]);
  const y = scaleLinear()
    .domain([0, 1])
    .range([RS - RM.bottom, RM.top]);
  const total = pts.reduce((s, p) => s + p.count, 0) || 1;
  const r = scaleSqrt().domain([0, total]).range([3, 18]);
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const c = armColor(arm);
  const square = armMarker(arm) === 'square';
  return (
    <div className="reli">
      <h3>Reliability: does confidence mean what it says?</h3>
      <p className="stats">
        <span>
          ECE <b className="num">{fmtProb(res?.calibration.ece)}</b>
        </span>
        <span>
          Brier <b className="num">{fmtProb(res?.calibration.brier)}</b>
        </span>
        <span>
          Bins with data <b className="num">{pts.length}</b>
        </span>
      </p>
      <div className="chart" style={{ maxWidth: 380 }}>
        <svg
          viewBox={`0 0 ${RS} ${RS}`}
          width="100%"
          role="img"
          aria-label={`Reliability diagram for ${armLabel(report, arm)}. ${pts
            .map(
              (p) =>
                `Confidence ${fmtProb(p.confidence)}: ${fmtShare(p.accuracy)} acceptable, ${p.count} decisions`,
            )
            .join('. ')}`}
        >
          <g className="grid">
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={RM.top} y2={RS - RM.bottom} />
                <line x1={RM.left} x2={RS - RM.right} y1={y(t)} y2={y(t)} />
              </g>
            ))}
          </g>
          {ticks.map((t) => (
            <g key={t}>
              <text x={x(t)} y={RS - RM.bottom + 16} textAnchor="middle">
                {t}
              </text>
              <text x={RM.left - 6} y={y(t)} dy="0.32em" textAnchor="end">
                {t}
              </text>
            </g>
          ))}
          <text className="axis-title" x={RS - RM.right} y={RS - 6} textAnchor="end">
            Stated confidence
          </text>
          <text
            className="axis-title"
            transform={`translate(12 ${RM.top}) rotate(-90)`}
            textAnchor="end"
          >
            Share acceptable
          </text>
          <line
            x1={x(0)}
            y1={y(0)}
            x2={x(1)}
            y2={y(1)}
            stroke="var(--ink-3)"
            strokeDasharray="4 4"
          />
          <text
            className="ref-label"
            transform={`translate(${x(0.66) - 8} ${y(0.66) - 8}) rotate(-45)`}
            textAnchor="middle"
          >
            perfectly calibrated
          </text>
          {pts.length > 1 && (
            <path
              className="series"
              d={pts
                .map((p, i) => `${i ? 'L' : 'M'} ${x(p.confidence)} ${y(p.accuracy)}`)
                .join(' ')}
              stroke={c}
            />
          )}
          {pts.map((p) => {
            const rr = r(p.count);
            return square ? (
              <rect
                key={p.lo}
                className="mark"
                x={x(p.confidence) - rr}
                y={y(p.accuracy) - rr}
                width={rr * 2}
                height={rr * 2}
                rx={2}
                fill={c}
                fillOpacity={0.85}
                stroke="var(--page)"
                strokeWidth={2}
              />
            ) : (
              <circle
                key={p.lo}
                className="mark"
                cx={x(p.confidence)}
                cy={y(p.accuracy)}
                r={rr}
                fill={c}
                fillOpacity={0.85}
                stroke="var(--page)"
                strokeWidth={2}
              />
            );
          })}
        </svg>
      </div>
      <p className="chart-caption">
        Each mark is a confidence bin; its area is the number of decisions in it. Marks below the
        diagonal are overconfident.
      </p>
    </div>
  );
}
