import { useState } from 'react';
import { scaleLinear } from 'd3-scale';
import type { Report, ShiftResult, ShiftTimeline, Strategy } from '../types';
import { armLabel } from '../lib/arms';
import { yDomain } from '../lib/curves';
import { fmtMoney, fmtPct, fmtShare, isNum } from '../lib/format';
import {
  MODE_LABEL,
  STRATEGIES,
  STRATEGY_LABEL,
  STRATEGY_STYLE,
  TRIGGER_LABEL,
  fmtWeeksAfter,
  getTimeline,
  regretArea,
  summarise,
  variantsOf,
  weeklyPath,
} from '../lib/transition';
import { DataDetails, Section, Segmented } from './ui';
import { useWidth } from '../hooks/useWidth';

const H = 300;
const M = { top: 22, right: 18, bottom: 34, left: 40 };

type Trigger = 'declared' | 'detected';
type Mode = 'sim' | 'live';

/** F11: something changed this week. Who decides until the model catches up, and what does the gap cost? */
export function Transition({ report }: { report: Report }) {
  const feature = report.features.find((f) => f.id === 'F11');
  const shifts = report.shifts ?? {};
  const names = Object.keys(shifts);
  const [name, setName] = useState(names[0] ?? '');
  const [trigger, setTrigger] = useState<Trigger>('declared');
  const [mode, setMode] = useState<Mode>('sim');
  const shift = shifts[name];
  const variants = shift ? variantsOf(shift) : [];
  const [variant, setVariant] = useState(variants[variants.length - 1] ?? '');
  const v = variants.includes(variant) ? variant : (variants[0] ?? '');
  const t = shift ? getTimeline(shift, trigger, mode, v) : undefined;

  return (
    <Section
      id="transition"
      fid="F11"
      feature={feature}
      title="Something changed this week. Who decides until the model catches up?"
      intro={
        <p>
          For the planner or ops lead on the week the business shifts. A deployed RL policy was
          trained on a world that just ended; retraining it takes time. Each strip is one strategy
          over a year of weekly decisions, scored against the best possible planner of whichever
          world is in force. The shaded area under 100% is profit given up.
        </p>
      }
    >
      {!shift || !t ? (
        <p className="empty">No regime-change results yet.</p>
      ) : (
        <>
          <div className="controls">
            <Segmented
              label="What changed"
              value={name}
              options={names.map((n) => ({ value: n, label: shifts[n]?.label ?? n }))}
              onChange={setName}
            />
          </div>
          <div className="controls">
            <Segmented
              label="Who notices"
              value={trigger}
              options={(['declared', 'detected'] as const).map((k) => ({
                value: k,
                label: TRIGGER_LABEL[k],
              }))}
              onChange={setTrigger}
            />
            <Segmented
              label="How RL is retrained"
              value={mode}
              options={(['sim', 'live'] as const).map((k) => ({ value: k, label: MODE_LABEL[k] }))}
              onChange={setMode}
            />
            {variants.length > 1 && (
              <Segmented
                label="Deployed RL"
                value={v}
                options={variants.map((x) => ({ value: x, label: armLabel(report, `ppo_${x}`) }))}
                onChange={setVariant}
              />
            )}
          </div>
          <ShiftNote shift={shift} />
          <TimelineChart shift={shift} t={t} />
          <Outcome shift={shift} t={t} mode={mode} />
          <Detection shift={shift} />
          <DataDetails summary="Show the strategies as a table">
            <table className="tbl">
              <caption>
                {shift.label}: {TRIGGER_LABEL[trigger].toLowerCase()},{' '}
                {MODE_LABEL[mode].toLowerCase()}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Strategy</th>
                  <th scope="col">Profit given up after the shift</th>
                  <th scope="col">Back to {shift.recover_at}% of optimal after</th>
                </tr>
              </thead>
              <tbody>
                {STRATEGIES.map((s) => (
                  <tr key={s}>
                    <th scope="row">{STRATEGY_LABEL[s]}</th>
                    <td>{fmtMoney(t.regret_usd[s])}</td>
                    <td>{fmtWeeksAfter(t.recover_week[s])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </DataDetails>
        </>
      )}
    </Section>
  );
}

function ShiftNote({ shift }: { shift: ShiftResult }) {
  return (
    <div className="shift-note">
      <p>
        <b>Week {shift.shift_week + 1}.</b>{' '}
        {shift.kind === 'announced' ? (
          <>An email lands in the inbox: “{shift.announcement}”</>
        ) : (
          <>Nothing in the inbox. Only the sales numbers change.</>
        )}
      </p>
      <p className="muted">
        What a planner writes into the bridge’s brief once it’s declared: “{shift.declared_note}”
      </p>
    </div>
  );
}

function TimelineChart({ shift, t }: { shift: ShiftResult; t: ShiftTimeline }) {
  // Draw at the real width so labels keep their size on a phone instead of shrinking with a viewBox.
  const [ref, W] = useWidth<HTMLElement>();
  const x = scaleLinear()
    .domain([0, shift.weeks])
    .range([M.left, W - M.right]);
  const values = STRATEGIES.flatMap((s) => t.weekly[s]).filter(isNum);
  const [y0, y1] = yDomain([...values, 100], 25);
  const y = scaleLinear()
    .domain([Math.min(y0, 0), Math.max(y1, 100)])
    .range([H - M.bottom, M.top]);
  const xs = (w: number) => x(w);
  const ys = (val: number) => y(val);
  const marks = [
    { week: shift.shift_week, label: 'the change' },
    ...(t.trigger_week !== null && t.trigger_week !== shift.shift_week
      ? [{ week: t.trigger_week, label: 'alarm' }]
      : []),
    ...(t.handback_week !== null
      ? [{ week: shift.shift_week + t.handback_week, label: 'hand-back' }]
      : []),
  ];
  const describe = STRATEGIES.map(
    (s) => `${STRATEGY_LABEL[s]}: ${fmtMoney(t.regret_usd[s])} given up`,
  ).join('; ');
  return (
    <figure className="chart" ref={ref}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        width="100%"
        role="img"
        aria-label={`Weekly score for each strategy over ${shift.weeks} weeks. ${describe}.`}
      >
        <g className="grid">
          {y.ticks(5).map((v) => (
            <g key={v}>
              <line x1={M.left} x2={W - M.right} y1={y(v)} y2={y(v)} />
              <text x={M.left - 6} y={y(v)} dy="0.32em" textAnchor="end" className="tick">
                {v}
              </text>
            </g>
          ))}
          {(W < 520 ? [0, shift.weeks - 1] : [0, 13, 26, 39, shift.weeks - 1])
            .filter((w) => w < shift.weeks)
            .map((w) => (
              <text key={w} x={x(w)} y={H - M.bottom + 18} textAnchor="middle" className="tick">
                week {w + 1}
              </text>
            ))}
        </g>
        <path
          d={regretArea(t.weekly.bridge, shift.shift_week, xs, ys)}
          fill="var(--arm-kev_rl)"
          opacity={0.1}
        />
        {marks.map((m, k) => {
          // a label too close to the previous one drops a line, so close marks never collide
          const prev = marks[k - 1];
          const low = prev !== undefined && Math.abs(x(m.week) - x(prev.week)) < 70;
          return (
            <g key={m.label} className="mark">
              <line
                x1={x(m.week)}
                x2={x(m.week)}
                y1={M.top - 8}
                y2={H - M.bottom}
                stroke="var(--rule-strong)"
                strokeDasharray="3 3"
              />
              <text x={x(m.week) + 4} y={low ? M.top + 6 : M.top - 10} className="tick">
                {m.label}
              </text>
            </g>
          );
        })}
        {STRATEGIES.map((s) => (
          <path
            key={s}
            d={weeklyPath(t.weekly[s], xs, ys)}
            fill="none"
            stroke={STRATEGY_STYLE[s].color}
            strokeWidth={STRATEGY_STYLE[s].width}
            strokeDasharray={STRATEGY_STYLE[s].dash}
            strokeLinejoin="round"
          />
        ))}
      </svg>
      <figcaption className="legend">
        {[...STRATEGIES].reverse().map((s) => (
          <span key={s} className="legend-item">
            <svg width="26" height="10" aria-hidden="true">
              <line
                x1="1"
                x2="25"
                y1="5"
                y2="5"
                stroke={STRATEGY_STYLE[s].color}
                strokeWidth={STRATEGY_STYLE[s].width}
                strokeDasharray={STRATEGY_STYLE[s].dash}
              />
            </svg>
            {STRATEGY_LABEL[s]}
          </span>
        ))}
      </figcaption>
    </figure>
  );
}

function Outcome({ shift, t, mode }: { shift: ShiftResult; t: ShiftTimeline; mode: Mode }) {
  const s = summarise(t);
  const how =
    mode === 'sim'
      ? `the simulator takes ${shift.rebuild_weeks} weeks to rebuild`
      : `RL learns only from live weeks, pooled over ${shift.live_units} products`;
  return (
    <div className="outcome">
      <p className="stats">
        <span>
          <b>{fmtMoney(s.bridge)}</b> given up with the bridge
        </span>
        <span>
          <b>{fmtMoney(s.rlOnly)}</b> with the better RL-only choice
        </span>
        <span>
          <b>{isNum(s.saving) ? fmtShare(s.saving) : '–'}</b> saved by bridging
        </span>
      </p>
      <p>
        Here {how}.{' '}
        {t.trigger_week === null ? 'Nobody notices the change, so nothing happens. ' : ''}
        {t.handback_week !== null
          ? `The bridge hands back to retrained RL ${fmtWeeksAfter(t.handback_week)} after the change.`
          : 'Retrained RL never beats the bridge within the year, so Kev keeps the calls.'}{' '}
        Lowest cost overall: <b>{STRATEGY_LABEL[s.best as Strategy]}</b>.
      </p>
    </div>
  );
}

function Detection({ shift }: { shift: ShiftResult }) {
  const rows = [
    { id: 'kev', label: 'Kev’s weekly alarm', note: 'reads the notes and the numbers' },
    { id: 'page_hinkley', label: 'Page–Hinkley test', note: 'demand numbers only' },
  ];
  return (
    <div className="detect">
      <h3>Would anyone notice?</h3>
      <p className="muted">
        Both detectors are tuned to raise the same rate of false alarms before the change, so the
        delays are a fair comparison.{' '}
        {shift.kind === 'silent'
          ? 'This change is silent: only the numbers carry it.'
          : 'The email is the only early signal; demand doesn’t move.'}
      </p>
      <div className="table-scroll">
        <table className="tbl">
          <thead>
            <tr>
              <th scope="col">Detector</th>
              <th scope="col">Caught it</th>
              <th scope="col">Typical delay</th>
              <th scope="col">False alarms per 100 weeks</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const d = shift.detection[r.id]?.matched;
              return (
                <tr key={r.id}>
                  <th scope="row">
                    {r.label}
                    <span className="cell-note">{r.note}</span>
                  </th>
                  <td>{d ? fmtShare(d.detection_rate) : '–'}</td>
                  <td>
                    {d && isNum(d.median_delay_weeks)
                      ? fmtWeeksAfter(Math.round(d.median_delay_weeks))
                      : 'never'}
                  </td>
                  <td>{d ? fmtPct(d.false_alarms_per_100_weeks, 1).replace('%', '') : '–'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
