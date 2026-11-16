import { useState } from 'react';
import type { Report } from '../types';
import {
  REGIME_LABEL,
  compareOrder,
  orderOptions,
  parseDeskState,
  posteriorRows,
} from '../lib/desk';
import { capitalize, fmtMoney, fmtNumber, horizonLabel, humanize } from '../lib/format';
import { Section } from './ui';

/** F3: step through one scenario as the planner sees it, make a call, then reveal the hidden truth. */
export function Desk({ report }: { report: Report }) {
  const feature = report.features.find((f) => f.id === 'F3');
  const weeks = report.desk?.weeks ?? [];
  const regimes = report.meta.world.regimes;
  const [step, setStep] = useState(0);
  const [picks, setPicks] = useState<Record<number, number>>({});
  const [revealed, setRevealed] = useState<Record<number, boolean>>({});
  const w = report.meta.world;

  const intro = (
    <p>
      For planners. One product, {horizonLabel(report.desk?.horizon ?? weeks.length)} of orders.
      Each week you get the stock position and a note from the inbox; the true demand regime stays
      hidden. Make the call, then reveal what was really going on and what the Bayes-optimal planner
      ordered. Notes come from sources of known reliability (
      {w.sources
        .map((s, i) => `${humanize(s)} ${Math.round((w.source_accuracy[i] ?? 0) * 100)}%`)
        .join(', ')}
      ).
    </p>
  );

  if (!weeks.length) {
    return (
      <Section id="desk" fid="F3" feature={feature} title="The desk" intro={intro}>
        <p className="empty">This export has no desk scenario.</p>
      </Section>
    );
  }

  const week = weeks[Math.min(step, weeks.length - 1)]!;
  const view = parseDeskState(week.state);
  const options = orderOptions(week.questions, w.order_menu);
  const mine = picks[step] ?? null;
  const isRevealed = !!revealed[step];
  const post = posteriorRows(week.posterior, regimes);
  const regimeIdx = regimes.indexOf(week.true_regime);
  const demandMean = w.demand_means[regimeIdx];

  return (
    <Section id="desk" fid="F3" feature={feature} title="Is this my world, or a toy?" intro={intro}>
      <div className="desk">
        <div className="desk-steps">
          <ol aria-label="Weeks in this scenario">
            {weeks.map((wk, i) => (
              <li key={wk.week}>
                <button
                  type="button"
                  className="step"
                  aria-current={i === step ? 'step' : undefined}
                  onClick={() => setStep(i)}
                >
                  Week {wk.week}
                  {revealed[i] && <span className="done-dot" aria-label="revealed" />}
                </button>
              </li>
            ))}
          </ol>
          <span className="desk-h" style={{ margin: 0 }}>
            {view.weekLine ?? `Week ${week.week} of ${weeks.length}`}
          </span>
        </div>

        <div className="desk-view">
          <p className="desk-h">What the planner sees</p>
          <p className="desk-stock">
            <span className="big num">{view.stock ?? '–'}</span>
            <span>units on hand</span>
          </p>
          {view.inbox.length > 0 && (
            <>
              <p className="desk-h">Inbox</p>
              <ul className="inbox">
                {[...view.inbox].reverse().map((n, i) => (
                  <li key={i} className={n.current ? 'current' : undefined}>
                    <span className="when">{capitalize(n.when || 'note')}</span>
                    {n.text}
                  </li>
                ))}
              </ul>
            </>
          )}
          {view.past.length > 0 && (
            <div className="table-scroll" style={{ marginBottom: 22 }}>
              <table className="tbl">
                <caption>Past weeks</caption>
                <thead>
                  <tr>
                    <th scope="col">Week</th>
                    <th scope="col">Ordered</th>
                    <th scope="col">Demand</th>
                    <th scope="col">Sold</th>
                    <th scope="col">Missed</th>
                  </tr>
                </thead>
                <tbody>
                  {view.past.map((p) => (
                    <tr key={p.week}>
                      <td>{p.week}</td>
                      <td>{p.ordered}</td>
                      <td>{p.demand}</td>
                      <td>{p.sold}</td>
                      <td>{p.missed}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <details className="brief">
            <summary>Read the full brief the agents get</summary>
            <p>{view.brief}</p>
          </details>
        </div>

        <div className="desk-call">
          <div>
            <p className="desk-h" id="desk-q">
              {week.questions?.order?.instructions ?? 'How many units do you order this week?'}
            </p>
            <div className="orders" role="radiogroup" aria-labelledby="desk-q">
              {options.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  role="radio"
                  aria-checked={mine === o.value}
                  aria-label={o.label}
                  className={`order-opt${isRevealed && o.value === week.bayes_order ? ' bayes-pick' : ''}`}
                  onClick={() => setPicks((p) => ({ ...p, [step]: o.value }))}
                >
                  {o.value}
                </button>
              ))}
            </div>
          </div>

          {!isRevealed ? (
            <div className="hidden-truth">
              <p>
                The demand regime this week is hidden from you and from every agent except the
                perfect-information planner.
              </p>
              <button
                type="button"
                className="btn hl-btn"
                onClick={() => setRevealed((r) => ({ ...r, [step]: true }))}
              >
                Reveal the hidden truth
              </button>
            </div>
          ) : (
            <div className="reveal reveal-enter" aria-live="polite">
              <p className="truth">
                <small>The hidden regime was</small>
                {REGIME_LABEL[week.true_regime] ?? capitalize(week.true_regime)}
                {demandMean !== undefined && (
                  <small style={{ marginTop: 6 }}>about {demandMean} units a week on average</small>
                )}
              </p>
              <div>
                <p className="desk-h">What the Bayes-optimal planner believed, given the notes</p>
                <div className="post-rows">
                  {post.map((r) => (
                    <div
                      key={r.regime}
                      className={`post-row${r.regime === week.true_regime ? ' is-true' : ''}`}
                    >
                      <span>
                        {REGIME_LABEL[r.regime] ?? r.regime}
                        {r.regime === week.true_regime && <span className="tag"> (true)</span>}
                      </span>
                      <span className="post-track">
                        <span
                          className="post-fill"
                          style={{ width: `${Math.max(0.5, r.p * 100)}%`, display: 'block' }}
                        />
                      </span>
                      <span className="num">{Math.round(r.p * 100)}%</span>
                    </div>
                  ))}
                </div>
              </div>
              <div className="facts">
                <p>
                  <b>{week.bayes_order}</b>Bayes ordered
                </p>
                <p>
                  <b>{week.demand}</b>Actual demand
                </p>
                <p>
                  <b>{fmtMoney(week.profit)}</b>Bayes profit
                </p>
              </div>
              <p className="verdict-line">{compareOrder(mine, week.bayes_order)}</p>
              {step < weeks.length - 1 && (
                <button type="button" className="btn ghost" onClick={() => setStep(step + 1)}>
                  Go to week {weeks[step + 1]!.week}
                </button>
              )}
              {step === weeks.length - 1 && (
                <p className="verdict-line">
                  Bayes made {fmtMoney(weeks.reduce((s, x) => s + (x.profit ?? 0), 0))} over the{' '}
                  {weeks.length} weeks
                  {Object.keys(picks).length > 0 &&
                    `; you matched its order in ${fmtNumber(
                      weeks.filter((x, i) => picks[i] === x.bayes_order).length,
                    )} of ${weeks.length}`}
                  .
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </Section>
  );
}
