import { useState } from 'react';
import type { Replay, ReplayDecision, Report } from '../types';
import { armColor, armLabel } from '../lib/arms';
import { REGIME_LABEL, posteriorRows } from '../lib/desk';
import { fmtMoney, fmtProb, horizonLabel, isNum } from '../lib/format';
import { defaultPair, disagreements, maxOrder, noteSourceLabel, replayArms } from '../lib/replay';
import { ArmName, Section, Segmented, Swatch } from './ui';

/** F7: one scenario, Kev vs PPO vs Bayes, week by week, with each agent's confidence. */
export function Replays({ report }: { report: Report }) {
  const feature = report.features.find((f) => f.id === 'F7');
  const replays = report.replays ?? [];
  const [idx, setIdx] = useState(0);
  const replay: Replay | undefined = replays[Math.min(idx, replays.length - 1)];
  const arms = replayArms(replay);
  const kevs = arms.filter((a) => a.startsWith('kev'));
  const ppos = arms.filter((a) => a.startsWith('ppo'));
  const def = defaultPair(arms);
  const [kevPick, setKev] = useState<string | null>(null);
  const [ppoPick, setPpo] = useState<string | null>(null);
  const kev = kevPick && kevs.includes(kevPick) ? kevPick : def.kev;
  const ppo = ppoPick && ppos.includes(ppoPick) ? ppoPick : def.ppo;
  const cols = ['bayes', kev, ppo].filter((a): a is string => !!a && arms.includes(a));
  const differ = kev && ppo ? new Set(disagreements(replay, kev, ppo)) : new Set<number>();
  const top = maxOrder(report);
  const regimes = report.meta.world.regimes;

  const intro = (
    <p>
      For anyone who wants to see a case rather than a chart. These are held-out scenarios where Kev
      and PPO disagreed most. Read down the weeks: the note, what was really happening, and what
      each agent ordered and how sure it was. Highlighted weeks are where the two learners disagree.
    </p>
  );

  if (!replay || !cols.length) {
    return (
      <Section id="replays" fid="F7" feature={feature} title="Decision replays" intro={intro}>
        <p className="empty">This export has no replays.</p>
      </Section>
    );
  }

  return (
    <Section
      id="replays"
      fid="F7"
      feature={feature}
      title="Show me a week where they disagree"
      intro={intro}
    >
      <div className="controls">
        {replays.length > 1 && (
          <Segmented
            label="Scenario"
            value={idx}
            options={replays.map((_, i) => ({
              value: i,
              label: `${i + 1}`,
            }))}
            onChange={setIdx}
          />
        )}
        {kevs.length > 1 && (
          <Segmented
            label="Kev arm"
            value={kev ?? ''}
            options={kevs.map((a) => ({
              value: a,
              label: armLabel(report, a).replace(/^Kev-4B,?\s*/, '') || a,
            }))}
            onChange={setKev}
          />
        )}
        {ppos.length > 1 && (
          <Segmented
            label="PPO arm"
            value={ppo ?? ''}
            options={ppos.map((a) => ({
              value: a,
              label: armLabel(report, a).replace(/^PPO,?\s*/, '') || a,
            }))}
            onChange={setPpo}
          />
        )}
      </div>
      <p className="chart-caption" style={{ marginBottom: 14 }}>
        Scenario seed {replay.seed}, {horizonLabel(replay.horizon)}.{' '}
        {kev && ppo
          ? `${armLabel(report, kev)} and ${armLabel(report, ppo)} disagree in ${differ.size} of ${replay.weeks.length} weeks.`
          : ''}
      </p>

      <div className="replay" role="table" aria-label="Week-by-week replay">
        <div className="replay-head" role="row">
          <span role="columnheader">Week, note and hidden truth</span>
          {cols.map((a) => (
            <span className="who" role="columnheader" key={a}>
              <Swatch arm={a} />
              {armLabel(report, a)}
            </span>
          ))}
        </div>
        {replay.weeks.map((w) => {
          const anyDec = Object.values(w.decisions)[0];
          const post = posteriorRows(anyDec?.posterior, regimes);
          const isDiff = differ.has(w.week);
          return (
            <div className={`replay-week${isDiff ? ' differ' : ''}`} role="row" key={w.week}>
              <div role="cell">
                <p className="week-no">
                  <span>Week {w.week}</span>
                  {isDiff && <span className="differ-tag">They disagree</span>}
                  {anyDec && <span>{anyDec.decidable ? 'Decidable call' : 'Judgement call'}</span>}
                </p>
                <p className="note">{w.note ? `“${w.note}”` : 'No note this week.'}</p>
                <p className="note-meta">
                  {noteSourceLabel(w.note_source)}
                  {w.note_claim ? `, claims ${w.note_claim}` : ''}. Really:{' '}
                  <b style={{ color: 'var(--ink)' }}>
                    {(REGIME_LABEL[w.true_regime] ?? w.true_regime).toLowerCase()}
                  </b>
                  , demand {w.demand}.
                </p>
                {post.length > 0 && (
                  <div
                    className="posterior-bar"
                    role="img"
                    aria-label={`Bayes belief: ${post
                      .map((p) => `${p.regime} ${Math.round(p.p * 100)}%`)
                      .join(', ')}`}
                    title={post.map((p) => `${p.regime} ${Math.round(p.p * 100)}%`).join(', ')}
                  >
                    {post.map((p) => (
                      <span
                        key={p.regime}
                        style={{
                          flex: Math.max(p.p, 0.001),
                          background: `var(--regime-${p.regime}, var(--ink-3))`,
                        }}
                      />
                    ))}
                  </div>
                )}
              </div>
              {cols.map((a) => (
                <Cell key={a} arm={a} label={armLabel(report, a)} d={w.decisions[a]} top={top} />
              ))}
            </div>
          );
        })}
        <div className="replay-total" role="row">
          <span role="cell" style={{ color: 'var(--ink-3)', fontSize: 'var(--t-xs)' }}>
            Profit over the scenario
          </span>
          {cols.map((a) => (
            <span role="cell" key={a}>
              <span className="replay-label-m">
                <Swatch arm={a} />
                {armLabel(report, a)}
              </span>
              <b>{fmtMoney(replay.totals[a])}</b>
            </span>
          ))}
        </div>
      </div>
      <p className="chart-caption">
        The thin bar under each note is the Bayes planner's belief over{' '}
        {regimes.map((r) => (REGIME_LABEL[r] ?? r).toLowerCase()).join(', ')} (faint to strong).
        “Matches Bayes” means the order is within 2% of the Bayes value.
      </p>
    </Section>
  );
}

function Cell({
  arm,
  label,
  d,
  top,
}: {
  arm: string;
  label: string;
  d: ReplayDecision | undefined;
  top: number;
}) {
  if (!d) {
    return (
      <div role="cell" className="cell-meta">
        no decision
      </div>
    );
  }
  return (
    <div role="cell">
      <span className="replay-label-m">
        <ArmName arm={arm} label={label} />
      </span>
      <p className="cell-order">
        {d.order}
        <small>units</small>
      </p>
      <div className="order-track" aria-hidden="true">
        <span style={{ width: `${(d.order / top) * 100}%`, background: armColor(arm) }} />
      </div>
      <div className="cell-meta">
        <span>
          Confidence{' '}
          <span className="num">{isNum(d.confidence) ? fmtProb(d.confidence) : '–'}</span>
        </span>
        <span className="ok-mark">
          <svg width={10} height={10} aria-hidden="true">
            {d.acceptable ? (
              <circle cx={5} cy={5} r={4.5} fill="currentColor" />
            ) : (
              <circle cx={5} cy={5} r={4} fill="none" stroke="currentColor" strokeWidth={1.5} />
            )}
          </svg>
          {d.acceptable ? 'Matches Bayes' : `Bayes ordered ${d.bayes_order}`}
        </span>
        <span>Week profit {fmtMoney(d.reward)}</span>
      </div>
    </div>
  );
}
