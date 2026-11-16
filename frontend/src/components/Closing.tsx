import type { Report } from '../types';
import { FAMILY_LABEL, armLabel, familyOf, sortArms, type Family } from '../lib/arms';
import { fmtHours, fmtMs, fmtPct, fmtUsd, horizonLabel } from '../lib/format';
import { ArmName, Section } from './ui';

/** F8: cost and latency, with accuracy alongside so the trade-off is on one row. */
export function Costs({ report }: { report: Report }) {
  const feature = report.features.find((f) => f.id === 'F8');
  const arms = sortArms(Object.keys(report.costs ?? {}));
  const hMax = String(Math.max(...report.meta.horizons));
  const families: Family[] = ['system-one', 'classic-rl'];
  return (
    <Section
      id="costs"
      fid="F8"
      feature={feature}
      title="What will it cost to train and to run?"
      intro={
        <p>
          For practitioners and buyers making a build-or-buy call. Training cost is paid once per
          horizon and seed; serving cost is paid on every decision. Accuracy sits in the last column
          so neither is read alone. GPU arms are priced at a stated H100 hourly rate; CPU arms are
          shown as free to serve, which flatters them only slightly.
        </p>
      }
    >
      {arms.length ? (
        <div className="table-scroll">
          <table className="tbl">
            <caption>
              Training totals across horizons and seeds; latency is per decision at evaluation.
            </caption>
            <thead>
              <tr>
                <th scope="col">Arm</th>
                <th scope="col">Hardware</th>
                <th scope="col">Training time</th>
                <th scope="col">Training cost</th>
                <th scope="col">Latency p50</th>
                <th scope="col">Latency p95</th>
                <th scope="col">Cost per 1k decisions</th>
                <th scope="col">% of optimal, {horizonLabel(hMax)}</th>
              </tr>
            </thead>
            <tbody>
              {families.flatMap((f) => {
                const members = arms.filter((a) => familyOf(a, report) === f);
                if (!members.length) return [];
                return [
                  <tr className="family-row" key={f}>
                    <th scope="rowgroup" colSpan={8}>
                      {FAMILY_LABEL[f]}
                    </th>
                  </tr>,
                  ...members.map((a) => {
                    const c = report.costs[a]!;
                    return (
                      <tr key={a}>
                        <th scope="row">
                          <ArmName arm={a} label={armLabel(report, a)} />
                        </th>
                        <td>{c.hardware}</td>
                        <td>{c.train_hours === 0 ? 'none' : fmtHours(c.train_hours)}</td>
                        <td>{fmtUsd(c.train_usd)}</td>
                        <td>{fmtMs(c.ms_per_decision_p50)}</td>
                        <td>{fmtMs(c.ms_per_decision_p95)}</td>
                        <td>{fmtUsd(c.usd_per_1k_decisions)}</td>
                        <td>{fmtPct(report.results[hMax]?.[a]?.pct_optimal.mean, 1)}</td>
                      </tr>
                    );
                  }),
                ];
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="empty">No cost data in this export.</p>
      )}
    </Section>
  );
}

/** F9: every design choice with its reasoning and the alternative rejected, plus the feature map itself. */
export function DecisionsLog({ report }: { report: Report }) {
  const feature = report.features.find((f) => f.id === 'F9');
  return (
    <Section
      id="decisions"
      fid="F9"
      feature={feature}
      title="How does the person behind this think?"
      intro={
        <p>
          For hiring managers asking how the person behind this thinks. Each design choice, the
          reasoning, and the alternative that was considered and rejected. The study is the excuse;
          this list is the work.
        </p>
      }
    >
      <ol className="log">
        {report.decisions.map((d) => (
          <li key={d.decision}>
            <h3>{d.decision}</h3>
            <div>
              <span className="k">Why</span>
              <p>{d.why}</p>
            </div>
            <div className="rejected">
              <span className="k">Rejected</span>
              <p>{d.rejected}</p>
            </div>
          </li>
        ))}
      </ol>
      <div className="fmap">
        <h3>This report's own feature map</h3>
        <p className="chart-caption" style={{ marginTop: 0, marginBottom: 12 }}>
          Every section above was planned against a reader and a need before it was built.
        </p>
        <div className="table-scroll">
          <table className="tbl">
            <thead>
              <tr>
                <th scope="col">Feature</th>
                <th scope="col">For</th>
                <th scope="col">Their need</th>
                <th scope="col">Why it's built this way</th>
              </tr>
            </thead>
            <tbody>
              {report.features.map((f) => (
                <tr key={f.id}>
                  <td>
                    <span className="num" style={{ color: 'var(--ink-3)', marginRight: 8 }}>
                      {f.id}
                    </span>
                    {f.name}
                  </td>
                  <td>{f.user}</td>
                  <td>{f.need}</td>
                  <td style={{ color: 'var(--ink-2)' }}>{f.why}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Section>
  );
}

/** F10 */
export function Limitations({ report }: { report: Report }) {
  const feature = report.features.find((f) => f.id === 'F10');
  return (
    <Section
      id="limitations"
      fid="F10"
      feature={feature}
      title="What can't this study tell me?"
      intro={
        <p>
          For everyone. A result is only as credible as its stated limits. Read these before quoting
          any number above.
        </p>
      }
    >
      <ul className="limits">
        {report.limitations.map((l) => (
          <li key={l}>{l}</li>
        ))}
      </ul>
    </Section>
  );
}

export function Footer({ report, source }: { report: Report; source: string }) {
  const m = report.meta;
  return (
    <footer className="footer">
      <dl>
        <dt>Profile</dt>
        <dd>{m.profile}</dd>
        <dt>Kev run</dt>
        <dd>{m.kev_run}</dd>
        <dt>Note writer</dt>
        <dd>{m.note_writer}</dd>
        <dt>Data file</dt>
        <dd>{source}</dd>
      </dl>
      <dl>
        <dt>Seeds</dt>
        <dd>{m.seeds.join(', ')}</dd>
        <dt>Scenarios</dt>
        <dd>{m.scenarios_per_horizon} held out per horizon</dd>
        <dt>Error budget</dt>
        <dd>{Math.round(m.error_budget * 100)}%</dd>
        <dt>By</dt>
        <dd>Carlos Donado</dd>
      </dl>
    </footer>
  );
}
