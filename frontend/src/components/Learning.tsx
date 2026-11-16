import type { Report } from '../types';
import { FAMILY_LABEL, armLabel, familyOf, sortArms } from '../lib/arms';
import { fmtDecisions, fmtDuration, fmtPct, horizonLabel } from '../lib/format';
import { LearningChart } from './LearningChart';
import { cleanPoints, curveArms, isStatic } from '../lib/curves';
import { ArmName, DataDetails, Section, Segmented, Swatch } from './ui';
import { Cast } from './Cast';

export interface ArmToggleProps {
  report: Report;
  arms: string[];
  visible: string[];
  onToggle: (arm: string) => void;
}

/** Legend that doubles as the arm toggles, grouped by family. Colour follows the arm, never its rank. */
export function ArmToggles({ report, arms, visible, onToggle }: ArmToggleProps) {
  const families = (['system-one', 'classic-rl'] as const).filter((f) =>
    arms.some((a) => familyOf(a, report) === f),
  );
  return (
    <div role="group" aria-label="Show or hide arms">
      {families.map((f) => (
        <div className="chip-group" key={f}>
          <span className="chip-group-label">{FAMILY_LABEL[f]}</span>
          <div className="chips">
            {arms
              .filter((a) => familyOf(a, report) === f)
              .map((a) => (
                <button
                  key={a}
                  type="button"
                  className="chip"
                  aria-pressed={visible.includes(a)}
                  onClick={() => onToggle(a)}
                >
                  <Swatch arm={a} />
                  {armLabel(report, a)}
                </button>
              ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export function HorizonControl({
  report,
  horizon,
  onChange,
}: {
  report: Report;
  horizon: string;
  onChange: (h: string) => void;
}) {
  return (
    <Segmented
      label="Planning horizon"
      value={horizon}
      options={report.meta.horizons.map((h) => ({ value: String(h), label: horizonLabel(h) }))}
      onChange={onChange}
    />
  );
}

interface Props {
  report: Report;
  horizon: string;
  onHorizon: (h: string) => void;
  visible: string[];
  onToggle: (arm: string) => void;
}

/** F4 */
export function Learning({ report, horizon, onHorizon, visible, onToggle }: Props) {
  const feature = report.features.find((f) => f.id === 'F4');
  const arms = curveArms(report, horizon);
  const curves = report.curves[horizon] ?? {};
  const seeds = Math.max(0, ...Object.values(curves).map((c) => c.seeds.length));
  return (
    <Section
      id="learning"
      fid="F4"
      feature={feature}
      title="How does each approach learn?"
      intro={
        <>
          <p>
            For ML practitioners weighing a decision model against classic RL. Each line is an arm's
            score on the same held-out scenarios as it consumes training decisions. Untrained scores
            sit in their own slot at zero, because a log axis can't show zero and the starting point
            is half the story (H1). Shaded bands are the range across {seeds}{' '}
            {seeds === 1 ? 'seed' : 'seeds'}; dotted lines are arms that don't train. A faint line
            joins each untrained score to the first trained one across the axis break.
          </p>
        </>
      }
    >
      <div className="controls">
        <HorizonControl report={report} horizon={horizon} onChange={onHorizon} />
      </div>
      <div className="controls">
        <ArmToggles report={report} arms={arms} visible={visible} onToggle={onToggle} />
      </div>
      {arms.length ? (
        <LearningChart report={report} horizon={horizon} visible={visible} />
      ) : (
        <p className="empty">No learning curves for this horizon yet.</p>
      )}
      <p className="chart-caption">
        Bayes-optimal is 100 and random is 0 by definition, so scores above 100 mean better than the
        Bayes planner on these scenarios (noise, or luck), and below 0 worse than ordering at
        random. The diamond marks where a PPO curve passes {armLabel(report, 'kev_rl')}.
      </p>
      <DataDetails summary="Show the curve data as a table">
        <table className="tbl">
          <caption>Mean % of optimal at each measured point, {horizonLabel(horizon)}</caption>
          <thead>
            <tr>
              <th scope="col">Arm</th>
              <th scope="col">Decisions</th>
              <th scope="col">% of optimal</th>
              <th scope="col">Seed range</th>
              <th scope="col">Compute so far</th>
            </tr>
          </thead>
          <tbody>
            {sortArms(Object.keys(curves)).flatMap((a) => {
              const c = curves[a]!;
              return cleanPoints(c.mean).map((p, i) => {
                const band = c.band.find((b) => b.decisions === p.decisions);
                return (
                  <tr key={`${a}-${p.decisions}`}>
                    <th scope="row">
                      {i === 0 ? <ArmName arm={a} label={armLabel(report, a)} /> : ''}
                    </th>
                    <td>{fmtDecisions(p.decisions)}</td>
                    <td>{fmtPct(p.score, 1)}</td>
                    <td>{band ? `${fmtPct(band.lo, 0)} to ${fmtPct(band.hi, 0)}` : '–'}</td>
                    <td>{isStatic(c) ? 'no training' : fmtDuration(p.seconds)}</td>
                  </tr>
                );
              });
            })}
          </tbody>
        </table>
      </DataDetails>
      <h3 className="strip-title">Who's competing</h3>
      <Cast report={report} />
    </Section>
  );
}
