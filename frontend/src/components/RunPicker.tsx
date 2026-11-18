import type { Run, RunSummary } from '../types';

const REPO = 'https://github.com/carlosdonado10/fast-and-slow';

function fmtRunDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}

function optionLabel(r: RunSummary): string {
  const smoke = r.profile === 'smoke' ? ', smoke test' : '';
  return `${r.label} (${fmtRunDate(r.created_at)}${smoke})`;
}

/** Every published run of the study; the report shows one at a time, the newest by default. */
export function RunPicker({
  run,
  runs,
  onPick,
}: {
  run: Run;
  runs: RunSummary[];
  onPick: (id: string) => void;
}) {
  const options = runs.some((r) => r.id === run.id) ? runs : [run, ...runs];
  return (
    <div className="runbar">
      <label className="runbar-pick">
        <span>Showing run</span>
        <select value={run.id} onChange={(e) => onPick(e.target.value)}>
          {options.map((r) => (
            <option key={r.id} value={r.id}>
              {optionLabel(r)}
            </option>
          ))}
        </select>
      </label>
      <p className="runbar-meta">
        Published {fmtRunDate(run.created_at)}
        {run.gpu ? `, trained on ${run.gpu}` : ''}
        {run.git_commit ? (
          <>
            , from code at{' '}
            <a href={`${REPO}/commit/${run.git_commit}`} target="_blank" rel="noreferrer">
              {run.git_commit}
            </a>
          </>
        ) : null}
        . {options.length === 1 ? 'The only run so far.' : `${options.length} runs published.`}
      </p>
    </div>
  );
}
