import { useEffect, useState } from 'react';
import type { Report as ReportData, Run, RunSummary } from './types';
import { loadRun, NoRunsYet, runFromUrl, type Loaded } from './lib/load';
import { RunPicker } from './components/RunPicker';
import { LEARNERS, sortArms } from './lib/arms';
import { Hero } from './components/Hero';
import { Scorecard } from './components/Scorecard';
import { Desk } from './components/Desk';
import { Learning } from './components/Learning';
import { Efficiency } from './components/Efficiency';
import { HorizonDial } from './components/HorizonDial';
import { Handoff } from './components/Handoff';
import { Replays } from './components/Replays';
import { Transition } from './components/Transition';
import { Costs, DecisionsLog, Footer, Limitations } from './components/Closing';

export function Report({
  report,
  run,
  runs = [],
  onPick,
}: {
  report: ReportData;
  run?: Run;
  runs?: RunSummary[];
  onPick?: (id: string) => void;
}) {
  const horizons = report.meta.horizons.map(String);
  // One horizon and one set of visible arms, shared by every chart that has them.
  const [horizon, setHorizon] = useState(horizons[horizons.length > 1 ? 1 : 0] ?? '1');
  const present = sortArms(
    new Set(Object.values(report.curves).flatMap((c) => Object.keys(c))),
  ).filter((a) => LEARNERS.includes(a));
  const [visible, setVisible] = useState<string[]>(present);
  const toggle = (arm: string) =>
    setVisible((v) => (v.includes(arm) ? v.filter((a) => a !== arm) : sortArms([...v, arm])));
  const smoke = report.meta.profile === 'smoke';

  // The page renders after the data loads, so the browser's own jump to #section has already
  // happened on an empty page. Do it once the sections exist, so shared links land.
  useEffect(() => {
    const id = decodeURIComponent(window.location.hash.slice(1));
    if (id) document.getElementById(id)?.scrollIntoView({ block: 'start', behavior: 'instant' });
  }, []);

  return (
    <>
      <a className="skip" href="#main">
        Skip to the report
      </a>
      {smoke && (
        <div className="smoke" role="note">
          <div className="wrap">
            <strong>Smoke-test data, not results.</strong>{' '}
            <span>
              These numbers come from a fake model run on a laptop to check the pipeline end to end.
              Every value is a placeholder until the full run replaces it.
            </span>
          </div>
        </div>
      )}
      <div className="wrap">
        <div className="topbar">
          <span>
            <b>Carlos Donado</b>
            <span className="brand-sub">{report.meta.title}, a research report</span>
          </span>
          <nav aria-label="Jump to">
            <a href="#scorecard">Scorecard</a>
            <a href="#desk">The desk</a>
            <a href="#learning">Learning</a>
            <a href="#handoff">Hand-off</a>
            <a href="#transition">Regime changes</a>
            <a href="#decisions">Decisions</a>
          </nav>
          <a className="pill" href="../../">
            Back to the portfolio
          </a>
        </div>
        {run && onPick && <RunPicker run={run} runs={runs} onPick={onPick} />}
        <Hero report={report} />
        <main id="main">
          <Scorecard report={report} />
          <Desk report={report} />
          <Learning
            report={report}
            horizon={horizon}
            onHorizon={setHorizon}
            visible={visible}
            onToggle={toggle}
          />
          <Efficiency
            report={report}
            horizon={horizon}
            onHorizon={setHorizon}
            visible={visible}
            onToggle={toggle}
          />
          <HorizonDial report={report} />
          <Handoff report={report} horizon={horizon} onHorizon={setHorizon} />
          <Replays report={report} />
          <Transition report={report} />
          <Costs report={report} />
          <DecisionsLog report={report} />
          <Limitations report={report} />
        </main>
        <Footer report={report} runId={run?.id ?? 'not published'} />
      </div>
    </>
  );
}

type State =
  | { kind: 'loading' }
  | { kind: 'ready'; data: Loaded }
  | { kind: 'empty' }
  | { kind: 'error'; message: string };

export default function App({
  load = loadRun,
}: {
  load?: (runId: string | null) => Promise<Loaded>;
}) {
  const [runId, setRunId] = useState<string | null>(() => runFromUrl());
  const [state, setState] = useState<State>({ kind: 'loading' });

  useEffect(() => {
    let live = true;
    load(runId)
      .then((data) => live && setState({ kind: 'ready', data }))
      .catch((e: unknown) => {
        if (!live) return;
        if (e instanceof NoRunsYet) setState({ kind: 'empty' });
        else setState({ kind: 'error', message: e instanceof Error ? e.message : String(e) });
      });
    return () => {
      live = false;
    };
  }, [load, runId]);

  // Back and forward move between runs the way they move between pages.
  useEffect(() => {
    const onPop = () => {
      setState({ kind: 'loading' });
      setRunId(runFromUrl());
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const pick = (id: string) => {
    const url = new URL(window.location.href);
    url.searchParams.set('run', id);
    url.hash = '';
    window.history.pushState(null, '', url);
    setState({ kind: 'loading' });
    setRunId(id);
  };

  if (state.kind === 'loading') {
    return (
      <div className="state" role="status">
        <p>Loading the report…</p>
      </div>
    );
  }
  if (state.kind === 'empty') {
    return (
      <div className="state" role="status">
        <div>
          <p>No runs have been published yet.</p>
          <p>
            Run the study, then publish it as a run with <code>fns publish</code> (section 7 of the
            Colab notebook). The newest run opens here.
          </p>
        </div>
      </div>
    );
  }
  if (state.kind === 'error') {
    return (
      <div className="state" role="alert">
        <div>
          <p>The report didn't load.</p>
          <p>{state.message}</p>
          <p>
            The report reads its runs from the API. Locally, start it with{' '}
            <code>docker compose up</code> in the repository, then reload.
          </p>
        </div>
      </div>
    );
  }
  const { run, runs } = state.data;
  return <Report key={run.id} report={run.report} run={run} runs={runs} onPick={pick} />;
}
