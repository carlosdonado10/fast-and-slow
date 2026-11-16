import { useEffect, useState } from 'react';
import type { Report as ReportData } from './types';
import { loadReport, type Loaded } from './lib/load';
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

export function Report({ report, source }: { report: ReportData; source: string }) {
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
        <Footer report={report} source={source} />
      </div>
    </>
  );
}

export default function App({ load = loadReport }: { load?: () => Promise<Loaded> }) {
  const [state, setState] = useState<
    { kind: 'loading' } | { kind: 'ready'; data: Loaded } | { kind: 'error'; message: string }
  >({ kind: 'loading' });

  useEffect(() => {
    let live = true;
    load()
      .then((data) => live && setState({ kind: 'ready', data }))
      .catch(
        (e: unknown) =>
          live && setState({ kind: 'error', message: e instanceof Error ? e.message : String(e) }),
      );
    return () => {
      live = false;
    };
  }, [load]);

  if (state.kind === 'loading') {
    return (
      <div className="state" role="status">
        <p>Loading the report…</p>
      </div>
    );
  }
  if (state.kind === 'error') {
    return (
      <div className="state" role="alert">
        <div>
          <p>The report data didn't load.</p>
          <p>{state.message}</p>
          <p>
            Export it with the backend's export stage into public/data/report.json, then reload.
          </p>
        </div>
      </div>
    );
  }
  return <Report report={state.data.report} source={state.data.source} />;
}
