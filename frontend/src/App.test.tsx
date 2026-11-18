import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App, { Report } from './App';
import { NoRunsYet } from './lib/load';
import type { Report as ReportData } from './types';
import { SECTIONS } from './components/sections';

const sample = JSON.parse(
  readFileSync(resolve(__dirname, './test/fixtures/report.sample.json'), 'utf8'),
) as ReportData;

const clone = (): ReportData => structuredClone(sample);

function renderReport(data: ReportData = clone()) {
  return render(<Report report={data} />);
}

describe('the report, from the sample JSON', () => {
  it('renders every section in the feature map, each with who it is for', () => {
    renderReport();
    for (const s of SECTIONS) {
      if (s.fid === 'F1') continue;
      const section = document.getElementById(s.id);
      expect(section, s.id).not.toBeNull();
      expect(section!.querySelector('.fid')).toHaveTextContent(s.fid);
      const feature = sample.features.find((f) => f.id === s.fid)!;
      expect(within(section!).getAllByText(feature.user).length).toBeGreaterThan(0);
    }
  });

  it('leads with the question and the computed answer (F1)', () => {
    renderReport();
    expect(screen.getByRole('heading', { level: 1, name: 'Fast and Slow' })).toBeInTheDocument();
    expect(screen.getByText(sample.meta.subtitle)).toBeInTheDocument();
    expect(screen.getByText('Not yet.')).toBeInTheDocument();
  });

  it('flags smoke-test data, and stays quiet for a full run', () => {
    const { unmount } = renderReport();
    expect(screen.getByText('Smoke-test data, not results.')).toBeInTheDocument();
    unmount();
    const full = clone();
    full.meta.profile = 'full';
    renderReport(full);
    expect(screen.queryByText('Smoke-test data, not results.')).toBeNull();
  });

  it('shows each hypothesis with its verdict, evidence and rule from the JSON (F2)', () => {
    renderReport();
    const card = document.getElementById('scorecard')!;
    for (const h of sample.hypotheses) {
      expect(within(card).getByText(h.statement)).toBeInTheDocument();
      expect(within(card).getByText(h.evidence)).toBeInTheDocument();
      expect(within(card).getByText(h.rule)).toBeInTheDocument();
    }
    const count = (v: string) => sample.hypotheses.filter((h) => h.verdict === v).length;
    expect(within(card).getAllByText('Supported').length).toBe(count('supported'));
    expect(within(card).getAllByText('Not supported').length).toBe(count('not supported'));
  });

  it('lets a planner make a call and reveal the hidden truth (F3)', async () => {
    const user = userEvent.setup();
    renderReport();
    const desk = document.getElementById('desk')!;
    const first = sample.desk.weeks[0]!;
    await user.click(within(desk).getByRole('radio', { name: /order 15 units/i }));
    await user.click(within(desk).getByRole('button', { name: 'Reveal the hidden truth' }));
    expect(within(desk).getByText('The hidden regime was')).toBeInTheDocument();
    expect(
      within(desk).getByText(first.bayes_order === 15 ? /Same call/ : /than the Bayes-optimal/),
    ).toBeInTheDocument();
    // walk to the last week and reveal it for the summary
    for (let i = 1; i < sample.desk.weeks.length; i++) {
      await user.click(within(desk).getByRole('button', { name: `Go to week ${i + 1}` }));
      await user.click(within(desk).getByRole('button', { name: 'Reveal the hidden truth' }));
    }
    expect(within(desk).getByText(/over the 4 weeks/)).toBeInTheDocument();
    await user.click(within(desk).getByRole('button', { name: /^Week 1/ }));
    expect(within(desk).getByText('The hidden regime was')).toBeInTheDocument();
  });

  it('draws learning curves with arm toggles and a horizon switch (F4)', async () => {
    const user = userEvent.setup();
    renderReport();
    const sec = document.getElementById('learning')!;
    expect(
      within(sec).getByRole('img', { name: /Learning curves, 4-week horizon/ }),
    ).toBeInTheDocument();
    const toggle = within(sec).getByRole('button', { name: 'PPO, numbers' });
    expect(toggle).toHaveAttribute('aria-pressed', 'true');
    await user.click(toggle);
    expect(toggle).toHaveAttribute('aria-pressed', 'false');
    await user.click(toggle);
    expect(toggle).toHaveAttribute('aria-pressed', 'true');

    await user.click(within(sec).getAllByRole('radio', { name: '1 week' })[0]!);
    expect(
      within(sec).getByRole('img', { name: /Learning curves, 1-week horizon/ }),
    ).toBeInTheDocument();
  });

  it('reads scores at a point with the keyboard and the pointer (F4 readout)', () => {
    renderReport();
    const sec = document.getElementById('learning')!;
    const slider = within(sec).getByRole('slider');
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(within(sec).getByText('Untrained')).toBeInTheDocument();
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    fireEvent.keyDown(slider, { key: 'ArrowLeft' });
    expect(within(sec).getByText(/Right column: training cost so far/)).toBeInTheDocument();
    fireEvent.keyDown(slider, { key: 'Escape' });
    expect(within(sec).queryByText(/Right column/)).toBeNull();
    fireEvent.pointerMove(slider, { clientX: 600, clientY: 100 });
    fireEvent.pointerLeave(slider);
    fireEvent.blur(slider);
  });

  it('shows thresholds, gain per 10x and crossovers (F4b)', () => {
    renderReport();
    const sec = document.getElementById('efficiency')!;
    expect(within(sec).getByText('Gain per 10× more decisions')).toBeInTheDocument();
    expect(within(sec).getAllByText(/passes/).length).toBeGreaterThan(0);
    expect(within(sec).getAllByText('never').length).toBeGreaterThan(0);
  });

  it('draws one small multiple per arm across horizons (F5)', () => {
    renderReport();
    const sec = document.getElementById('horizon')!;
    expect(within(sec).getAllByRole('figure').length).toBe(7);
  });

  it('turns calibration into a hand-off answer at each budget (F6)', async () => {
    const user = userEvent.setup();
    renderReport();
    const sec = document.getElementById('handoff')!;
    expect(
      within(sec).getByText(/At a 5% error budget, Kev-4B \+ context could take/),
    ).toBeInTheDocument();
    await user.click(within(sec).getByRole('radio', { name: '10%' }));
    expect(within(sec).getByText(/At a 10% error budget/)).toBeInTheDocument();
    await user.click(within(sec).getByRole('radio', { name: /PPO, parsed signals/ }));
    expect(within(sec).getByText(/PPO, parsed signals could take/)).toBeInTheDocument();
    expect(
      within(sec).getByRole('img', { name: /Reliability diagram for PPO, parsed signals/ }),
    ).toBeInTheDocument();
    // keyboard on the segmented control
    const tenPct = within(sec).getByRole('radio', { name: '10%' });
    fireEvent.keyDown(tenPct, { key: 'ArrowLeft' });
    expect(within(sec).getByText(/At a 5% error budget/)).toBeInTheDocument();
    fireEvent.keyDown(within(sec).getByRole('radio', { name: '5%' }), { key: 'ArrowRight' });
    expect(within(sec).getByText(/At a 10% error budget/)).toBeInTheDocument();
  });

  it('replays a scenario side by side (F7)', async () => {
    const user = userEvent.setup();
    renderReport();
    const sec = document.getElementById('replays')!;
    expect(within(sec).getAllByRole('row').length).toBeGreaterThan(4);
    expect(within(sec).getByText(/disagree in \d of 4 weeks/)).toBeInTheDocument();
    await user.click(within(sec).getByRole('radio', { name: '2' }));
    expect(
      within(sec).getByText(new RegExp(`seed ${sample.replays[1]!.seed}`)),
    ).toBeInTheDocument();
    await user.click(within(sec).getByRole('radio', { name: 'zero-shot' }));
    expect(within(sec).getAllByText('Kev-4B, zero-shot').length).toBeGreaterThan(0);
  });

  it('lists costs, decisions and limitations straight from the JSON (F8–F10)', () => {
    renderReport();
    const costs = document.getElementById('costs')!;
    expect(within(costs).getAllByRole('row').length).toBe(1 + 2 + Object.keys(sample.costs).length);
    const log = document.getElementById('decisions')!;
    for (const d of sample.decisions) expect(within(log).getByText(d.decision)).toBeInTheDocument();
    const lim = document.getElementById('limitations')!;
    for (const l of sample.limitations) expect(within(lim).getByText(l)).toBeInTheDocument();
  });
});

describe('null and missing data', () => {
  it('renders with nulls, missing horizons and empty sections', () => {
    const d = clone();
    d.replays = [];
    d.desk = { horizon: 4, weeks: [] };
    d.costs = {};
    d.horizon_dial = {};
    d.curves['4'] = {};
    d.efficiency['4'] = {};
    for (const r of Object.values(d.results['4']!)) {
      r.pct_optimal.mean = null;
      r.calibration.ece = null;
      r.handoff.decidable.automation = {};
    }
    renderReport(d);
    expect(screen.getByText('This export has no replays.')).toBeInTheDocument();
    expect(screen.getByText('This export has no desk scenario.')).toBeInTheDocument();
    expect(screen.getByText('No cost data in this export.')).toBeInTheDocument();
    expect(screen.getByText('No horizon results yet.')).toBeInTheDocument();
    expect(screen.getByText('No learning curves for this horizon yet.')).toBeInTheDocument();
    expect(
      screen.getByText('No PPO curve passes a Kev curve at this horizon.'),
    ).toBeInTheDocument();
  });
});

describe('loading runs', () => {
  const summary = (id: string, label: string) => ({
    id,
    label,
    created_at: '2026-11-18T02:15:00+00:00',
    profile: 'full',
    git_commit: 'abc1234',
    gpu: 'NVIDIA A100-SXM4-40GB',
    verdicts: {},
  });
  const runs = [summary('new', 'Second A100 run'), summary('old', 'First A100 run')];
  const loader = vi.fn((id: string | null) =>
    Promise.resolve({ run: { ...summary(id ?? 'new', 'x'), report: clone() }, runs }),
  );

  it('shows a loading state, then the newest run with its provenance', async () => {
    window.history.replaceState(null, '', '/');
    render(<App load={loader} />);
    expect(screen.getByRole('status')).toHaveTextContent('Loading the report');
    expect(await screen.findByRole('heading', { level: 1 })).toBeInTheDocument();
    expect(loader).toHaveBeenLastCalledWith(null);
    expect(screen.getByText(/trained on NVIDIA A100-SXM4-40GB/)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'abc1234' })).toHaveAttribute(
      'href',
      'https://github.com/carlosdonado10/fast-and-slow/commit/abc1234',
    );
    expect(screen.getByText(/2 runs published/)).toBeInTheDocument();
  });

  it('switches runs from the picker, keeps it in the address, and follows back', async () => {
    window.history.replaceState(null, '', '/');
    render(<App load={loader} />);
    const picker = await screen.findByRole('combobox', { name: 'Showing run' });
    await userEvent.selectOptions(picker, 'old');
    expect(window.location.search).toBe('?run=old');
    await waitFor(() => expect(loader).toHaveBeenLastCalledWith('old'));
    await screen.findByRole('combobox', { name: 'Showing run' });
    window.history.replaceState(null, '', '/?run=new');
    fireEvent.popState(window);
    await waitFor(() => expect(loader).toHaveBeenLastCalledWith('new'));
  });

  it('lists the open run even when the list leaves it out', async () => {
    window.history.replaceState(null, '', '/?run=lonely');
    render(
      <App
        load={(id) =>
          Promise.resolve({
            run: { ...summary(id!, 'Lonely run'), gpu: '', git_commit: '', report: clone() },
            runs: [],
          })
        }
      />,
    );
    expect(await screen.findByRole('option', { name: /Lonely run/ })).toBeInTheDocument();
    expect(screen.getByText(/The only run so far/)).toBeInTheDocument();
  });

  it('says how to publish when there are no runs yet', async () => {
    render(<App load={() => Promise.reject(new NoRunsYet())} />);
    expect(await screen.findByText('No runs have been published yet.')).toBeInTheDocument();
    expect(screen.getByText(/fns publish/)).toBeInTheDocument();
  });

  it('explains how to fix a failed load', async () => {
    render(
      <App
        load={() =>
          Promise.reject(new Error('The runs API answered api/runs/latest with HTTP 503.'))
        }
      />,
    );
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('HTTP 503');
    expect(alert).toHaveTextContent('docker compose up');
  });
});
