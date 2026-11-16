import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Report } from './App';
import type { Report as ReportData } from './types';

const sample = JSON.parse(
  readFileSync(resolve(__dirname, '../public/data/report.sample.json'), 'utf8'),
) as ReportData;

describe('regime changes (F11)', () => {
  it('switches shift, trigger and retraining mode and updates the story', async () => {
    const user = userEvent.setup();
    render(<Report report={structuredClone(sample)} source="sample" />);
    const sec = document.getElementById('transition')!;
    expect(within(sec).getByText(/An email lands in the inbox/)).toBeInTheDocument();
    expect(
      within(sec).getByRole('img', { name: /Weekly score for each strategy/ }),
    ).toBeInTheDocument();
    expect(within(sec).getByText('Would anyone notice?')).toBeInTheDocument();

    await user.click(within(sec).getByRole('radio', { name: /competitor/i }));
    expect(within(sec).getByText(/Only the sales numbers change/)).toBeInTheDocument();
    await user.click(within(sec).getByRole('radio', { name: 'Learn from live weeks' }));
    expect(within(sec).getByText(/pooled over \d+ products/)).toBeInTheDocument();
    await user.click(within(sec).getByRole('radio', { name: /alarm flags it/ }));
    expect(within(sec).getByText(/given up with the bridge/)).toBeInTheDocument();
    expect(within(sec).getAllByText('Bridge: Kev until retrained RL wins').length).toBeGreaterThan(
      0,
    );
  });

  it('says so when there are no regime-change results', () => {
    const r = structuredClone(sample);
    delete r.shifts;
    render(<Report report={r} source="sample" />);
    expect(
      within(document.getElementById('transition')!).getByText('No regime-change results yet.'),
    ).toBeInTheDocument();
  });

  it('handles an unnoticed change and a bridge that never hands back', () => {
    const r = structuredClone(sample);
    const shift = Object.values(r.shifts!)[0]!;
    for (const tl of Object.values(shift.timelines)) {
      tl.trigger_week = null;
      tl.handback_week = null;
      tl.bridge_saving = null;
    }
    shift.detection = {};
    render(<Report report={r} source="sample" />);
    const sec = document.getElementById('transition')!;
    expect(within(sec).getByText(/Nobody notices the change/)).toBeInTheDocument();
    expect(within(sec).getByText(/never beats the bridge/)).toBeInTheDocument();
    expect(screen.getAllByText('never').length).toBeGreaterThan(0);
  });
});
