import type { Report, Run, RunSummary } from '../types';

/** Relative, so the build works under any sub-path (the lab serves it at /p/fast-and-slow/). */
export const API = 'api';

export interface Loaded {
  run: Run;
  runs: RunSummary[];
}

/** No run has been published yet: an empty state with directions, not an error. */
export class NoRunsYet extends Error {
  constructor() {
    super('No runs have been published yet.');
    this.name = 'NoRunsYet';
  }
}

function looksLikeReport(x: unknown): x is Report {
  if (!x || typeof x !== 'object') return false;
  const r = x as Partial<Report>;
  return !!r.meta && !!r.results && Array.isArray(r.hypotheses);
}

async function getJson(fetcher: typeof fetch, url: string): Promise<unknown> {
  let res: Response;
  try {
    res = await fetcher(url, { cache: 'no-cache' });
  } catch (e) {
    throw new Error(
      `Couldn't reach the runs API (${url}): ${e instanceof Error ? e.message : String(e)}.`,
      { cause: e },
    );
  }
  if (res.status === 404) return undefined;
  if (!res.ok) throw new Error(`The runs API answered ${url} with HTTP ${res.status}.`);
  try {
    return await res.json();
  } catch (e) {
    throw new Error(`${url} didn't return JSON. Is the API behind this address?`, { cause: e });
  }
}

/** The run to show (the newest when runId is null) and every run's summary, for the picker. */
export async function loadRun(
  runId: string | null,
  fetcher: typeof fetch = fetch,
): Promise<Loaded> {
  const path = runId ? `${API}/runs/${encodeURIComponent(runId)}` : `${API}/runs/latest`;
  const [run, runs] = await Promise.all([getJson(fetcher, path), getJson(fetcher, `${API}/runs`)]);
  if (run === undefined) {
    if (runId) throw new Error(`There is no run "${runId}". Pick another from the list.`);
    throw new NoRunsYet();
  }
  const r = run as Partial<Run>;
  if (!looksLikeReport(r.report)) throw new Error(`${path} didn't return a report.`);
  return { run: r as Run, runs: Array.isArray(runs) ? (runs as RunSummary[]) : [] };
}

/** ?run=<id> in the address bar, so a link opens the same run. */
export function runFromUrl(search: string = window.location.search): string | null {
  return new URLSearchParams(search).get('run');
}
