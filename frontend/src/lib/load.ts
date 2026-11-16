import type { Report } from '../types';

export const REPORT_URLS = ['data/report.json', 'data/report.sample.json'] as const;

export interface Loaded {
  report: Report;
  source: string;
}

function looksLikeReport(x: unknown): x is Report {
  if (!x || typeof x !== 'object') return false;
  const r = x as Partial<Report>;
  return !!r.meta && !!r.results && Array.isArray(r.hypotheses);
}

/**
 * Fetch the real report, falling back to the sample. Relative URLs, so the build works
 * under any sub-path (the lab serves it at /p/fast-and-slow/).
 */
export async function loadReport(
  fetcher: typeof fetch = fetch,
  urls: readonly string[] = REPORT_URLS,
): Promise<Loaded> {
  const problems: string[] = [];
  for (const url of urls) {
    try {
      const res = await fetcher(url, { cache: 'no-cache' });
      if (!res.ok) {
        problems.push(`${url}: HTTP ${res.status}`);
        continue;
      }
      const data: unknown = await res.json();
      if (looksLikeReport(data)) return { report: data, source: url };
      problems.push(`${url}: not a report file`);
    } catch (e) {
      problems.push(`${url}: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
  throw new Error(`Couldn't load the report data. Tried ${problems.join('; ')}.`);
}
