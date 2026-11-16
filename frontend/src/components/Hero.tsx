import type { Report } from '../types';
import { heroAnswer, normalizeVerdict, verdictStyle } from '../lib/verdict';
import { fmtDate, horizonLabel } from '../lib/format';
import { VerdictGlyph } from './ui';
import { SECTIONS } from './sections';

/** F1: the thesis and the answer, one line each, then where to go next. */
export function Hero({ report }: { report: Report }) {
  const { meta, hypotheses, features } = report;
  const answer = heroAnswer(hypotheses);
  const featureById = new Map(features.map((f) => [f.id, f]));
  return (
    <header className="hero" id="top">
      <h1 className="hero-title" aria-label={meta.title}>
        <span>Fast</span>
        <span className="and" aria-hidden="true">
          and
        </span>
        <span className="slow">Slow</span>
      </h1>
      <div className="hero-grid">
        <div>
          <dl className="qa">
            <div>
              <dt>The question</dt>
              <dd>{meta.subtitle}</dd>
            </div>
            <div>
              <dt>The answer{meta.profile === 'smoke' ? ', from smoke-test data' : ''}</dt>
              <dd>
                <span className="answer-lead">{answer.lead}</span>{' '}
                <span className="answer-detail">{answer.detail}</span>
              </dd>
            </div>
            <div>
              <dt>Five hypotheses, written before the runs</dt>
              <dd className="verdict-row">
                {hypotheses.map((h) => {
                  const s = verdictStyle(h.verdict);
                  return (
                    <a
                      key={h.id}
                      className="verdict-chip"
                      href={`#hyp-${h.id}`}
                      aria-label={`${h.id}: ${s.label}`}
                    >
                      <VerdictGlyph tone={s.tone} size={12} />
                      <b>{h.id}</b> {s.label.toLowerCase()}
                    </a>
                  );
                })}
              </dd>
            </div>
          </dl>
        </div>
        <nav className="toc" aria-label="Report sections">
          <h2>In this report, and who it's for</h2>
          <ol>
            {SECTIONS.filter((s) => s.fid !== 'F1').map((s) => (
              <li key={s.id}>
                <a href={`#${s.id}`}>
                  <span className="toc-id">{s.fid}</span>
                  <span className="toc-name">{s.title}</span>
                  <span className="toc-user">{featureById.get(s.fid)?.user ?? ''}</span>
                </a>
              </li>
            ))}
          </ol>
        </nav>
      </div>
      <p className="meta-line">
        <span>
          <b>Horizons</b> {meta.horizons.map(horizonLabel).join(', ')}
        </span>
        <span>
          <b>Seeds</b> {meta.seeds.length}
        </span>
        <span>
          <b>Held-out scenarios</b> {meta.scenarios_per_horizon} per horizon
        </span>
        <span>
          <b>Exported</b> {fmtDate(meta.generated_at)}
        </span>
        <span>
          <b>Verdicts</b>{' '}
          {hypotheses.filter((h) => normalizeVerdict(h.verdict) === 'supported').length} of{' '}
          {hypotheses.length} supported
        </span>
      </p>
    </header>
  );
}
