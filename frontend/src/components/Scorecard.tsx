import type { Report } from '../types';
import { verdictStyle } from '../lib/verdict';
import { Section, VerdictGlyph } from './ui';

/** F2: each hypothesis with its verdict, the evidence number and the rule written before the runs. */
export function Scorecard({ report }: { report: Report }) {
  const feature = report.features.find((f) => f.id === 'F2');
  return (
    <Section
      id="scorecard"
      fid="F2"
      feature={feature}
      title="Were the verdicts decided before anyone looked?"
      intro={
        <p>
          For practitioners and hiring managers. Each rule was fixed in the spec before any run; the
          export step applies it to the numbers, and nothing here was judged by hand.
        </p>
      }
    >
      <ol className="hyps">
        {report.hypotheses.map((h) => {
          const s = verdictStyle(h.verdict);
          return (
            <li className="hyp" key={h.id} id={`hyp-${h.id}`}>
              <span className="hyp-id">{h.id}</span>
              <div>
                <p className={`verdict ${s.tone}`}>
                  <VerdictGlyph tone={s.tone} />
                  <span className="verdict-text">{s.label}</span>
                </p>
                <h3>{h.statement}</h3>
                <p className="hyp-evidence">{h.evidence}</p>
              </div>
              <p className="hyp-rule">
                <b>The rule</b>
                {h.rule}
              </p>
            </li>
          );
        })}
      </ol>
    </Section>
  );
}
