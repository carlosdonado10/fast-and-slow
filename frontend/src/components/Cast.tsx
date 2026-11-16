import type { Report } from '../types';
import { FAMILY_LABEL, familyOf, sortArms, type Family } from '../lib/arms';
import { ArmName } from './ui';

const FAMILIES: Family[] = ['system-one', 'classic-rl', 'reference'];

/** The arms, grouped by family: the legend every later chart relies on. */
export function Cast({ report }: { report: Report }) {
  const arms = sortArms(Object.keys(report.arms));
  return (
    <div className="cast" role="list" aria-label="The arms compared in this study">
      {FAMILIES.map((f) => {
        const members = arms.filter((a) => familyOf(a, report) === f);
        if (!members.length) return null;
        return (
          <div className="cast-col" key={f} role="listitem">
            <h3>{FAMILY_LABEL[f]}</h3>
            <ul>
              {members.map((a) => {
                const info = report.arms[a]!;
                return (
                  <li key={a}>
                    <ArmName arm={a} label={info.label} />
                    <p>
                      {info.role}. Sees: {info.sees.toLowerCase()}.
                    </p>
                  </li>
                );
              })}
            </ul>
          </div>
        );
      })}
    </div>
  );
}
