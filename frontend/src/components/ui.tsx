import { useId, type ReactNode } from 'react';
import { armColor, armDash, armMarker } from '../lib/arms';
import type { Feature } from '../types';
import type { VerdictStyle } from '../lib/verdict';

/* ---------------------------------------------------------------- Section */

interface SectionProps {
  id: string;
  feature: Feature | undefined;
  fid: string;
  title: string;
  intro: ReactNode;
  children: ReactNode;
}

/** Every section shows who it is for and the question it answers: the feature map, made visible. */
export function Section({ id, feature, fid, title, intro, children }: SectionProps) {
  const headingId = `${id}-title`;
  return (
    <section className="sec" id={id} aria-labelledby={headingId}>
      <aside className="sec-margin" aria-label={`${fid}: who this section is for`}>
        <span className="fid">{fid}</span>
        {feature && <span className="fname">{feature.name}</span>}
        {feature && (
          <dl>
            <div>
              <dt>For</dt>
              <dd>{feature.user}</dd>
            </div>
          </dl>
        )}
      </aside>
      <div className="sec-body">
        <h2 className="sec-title" id={headingId}>
          {title}
        </h2>
        <div className="sec-intro">{intro}</div>
        <div className="sec-content">{children}</div>
        {feature?.why && (
          <p className="sec-why">
            <b>Why this view: </b>
            {feature.why}
          </p>
        )}
      </div>
    </section>
  );
}

/* ---------------------------------------------------------------- Segmented */

interface SegmentedProps<T extends string | number> {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  hideLabel?: boolean;
}

/** A radio group drawn as a segmented control; arrow keys move the selection. */
export function Segmented<T extends string | number>({
  label,
  value,
  options,
  onChange,
  hideLabel,
}: SegmentedProps<T>) {
  const id = useId();
  const idx = Math.max(
    0,
    options.findIndex((o) => o.value === value),
  );
  const move = (d: number, el: HTMLElement) => {
    const next = options[(idx + d + options.length) % options.length];
    if (!next) return;
    onChange(next.value);
    const btns = el.parentElement?.querySelectorAll('button');
    btns?.[(idx + d + options.length) % options.length]?.focus();
  };
  return (
    <div className="seg-wrap">
      <span id={id} className={hideLabel ? 'sr-only' : 'control-label'}>
        {label}
      </span>
      <div className="seg" role="radiogroup" aria-labelledby={id}>
        {options.map((o) => (
          <button
            key={String(o.value)}
            type="button"
            role="radio"
            aria-checked={o.value === value}
            tabIndex={o.value === value ? 0 : -1}
            onClick={() => onChange(o.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                e.preventDefault();
                move(1, e.currentTarget);
              } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                e.preventDefault();
                move(-1, e.currentTarget);
              }
            }}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- Swatch */

/** Colour plus a family shape (circle = System One, square = classic RL, dash = reference). */
export function Swatch({ arm, size = 10 }: { arm: string; size?: number }) {
  const m = armMarker(arm);
  const c = armColor(arm);
  if (m === 'none') {
    return (
      <svg className="swatch" width={size + 8} height={size} aria-hidden="true">
        <line
          x1={0}
          x2={size + 8}
          y1={size / 2}
          y2={size / 2}
          stroke={c}
          strokeWidth={2}
          strokeDasharray={armDash(arm)}
        />
      </svg>
    );
  }
  return (
    <svg className="swatch" width={size} height={size} aria-hidden="true">
      {m === 'circle' ? (
        <circle cx={size / 2} cy={size / 2} r={size / 2} fill={c} />
      ) : (
        <rect width={size} height={size} rx={1.5} fill={c} />
      )}
    </svg>
  );
}

export function ArmName({ arm, label }: { arm: string; label: string }) {
  return (
    <span className="arm-name">
      <Swatch arm={arm} />
      {label}
    </span>
  );
}

/* ---------------------------------------------------------------- Verdict glyph */

export function VerdictGlyph({ tone, size = 14 }: { tone: VerdictStyle['tone']; size?: number }) {
  const r = size / 2 - 1;
  const c = size / 2;
  return (
    <svg
      className="glyph"
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      aria-hidden="true"
    >
      {tone === 'held' && <circle cx={c} cy={c} r={r} fill="currentColor" />}
      {tone === 'failed' && (
        <>
          <circle cx={c} cy={c} r={r} fill="none" stroke="currentColor" strokeWidth={1.5} />
          <line
            x1={c - r * 0.7}
            y1={c + r * 0.7}
            x2={c + r * 0.7}
            y2={c - r * 0.7}
            stroke="currentColor"
            strokeWidth={1.5}
          />
        </>
      )}
      {tone === 'mixed' && (
        <>
          <circle cx={c} cy={c} r={r} fill="none" stroke="currentColor" strokeWidth={1.5} />
          <path d={`M ${c} ${c - r} A ${r} ${r} 0 0 1 ${c} ${c + r} Z`} fill="currentColor" />
        </>
      )}
      {tone === 'open' && (
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke="currentColor"
          strokeWidth={1.5}
          strokeDasharray="2 2"
        />
      )}
    </svg>
  );
}

/* ---------------------------------------------------------------- Data table fallback */

export function DataDetails({ summary, children }: { summary: string; children: ReactNode }) {
  return (
    <details className="data-details">
      <summary>{summary}</summary>
      <div className="table-scroll">{children}</div>
    </details>
  );
}
