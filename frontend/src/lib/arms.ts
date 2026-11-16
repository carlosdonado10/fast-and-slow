import type { ArmGroup, Report } from '../types';

export const SYSTEM_ONE = ['kev_zero', 'kev_icl', 'kev_rl'] as const;
export const CLASSIC = ['ppo_num', 'ppo_emb', 'ppo_sig'] as const;
export const REFERENCES = ['bayes', 'oracle', 'basestock', 'random'] as const;
export const LEARNERS: readonly string[] = [...SYSTEM_ONE, ...CLASSIC];

/** Fixed order everywhere, so an arm keeps its position and its colour across charts. */
export const ARM_ORDER: readonly string[] = [...SYSTEM_ONE, ...CLASSIC, ...REFERENCES];

export type Family = ArmGroup;

export function familyOf(arm: string, report?: Pick<Report, 'arms'>): Family {
  const g = report?.arms[arm]?.group;
  if (g === 'system-one' || g === 'classic-rl' || g === 'reference') return g;
  if (arm.startsWith('kev')) return 'system-one';
  if (arm.startsWith('ppo')) return 'classic-rl';
  return 'reference';
}

export function isLearner(arm: string): boolean {
  return arm.startsWith('kev') || arm.startsWith('ppo');
}

export function sortArms<T extends string>(arms: Iterable<T>): T[] {
  const rank = (a: string) => {
    const i = ARM_ORDER.indexOf(a);
    return i === -1 ? ARM_ORDER.length : i;
  };
  return [...arms].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
}

export function armLabel(report: Pick<Report, 'arms'>, arm: string): string {
  return report.arms[arm]?.label ?? arm;
}

/** CSS colour for an arm; the tokens live in index.css and switch with the colour scheme. */
export function armColor(arm: string): string {
  return ARM_ORDER.includes(arm) ? `var(--arm-${arm})` : 'var(--ink-3)';
}

/** Reference arms are quiet grey dashes; learners are solid. */
export function armDash(arm: string): string | undefined {
  switch (arm) {
    case 'bayes':
      return '6 4';
    case 'oracle':
      return '2 3';
    case 'basestock':
      return '10 3 2 3';
    case 'random':
      return '1 4';
    default:
      return undefined;
  }
}

/** Secondary encoding of family, so identity never rests on colour alone. */
export function armMarker(arm: string): 'circle' | 'square' | 'none' {
  const f = familyOf(arm);
  return f === 'system-one' ? 'circle' : f === 'classic-rl' ? 'square' : 'none';
}

export const FAMILY_LABEL: Record<Family, string> = {
  'system-one': 'System One (Kev)',
  'classic-rl': 'Classic RL (PPO)',
  reference: 'Reference points',
};
