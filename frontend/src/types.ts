/** The data contract: report.json as written by backend/src/fast_and_slow/export/report.py.
 * inf/nan are exported as null, so every derived number may be null. */

export type Num = number | null;
export type ArmGroup = 'reference' | 'system-one' | 'classic-rl';
export type Verdict = 'supported' | 'not supported' | 'inconclusive' | 'pending';

export interface World {
  regimes: string[];
  demand_means: number[];
  regime_prior?: number[];
  stay_prob: number;
  max_demand?: number;
  order_menu: number[];
  inventory_cap?: number;
  start_inventory?: number;
  price: number;
  unit_cost: number;
  holding_cost: number;
  stockout_penalty: number;
  fixed_order_cost: number;
  salvage_value: number;
  note_prob?: number;
  distractor_prob?: number;
  sources: string[];
  source_accuracy: number[];
  source_mix?: number[];
}

export interface Meta {
  title: string;
  subtitle: string;
  profile: string;
  generated_at: string;
  horizons: number[];
  seeds: number[];
  kev_run: string;
  note_writer: string;
  scenarios_per_horizon: number;
  error_budget: number;
  thresholds: number[];
  world: World;
}

export interface ArmInfo {
  label: string;
  group: ArmGroup;
  sees: string;
  role: string;
}

export interface MeanCI {
  mean: Num;
  ci: [Num, Num] | Num[];
}

export interface ReliabilityBin {
  lo: number;
  hi: number;
  confidence: Num;
  accuracy: Num;
  count: number;
}

export interface HandoffSplit {
  n: number;
  accuracy: Num;
  automation: Record<string, Num>;
}

export interface ArmResult {
  seeds: number;
  return_mean: Num;
  pct_optimal: MeanCI;
  inbox_value: MeanCI;
  pct_of_perfect_info: Num;
  acceptable_rate: Num;
  mean_regret: Num;
  calibration: { ece: Num; brier: Num; bins: ReliabilityBin[] };
  regime: { brier_vs_truth: Num; brier_vs_bayes: Num; bayes_brier_vs_truth: Num } | null;
  handoff: { all: HandoffSplit; decidable: HandoffSplit; judgement: HandoffSplit };
  latency_ms: { p50: Num; p95: Num };
}

export interface CurvePoint {
  decisions: number;
  score: Num;
  seconds?: Num;
}

export interface BandPoint {
  decisions: number;
  lo: Num;
  hi: Num;
}

export interface Curve {
  mean: CurvePoint[];
  seeds: CurvePoint[][];
  band: BandPoint[];
}

export interface EfficiencySummary {
  start: Num;
  final: Num;
  to_threshold: Record<string, Num>;
  gain_per_decade: Num;
  aulc: Num;
}

export interface Efficiency extends EfficiencySummary {
  per_seed?: EfficiencySummary[];
}

export type EfficiencyByArm = Record<string, Efficiency> & { _crossover?: Record<string, Num> };

export interface DialPoint {
  horizon: number;
  pct_optimal: Num;
  ci: Num[];
  inbox_value: Num;
}

export interface CostRow {
  hardware: string;
  train_hours: Num;
  train_usd: Num;
  ms_per_decision_p50: Num;
  ms_per_decision_p95: Num;
  usd_per_1k_decisions: Num;
}

export interface ReplayDecision {
  order: number;
  confidence: Num;
  acceptable: boolean;
  inventory: number;
  decidable: boolean;
  bayes_order: number;
  posterior: Num[];
  reward: Num;
}

export interface ReplayWeek {
  week: number;
  note: string | null;
  note_source: string | null;
  note_claim: string | null;
  true_regime: string;
  demand: number;
  decisions: Record<string, ReplayDecision>;
}

export interface Replay {
  horizon: number;
  seed: number;
  weeks: ReplayWeek[];
  totals: Record<string, Num>;
}

export interface DeskQuestion {
  type: string;
  instructions: string;
  criteria: Record<string, string>;
}

export interface DeskWeek {
  week: number;
  state: string;
  questions: Record<string, DeskQuestion>;
  posterior: Record<string, Num>;
  true_regime: string;
  bayes_order: number;
  demand: number;
  profit: Num;
}

export interface Hypothesis {
  id: string;
  statement: string;
  rule: string;
  verdict: string;
  evidence: string;
}

export interface Feature {
  id: string;
  name: string;
  user: string;
  need: string;
  why: string;
}

export interface DesignDecision {
  decision: string;
  why: string;
  rejected: string;
}

export type Strategy = 'stale' | 'retrain' | 'kev' | 'bridge';

export interface ShiftTimeline {
  trigger: 'declared' | 'detected';
  mode: 'sim' | 'live';
  variant: string;
  trigger_week: number | null;
  weekly: Record<Strategy, Num[]>;
  regret_usd: Record<Strategy, Num>;
  recover_week: Record<Strategy, number | null>;
  handback_week: number | null;
  bridge_saving: Num;
}

export interface DetectorSummary {
  threshold: Num;
  false_alarms_per_100_weeks: Num;
  detection_rate: Num;
  median_delay_weeks: Num;
  delays: (number | null)[];
}

export interface ShiftResult {
  label: string;
  kind: 'announced' | 'silent';
  declared_note: string;
  announcement: string | null;
  weeks: number;
  shift_week: number;
  horizon: number;
  rebuild_weeks: number;
  live_units: number;
  recover_at: number;
  references: Record<string, Num>;
  kev: { declared: Num; stale_brief: Num };
  ppo: Record<string, { pre_shift_score: Num; curve: { decisions: number; score: Num }[] }>;
  detection: Record<
    string,
    { matched: DetectorSummary; mean_score_by_week: Num[]; 'at_0.5'?: DetectorSummary }
  >;
  timelines: Record<string, ShiftTimeline>;
}

export interface Report {
  meta: Meta;
  arms: Record<string, ArmInfo>;
  references: Record<string, Record<string, Num>>;
  results: Record<string, Record<string, ArmResult>>;
  curves: Record<string, Record<string, Curve>>;
  efficiency: Record<string, EfficiencyByArm>;
  horizon_dial: Record<string, DialPoint[]>;
  handoff_budgets: number[];
  costs: Record<string, CostRow>;
  replays: Replay[];
  desk: { horizon: number; weeks: DeskWeek[] };
  hypotheses: Hypothesis[];
  features: Feature[];
  decisions: DesignDecision[];
  limitations: string[];
  shifts?: Record<string, ShiftResult>;
}

/** One published run, as listed by GET /api/runs (backend/src/fast_and_slow/store.py). */
export interface RunSummary {
  id: string;
  label: string;
  created_at: string;
  profile: string;
  git_commit: string;
  gpu: string;
  verdicts: Record<string, string>;
}

/** A run with its report: GET /api/runs/latest or /api/runs/{id}. */
export interface Run extends RunSummary {
  report: Report;
}
