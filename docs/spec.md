# Fast and Slow: spec

> Can a "System One" decision model (JEV-style) take over the planning decisions that are actually decidable, and how does it learn compared with classic reinforcement learning?

Status: v1 spec, 2026-11-15. Owner: Carlos Donado.

---

## 1. The use case

### Who it's for
**Primary: demand and supply planners** (S&OP, inventory and replenishment planners) and the people who buy tools for them.
**Secondary: ML practitioners in RL-heavy fields** who want evidence on whether decision models disrupt classic RL. The portfolio audience, hiring managers, reads it through the planner's eyes.

### The problem
A planner's week mixes two kinds of decisions:
1. **Decidable calls.** Given the numbers and the signals, one action dominates: it would still be right whatever the hidden truth turned out to be. These can be computed, so a model should own them.
2. **Judgement calls.** The evidence is genuinely ambiguous, and the right call depends on things nobody can compute: a relationship, a risk appetite, a hunch about a customer.

Today's tools handle neither well. Rule- and optimisation-based planning software ignores qualitative signals (a sales rep's note, a supplier rumour), so planners re-check everything by hand. Classic RL can learn policies, but it needs a simulator, huge amounts of experience and numeric inputs, and it can't explain its confidence.

**The promise of JEV-style models:** they read the messy text, return a *calibrated* probability over typed options, and run in milliseconds. If the calibration is honest, the model can **take the decidable calls and route the judgement calls to a person**. That is the product thesis this project tests.

### What this project is
A **research study with an interactive report.** It isn't a product to deploy into a planning stack; it produces evidence and makes it explorable. The planner-facing question is "what could I safely hand off?" The practitioner-facing question is "how do these models learn compared with RL?"

### Hypotheses (from Carlos; the study is designed to falsify them)
| ID | Hypothesis | How it's measured |
|---|---|---|
| H1 | A JEV-style model is **strong from the start** (zero-shot) where RL starts near random. | Score at 0 training decisions, Kev vs PPO. |
| H2 | A JEV-style model **improves more slowly** with training, and RL eventually catches up or overtakes. | **Sample efficiency**: decisions needed to reach 80/90/95% of optimal; gain per 10× more experience; the crossover point in decisions consumed. |
| H3 | A JEV-style model is **good at one-shot decisions, weaker at long-horizon (multi-step) ones**. | Kev's gap to optimal as the horizon grows from 1 to 4 to 12. |
| H4 | Calibrated confidence lets a JEV-style model **hand off safely**: it is confident on decidable calls and unsure on judgement calls. | Automation rate at a 5% error budget; split by decidable vs judgement. |
| H5 | **Context without fine-tuning** (rules and worked examples in the prompt) recovers part of the fine-tuning gain for free. | In-context arm vs zero-shot and RL-fine-tuned. |
| H6 | **A System One model bridges regime changes.** When the business changes, a deployed RL policy is quietly wrong until it's retrained; a context-reading model can take over during the transition and hand back once retrained RL overtakes it. | Transition regret: bridge vs stale RL vs retrain-without-bridge, for announced and silent shifts, with retraining paid by a rebuilt simulator or by live experience; plus detection delay and false alarms, Kev vs a statistical detector. See section 2b. |

---

## 2. The environment: a planner's replenishment desk

One product, H weekly periods (**the horizon dial: H ∈ {1, 4, 12}**). Each week the planner reads the week's notes, sees inventory and recent sales, and orders stock.

- **Hidden truth:** a demand regime `z ∈ {low, normal, surge}` with mean weekly demand {3, 10, 24}. It shifts often (stay probability 0.5), so last week's sales are a weak guide, and it is never shown.
- **Demand:** Poisson around the regime mean, capped at 40. Unmet demand is lost.
- **Signals:** most weeks bring a note. Each note comes from a source with known reliability (rumour 0.65, sales rep 0.80, confirmed customer order 0.95) and claims "down / flat / up" (that is, low / normal / surge). Some notes are irrelevant distractors. An open LLM writes the note text from those hidden attributes, so the words vary but the information content is exactly known.
- **Decision:** order quantity, one of {0, 5, 10, 15, 20, 25, 30, 40}. Arrives before the week's demand. Stock starts at 0.
- **Economics:** price 14, unit cost 5, holding 4 per unit carried, a stockout penalty of 10 per unit of lost sale, a **fixed order cost of 10** (which rewards batching and so creates real multi-step tradeoffs), and no salvage value at the end.

**Why this design (decisions, with reasoning):**
- **One environment with a horizon dial**, not separate tasks for one-shot and multi-step, so H3 is a controlled comparison. Only the horizon changes.
- **Synthetic but ground-truthed.** Because we know every note's true reliability, we can compute the **Bayes-optimal planner** exactly. "% of optimal" then has a real denominator, and calibration is measured against truth, not against a labeller.
- **Small enough to solve exactly:** a belief grid times inventory times actions, solved by dynamic programming in numpy. Real planning is bigger. Being solvable is what makes the measurements trustworthy, and that's the trade we're making.
- **Fixed order cost and inventory carry-over** make H > 1 a genuine planning problem (when to batch, how much to carry, end-of-horizon run-down), not H independent one-shot calls.
- **Tuned so that reading the inbox matters.** The first world (persistent regimes, mild costs) let the numbers-only base-stock tool reach 99% of the Bayes optimum at H=1: there was nothing left for a reader to win, and the study would have been moot. The tuned world (regimes that shift, sharper costs, more reliable sources) leaves base-stock at 76–90% of optimal and makes only 23–50% of its orders acceptable, while perfect information is worth 1.6–3.7× the Bayes return. About half of all decisions are decidable.
- **Notes written by an open model from a different family** (Phi-4-mini) than the model under test (Qwen-based Kev), so the text style doesn't favour the tested model. Paraphrases are split into train and test, so evaluation uses wording the agent never trained on.

### Decidable vs judgement, made precise
For each decision we compute the Bayes posterior over regimes. A decision is **decidable** if the Bayes-optimal action is also the full-information optimal action under *every* regime with posterior ≥ 0.15. Knowing the truth wouldn't change the call, so a dominant strategy exists. Otherwise it is a **judgement call**. This label never reaches the agents; it only drives the analysis for H4.

---

## 2b. Regime changes: who decides while the model catches up? (H6)

**The planner's problem.** A deployed RL policy encodes the world it was trained on. When the business changes (a new contract, a new competitor), the policy keeps making confident decisions for a world that no longer exists. Retraining fixes that, but it isn't free: someone has to notice, update the simulator or gather new experience, and retrain. Experimentation on a live business is expensive. **The question is who makes the calls in between.**

**The proposal being tested.** Use the System One model as the *bridge*:
1. **Detect:** each week it answers a typed question, "has something changed that the usual playbook doesn't cover?"
2. **Cover:** once a change is declared (by a planner, or by that alarm), it makes the calls with the new situation written into its brief.
3. **Hand back:** when retrained RL scores better on the new world than the bridge does, RL takes over again.

**Shifts tested** (both happen at week 20 of a 52-week deployment of monthly, H=4 plans):
| Shift | What changes | How it shows | Why it's here |
|---|---|---|---|
| **Announced economics change** | A key account adds late-delivery penalties: stockout penalty 10 → 30 | An email in the inbox on the week it starts | Readable as text. A numbers-only detector can't see it at all |
| **Silent demand shift** | A competitor enters: every regime's demand drops 40% | Only in the sales numbers | Tests detection from data, where a statistical detector should be strong |

**Triggers compared:** a planner declares the change on the week it happens (the best case for the bridge), or the Kev alarm fires (realistic: delay and false alarms count against it).

**Retraining, two ways** (one warm-started PPO retraining curve on the new world, mapped onto the calendar two ways):
- **Rebuilt simulator:** the company updates its simulator, which takes `rebuild_weeks` (default 6); offline retraining is then fast. RL is back at full strength once the rebuild is done.
- **Live experience only:** no simulator update; RL learns only from the weeks it actually plays, pooled across `live_units` products or sites (default 50). By week *w* after the shift it has seen 50 × 4 × *w* decisions of the new world.

**Arms over the 52 weeks:** stale PPO (never retrained), PPO retrained (sim / live), Kev with the updated brief (from the trigger on), and the **bridge** (stale PPO until the trigger, Kev until retrained PPO beats it, then PPO). The perfect-information and Bayes planners of each world are the references.

**Metrics:**
- % of the new world's optimum, week by week (the transition timeline);
- **transition regret**: profit lost against Bayes, summed from the shift to week 52;
- **weeks to recover** (rolling score back within 90% of optimal);
- **regret saved by bridging**;
- for detection: **delay** (weeks from the shift to the first alarm) and **false-alarm rate** (alarms per 100 pre-shift weeks), Kev vs a Page–Hinkley detector on demand.

**H6 verdict rule (pre-registered):** supported if, for both shifts and both retraining modes, the planner-declared bridge cuts transition regret by ≥ 25% against the best non-bridge arm; not supported if it cuts it by less than 5% anywhere; otherwise inconclusive. Detection is reported alongside as the cost of not having a planner in the loop.

**Decisions:**
- *Shifts at the episode boundary, plans of H=4.* A plan starts in one world, which keeps the Bayes reference exact. Rejected: a mid-plan shift (no clean optimum).
- *One retraining curve, two calendar mappings.* The same PPO, warm-started on the new world; the two modes differ only in how fast experience arrives. Rejected: two separate trainings, which would mix the effect of the mode with training noise.
- *Pooling live experience across 50 units.* A real planning team runs one policy over many SKUs or sites. With a single unit, live retraining would never recover within a year, which is true but uninformative.
- *Cheap on the budget:* no Kev training. Kev only evaluates on the new world and runs the weekly alarm (minutes of GPU); PPO retraining runs on the laptop.

**Report feature F11, the transition timeline.** For the planner deciding what to do the week something changes: a 52-week strip per arm with the shift marked, the bridge's hand-back point, the shaded regret area, a toggle between shifts and retraining modes, and a detection panel (delay vs false alarms, Kev vs statistics). *Why this view:* the question is about time ("how long am I exposed, and what does it cost?"), so the x axis is calendar weeks, not decisions.

## 3. The arms (what learns, and how)

| Arm | What it is | What it sees | Why it's here |
|---|---|---|---|
| `bayes` | Exact Bayes-optimal DP planner | Note attributes (perfect parsing) + numbers | The 100% reference |
| `oracle` | Perfect-information planner | The hidden regime | Upper bound: the value of knowing the truth |
| `basestock` | Order-up-to policy tuned on numbers | Numbers only | Today's planning tool, the status quo |
| `random` | Uniform random orders | – | The 0% reference |
| `kev_zero` | Kev-4B out of the box | Text state (brief, numbers, notes) | H1, H3 |
| `kev_icl` | Kev-4B + rules and worked examples in the state | Text state + context | H5: context, no training |
| `kev_rl` | Kev-4B, LoRA + PPO on environment reward | Text state | H2: how a decision model learns |
| `ppo_num` | Classic PPO (MLP) | Numbers only | Classic RL as deployed today |
| `ppo_emb` | Classic PPO | Numbers + text embeddings of the notes (bge-small) | A fair fight: RL that can "read" |
| `ppo_sig` | Classic PPO | Numbers + parsed note attributes | Ablation: RL with perfect reading. Separates "can read" from "can learn" |

**Model choice: Kev-4B** (Apache-2.0, a JEV-compatible API, Qwen3.5-4B base + LoRA + pointer head, a calibrated temperature). The README recommends it as the default size; at about 18 ms per request on an H100 it fits LoRA RL within the budget.

**How Kev is fine-tuned with RL:** Kev doesn't generate text; a pointer head scores each option. Its softmax over the 7 order options *is* a policy, so we run **PPO directly on the pointer-head logits**, training the LoRA adapter and head, starting from the released weights (`init_from`, which keeps what Kev knows). Rollouts are batched and gradient-free; updates recompute the log-probs with gradients and use a clipped objective. Advantages are group-relative: G rollouts share the same scenario seed, so the exogenous randomness is identical.

---

## 4. Metrics

| Metric | Definition | Serves |
|---|---|---|
| **% of optimal** (headline) | `100 × (R − R_random) / (R_bayes − R_random)` on a fixed held-out set of 300 scenarios per horizon, with bootstrap 95% CI | H1–H3, H5 |
| **Inbox value captured** (planner scale) | `100 × (R − R_basestock) / (R_bayes − R_basestock)`: 0 = the numbers-only tool, 100 = reading every note perfectly. It answers "did reading the inbox beat my current tool?", which the random-anchored scale hides because any sane policy scores high on it | Planner framing, F6 |
| % of perfect information | Same, with `oracle` in the denominator | Context |
| Learning curve | % of optimal vs **decisions consumed** (log x), plus GPU-hours | H1, H2 |
| **Sample efficiency** | (a) **decisions-to-threshold**: training decisions until the eval score first reaches 80 / 90 / 95% of optimal (base-stock already sits at 76–90%, so lower thresholds say nothing), interpolated on log-decisions (∞ if never reached); (b) **learning rate**: score gained per 10× more decisions, the slope of the curve in log space; (c) **area under the learning curve** over log-decisions, normalised to [0, 1], one number for "good early *and* improving"; (d) **crossover**: the decision count where the PPO curve passes the Kev curve (if it does). Every value comes with a seed spread | H1, H2: the only direct test of "starts strong, trains slower" |
| **Calibration** | ECE and Brier for (a) "is my chosen action acceptable?" (within 2% of Bayes value) and (b) the regime question vs the Bayes posterior and the truth | H4 |
| **Automation @ 5% error** | Largest share of decisions the agent can take, ordered by confidence, with ≤ 5% unacceptable. Split by decidable vs judgement | H4 (the planner's question) |
| **Cost & latency** | ms per decision (p50/p95), GPU-hours to train, $ at a stated H100 rate, $ per 1k decisions served | Practitioner tradeoff |

Every hypothesis gets a verdict that the export step computes from these numbers: **supported, not supported or inconclusive**, with the rule written next to it in the report.

---

## 5. Budget (~100 Colab compute units)

The first plan assumed about 25 H100-hours. Colab bills in compute units (CU), and an H100 costs about 15 CU an hour, so 100 CU buys roughly 6–7 H100-hours. **Decision: spend the GPU only where the question needs it, and price everything before spending.**

| Stage | Where | Plan |
|---|---|---|
| Note bank (Phi-4-mini, ~600 notes) | GPU | once, ~10 min |
| Kev zero-shot + in-context | GPU | every horizon (1, 4, 12); cheap, no training |
| Kev RL | GPU | **H=1 and H=12 only, one seed**, 60 updates. The two ends of the dial are enough for H2's learning-speed question and H3's interaction with horizon |
| Classic PPO | **laptop CPU, 0 CU** | 3 variants × 3 horizons × **3 seeds** (it's free, so it gets the seeds) |
| Bayes and oracle DP | CPU | minutes |

**How spend is controlled:**
- `fns kev-check` measures Kev's real speed on the GPU you were given.
- `fns plan` prices each stage in hours and CU against the budget. When the plan is over budget, it says how many RL updates fit.
- `fns kev-rl --max-minutes` pauses at a checkpoint, so each session's spend has a cap.
- With guessed speeds, the lean plan comes to about 28 CU, leaving room to add H=4 RL or a second seed once the real speed is known.

**What we give up:** a single Kev RL seed means no seed spread for Kev+RL, so H2's verdict carries that caveat in the report. RL at H=4 is also dropped unless the measured speed leaves room for it.

The config has a `smoke` profile (runs on a laptop in seconds, with a fake Kev) and a `full` profile (Colab). Every stage is resumable and checkpointed to Google Drive.

---

## 6. Feature map (the report)

Each feature: **who it's for → what need it serves → why it is built this way.**

| # | Feature | User | Need | Reasoning |
|---|---|---|---|---|
| F1 | **The question**: a hero stating the thesis and the answer in one line each | Everyone, 10 seconds | "What did you find?" | Lead with the verdict. Researchers skim, and hiring managers skim harder |
| F2 | **Hypothesis scorecard**: H1–H5, each with a verdict, the evidence number and the rule | Practitioner, hiring manager | Trust: was this decided before or after looking? | Pre-registered rules shown next to verdicts prevent post-hoc storytelling |
| F3 | **The desk** (environment explorer): step through one scenario; see the note, the numbers, then reveal the hidden truth and the Bayes posterior | Planner | "Is this my world, or a toy?" | Grounds every later chart in a concrete decision |
| F4 | **Learning curves**: % of optimal vs decisions consumed (log x), arm toggles, horizon tabs, a CI band, a "cost so far" readout | Practitioner | H1/H2: how each approach learns | Log x is the honest scale: RL needs orders of magnitude more experience, and a linear axis would hide Kev's starting point |
| F4b | **Sample-efficiency strip** under the curves: decisions-to-50/80/90% markers on the x axis per arm, a "gain per 10× data" bar and the crossover marker | Practitioner, buyer | "How much experience (≈ time, money, simulator work) before this is useful?" | In planning, experience is the scarce thing: you can't replay 300k weeks of a real business. Decisions-to-threshold turns the curves into the number a buyer budgets with |
| F5 | **Horizon dial chart**: each arm's final score as H goes 1 → 4 → 12 | Both | H3: does one-shot strength survive planning? | One small multiple per arm; the slope is the finding |
| F6 | **Hand-off view**: automation @ error budget with a slider (1–10%), split decidable vs judgement, beside a reliability diagram | Planner | "What could I safely hand off?" | Turns calibration (an abstract metric) into the planner's actual decision |
| F7 | **Decision replays**: one scenario, Kev vs PPO vs Bayes side by side, week by week, with each agent's confidence | Both | "Show me a case where they differ" | One concrete disagreement is worth more than a chart |
| F8 | **Cost & latency table** | Practitioner, buyer | Build vs buy, serving cost | Accuracy without cost is half a decision |
| F9 | **Decisions log**: every design choice in this spec, with reasoning and the alternative rejected | Hiring manager | "How does this person think?" | This is the PM artifact; the study is the excuse |
| F10 | **Limitations** | Everyone | Honesty | Synthetic world, one product, one model family, a small horizon. Saying so is part of being credible |

**Format decision:** a static, precomputed report (React reads a `report.json` exported by the backend). No live inference. **Why:** it's cheap to host on the shared lab box (CPU-only, one stack at a time), loads instantly, and is reproducible. A live playground would need a GPU or a slow CPU Kev for little added insight. Revisit once the results are in.

---

## 7. Architecture

```
fast-and-slow/
  docs/spec.md               this file
  backend/                   uv project (Python 3.12), package `fast_and_slow`
    src/fast_and_slow/
      config.py              typed configs + profiles (smoke, full)
      sim/                   world dynamics, signal model, gymnasium env, text rendering
      notes/                 note-bank generation (templates offline, Phi-4-mini on GPU) + train/test split
      oracle/                Bayes belief filter + DP, perfect-information DP, decidability labels
      agents/                baselines, Kev (zero-shot / in-context / RL), classic PPO (num / emb / sig)
      eval/                  fixed scenario sets, evaluation runner, metrics
      export/                runs/ → report.json (+ hypothesis verdicts)
      cli.py                 `fns <stage>` entry points used by the notebook
    tests/                   pytest; fast and offline (a fake Kev model)
  notebooks/colab.ipynb      THE entry point: clone, uv sync, pick a profile, run stages, sync to Drive
  frontend/                  Vite + React + TypeScript report; reads public/data/report.json
```

- **Python with uv:** `uv sync` (CPU) or `uv sync --extra gpu` (Kev, transformers). Kev is installed from Git at the `kev-1.0` tag, because PyPI's `kev` is an unrelated package.
- **Notebook is thin:** every cell is one CLI call. Logic lives in the package, which is tested and diffable.
- **Frontend:** pnpm + Vite + React + TS, Python-free, built to static files that the lab serves at `/p/fast-and-slow/`.
