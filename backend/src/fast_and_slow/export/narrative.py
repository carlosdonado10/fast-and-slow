"""The words the report shows next to the numbers: arm descriptions, hypotheses with their pre-registered verdict rules,
the feature map and the decisions log. One source of truth for the frontend (mirrors docs/spec.md)."""

from __future__ import annotations

ARMS: dict[str, dict[str, str]] = {
    "random": {"label": "Random", "group": "reference", "sees": "Nothing", "role": "The 0% anchor"},
    "basestock": {
        "label": "Base-stock tool",
        "group": "reference",
        "sees": "Numbers only",
        "role": "Today's planning software: an order-up-to rule tuned on history",
    },
    "bayes": {
        "label": "Bayes-optimal",
        "group": "reference",
        "sees": "Numbers + perfectly parsed notes",
        "role": "The 100% anchor: exact dynamic programming over beliefs",
    },
    "oracle": {
        "label": "Perfect information",
        "group": "reference",
        "sees": "The hidden demand regime",
        "role": "Upper bound: what knowing the truth is worth",
    },
    "kev_zero": {
        "label": "Kev-4B, zero-shot",
        "group": "system-one",
        "sees": "Text: brief, numbers, inbox",
        "role": "A JEV-style decision model out of the box",
    },
    "kev_icl": {
        "label": "Kev-4B + context",
        "group": "system-one",
        "sees": "Text + worked examples",
        "role": "No training: rules and an experienced planner's examples in the prompt",
    },
    "kev_rl": {
        "label": "Kev-4B + RL",
        "group": "system-one",
        "sees": "Text: brief, numbers, inbox",
        "role": "LoRA + PPO on the decision head, trained on simulated weeks",
    },
    "ppo_num": {
        "label": "PPO, numbers",
        "group": "classic-rl",
        "sees": "Numbers only",
        "role": "Classic RL as deployed in planning today",
    },
    "ppo_emb": {
        "label": "PPO, numbers + text",
        "group": "classic-rl",
        "sees": "Numbers + note embeddings",
        "role": "Classic RL that can 'read' (a fair fight)",
    },
    "ppo_sig": {
        "label": "PPO, parsed signals",
        "group": "classic-rl",
        "sees": "Numbers + note attributes",
        "role": "Classic RL with a perfect parser: separates reading from learning",
    },
}

HYPOTHESES: list[dict[str, str]] = [
    {
        "id": "H1",
        "statement": "A JEV-style model is strong from the start, where RL starts near random.",
        "rule": "Supported if Kev zero-shot beats every PPO variant's untrained score by >= 20 points of % optimal "
        "(mean over horizons); not supported if it doesn't beat them at all.",
    },
    {
        "id": "H2",
        "statement": "A JEV-style model improves more slowly with training; RL catches up or overtakes.",
        "rule": "Supported if Kev+RL gains fewer points per 10x data than the best PPO variant in most horizons and PPO "
        "reaches Kev+RL's final score (a crossover); not supported if Kev+RL learns at least as fast everywhere.",
    },
    {
        "id": "H3",
        "statement": "A JEV-style model is good at one-shot decisions, weaker at multi-step planning.",
        "rule": "Supported if Kev's gap to optimal (zero-shot and in-context, averaged) grows by >= 5 points from the "
        "shortest to the longest horizon; not supported if it shrinks or stays within 2 points.",
    },
    {
        "id": "H4",
        "statement": "Calibrated confidence lets it hand off safely: sure on decidable calls, unsure on judgement calls.",
        "rule": "Supported if, at a 5% error budget, the best Kev arm automates >= 10 points more of the decidable calls "
        "than of the judgement calls and its ECE is <= 0.10; not supported if it automates no more decidable calls.",
    },
    {
        "id": "H5",
        "statement": "Context without fine-tuning recovers part of the fine-tuning gain for free.",
        "rule": "Supported if Kev+context gains over zero-shot >= 25% of what Kev+RL gains (and more than 0); "
        "not supported if context doesn't help at all.",
    },
    {
        "id": "H6",
        "statement": "A System One model bridges regime changes: it covers the transition while RL is retrained.",
        "rule": "Supported if, when a planner declares the change, the bridge (Kev until retrained RL beats it) cuts "
        "transition regret by >= 25% vs the better RL-only strategy, for both shifts, both retraining modes and "
        "every PPO variant; not supported if it saves < 5% anywhere.",
    },
]

FEATURES: list[dict[str, str]] = [
    {
        "id": "F1",
        "name": "The question",
        "user": "Everyone, first 10 seconds",
        "need": "What did you find?",
        "why": "Lead with the verdict. Researchers skim, and hiring managers skim harder.",
    },
    {
        "id": "F2",
        "name": "Hypothesis scorecard",
        "user": "Practitioner, hiring manager",
        "need": "Trust: was this decided before or after looking?",
        "why": "Rules written before the runs, shown next to each verdict, prevent post-hoc storytelling.",
    },
    {
        "id": "F3",
        "name": "The desk",
        "user": "Planner",
        "need": "Is this my world, or a toy?",
        "why": "Step through one scenario and reveal the hidden truth; grounds every later chart in a concrete decision.",
    },
    {
        "id": "F4",
        "name": "Learning curves",
        "user": "Practitioner",
        "need": "How does each approach learn?",
        "why": "Log x is the honest scale: RL needs orders of magnitude more experience, and a linear axis would hide "
        "Kev's starting point.",
    },
    {
        "id": "F4b",
        "name": "Sample-efficiency strip",
        "user": "Practitioner, buyer",
        "need": "How much experience before this is useful?",
        "why": "In planning, experience is the scarce thing. Decisions-to-threshold is the number a buyer budgets with.",
    },
    {
        "id": "F5",
        "name": "Horizon dial",
        "user": "Both",
        "need": "Does one-shot strength survive planning?",
        "why": "One small multiple per arm; the slope is the finding.",
    },
    {
        "id": "F6",
        "name": "Hand-off view",
        "user": "Planner",
        "need": "What could I safely hand off?",
        "why": "Turns calibration, an abstract metric, into the planner's actual decision.",
    },
    {
        "id": "F7",
        "name": "Decision replays",
        "user": "Both",
        "need": "Show me a case where they differ.",
        "why": "One concrete disagreement is worth more than a chart.",
    },
    {
        "id": "F8",
        "name": "Cost & latency",
        "user": "Practitioner, buyer",
        "need": "Build vs buy, serving cost",
        "why": "Accuracy without cost is half a decision.",
    },
    {
        "id": "F9",
        "name": "Decisions log",
        "user": "Hiring manager",
        "need": "How does this person think?",
        "why": "Every design choice with its reasoning and the alternative rejected.",
    },
    {
        "id": "F11",
        "name": "Transition timeline",
        "user": "Planner, ops lead",
        "need": "Something changed this week: who decides until the model catches up, and what does the gap cost?",
        "why": "The question is about time, so the x axis is calendar weeks: a strip per strategy with the shift, the "
        "bridge's hand-back and the shaded regret, plus how fast each detector would have noticed.",
    },
    {
        "id": "F10",
        "name": "Limitations",
        "user": "Everyone",
        "need": "Honesty",
        "why": "Synthetic world, one product, one model family, small horizons. Saying so is part of being credible.",
    },
]

DECISIONS: list[dict[str, str]] = [
    {
        "decision": "One environment with a horizon dial (1, 4, 12 weeks)",
        "why": "Makes 'one-shot vs multi-step' a controlled variable: only the horizon changes.",
        "rejected": "Separate one-shot and multi-step tasks, where task differences would confound the comparison.",
    },
    {
        "decision": "A synthetic, ground-truthed world",
        "why": "Every note's reliability is known, so the Bayes-optimal planner, % of optimal and calibration are exact.",
        "rejected": "Real demand data with generated notes: realistic, but no ground truth to score against.",
    },
    {
        "decision": "Notes written by Phi-4-mini, a different model family from Kev's Qwen base",
        "why": "The text style can't favour the model under test. Train and test use different wording.",
        "rejected": "Templates only (too regular to test reading) or Qwen-written notes (a home-field advantage).",
    },
    {
        "decision": "Kev-4B as the open JEV stand-in",
        "why": "Apache-2.0, a JEV-compatible API, calibrated by default, and small enough to RL-train in budget.",
        "rejected": "Nimble-9B (slower to train) and the closed JEV API (no fine-tuning, no weights).",
    },
    {
        "decision": "PPO directly on Kev's pointer-head logits",
        "why": "Kev scores options instead of generating text, so its option distribution is already a policy.",
        "rejected": "Supervised fine-tuning on Bayes labels: that's imitation, not the RL comparison asked for.",
    },
    {
        "decision": "Three PPO variants (numbers / embeddings / parsed signals)",
        "why": "Separates 'can it read?' from 'can it learn?', the two things a JEV-style model bundles.",
        "rejected": "A single PPO baseline, which would leave the reason for any gap unknown.",
    },
    {
        "decision": "Two score scales: % of optimal and inbox value captured",
        "why": "Practitioners expect random = 0 / optimal = 100; planners ask whether reading beat their numbers-only tool.",
        "rejected": "One scale: random-anchored scores look high for any sane policy and hide the planner's question.",
    },
    {
        "decision": "Decidable vs judgement labels from the oracle",
        "why": "A decision is decidable when knowing the truth wouldn't change the call: a dominant strategy exists.",
        "rejected": "Labelling by the model's own confidence, which would make H4 circular.",
    },
    {
        "decision": "Regime changes measured as a bridge, not a contest (H6)",
        "why": "The realistic choice after a change isn't Kev or RL forever: it's who decides while retraining catches up.",
        "rejected": "Comparing stale RL with Kev only, which ignores that companies do retrain.",
    },
    {
        "decision": "One retraining curve, two calendar mappings (rebuilt simulator vs live experience)",
        "why": "Same warm-started PPO; the modes differ only in how fast new-world experience arrives, so training "
        "noise can't masquerade as a difference between them.",
        "rejected": "Two separate trainings.",
    },
    {
        "decision": "An announced shift and a silent one",
        "why": "Text-readable changes are where a reading model should shine; silent ones test whether it notices "
        "from numbers, against a statistical detector calibrated to the same false-alarm rate.",
        "rejected": "Only announced shifts, which would flatter the model.",
    },
    {
        "decision": "A static report, no live inference",
        "why": "Cheap on the shared CPU-only lab box, instant to load and reproducible.",
        "rejected": "A live playground: it needs a GPU for little added insight; revisit after the results.",
    },
]

LIMITATIONS: list[str] = [
    "A synthetic world with one product and three demand regimes; real planning has many SKUs, lead times and constraints.",
    "One decision-model family (Kev on Qwen3.5) at one size; the closed JEV may behave differently.",
    "Horizons up to 12 weeks. Longer plans could widen or close the multi-step gap.",
    "Regime changes are two stylised shifts at a plan boundary; real changes arrive gradually and mixed.",
    "Live retraining pools 50 identical products; real portfolios are heterogeneous.",
    "RL budgets are bounded (~100 Colab compute units): one Kev+RL seed, at two horizons, and its curves may not have converged.",
    "Notes are generated text. They vary in wording but not in the messiness of real inboxes.",
]
