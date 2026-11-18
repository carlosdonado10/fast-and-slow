# Fast and Slow

**Can a "System One" decision model take over a planner's decidable calls, and how does it learn compared with classic reinforcement learning?**

A research project with an interactive report. A JEV-style decision model ([Kev-4B](https://github.com/jaredpalmer/kev), the open alternative to TypeSafe's Jev) is compared with classic PPO on a simulated replenishment desk, where a planner reads an inbox of notes of mixed reliability and orders stock each week.

- **Who it's for:** demand and supply planners, and the people who buy their tools; also ML practitioners in fields where RL is common.
- **The product thesis:** planners should spend their judgement on calls that can't be settled by computation. When a dominant strategy exists, a model with honest confidence should take the call.
- **The spec** (use case, hypotheses, metrics, the feature map and every design decision with its reasoning): [`docs/spec.md`](docs/spec.md).

## Hypotheses under test

| | Hypothesis |
|---|---|
| H1 | A JEV-style model is strong from the start, where RL starts near random |
| H2 | It improves more slowly with training; RL catches up (measured by sample efficiency) |
| H3 | It's good at one-shot decisions and weaker at multi-step planning |
| H4 | Calibrated confidence lets it hand off safely: sure on decidable calls, unsure on judgement calls |
| H5 | Context in the prompt, without fine-tuning, recovers part of the fine-tuning gain |
| H6 | When the business changes, it bridges the transition (detects the change, makes the calls) until retrained RL overtakes it |

Each verdict is computed by the export step from rules written down before the runs (`backend/src/fast_and_slow/export/narrative.py`).

## Layout

```
docs/spec.md            the use case, feature map and decisions log
backend/                Python (uv) package `fast_and_slow`: simulator, exact Bayes planner, agents, RL, metrics, export
notebooks/colab.ipynb   the single entry point for the GPU study (thin: one CLI call per cell)
frontend/               React + Vite report that reads report.json
```

## Run it

**On a laptop, in about 15 seconds** (fake Kev, template notes, tiny budgets; checks the whole pipeline):

```bash
cd backend
uv sync
uv run fns all --profile smoke --out ../frontend/public/data/report.json
```

**The real study** runs on Colab with an A100 (H100s are rarely available, and an A100 has room for Kev-4B): open `notebooks/colab.ipynb`, set the repo URL, and run the cells from top to bottom. Each stage is resumable and writes to Google Drive.

**The report:**

```bash
cd frontend
pnpm install
pnpm dev
```

## Quality gates

`pre-commit` runs on every commit (install it once with `uv run --directory backend pre-commit install`):

- black, ruff and strict mypy on the backend
- pytest with a coverage gate of 80% or more (currently about 99%)
- eslint, prettier, tsc and vitest coverage on the frontend
- detect-secrets
- blocking `.env` files, large files (over 500 KB), and model weights or run outputs
- stripping notebook outputs
- standard file-hygiene checks
- no direct commits to `main`

GPU-only adapters (the Kev loader, the note writer and the sentence embedder) are excluded from coverage. The logic they feed is tested through a fake policy that has the same interface; on Colab, `fns kev-check` covers the real path before any long run.
