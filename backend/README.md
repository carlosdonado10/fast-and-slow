# fast_and_slow (backend)

The study package for [Fast and Slow](../README.md): the simulator, the exact Bayes planner, the Kev and PPO agents, the metrics and the report export, behind the `fns` CLI.

```bash
uv sync                                   # CPU: everything except real Kev / note writer / sentence encoder
uv sync --extra gpu                       # + Kev-4B, Phi-4-mini note writer, bge-small embedder
uv run fns all --profile smoke            # the whole pipeline in ~15 s with a fake Kev
uv run pytest --cov                       # tests (coverage gate: 80%)
```

The spec is in [`docs/spec.md`](../docs/spec.md). The GPU entry point is [`notebooks/colab.ipynb`](../notebooks/colab.ipynb).
