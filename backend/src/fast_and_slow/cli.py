"""`fns` command line: one command per study stage. The Colab notebook calls these, nothing else.

fns notes      --profile full          # write the note bank (Phi-4-mini on GPU, templates on smoke)
fns refs       --profile full          # Bayes / oracle / base-stock / random references
fns kev        --profile full          # Kev zero-shot + in-context evaluations
fns kev-rl     --profile full --horizon 4 --seed 0
fns ppo        --profile full --variant emb --horizon 4 --seed 0
fns export     --profile full          # build report.json (runs/<profile>/report.json)
fns publish    --profile full --api https://.../api --label "first A100 run"   # store it as a new run
fns all        --profile smoke         # everything, in order (laptop smoke test)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path

import typer

from .config import load
from .export.report import export as export_report
from .study import Study

app = typer.Typer(add_completion=False, no_args_is_help=True)
Profile = typer.Option("smoke", "--profile", "-p", help="smoke | full | path to a YAML override")
Runs = typer.Option(Path("runs"), "--runs", help="where outputs go (a Drive folder on Colab)")
Force = typer.Option(False, "--force", help="redo work that already has outputs")
ReportFile = typer.Option(None, "--report", help="an existing report.json (default: export one now)")
Out = typer.Option(None, "--out", help="where report.json goes (default: runs/<profile>/report.json)")


def _study(profile: str, runs: Path) -> Study:
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s")
    logging.getLogger("fast_and_slow").setLevel(logging.INFO)  # our progress lines, not every library's HTTP requests
    cfg = load(profile)
    return Study(cfg, runs)


@app.command()
def notes(profile: str = Profile, runs: Path = Runs, force: bool = Force) -> None:
    """Write (or reuse) the note bank."""
    typer.echo(_study(profile, runs).notes(force))


@app.command()
def refs(profile: str = Profile, runs: Path = Runs, force: bool = Force) -> None:
    """Reference arms on the held-out scenarios: random, base-stock, Bayes, perfect information."""
    _study(profile, runs).references(force)


@app.command()
def kev(profile: str = Profile, runs: Path = Runs, force: bool = Force) -> None:
    """Kev zero-shot and in-context evaluations (no training)."""
    s = _study(profile, runs)
    s.kev_static("kev_zero", force)
    s.kev_static("kev_icl", force)


@app.command("kev-rl")
def kev_rl(
    profile: str = Profile,
    runs: Path = Runs,
    horizon: int = typer.Option(..., "--horizon"),
    seed: int = typer.Option(0, "--seed"),
    max_minutes: float = typer.Option(0, "--max-minutes", help="pause at a checkpoint after this long (0 = no limit)"),
    force: bool = Force,
) -> None:
    """RL fine-tune Kev (LoRA + PPO on the decision head) for one horizon and seed. Resumes from its checkpoint."""
    finished = _study(profile, runs).kev_rl(horizon, seed, force, max_minutes or None)
    typer.echo("finished" if finished else "paused at a checkpoint: run the same command to continue")


@app.command()
def ppo(
    profile: str = Profile,
    runs: Path = Runs,
    variant: str = typer.Option(..., "--variant", help="num | emb | sig"),
    horizon: int = typer.Option(..., "--horizon"),
    seed: int = typer.Option(0, "--seed"),
    force: bool = Force,
) -> None:
    """Train and evaluate one classic PPO run."""
    _study(profile, runs).ppo(variant, horizon, seed, force)


@app.command()
def shift(
    profile: str = Profile,
    runs: Path = Runs,
    part: str = typer.Option(..., "--part", help="cpu: references + PPO retraining (laptop) | gpu: Kev + the alarm"),
    name: str = typer.Option("", "--name", help="one shift (default: every shift in the config)"),
    force: bool = Force,
) -> None:
    """H6 regime changes: measure what the transition timelines need."""
    from .shift_study import run_shift

    s = _study(profile, runs)
    for n in [name] if name else s.cfg.shift.shifts:
        typer.echo(run_shift(s, n, part, force))


@app.command("ppo-all")
def ppo_all(profile: str = Profile, runs: Path = Runs, force: bool = Force) -> None:
    """Every PPO run in the plan. CPU only: meant for a laptop, so it costs no Colab units."""
    s = _study(profile, runs)
    todo = s.ppo_runs()
    for i, (v, h, seed) in enumerate(todo, 1):
        typer.echo(f"[{i}/{len(todo)}] ppo_{v} H{h} seed{seed}")
        s.ppo(v, h, seed, force)


@app.command()
def export(
    profile: str = Profile,
    runs: Path = Runs,
    out: Path | None = Out,
) -> None:
    """Build report.json (reports are published as runs with `fns publish`, never committed)."""
    s = _study(profile, runs)
    typer.echo(export_report(s, out or s.root / "report.json"))


@app.command()
def publish(
    profile: str = Profile,
    runs: Path = Runs,
    api: str = typer.Option(
        ..., "--api", envvar="FNS_API", help="the runs API, e.g. https://example.org/p/fast-and-slow/api"
    ),
    token: str = typer.Option(..., "--token", envvar="UPLOAD_TOKEN", help="the API's upload token (or $UPLOAD_TOKEN)"),
    label: str = typer.Option("", "--label", help="a name for this run in the report's run picker"),
    report: Path | None = ReportFile,
    commit: str = typer.Option("", "--commit", help="the code that produced the run (default: this checkout's HEAD)"),
    gpu: str = typer.Option("", "--gpu", help="the GPU it ran on (default: from kev-check's speed.json)"),
) -> None:
    """Publish the report as a new run: the report site lists every run and opens the newest.

    An older report.json works as is (--report); pass --commit when it came from different code than this checkout."""
    from .publish import git_commit, gpu_name, post_run

    s = _study(profile, runs)
    path = report or export_report(s, s.root / "report.json")
    body = {
        "report": json.loads(path.read_text()),
        "label": label,
        "git_commit": commit or git_commit(Path(__file__).parent),
        "gpu": gpu or gpu_name(s.root),
    }
    try:
        run_id = post_run(api, token, body)
    except RuntimeError as e:
        typer.echo(f"not published: {e}\nThe report is still saved at {path}; run this again when the API is up.")
        raise typer.Exit(code=1) from e
    typer.echo(f"published run {run_id}")


@app.command("all")
def run_all(
    profile: str = Profile,
    runs: Path = Runs,
    out: Path | None = Out,
    force: bool = Force,
) -> None:
    """Every stage in order. Meant for the smoke profile; on Colab the notebook runs stages one by one."""
    s = _study(profile, runs)
    s.notes(force)
    s.references(force)
    s.kev_static("kev_zero", force)
    s.kev_static("kev_icl", force)
    for h, seed in s.kev_rl_runs():
        s.kev_rl(h, seed, force)
    for v, h, seed in s.ppo_runs():
        s.ppo(v, h, seed, force)
    from .shift_study import run_shift

    for n in s.cfg.shift.shifts:
        run_shift(s, n, "cpu", force)
        run_shift(s, n, "gpu", force)
    typer.echo(export_report(s, out or s.root / "report.json"))


@app.command()
def status(profile: str = Profile, runs: Path = Runs) -> None:
    """What's done and what's left (Colab sessions disconnect; every stage resumes from here)."""
    s = _study(profile, runs)
    c = s.cfg
    rows = [("notes", s.root / "notes" / "bank.json")]
    rows += [(f"refs H{h}", s.root / "refs" / f"H{h}.json") for h in c.eval.horizons]
    for arm in ("kev_zero", "kev_icl"):
        rows += [(f"{arm} H{h}", s.root / "evals" / arm / f"H{h}" / "seed0.json") for h in c.eval.horizons]
    for h, seed in s.kev_rl_runs():
        ck = s.root / "checkpoints" / "kev_rl" / f"H{h}" / f"seed{seed}" / "trainer.json"
        out = s.root / "evals" / "kev_rl" / f"H{h}" / f"seed{seed}.json"
        tag = ""
        if ck.exists() and not out.exists():
            tag = f"  (paused at update {json.loads(ck.read_text())['update']}/{c.kev.rl_updates})"
        rows.append((f"kev_rl H{h} seed{seed}{tag}", out))
    for v, h, seed in s.ppo_runs():
        rows.append((f"ppo_{v} H{h} seed{seed}", s.root / "evals" / f"ppo_{v}" / f"H{h}" / f"seed{seed}.json"))
    for n in c.shift.shifts:
        d = s.root / "shifts" / n
        rows.append((f"shift {n}: kev + alarm (gpu)", d / "detect.json"))
        rows += [(f"shift {n}: ppo_{v} retrain (cpu)", d / f"ppo_{v}.json") for v in c.shift.ppo_variants]
    done = sum(p.exists() for _, p in rows)
    for name, p in rows:
        typer.echo(f"{'done' if p.exists() else '    '}  {name}")
    typer.echo(f"{done}/{len(rows)} stages complete")


@app.command("kev-check")
def kev_check(profile: str = Profile, runs: Path = Runs, batch: int = typer.Option(16, "--batch")) -> None:
    """Preflight on the GPU: load Kev, check gradients reach the adapter, and measure real speed for `fns plan`."""
    import time

    import torch

    from .plan import Speed
    from .sim.desk import Episode
    from .sim.render import ORDER_Q, request
    from .sim.world import sample_scenario

    s = _study(profile, runs)
    w = s.cfg.world
    typer.echo("kev-check 1/4: loading Kev-4B (the first time downloads about 8 GB)")
    t0 = time.perf_counter()
    pol = s.policy(fresh=True)
    typer.echo(f"loaded {pol.name} in {time.perf_counter() - t0:.1f}s")
    reqs = []
    for i in range(batch):
        sc = sample_scenario(w, 12, 100 + i)
        ep = Episode(w, sc, s.bank.texts(sc, "test"))
        for _ in range(i % 12):
            ep.step(10)
        reqs.append(request(w, ep.observe()))
    order_only = [{**r, "questions": {ORDER_Q: r["questions"][ORDER_Q]}} for r in reqs]

    def timed(fn: Callable[[], object], repeat: int = 2) -> float:
        fn()  # warm-up (kernels, caches)
        t = time.perf_counter()
        for _ in range(repeat):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / repeat / len(reqs)

    typer.echo(f"kev-check 2/4: timing {len(reqs)} requests with both questions")
    both = timed(lambda: pol.score(reqs))
    typer.echo("kev-check 3/4: timing the order question alone")
    order = timed(lambda: pol.score(order_only))

    micro = s.cfg.kev.rl_micro_batch

    def step() -> None:  # the trainer's pattern: forward/backward in micro-batches, gradients accumulate
        for k in range(0, len(order_only), micro):
            zs = pol.score(order_only[k : k + micro], grad=True)
            loss = torch.stack([-torch.log_softmax(z[ORDER_Q].float(), -1)[0] for z in zs]).sum() / len(order_only)
            loss.backward()  # type: ignore[no-untyped-call]

    typer.echo(
        f"kev-check 4/4: timing a training step ({micro} requests per pass) and checking gradients reach the adapter"
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train = timed(step)
    if torch.cuda.is_available():
        peak, total = (
            torch.cuda.max_memory_allocated() / 2**30,
            torch.cuda.get_device_properties(0).total_memory / 2**30,
        )
        typer.echo(f"peak GPU memory in training: {peak:.1f} of {total:.0f} GB")
    grads = sum(p.grad is not None for g in pol.param_groups() for p in g["params"])
    if not grads:
        typer.echo("no gradient reached the trainable weights: RL cannot run")
        raise typer.Exit(code=1)
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    speed = Speed(order, both, train, gpu)
    speed.save(s.root / "speed.json")
    typer.echo(f"gradient reached {grads} trainable tensors: RL can run")
    typer.echo(
        f"per request: order {order * 1000:.0f} ms, order+regime {both * 1000:.0f} ms, "
        f"train step {train * 1000:.0f} ms  ({gpu}) -> saved for `fns plan`"
    )


@app.command()
def plan(
    profile: str = Profile,
    runs: Path = Runs,
    cu_per_hour: float = typer.Option(0, "--cu-per-hour", help="your GPU's rate from Colab's resources panel"),
    budget: float = typer.Option(0, "--budget", help="compute units you're willing to spend"),
    session_cu: float = typer.Option(0, "--session-cu", help="compute units you'll spend per Colab session"),
) -> None:
    """Estimated GPU hours and Colab compute units per stage, from kev-check's measured speed."""
    from .plan import Speed, render

    s = _study(profile, runs)
    e = s.cfg.eval
    if cu_per_hour:  # remembered for the report's cost table (GPU time priced at what it actually cost)
        (s.root / "rate.json").write_text(json.dumps({"cu_per_hour": cu_per_hour}))
    typer.echo(
        render(
            s.cfg,
            Speed.load(s.root / "speed.json"),
            cu_per_hour or e.cu_per_hour,
            budget or e.budget_cu,
            session_cu=session_cu,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    app()
