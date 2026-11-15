"""`fns` command line: one command per study stage. The Colab notebook calls these, nothing else.

fns notes      --profile full          # write the note bank (Phi-4-mini on GPU, templates on smoke)
fns refs       --profile full          # Bayes / oracle / base-stock / random references
fns kev        --profile full          # Kev zero-shot + in-context evaluations
fns kev-rl     --profile full --horizon 4 --seed 0
fns ppo        --profile full --variant emb --horizon 4 --seed 0
fns export     --profile full --out ../frontend/public/data/report.json
fns all        --profile smoke         # everything, in order (laptop smoke test)
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from .config import load
from .export.report import export as export_report
from .study import Study

app = typer.Typer(add_completion=False, no_args_is_help=True)
Profile = typer.Option("smoke", "--profile", "-p", help="smoke | full | path to a YAML override")
Runs = typer.Option(Path("runs"), "--runs", help="where outputs go (a Drive folder on Colab)")
Force = typer.Option(False, "--force", help="redo work that already has outputs")
Out = typer.Option(Path("../frontend/public/data/report.json"), "--out", help="where report.json goes")


def _study(profile: str, runs: Path) -> Study:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
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
    force: bool = Force,
) -> None:
    """RL fine-tune Kev (LoRA + PPO on the decision head) for one horizon and seed."""
    _study(profile, runs).kev_rl(horizon, seed, force)


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
def export(
    profile: str = Profile,
    runs: Path = Runs,
    out: Path = Out,
) -> None:
    """Build report.json for the frontend."""
    typer.echo(export_report(_study(profile, runs), out))


@app.command("all")
def run_all(
    profile: str = Profile,
    runs: Path = Runs,
    out: Path = Out,
    force: bool = Force,
) -> None:
    """Every stage in order. Meant for the smoke profile; on Colab the notebook runs stages one by one."""
    s = _study(profile, runs)
    s.notes(force)
    s.references(force)
    s.kev_static("kev_zero", force)
    s.kev_static("kev_icl", force)
    for h in s.cfg.eval.horizons:
        for seed in s.cfg.seeds:
            s.kev_rl(h, seed, force)
            for v in s.cfg.classic.variants:
                s.ppo(v, h, seed, force)
    typer.echo(export_report(s, out))


@app.command()
def status(profile: str = Profile, runs: Path = Runs) -> None:
    """What's done and what's left (Colab sessions disconnect; every stage resumes from here)."""
    s = _study(profile, runs)
    c = s.cfg
    rows = [("notes", s.root / "notes" / "bank.json")]
    rows += [(f"refs H{h}", s.root / "refs" / f"H{h}.json") for h in c.eval.horizons]
    for arm in ("kev_zero", "kev_icl"):
        rows += [(f"{arm} H{h}", s.root / "evals" / arm / f"H{h}" / "seed0.json") for h in c.eval.horizons]
    for h in c.eval.horizons:
        for seed in c.seeds:
            rows.append((f"kev_rl H{h} seed{seed}", s.root / "evals" / "kev_rl" / f"H{h}" / f"seed{seed}.json"))
            for v in c.classic.variants:
                rows.append((f"ppo_{v} H{h} seed{seed}", s.root / "evals" / f"ppo_{v}" / f"H{h}" / f"seed{seed}.json"))
    done = sum(p.exists() for _, p in rows)
    for name, p in rows:
        typer.echo(f"{'done' if p.exists() else '    '}  {name}")
    typer.echo(f"{done}/{len(rows)} stages complete")


@app.command("kev-check")
def kev_check(profile: str = Profile, runs: Path = Runs) -> None:
    """One-minute preflight before hours of GPU time: load Kev, score a request, take one gradient step."""
    import time

    import torch

    from .sim.desk import Episode
    from .sim.render import ORDER_Q, request
    from .sim.world import sample_scenario

    s = _study(profile, runs)
    w = s.cfg.world
    t0 = time.perf_counter()
    pol = s.policy(fresh=True)
    typer.echo(f"loaded {pol.name} in {time.perf_counter() - t0:.1f}s")
    sc = sample_scenario(w, 4, 1)
    req = request(w, Episode(w, sc, s.bank.texts(sc, "test")).observe())
    t0 = time.perf_counter()
    out = pol.score([req])[0]
    probs = torch.softmax(out[ORDER_Q].float(), -1)
    typer.echo(
        f"scored in {(time.perf_counter() - t0) * 1000:.0f} ms; order probs {[round(float(p), 3) for p in probs]}"
    )
    z = pol.score([req], grad=True)[0][ORDER_Q]
    loss = -torch.log_softmax(z.float(), -1)[0]
    loss.backward()  # type: ignore[no-untyped-call]
    grads = sum(p.grad is not None for g in pol.param_groups() for p in g["params"])
    if not grads:
        raise typer.Exit(code=1)
    typer.echo(f"gradient reached {grads} trainable tensors: RL can run")


if __name__ == "__main__":  # pragma: no cover
    app()
