"""Kev-4B as a DecisionPolicy (GPU only; excluded from coverage, exercised on Colab by the `kev-smoke` stage).

Built on Kev's own internals rather than its HTTP server, because RL needs gradients:
- kev.data.materialize turns a System One request into Kev's internal record, exactly as the server does;
- DecisionModel.forward_batch returns differentiable pointer-head logits (temperature applied in eval mode);
- the LoRA adapter is kept unmerged (LoadOptions(merge=False)) so its weights and the pointer head can be trained.

The model stays in eval mode even while training: that keeps dropout off and the fitted temperature applied, so the
log-probabilities used for PPO ratios are the same ones the agent acted on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..sim.render import question_keys
from .policy import Request


class KevPolicy:
    def __init__(self, run: str, dtype: str = "bf16", device: str = "cuda"):
        from kev.checkpoint import Checkpoint, LoadOptions

        self.name = run
        self.device = device
        ckpt = Checkpoint(run)
        torch_dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        self.tok, self.model = ckpt.load(device, LoadOptions(dtype=torch_dtype, merge=False))
        self.model.eval()
        self.temperature = float(self.model.head.temperature)
        self.adapter = [p for n, p in self.model.named_parameters() if "lora_" in n]
        self.head = list(self.model.head.parameters())
        for p in self.model.parameters():
            p.requires_grad_(False)
        for p in [*self.adapter, *self.head]:
            p.requires_grad_(True)
        if hasattr(self.model.lm, "gradient_checkpointing_enable"):
            self.model.lm.gradient_checkpointing_enable()

    def _record(self, req: Request) -> tuple[dict[str, Any], list[str]]:
        from kev.data import materialize

        first = {
            qid: (False if q["type"] == "noul" else next(iter(q["criteria"]))) for qid, q in req["questions"].items()
        }
        labelled = {
            "state": req["state"],
            "questions": {
                qid: {**q, "label": first[qid], "src": "fast-and-slow"} for qid, q in req["questions"].items()
            },
        }
        rec = materialize(labelled)
        for q, (qid, src) in zip(rec["questions"], req["questions"].items(), strict=True):
            if list(q["keys"]) != question_keys(src):
                raise RuntimeError(f"Kev reordered options for {qid}: {q['keys']}")
        return rec, list(req["questions"])

    def score(self, requests: list[Request], grad: bool = False) -> list[dict[str, torch.Tensor]]:
        recs = [self._record(r) for r in requests]
        encs = [self.model.encode(self.tok, rec) for rec, _ in recs]
        with torch.set_grad_enabled(grad):
            logits = self.model.forward_batch(encs)
        return [
            {qid: z.float() for qid, z in zip(qids, per_q, strict=True)}
            for (_, qids), per_q in zip(recs, logits, strict=True)
        ]

    def param_groups(self) -> list[dict[str, Any]]:
        return [{"params": self.adapter, "role": "adapter"}, {"params": self.head, "role": "head"}]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "adapter": {n: p.detach().cpu() for n, p in self.model.named_parameters() if "lora_" in n},
                "head": self.model.head.state_dict(),
            },
            path / "kev_rl_weights.pt",
        )

    def load(self, path: Path) -> None:
        state = torch.load(path / "kev_rl_weights.pt", map_location=self.device)
        params = dict(self.model.named_parameters())
        with torch.no_grad():
            for n, t in state["adapter"].items():
                params[n].copy_(t.to(params[n].device, params[n].dtype))
        self.model.head.load_state_dict(state["head"])
