"""Published runs, kept in MongoDB: one document per run of the study, holding its report and what produced it.

Why a document store: a report is one nested JSON object (curves, verdicts, replays) that the frontend reads whole and
nobody queries inside. A run is naturally one document; the list view needs only a small summary of each, which a
projection serves without loading the reports. Runs are append-only: a re-run from Colab adds a run, never edits one.

This module is light on purpose (pymongo, no numpy or torch), so the API container stays small.
"""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from typing import Any

from pymongo import DESCENDING
from pymongo.collection import Collection

REQUIRED_KEYS = ("meta", "results", "hypotheses")
SUMMARY_FIELDS = {"_id": 1, "label": 1, "created_at": 1, "profile": 1, "git_commit": 1, "gpu": 1, "verdicts": 1}


class InvalidReport(ValueError):
    pass


def check_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise InvalidReport("a report is a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in report]
    if missing:
        raise InvalidReport(f"not a report: missing {', '.join(missing)}")
    if not isinstance(report["hypotheses"], list):
        raise InvalidReport("not a report: hypotheses must be a list")
    return report


def new_run_id(now: datetime) -> str:
    """Sortable and readable: 2026-11-18T0215-3f9a (a short random tail keeps two publishes in a minute apart)."""
    return f"{now:%Y-%m-%dT%H%M}-{secrets.token_hex(2)}"


class RunStore:
    def __init__(self, runs: Collection[dict[str, Any]]):
        self.runs = runs

    def ensure_indexes(self) -> None:
        self.runs.create_index([("created_at", DESCENDING)])

    def add(
        self,
        report: dict[str, Any],
        label: str = "",
        git_commit: str = "",
        gpu: str = "",
        now: datetime | None = None,
    ) -> str:
        check_report(report)
        now = now or datetime.now(UTC)
        run_id = new_run_id(now)
        meta = report["meta"]
        self.runs.insert_one(
            {
                "_id": run_id,
                "label": label or f"{meta.get('profile', 'run')} run",
                "created_at": now,
                "profile": meta.get("profile", ""),
                "git_commit": git_commit,
                "gpu": gpu,
                "verdicts": {h.get("id", "?"): h.get("verdict", "") for h in report["hypotheses"]},
                "report": report,
            }
        )
        return run_id

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        docs = self.runs.find({}, SUMMARY_FIELDS).sort("created_at", DESCENDING).limit(limit)
        return [_summary(d) for d in docs]

    def get(self, run_id: str) -> dict[str, Any] | None:
        doc = self.runs.find_one({"_id": run_id})
        return None if doc is None else {**_summary(doc), "report": doc["report"]}

    def latest(self) -> dict[str, Any] | None:
        doc = self.runs.find_one({}, {"_id": 1}, sort=[("created_at", DESCENDING)])
        return None if doc is None else self.get(doc["_id"])


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    created = doc["created_at"]
    if created.tzinfo is None:  # MongoDB hands datetimes back naive; they were stored as UTC
        created = created.replace(tzinfo=UTC)
    return {
        "id": doc["_id"],
        "label": doc.get("label", ""),
        "created_at": created.isoformat(),
        "profile": doc.get("profile", ""),
        "git_commit": doc.get("git_commit", ""),
        "gpu": doc.get("gpu", ""),
        "verdicts": doc.get("verdicts", {}),
    }
