from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import mongomock
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from fast_and_slow import publish as pub
from fast_and_slow.api import create_app
from fast_and_slow.cli import app as cli
from fast_and_slow.store import InvalidReport, RunStore, check_report

TOKEN = "test-token"


def report(profile: str = "full", verdict: str = "supported") -> dict[str, Any]:
    return {
        "meta": {"profile": profile},
        "results": {},
        "hypotheses": [{"id": "H1", "verdict": verdict}, {"id": "H2", "verdict": "inconclusive"}],
    }


@pytest.fixture
def store() -> RunStore:
    s = RunStore(mongomock.MongoClient()["fast_and_slow"]["runs"])
    s.ensure_indexes()
    return s


@pytest.fixture
def client(store: RunStore) -> TestClient:
    with TestClient(create_app(store, upload_token=TOKEN)) as c:
        yield c  # type: ignore[misc]


def test_runs_are_listed_newest_first_with_summaries_only(store: RunStore) -> None:
    t0 = datetime(2026, 11, 18, 2, 0, tzinfo=UTC)
    old = store.add(report(verdict="not supported"), label="first", now=t0)
    new = store.add(report(), git_commit="abc1234", gpu="NVIDIA A100", now=t0 + timedelta(hours=1))
    runs = store.list()
    assert [r["id"] for r in runs] == [new, old]
    assert "report" not in runs[0]
    assert runs[0]["verdicts"] == {"H1": "supported", "H2": "inconclusive"}
    assert runs[0]["label"] == "full run" and runs[1]["label"] == "first"
    assert runs[0]["created_at"].startswith("2026-11-18T03:00") and runs[0]["created_at"].endswith("+00:00")
    assert old.startswith("2026-11-18T0200-")
    latest = store.latest()
    assert latest is not None and latest["id"] == new and latest["report"]["meta"]["profile"] == "full"
    assert store.get("nope") is None


def test_an_empty_store_has_no_latest_run(store: RunStore) -> None:
    assert store.latest() is None and store.list() == []


@pytest.mark.parametrize("bad", [[], {"meta": {}}, {"meta": {}, "results": {}, "hypotheses": {}}])
def test_only_reports_are_stored(bad: Any) -> None:
    with pytest.raises(InvalidReport):
        check_report(bad)


def test_api_reads_are_public_and_writes_need_the_token(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/runs").json() == []
    assert client.get("/api/runs/latest").status_code == 404

    body = {"report": report(), "label": "A100 run", "gpu": "NVIDIA A100"}
    assert client.post("/api/runs", json=body).status_code == 401
    assert client.post("/api/runs", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    ok = client.post("/api/runs", json=body, headers={"Authorization": f"Bearer {TOKEN}"})
    assert ok.status_code == 201
    run_id = ok.json()["id"]

    runs = client.get("/api/runs").json()
    assert [r["id"] for r in runs] == [run_id] and runs[0]["label"] == "A100 run"
    assert client.get(f"/api/runs/{run_id}").json()["report"]["meta"]["profile"] == "full"
    assert client.get("/api/runs/latest").json()["id"] == run_id
    assert client.get("/api/runs/missing").status_code == 404


def test_api_rejects_non_reports_and_oversized_uploads(client: TestClient) -> None:
    auth = {"Authorization": f"Bearer {TOKEN}"}
    bad = client.post("/api/runs", json={"report": {"meta": {}}}, headers=auth)
    assert bad.status_code == 422 and "missing" in bad.json()["detail"]
    big = client.post("/api/runs", json={"report": report()}, headers={**auth, "Content-Length": str(20 * 2**20)})
    assert big.status_code in (413, 400)


def test_uploads_are_off_without_a_server_token(store: RunStore) -> None:
    with TestClient(create_app(store, upload_token="")) as c:
        r = c.post("/api/runs", json={"report": report()}, headers={"Authorization": "Bearer x"})
    assert r.status_code == 503


def test_health_reports_an_unreachable_database(store: RunStore) -> None:
    class Down:
        def command(self, *_: Any) -> None:
            raise ConnectionError("no route")

    class Coll:
        database = Down()

        def create_index(self, *_: Any, **__: Any) -> None:
            pass

    with TestClient(create_app(RunStore(Coll()), upload_token=TOKEN)) as c:  # type: ignore[arg-type]
        assert c.get("/api/health").status_code == 503


def test_publish_sends_the_report_to_the_api(
    tmp_path: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: dict[str, Any] = {}

    def fake_post(api: str, token: str, body: dict[str, Any], timeout: float = 60.0) -> str:
        sent.update(api=api, token=token, body=body)
        r = client.post("/api/runs", json=body, headers={"Authorization": f"Bearer {token}"})
        if r.status_code != 201:
            raise RuntimeError(r.json()["detail"])
        return str(r.json()["id"])

    monkeypatch.setattr(pub, "post_run", fake_post)
    rep = tmp_path / "r.json"
    rep.write_text(json.dumps(report()))
    base = ["--profile", "smoke", "--runs", str(tmp_path), "--api", "http://x/api", "--label", "smoke"]
    ok = CliRunner().invoke(cli, ["publish", *base, "--token", TOKEN, "--report", str(rep)])
    assert ok.exit_code == 0, ok.output
    assert "published run" in ok.output and sent["body"]["label"] == "smoke"
    assert client.get("/api/runs").json()[0]["label"] == "smoke"

    older = ["--commit", "3438254", "--gpu", "NVIDIA A100-SXM4-40GB"]
    again = CliRunner().invoke(cli, ["publish", *base, "--token", TOKEN, "--report", str(rep), *older])
    assert again.exit_code == 0, again.output
    assert sent["body"]["git_commit"] == "3438254" and sent["body"]["gpu"] == "NVIDIA A100-SXM4-40GB"

    refused = CliRunner().invoke(cli, ["publish", *base, "--token", "wrong", "--report", str(rep)])
    assert refused.exit_code == 1 and "not published" in refused.output and str(rep) in refused.output


def test_post_run_explains_http_and_network_failures(store: RunStore) -> None:
    with pytest.raises(RuntimeError, match="couldn't reach"):
        pub.post_run("http://127.0.0.1:9/api", TOKEN, {"report": report()}, timeout=2)


def test_post_run_reports_the_servers_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import urllib.error

    def refuse(*_: Any, **__: Any) -> None:
        raise urllib.error.HTTPError(
            "http://x/api/runs", 401, "Unauthorized", {}, io.BytesIO(b'{"detail": "bad token"}')  # type: ignore[arg-type]
        )

    monkeypatch.setattr(pub.urllib.request, "urlopen", refuse)
    with pytest.raises(RuntimeError, match=r"HTTP 401.*bad token"):
        pub.post_run("http://x/api", TOKEN, {})


def test_post_run_returns_the_new_id(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    class Res(io.BytesIO):
        def __enter__(self) -> Res:
            return self

        def __exit__(self, *_: object) -> None:
            pass

    seen: dict[str, Any] = {}

    def ok(req: Any, timeout: float) -> Res:
        seen.update(url=req.full_url, auth=req.headers["Authorization"])
        return Res(b'{"id": "2026-11-18T0200-ab12"}')

    monkeypatch.setattr(pub.urllib.request, "urlopen", ok)
    assert pub.post_run("http://x/api/", TOKEN, {}) == "2026-11-18T0200-ab12"
    assert seen == {"url": "http://x/api/runs", "auth": f"Bearer {TOKEN}"}


def test_run_metadata_helpers(tmp_path: Path) -> None:
    assert pub.gpu_name(tmp_path) == ""
    (tmp_path / "speed.json").write_text(json.dumps({"gpu": "NVIDIA A100-SXM4-40GB"}))
    assert pub.gpu_name(tmp_path) == "NVIDIA A100-SXM4-40GB"
    assert len(pub.git_commit(Path(__file__).parent)) >= 7
    assert pub.git_commit(tmp_path) == ""
