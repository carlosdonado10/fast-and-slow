"""The report's read API, plus one write route that Colab (or a laptop) uses to publish a run.

    GET  /api/health          the stack's readiness check (pings MongoDB)
    GET  /api/runs            run summaries, newest first (no reports: the run picker needs only these)
    GET  /api/runs/latest     the newest run with its report
    GET  /api/runs/{id}       one run with its report
    POST /api/runs            publish a run; needs `Authorization: Bearer <UPLOAD_TOKEN>`

Reads are public (the report is public); writes need the upload token. MongoDB is never exposed: only this service
talks to it, inside the compose network. Run it with `uvicorn fast_and_slow.api:app` (env: MONGO_URI, UPLOAD_TOKEN).
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from .store import InvalidReport, RunStore

MAX_UPLOAD_BYTES = 12 * 2**20  # MongoDB caps a document at 16 MB; a full report is well under 1 MB


class Publish(BaseModel):
    report: dict[str, Any]
    label: str = Field("", max_length=120)
    git_commit: str = Field("", max_length=64)
    gpu: str = Field("", max_length=120)


def _runs(request: Request) -> RunStore:
    return request.app.state.store


Store = Annotated[RunStore, Depends(_runs)]  # module level: FastAPI resolves string annotations from module globals


def create_app(store: RunStore | None = None, upload_token: str | None = None) -> FastAPI:
    """With no store, connect to MONGO_URI at startup (the container); tests pass an in-memory one."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        client: Any = None
        if store is None:
            from pymongo import MongoClient

            client = MongoClient(
                os.environ.get("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=3000
            )
            app.state.store = RunStore(client[os.environ.get("MONGO_DB", "fast_and_slow")]["runs"])
        else:
            app.state.store = store
        app.state.store.ensure_indexes()
        yield
        if client is not None:
            client.close()

    app = FastAPI(title="Fast and Slow runs", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")
    token = upload_token if upload_token is not None else os.environ.get("UPLOAD_TOKEN", "")

    def authorised(authorization: str = Header("")) -> None:
        if not token:
            raise HTTPException(503, "Uploads are switched off: the server has no UPLOAD_TOKEN set.")
        scheme, _, given = authorization.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(given.encode(), token.encode()):
            raise HTTPException(401, "A valid upload token is needed to publish a run.")

    @app.get("/api/health")
    def health(store: Store) -> dict[str, str]:
        try:
            store.runs.database.command("ping")
        except Exception as e:
            raise HTTPException(503, f"MongoDB is not reachable: {type(e).__name__}") from e
        return {"status": "ok"}

    @app.get("/api/runs")
    def list_runs(store: Store) -> list[dict[str, Any]]:
        return store.list()

    @app.get("/api/runs/latest")
    def latest_run(store: Store) -> dict[str, Any]:
        run = store.latest()
        if run is None:
            raise HTTPException(404, "No runs have been published yet.")
        return run

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str, store: Store) -> dict[str, Any]:
        run = store.get(run_id)
        if run is None:
            raise HTTPException(404, f"There is no run {run_id!r}.")
        return run

    @app.post("/api/runs", status_code=201, dependencies=[Depends(authorised)])
    def publish(body: Publish, request: Request, store: Store) -> dict[str, str]:
        if int(request.headers.get("content-length", "0")) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "That report is too large to store as one run.")
        try:
            run_id = store.add(body.report, body.label, body.git_commit, body.gpu)
        except InvalidReport as e:
            raise HTTPException(422, str(e)) from e
        return {"id": run_id}

    return app


app = create_app()
