"""Red-team web API (contract draft).

Catalog and configuration validation are implemented. The remaining routes fix
their request and response models so the web UI can be built against mocks;
they answer 501 until the red-team runner stores episodes, traces and reports.
"""
from fastapi import APIRouter, HTTPException, Query

from wms.redteam.catalog import load_catalog
from wms.redteam.schema import (Catalog, Draft, EpisodeDetail, EpisodePage, ExperimentConfig, Matrix, Report,
                                RunProgress, TargetRef, TargetTestResult, Trace, ValidationResult, Verdict)
from wms.redteam.validation import URL, validate_config


def _pending(what):
    raise HTTPException(501, f"{what} is part of the red-team API contract and is not implemented yet")


def create_redteam_router(services):
    router = APIRouter(prefix="/api/v1", tags=["redteam"])

    @router.get("/catalog", response_model=Catalog)
    def catalog():
        return load_catalog()

    @router.post("/experiments/validate", response_model=ValidationResult)
    def validate(config: ExperimentConfig):
        return validate_config(config, load_catalog())

    @router.post("/targets/test", response_model=TargetTestResult)
    def test_target(target: TargetRef):
        known = {t.id: t for t in load_catalog().targets}
        if target.id not in known:
            raise HTTPException(404, f"Unknown target: {target.id}")
        if known[target.id].requires_endpoint and not URL.match((target.endpoint or "").strip()):
            return TargetTestResult(ok=False, message="请填写有效的策略服务地址")
        _pending("Target connectivity test")

    @router.get("/experiments/drafts/{draft_id}", response_model=Draft)
    def get_draft(draft_id: str):
        _pending("Draft storage")

    @router.put("/experiments/drafts/{draft_id}", response_model=Draft)
    def put_draft(draft_id: str, config: ExperimentConfig):
        _pending("Draft storage")

    @router.post("/runs/{run_id}/pause", response_model=RunProgress)
    def pause(run_id: str):
        _pending("Pause")

    @router.post("/runs/{run_id}/resume", response_model=RunProgress)
    def resume(run_id: str):
        _pending("Resume")

    @router.get("/runs/{run_id}/progress", response_model=RunProgress)
    def progress(run_id: str):
        _pending("Run progress")

    @router.get("/runs/{run_id}/events", responses={200: {"content": {"text/event-stream": {}}, "description": "Server-sent RunEvent objects"}})
    def events(run_id: str):
        _pending("Run event stream")

    @router.get("/runs/{run_id}/episodes", response_model=EpisodePage)
    def episodes(run_id: str, task: str | None = None, strategy: str | None = None, defense: str | None = None,
                 verdict: Verdict | None = None, q: str | None = None, include_baseline: bool = True,
                 cursor: str | None = None, limit: int = Query(100, ge=1, le=500)):
        _pending("Episode listing")

    @router.get("/runs/{run_id}/episodes/{episode_id}", response_model=EpisodeDetail)
    def episode(run_id: str, episode_id: str):
        _pending("Episode detail")

    @router.get("/runs/{run_id}/episodes/{episode_id}/trace", response_model=Trace)
    def trace(run_id: str, episode_id: str):
        _pending("Episode trace")

    @router.get("/runs/{run_id}/report", response_model=Report)
    def report(run_id: str):
        _pending("Vulnerability report")

    @router.get("/runs/{run_id}/matrix", response_model=Matrix)
    def matrix(run_id: str, defense: str = "none"):
        _pending("Comparison matrix")

    return router
