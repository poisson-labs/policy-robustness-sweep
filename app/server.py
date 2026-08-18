"""app/server.py — the Modal ASGI app (M2-02). Thin: routes → app/core.ProbeService.

Endpoints (spec §3/§7):
- POST /probe      validated+clamped world → cached .rrd URL ($0) or live probe
- GET  /rrd/{key}  canonical replay bytes from the Volume (never deleted once served)
- GET  /manifest   the committed KM cell manifest + boundary uncertainty (static)
- GET  /receipt    measured numbers only (from committed measurement records)
- GET  /health     mode + limits + today's counters (the degradation banner source)
- GET  /           mounts the G5 test page for now; the M3 bundle replaces it

    uv run modal deploy app/server.py
"""

import json
from pathlib import Path
from typing import Any

import modal

VOLUME_MOUNT = "/vol"
checkpoints = modal.Volume.from_name("opw-checkpoints", create_if_missing=True, version=2)

RECEIPT_SOURCES = [
    "docs/measurements/2026-08-13-g2-full-run.md",
    "docs/measurements/2026-08-13-g4-full-sweep.md",
    "docs/measurements/2026-08-18-probe-latency.md",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi[standard]==0.141.1", "pydantic==2.13.4")
    .add_local_dir("docs/measurements", remote_path="/assets/measurements", copy=True)
    .add_local_dir("web/g5", remote_path="/assets/g5", copy=True)
    .add_local_python_source("configs", "app", copy=True)
)

app = modal.App("opw-app")


@app.function(
    image=image,
    volumes={VOLUME_MOUNT: checkpoints},
    scaledown_window=300,
    min_containers=0,
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def serve():  # type: ignore[no-untyped-def]
    import contextvars

    from fastapi import FastAPI, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from app.core import Limits, ProbeService

    request_ip: contextvars.ContextVar[str] = contextvars.ContextVar("request_ip", default="?")

    probe_cls = modal.Cls.from_name("opw-probe-live", "Probe")
    rrd_dir = Path(VOLUME_MOUNT) / "rrd"
    analytics = Path(VOLUME_MOUNT) / "analytics" / "probes.jsonl"

    def backend(world_json: str) -> dict[str, Any]:
        return probe_cls().probe.remote(world_json)

    def rrd_exists(key: str) -> bool:
        checkpoints.reload()
        return (rrd_dir / f"{ProbeService.safe_key(key)}.rrd").exists()

    service = ProbeService(
        limits=Limits(),  # probe_cost_usd stays None until a billed number exists
        backend=backend,
        rrd_exists=rrd_exists,
        rrd_url=lambda key: f"/rrd/{ProbeService.safe_key(key)}.rrd",
        analytics_path=analytics,
    )

    api = FastAPI(title="One Policy, 400 Worlds — probe API")

    @api.post("/probe")
    async def probe(request: Request) -> JSONResponse:
        # Read the body async (cheap), then run the BLOCKING service in the threadpool —
        # the first deploy ran blocking Modal calls on the event loop and Modal cancelled
        # the stalled input (500). Body parsed by hand so garbage JSON degrades to
        # defaults (200) rather than FastAPI's 422 — hostile input is data, not an error.
        try:
            raw_bytes = await request.body()
        except Exception:
            raw_bytes = b""
        try:
            parsed = json.loads(raw_bytes or b"{}")
        except Exception:
            parsed = {}
        raw: dict[str, Any] = parsed if isinstance(parsed, dict) else {}
        ip = request_ip.get()
        return await run_in_threadpool(_probe, raw, ip)

    def _probe(raw: dict[str, Any], ip: str) -> JSONResponse:
        status, out = service.handle(raw, ip=ip)
        return JSONResponse(out, status_code=status)

    @api.middleware("http")
    async def capture_ip(request: Request, call_next):  # type: ignore[no-untyped-def]
        fwd = request.headers.get("x-forwarded-for")
        ip = fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")
        token = request_ip.set(ip)
        try:
            return await call_next(request)
        finally:
            request_ip.reset(token)

    @api.get("/rrd/{name}")
    def rrd(name: str) -> FileResponse:
        from fastapi import HTTPException

        if "/" in name or ".." in name or not name.endswith(".rrd"):
            raise HTTPException(status_code=404)
        checkpoints.reload()
        path = rrd_dir / name
        if not path.exists():
            raise HTTPException(status_code=404)
        return FileResponse(path, media_type="application/octet-stream")

    @api.get("/manifest")
    def manifest() -> JSONResponse:
        m = json.loads(
            Path("/assets/measurements/2026-08-13-cell-survival-manifest.json").read_text()
        )
        b = json.loads(
            Path("/assets/measurements/2026-08-15-boundary-uncertainty.json").read_text()
        )
        return JSONResponse({"cells": m, "boundary": b})

    @api.get("/receipt")
    def receipt() -> JSONResponse:
        # Measured numbers only: served verbatim from the committed measurement records.
        docs = {Path(p).name: Path("/assets/measurements") / Path(p).name for p in RECEIPT_SOURCES}
        return JSONResponse(
            {
                "policy": "every number here is measured; missing values are MEASURED_TBD",
                "sources": {k: v.read_text() for k, v in docs.items() if v.exists()},
                "probe_cost_usd": service.limits.probe_cost_usd,
                "idle_cost_usd": 0.0,
                "note_idle": (
                    "scale-to-zero outside launch week; launch week keeps "
                    "min_containers=1 (disclosed)"
                ),
            }
        )

    @api.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {
                "mode": service.mode(),
                "banner": service.banner(),
                "limits": {
                    "per_ip_per_minute": service.limits.per_ip_per_minute,
                    "global_concurrency": service.limits.global_concurrency,
                    "daily_spend_cap_usd": service.limits.daily_spend_cap_usd,
                },
                "probes_today": service.ledger.probes_today(),
                "spent_today_usd": service.ledger.spent_today(),
            }
        )

    api.mount("/", StaticFiles(directory="/assets/g5", html=True), name="page")
    return api
