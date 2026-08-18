"""The live probe (M2-01): one world in → canonical .rrd out, on a scale-to-zero GPU
container, with the JAX compilation cache WARMED AT IMAGE BUILD so cold containers skip
the ~100 s compile measured in the batch runs.

Cache design (verified against docs.jax.dev persistent_compilation_cache, 2026-08-15):
JAX_COMPILATION_CACHE_DIR + XLA GPU caches (jax_persistent_cache_enable_xla_caches) —
the cache key includes the GPU name, so the warm-up build step runs on the SAME GPU
class as serving (gpu= on run_function). The warm-up traces the exact probe program
(same shapes, same jit) so serving hits the cache byte-for-byte.

Physics/logging are the shared code paths (sweep/injected_env.push_step, probe/
rrd_logger) — the probe simulates the same physics G3 verified and writes the same
artifact the replay batch writes. Deterministic policy; env reset PRNG = the only
stochasticity source (spec §6). Canonical trajectory persisted beside the .rrd
(clip = replay = probe, by construction — Session 21).

    uv run modal deploy probe/live.py
    uv run modal run probe/live.py::measure --run-id 20260813T155006Z   # latency study
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import modal

from train.modal_app import VOLUME_MOUNT, base_image, checkpoints

RUN_ID = "20260813T155006Z"  # G2 checkpoint run (v0 = baseline policy)
CHECKPOINT = "converged"
CACHE_DIR = "/jax-cache"

# Warm-up traces the exact program the probe runs; the compiled artifacts land in the
# image layer at CACHE_DIR. Must run on the serving GPU class (cache key = GPU name).
PROBE_GPU = "A100-80GB"


def _expose_ptxas() -> None:
    """Image-build step: put the pip-wheel ptxas (nvidia-cuda-nvcc-cu12, already
    installed by jax[cuda12]) on PATH so XLA's kernel-cache subprocess mode can spawn
    it (Session 24: ENABLE_XLA_CACHES=all failed with RET_CHECK process.Start())."""
    import glob
    import os

    hits = glob.glob("/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin/ptxas")
    assert hits, "ptxas not found in the nvidia-cuda-nvcc wheel"
    os.symlink(hits[0], "/usr/local/bin/ptxas")
    print("ptxas exposed:", hits[0])


def _warm_compile_cache() -> None:
    """Image-build step: run one probe end-to-end so every jit/scan is compiled into the
    persistent cache baked into this layer."""
    import os

    os.environ["JAX_COMPILATION_CACHE_DIR"] = CACHE_DIR
    os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
    from probe.probe_core import ProbeRuntime

    rt = ProbeRuntime(run_id=RUN_ID, checkpoint=CHECKPOINT, volume_root=None)
    # No Volume at build time: load params from the baked-in copy (see image below).
    rt.warm_up()
    n = sum(len(files) for _, _, files in os.walk(CACHE_DIR))
    print(f"compile cache warmed: {n} files under {CACHE_DIR}")
    assert n > 0, "compile cache is empty after warm-up — caching is not engaging"


probe_image = (
    base_image.uv_pip_install("pydantic==2.13.4", "rerun-sdk==0.36.0")
    .env(
        {
            "JAX_COMPILATION_CACHE_DIR": CACHE_DIR,
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
            # "all" also persists XLA's GPU kernel + autotune caches — the residual
            # 18 s of cold "sim" measured in Session 24. Needs ptxas on PATH (below).
            # XLA kernel-cache persistence ("ENABLE_XLA_CACHES=all") is OFF: its
            # subprocess ptxas/linker path fails in this image even with ptxas on PATH
            # and XLA's own suggested flag (builds 1,3,4 — Session 24/25). Time-boxed.
            # Lever 2 (autotune_level=0) MEASURED and REVERTED: cold sim stayed
            # 18-21 s on genuinely cold containers, warm sim unchanged — the residual
            # cold cost is not autotuning (DEVLOG Session 25). Next lever: memory
            # snapshot (container-ready + runtime-init) and a compile-cache hit audit.
        }
    )
    # Policy params baked into the image so warm-up (no Volume at build) and serving
    # (no Volume read on the hot path) both load from disk instantly.
    .add_local_dir("probe/params", remote_path="/params", copy=True)
    .add_local_python_source("train", "sweep", "configs", "probe", "reduce", copy=True)
    # run_function steps import this module in-container → local source must already
    # be present (first attempt ordered these the other way: ModuleNotFoundError).
    .run_function(_warm_compile_cache, gpu=PROBE_GPU)
)

app = modal.App("opw-probe-live")


@app.function(
    image=probe_image,
    gpu=PROBE_GPU,
    timeout=120,
    volumes={VOLUME_MOUNT: checkpoints},
    scaledown_window=60,
)
@modal.concurrent(max_inputs=1)
def probe(world_json: str) -> dict[str, Any]:
    """One probe. `world_json` is a serialized WorldConfig (already clamped by app/)."""
    from probe.probe_core import ProbeRuntime

    t0 = time.monotonic()
    rt = ProbeRuntime.get(run_id=RUN_ID, checkpoint=CHECKPOINT, volume_root=Path(VOLUME_MOUNT))
    t_ready = time.monotonic()
    result = rt.run(json.loads(world_json))
    result["timings_s"]["container_ready_s"] = round(t_ready - t0, 3)
    result["timings_s"]["total_s"] = round(time.monotonic() - t0, 3)
    checkpoints.commit()
    return result


@app.local_entrypoint()
def measure(cold_starts: int = 5, warm_calls: int = 20) -> None:
    """Latency study: N cold starts (each waits out the scaledown window) + N warm calls.
    Prints per-call timings + p50/p95; the caller writes docs/measurements/."""
    from configs.world import WorldConfig

    def call(seed: int) -> dict[str, Any]:
        w = WorldConfig.clamped(friction=0.5, push_magnitude_pct_bw=90.0, seed=seed)
        t0 = time.monotonic()
        r = probe.remote(w.model_dump_json())
        r["client_wall_s"] = round(time.monotonic() - t0, 3)
        return r

    cold: list[dict[str, Any]] = []
    for i in range(cold_starts):
        if i:
            time.sleep(75)  # > scaledown_window → next call is a genuine cold start
        cold.append(call(seed=100 + i))
        print("cold", json.dumps(cold[-1]["timings_s"] | {"wall": cold[-1]["client_wall_s"]}))
    warm: list[dict[str, Any]] = [call(seed=200 + i) for i in range(warm_calls)]
    for r in warm:
        print("warm", json.dumps(r["timings_s"] | {"wall": r["client_wall_s"]}))

    def pct(xs: list[float], q: float) -> float:
        xs = sorted(xs)
        return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]

    cw = [r["client_wall_s"] for r in cold]
    ww = [r["client_wall_s"] for r in warm]
    print(
        json.dumps(
            {
                "cold": {"n": len(cw), "p50": pct(cw, 0.5), "p95": pct(cw, 0.95), "max": max(cw)},
                "warm": {"n": len(ww), "p50": pct(ww, 0.5), "p95": pct(ww, 0.95), "max": max(ww)},
                "budget": {"warm_s": 10, "cold_s": 20, "spawn_poll_trigger_cold_p95_s": 30},
            },
            indent=2,
        )
    )
