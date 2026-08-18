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


@app.cls(
    image=probe_image,
    gpu=PROBE_GPU,
    timeout=120,
    volumes={VOLUME_MOUNT: checkpoints},
    scaledown_window=60,
    # Lever 4 (M2-01b): GPU memory snapshot. The persistent compile cache is
    # STRUCTURALLY defeated for jit(rollout) on this mjx build — its FFI custom calls
    # bake host/device pointers into the HLO as constants, so the cache key differs in
    # every process (Sessions 25/26; audit code lived here, now in git history). A
    # snapshot taken AFTER the JIT restores a process that already holds the compiled
    # executable, skipping compile entirely.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
@modal.concurrent(max_inputs=1)
class Probe:
    @modal.enter(snap=True)
    def compile(self) -> None:
        from probe.probe_core import ProbeRuntime

        t0 = time.monotonic()
        self.rt = ProbeRuntime.get(run_id=RUN_ID, checkpoint=CHECKPOINT, volume_root=None)
        self.rt.warm_up()  # JIT + first run happen BEFORE the snapshot
        self.snap_prep_s = round(time.monotonic() - t0, 3)

    @modal.method()
    def probe(self, world_json: str) -> dict[str, Any]:
        """One probe. `world_json` is a serialized WorldConfig (already clamped by app/)."""
        t0 = time.monotonic()
        result = self.rt.run(json.loads(world_json))
        result["timings_s"]["container_ready_s"] = 0.0
        result["timings_s"]["snapshot_prep_s"] = self.snap_prep_s
        result["timings_s"]["total_s"] = round(time.monotonic() - t0, 3)
        checkpoints.commit()
        return result


@app.function(image=probe_image, gpu=PROBE_GPU, timeout=600, volumes={VOLUME_MOUNT: checkpoints})
def dump_hlo(tag: str) -> str:
    """Audit helper: write this process's lowered HLO for the canonical world to the
    Volume so two processes' programs can be diffed (M2-01b cache-key hunt)."""
    from configs.world import WorldConfig
    from probe.probe_core import ProbeRuntime

    rt = ProbeRuntime.get(run_id=RUN_ID, checkpoint=CHECKPOINT, volume_root=Path(VOLUME_MOUNT))
    text = rt.hlo_text(WorldConfig.clamped(friction=0.5, push_magnitude_pct_bw=90.0))
    out = Path(VOLUME_MOUNT) / "audit" / f"hlo-{tag}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    checkpoints.commit()
    return f"{len(text)} chars"


@app.function(image=probe_image, gpu=PROBE_GPU, timeout=600)
def model_leaf_audit() -> list[str]:
    """Audit: which mjx.Model leaves are int32 arrays of the suspicious shapes seen as
    per-process constants in the HLO diff (18/40/14/12-wide int32 pointer pairs)?"""
    import jax
    import numpy as np

    from probe.probe_core import ProbeRuntime

    rt = ProbeRuntime.get(run_id=RUN_ID, checkpoint=CHECKPOINT, volume_root=None)
    out: list[str] = []
    leaves = jax.tree_util.tree_leaves_with_path(rt.base_model)
    out.append(f"model leaves: {len(leaves)}")
    kinds: dict[str, int] = {}
    for path, leaf in leaves:
        name = jax.tree_util.keystr(path)
        if not hasattr(leaf, "shape"):
            out.append(f"NON-ARRAY LEAF {name}: {type(leaf).__name__} {str(leaf)[:60]}")
            continue
        arr = np.asarray(leaf)
        kinds[str(arr.dtype)] = kinds.get(str(arr.dtype), 0) + 1
        if (
            arr.dtype.kind in "iu"
            and arr.size
            and abs(arr.ravel().astype(np.int64)).max() > 1 << 20
        ):
            out.append(f"BIG-INT LEAF {name} {arr.dtype} {arr.shape} max={int(abs(arr).max())}")
    out.append(f"dtype census: {kinds}")
    # env-level state that the closure captures: anything with pointer-like ints?
    for attr in ("_mjx_model", "mjx_model", "_mj_model", "mj_model"):
        obj = getattr(rt.env, attr, None)
        out.append(f"env.{attr}: {type(obj).__name__}")
    return out


@app.local_entrypoint()
def measure(cold_starts: int = 5, warm_calls: int = 20) -> None:
    """Latency study: N cold starts (each waits out the scaledown window) + N warm calls.
    Prints per-call timings + p50/p95; the caller writes docs/measurements/."""
    from configs.world import WorldConfig

    def call(seed: int) -> dict[str, Any]:
        w = WorldConfig.clamped(friction=0.5, push_magnitude_pct_bw=90.0, seed=seed)
        t0 = time.monotonic()
        r = Probe().probe.remote(w.model_dump_json())
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
