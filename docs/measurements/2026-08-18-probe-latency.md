# Live probe latency — first measurement (M2-01)

Source: `probe/live.py::measure` against the deployed `opw-probe-live` app
(image with the JAX persistent compilation cache warmed at build; A100-80GB;
scaledown_window 60 s), 2026-08-18. Study: 5 cold starts (each after a 75 s idle so the
container had scaled to zero) + 20 warm calls, one boundary world
(μ 0.5, 90%BW, varying seed). Wall = client-observed round trip.

## Results

| | n | p50 | p95 | max | budget (spec §7) |
|---|---|---|---|---|---|
| cold | 5 | 39.5 s | 60.0 s | 60.0 s | < 20 s |
| warm | 20 | 7.2 s | 8.9 s | 9.7 s | < 10 s |

Warm breakdown (per call, ±small): runtime already resident; **sim 0.52 s**; `.rrd`
logging + save **4.1–4.5 s**; function total 4.6–5.0 s; client wall 6.8–9.7 s
(Modal request overhead ≈ 2–4 s).

Cold breakdown (first cold, representative): container ready 7.5 s (image pull +
process start), runtime init 2.2 s (params + env + policy build), **sim 18.0 s**,
logging 4.2 s, function total 30.4 s, wall 37.8 s. Warm-up at image build had reported
"compile + run 99.5 s" and left 24 files in the cache.

## Reading

- **Warm passes the budget** (p95 8.9 s < 10 s). The dominant warm cost is no longer
  physics (0.5 s) but the Rerun logging/save (~4.3 s) — a CPU loop over 250 steps ×
  every geom; that is the next optimization target and it is not GPU work.
- **Cold misses the budget** (p50 39.5 s, p95 60 s vs < 20 s). The compilation cache
  is doing *some* work — cold sim is 18 s, not the 99.5 s uncached compile — but it is
  not eliminating compile: 18 s of "sim" on a cold container is cache-load + residual
  compilation of parts the persistent cache does not cover (XLA autotuning of GPU
  fusions is the usual suspect; the kernel-cache mode that would capture it needs a
  ptxas subprocess this image cannot start — see DEVLOG Session 24). Container ready
  7.5 s and Modal overhead make up the rest.
- The p95 cold of 60 s comes from one slow start (image pull variance).

## Decisions triggered (recorded in DEVLOG Session 24)

- §14 #4 min-warm trigger: **p95 cold (60 s) > 20 s → `min_containers=1` for launch
  week, disclosed in the receipt as a line item** — unless the cold path is fixed
  before launch (M2-01b, below). Pure scale-to-zero remains the steady-state target.
- spawn+poll trigger: p95 cold 60 s > 30 s → **the synchronous request path is at
  risk on cold starts**; the 150 s platform cap still holds every observed cold call,
  so sync remains viable *with* min-warm. Spawn+poll is built only if the cold fix
  fails and min-warm is rejected.

## Follow-ups (M2-01b)

1. Fix the residual cold compile: get the XLA autotune/kernel cache to persist (install
   the CUDA toolkit's ptxas so `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all` can spawn
   it), or pin autotuning off (`xla_gpu_autotune_level=0`) and measure the sim-time
   cost; re-measure cold.
2. Cut the 4.3 s logging: log geom transforms per body (batch) instead of per geom per
   step; measure.
3. Memory snapshot (`enable_memory_snapshot`, G1 finding: "3–10×" claimed) as the
   reserve lever for container-ready + runtime-init (~9.7 s together).
