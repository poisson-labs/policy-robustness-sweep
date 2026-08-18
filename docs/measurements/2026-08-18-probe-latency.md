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

## Lever 1 — columnar Rerun logging (measured 2026-08-18, same study design)

Change: per-geom-per-step `rr.log` loop (~7500 calls) → `rr.send_columns` per entity
(~30 calls) + columnar arrow. Compile-cache config unchanged (JAX executable cache; the
XLA kernel-cache mode was time-boxed after three failed builds — see DEVLOG Session 25).

| | n | p50 | p95 | max | budget |
|---|---|---|---|---|---|
| cold | 5 | 35.4 s | 37.5 s | 37.5 s | < 20 s |
| warm | 20 | **1.8 s** | **2.1 s** | 2.1 s | < 10 s |

Warm breakdown: sim 0.56 s, **log 0.21–0.27 s** (was 4.1–4.5 s), function total
0.77–0.84 s, wall 1.6–2.1 s. Cold breakdown unchanged in shape: container ready
5.6–9.7 s, runtime init 2.0–3.8 s, **sim 12.2–21.1 s** (the residual compile —
lever 2's target), log ≤ 0.6 s.

Reading: warm is now dominated by Modal request overhead (~1 s), not by our code. The
cold miss is entirely the residual GPU compile inside "sim".

## Lever 2 — XLA GPU autotuning off (`xla_gpu_autotune_level=0`) — measured, REVERTED

Build-time compile fell 97.6 → 82.2 s (autotuning ≈ 15 s of the one-time compile), but
on genuinely cold containers **sim stayed 18.4–20.7 s** (one 11.3 s), warm sim unchanged
at 0.55 s. Cold p50 39.3 s / p95 68.9 s (one start hit 34 s runtime-init + 39 s
container-ready — image-pull/host variance, the largest single cold cost observed).
Conclusion: the residual cold cost is NOT autotuning; autotune-off buys nothing at
runtime and forgoes an optimization → reverted. Study note: one "cold" row (0.5 s sim)
was a container that had not scaled down in 75 s — real cold starts are the ≥18 s rows.

Where cold time actually goes (from the three clean cold rows): container ready
6.3–9.1 s, runtime init 2.6–2.9 s, sim 18.4–20.7 s (residual compile/cache-load),
Modal overhead ~5–8 s. Next levers: (a) audit whether the persistent cache is being HIT
at all in the serving process (log JAX cache hits/misses on cold), (b) memory snapshot
for container-ready + init.

## Lever 3 — cache-hit audit: root cause (measured 2026-08-18)

Instrumented the serving process (JAX cache logger, baked-key listing, key-component
hashes, HLO dumps from two processes):
- small programs → persistent-cache HITS; **`jit(rollout)` → MISS on every cold start**,
  writing a different key per process (≥4 keys observed for identical code).
- key components across two cold processes: XLA flags / backend / compile_options /
  accelerator_config / jax_lib all SAME; **`computation` DIFFERS**.
- HLO diff (4231 lines each): **12 differing lines — int32 constants (18/40/14/12-wide)
  holding 64-bit host pointers split into halves**, operands of the two
  `stablehlo.custom_call @jax_callable_variadic_tuple…` sites = mujoco-mjx 3.11.0's FFI
  bridge. Engine property, not our code: on this build the persistent compile cache
  structurally cannot hit the rollout program across processes.
- Traced-params change (params as jit arg, not closure) kept and correct (baked
  jit_rollout keys 2 → 1); the FFI pointers remained.

## Lever 4 — GPU memory snapshot (measured 2026-08-18) — THE FIX

`@app.cls(enable_memory_snapshot=True, experimental_options={"enable_gpu_snapshot":
True})`, JIT + warm-up in `@modal.enter(snap=True)`. Snapshot creation (one-time, at
deploy's first call): 165.8 s wall (snapshot_prep 18–27 s inside).

| | n | p50 | p95 | max | budget |
|---|---|---|---|---|---|
| cold (restored) | 3 genuine | 24.1 s | 35.1 s | 35.1 s | < 20 s |
| warm | 20 | 2.8 s | 5.2 s | 36.0 s* | < 10 s |

**On restored cold containers: sim 0.51–0.52 s, function total 0.7–1.6 s — the compile
is GONE.** All remaining cold wall (23–35 s) is snapshot restore + Modal scheduling,
outside our code. (*one warm outlier at 36 s: a call that landed on a fresh restore.)
Two study "cold" rows (1.8 s, 2.2 s walls) hit still-live containers — genuine restores
are the three ≥ 23 s rows.

## Standing after M2-01b

- Warm: **p50 2.8 s** (from 7.2 s at M2-01) — passes.
- Cold: **p50 ≈ 24 s** (from 39.5 s), p95 35 s — still misses < 20 s, but the residual is
  100% platform (snapshot restore), 0% our code. Snapshot restore time is Modal's
  number; the only lever left on our side would be image size (smaller image →
  faster restore) — worth one measured try in M4, not now.
- §14 #4: cold p95 35 s > 20 s → **min_containers=1 for launch week stands**,
  receipt-disclosed. Steady state: scale-to-zero. Post framing is now sharper: "a cold
  probe is ~24 s of *platform* restore + 1 s of physics; during launch week we keep one
  warm."
- spawn+poll trigger (p95 cold > 30 s): still nominally tripped at 35 s, but every
  cold call remains ≪ 150 s → sync path holds with min-warm; spawn+poll not built.
