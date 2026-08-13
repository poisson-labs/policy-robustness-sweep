# G1 — Modal capability verification

**Gate:** spec §4 G1. **Method:** every finding below was read from Modal's live
documentation on 2026-08-13 (fetch date). Each section cites the URL it was read from.
Findings from secondary (non-modal.com) sources are explicitly marked UNVERIFIED and must
be re-checked against Modal's own docs/CLI before being relied on. Nothing here comes from
training-data assumptions; where an assumption conflicted with the docs, the docs won and
the discrepancy is listed at the bottom and in DEVLOG.

## 1. Web endpoints / ASGI (probe path + app serving)

Source: https://modal.com/docs/guide/webhooks

- Decorators: `@modal.asgi_app()` (FastAPI/Starlette — our `app/` path),
  `@modal.fastapi_endpoint()`, `@modal.wsgi_app()`, `@modal.web_server(port)`.
- Request bodies up to 4 GiB; response bodies unlimited in size; WebSocket messages up to
  2 MiB.
- Per-container request concurrency via `@modal.concurrent(max_inputs=N)`; beyond that,
  Modal scales horizontally with more containers.
- Account-level rate limit: "for a new account, this is set to 200 Function calls or HTTP
  requests per second, with a burst multiplier of 5 seconds"; excess requests get HTTP 429.
  **Launch-relevant:** this is an account-level ceiling on top of our own per-IP limiting;
  429 behavior must be folded into the degradation mode (spec §7) and the M4 cap drill.

## 2. Max request duration (probe path budget)

Source: https://modal.com/docs/guide/webhook-timeouts

- Hard cap: "all Web Function types have a maximum HTTP request timeout of 150 seconds."
- Past 150 s, Modal issues an HTTP 303 redirect chain (browser-dependent, breaks under
  CORS); docs recommend spawn + poll (endpoint returns a call ID; a poll endpoint returns
  202 while pending) for genuinely long work.
- **Implication for us:** the probe budget (spec §7: < 10 s warm / < 20 s cold target) fits
  comfortably inside a single 150 s request window. A synchronous `POST /probe` is viable;
  spawn+poll is the documented fallback if measured cold paths ever threaten the window.
  No redirect-chain reliance — it breaks under CORS, which a fetch()-driven frontend hits.

Source: https://modal.com/docs/guide/timeouts

- Non-web Function execution timeout: default 300 s, configurable 1 s – 24 h via
  `@app.function(timeout=...)`; timeout raises `modal.exception.FunctionTimeoutError`
  (catchable); enforcement is "at least as long", may overrun by seconds.
- **Implication:** train/ and sweep/ runs configure explicit long timeouts; the probe
  function sets a short one so a hung rollout can't burn GPU-seconds silently.

## 3. Scale-to-zero billing (the receipt's idle-$0 claim)

Source: https://modal.com/pricing (rates as published on fetch date; the receipt will use
billed amounts from our actual runs, not this table)

- Per-second billing; "you never pay for idle resources." Scale-to-zero idle cost is $0 —
  the spec §2 receipt claim is supported as documented behavior; the receipt itself still
  reports our measured bill.
- GPU $/s (published): T4 $0.000164 · L40S $0.000542 · A100-40GB $0.000583 ·
  A100-80GB $0.000694 · H100 SXM5 $0.001097 · H200 $0.001261 · B200 $0.001736 ·
  B300 $0.001972. CPU $0.0000131 per physical core-second (min 0.125 cores);
  memory $0.00000222 per GiB-second.
- Volume storage: $0.09/GiB/month with 1 TiB/month included free (per pricing page).
- Free credits: Starter $30/month, Team $100/month — relevant to the M0 budget question in
  NOTES.md.
- Egress charges: not mentioned on the pricing page (re-check before launch; .rrd serving
  volume at HN scale is the concern).

## 4. Volumes + serving `.rrd` (feeds §14 decision #2)

Source: https://modal.com/docs/guide/volumes

- `modal.Volume.from_name(name, create_if_missing=True)`, mounted via
  `@app.function(volumes={"/mount/path": volume})`. Volumes v2 via `version=2`.
- Consistency: explicit `.commit()` (plus automatic background commits "every few
  seconds"); other containers see changes only after `.reload()`; concurrent same-file
  writes are last-write-wins. **Implication:** probe writes `.rrd` then commits before
  returning its URL; the app container must reload before serving a fresh file — or we
  route the bytes back through the probe function's return path / have the app read after
  an explicit reload. Cache design must respect this.
- Limits: v1 — 500,000 inode hard cap, ~50,000 files recommended; v2 — no total file
  limit, < 1 TiB per file, 262,144 files per directory. **Use v2** — a popular probe day
  plus the sweep's replay set could plausibly exceed v1's comfortable range.
- The volumes guide documents no direct public-HTTP access to Volume files.

Serving patterns actually documented:
- ASGI app reads from its mounted Volume and serves bytes (FastAPI `FileResponse` /
  `StaticFiles`); static frontend assets baked into the image via `Image.add_local_dir`
  (pattern shown in https://modal.com/docs/examples/doc_ocr_webapp and
  https://modal.com/docs/guide/webhooks).
- `modal.CloudBucketMount` (S3 / Cloudflare R2 / GCS) — read/write bucket mounts
  (source: https://modal.com/docs/guide/cloud-bucket-mounts). Writing `.rrd`s to R2 and
  serving through R2's public/CDN layer is the documented escape hatch if serving through
  the app becomes a latency or egress problem. Constraints: sequential-write oriented
  (no append/seek-and-write), credentials via Modal Secrets.
- UNVERIFIED (secondary sources — a modal-client code walkthrough and a release-notes
  aggregator, not modal.com docs): Volume API may expose pre-signed HTTP GET URLs for
  files; a `@app.server()` primitive may have landed mid-2026. Re-verify both against
  modal.com reference docs before letting either into the design; neither is needed for
  the default plan.

**Recommendation for §14 #2 (decision itself needs Taylor's sign-off, per working
agreement):** file-from-Volume (v2) served by the FastAPI app, `.rrd` URL returned by
`POST /probe` — the spec's default, confirmed boring and documented. R2 CloudBucketMount
held as the CDN upgrade path. No streaming: nothing in the fetched docs makes streaming
`.rrd` simpler or cheaper than file-then-fetch, and the viewer consumes a file URL anyway.

## 5. `.map` / batching API shape (sweep + reduce fan-out)

Source: https://modal.com/docs/guide/scale

- `Function.map()` over independent inputs (`return_exceptions=True` available),
  `.starmap()` for tuple-unpacking, `.spawn()` for async jobs.
- Quotas: 2,000 pending inputs per Function; 25,000 total (running + pending); 1,000
  concurrent inputs per `.map` invocation; 1M pending for `.spawn`.
- Autoscaler knobs: `max_containers`, `min_containers`, `buffer_containers`,
  `scaledown_window`; runtime changes via `Function.update_autoscaler()` (the documented
  mechanism for §14 #4 launch-day min-warm without redeploying).
- **Implication for sweep:** our sweep is vmap-batched on a single GPU (spec §3), so
  `.map` quotas are irrelevant to the grid itself; `.map`/`.spawn` matter for reduce/render
  fan-out and stay far under quota there.

Source: https://modal.com/docs/guide/dynamic-batching

- `@modal.batched(max_batch_size=..., wait_ms=...)` on async functions; list-in/list-out,
  equal lengths; fires on size or wait threshold. Noted for completeness — not needed for
  v1 (probes are single rollouts; sweep is vmap inside one function).

## 6. Cold start + JIT tax (probe latency budget, spec §13 risk)

Source: https://modal.com/docs/guide/cold-start

- Containers boot "in about one second"; cold-start latency = queueing + initialization.
- `scaledown_window` default 60 s, configurable up to 20 min — a free-ish warm window
  after each probe burst.
- Recommended: move heavy init into the image or `@modal.enter`; our JAX persistent
  compilation cache baked into the image (kickoff §4) is exactly this pattern.

Source: https://modal.com/docs/guide/memory-snapshot

- `enable_memory_snapshot=True`; "initialization-heavy Functions often start up 3-10x
  faster." GPU-state snapshots are **alpha** (`experimental_options=
  {"enable_gpu_snapshot": True}`) — snapshots also capture PRNG state (a determinism
  hazard worth noting for our seed policy) and are invalidated by code changes.
- **Implication:** CPU memory snapshot is a candidate lever if the measured probe
  cold-start misses budget; GPU snapshot is alpha and stays off the critical path.
  Decision deferred until the G3/G4 measured numbers exist.

## 7. Images + GPUs (train/sweep/probe containers)

Source: https://modal.com/docs/guide/images

- Chained builder from `modal.Image.debian_slim()`; **`Image.uv_pip_install()` is the
  recommended installer** (`pip_install` is the fallback); `apt_install`, `env`,
  `run_commands`, `run_function` for build steps; `add_local_dir`/`add_local_file`/
  `add_local_python_source` for local code; per-layer caching, `force_build=True`.
- Docs strongly recommend tight pins (e.g. `"torch==2.0"`-style) — matches our pinning
  policy; the JAX/CUDA pins will be recorded in README when the image lands (G2).

Source: https://modal.com/docs/guide/gpu

- `@app.function(gpu="...")` with types: T4, L4, A10, L40S, A100 / A100-40GB / A100-80GB,
  RTX-PRO-6000, H100 (H100! opts out of free H200 upgrade), H200, B200, B300; multi-GPU
  via `"H100:8"`; fallback lists like `gpu=["H100", "A100-40GB:2"]`.
- Hardware choice for sweep (§14 #3) stays a G4-measured decision; the fallback-list
  feature is useful if the first choice has capacity issues.

## Assumption-vs-docs discrepancies (docs won)

1. Web request cap is 150 s with a 303-redirect continuation mechanism — not a simple
   fixed timeout as assumed; the redirect chain's CORS incompatibility is the load-bearing
   detail for a fetch()-based frontend.
2. `Image.uv_pip_install()` now exists and is the recommended install path — pre-verification
   assumption was pip-based image builds with uv only as a local tool.
3. Volumes now have a v2 with materially different limits (v1's 500k-inode cap vs v2's
   per-directory limits); v1/v2 choice is a real decision, resolved above as v2.
4. Account-level rate limit (200 req/s, burst ×5) exists above any app-level limiting —
   launch-traffic planning assumed only self-imposed limits.
