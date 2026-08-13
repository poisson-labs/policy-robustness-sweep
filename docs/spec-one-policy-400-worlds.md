# Project Spec: "One Policy, 400 Worlds" / "A Robot You Can Shove"

**Poisson Labs — technical specification for the flagship build.**

---

## 1. Summary

A live web artifact backed by real physics: a trained quadruped locomotion policy (MuJoCo Playground, Unitree Go1) whose failure boundary is mapped across a friction × lateral-push grid (precomputed, statistical), and which readers can probe at any off-grid world via sliders that trigger on-demand MJX rollouts in scale-to-zero Modal containers, with Rerun 3D replays and a measured cost receipt. Ships as a blog post whose central artifact is the experiment itself.

## 2. Goals / Non-Goals

**Goals (v1):**

- Precomputed robustness surface: 20×20 grid (friction μ × lateral push, % bodyweight), **16 seeds/cell**, censored-survival statistics, variance figure, diverged-region rendering.
- Live probe path: reader config → Modal GPU/CPU function → rollout → `.rrd` → embedded pinned Rerun viewer. Cached by rounded config. Visible cold-start/cost telemetry in UI.
- Perturbation set v1: push **magnitude, direction (dial), timing (slider)** + global friction. (Payload, gravity presets, leg damage, ice placement: deferred to devlog drip.)
- Shareable world URLs (config encoded in query params; page reconstructs schematic + result).
- Receipt: measured-only — sweep cost/time, per-probe cost, idle cost ($0.00), JIT/cold-start breakdown.
- Read-only degradation mode (spend cap / rate limit hit → map + cached replays stay fully functional; honest banner).
- Rendered video artifacts (see `render/`, §3): failure-centered clips (auto-trimmed push→fall, slow-mo), overlay/ghost renders (multi-rollout superposition: 16 seeds in one clip; same world across checkpoints), tiled composites (cross-boundary; 16-seed). Triple duty: launch kit, mobile clip-fallback, per-cell quick-look clips.
- Repo shaped as reusable pattern: `config → executor → reducer → artifact` separation, one-command sweep reproduction.

**Stretch (v1 if timeline allows, else first drip items after launch):**

- Velocity-command presets as a probe parameter (amble/march/fast/turning; clamped set) — the policy is natively command-conditioned, so this is a config field, not new architecture.
- Checkpoint selector on the probe path ("newborn / half-trained / final") — requires: cache key includes checkpoint id; G3 equivalence passed per exposed checkpoint. Heatmap/sweep remains final-checkpoint-only in v1; earlier-checkpoint sweeps are the training-progression figure (own post/devlog).

**Non-goals (v1, explicitly):**

- Crowd layer / persistent reader dots / leaderboard → **v1.1** (separate release; design persistence interface now, implement later).
- Failure-taxonomy clustering, adaptive sampling, VLM grading, drone plant → separate posts.
- Real-time interactive control (joystick/timed shove) → architecturally excluded; probes are parametric configs only.
- In-scene 3D editing; the Rerun viewer is output-only. Input is page UI + 2D SVG schematic.
- Training-time content beyond one devlog GIF.

## 3. System Architecture

Components (all Modal-deployed unless noted):

1. **`train/`** — Playground Go1 joystick-locomotion training (Brax PPO per Playground recipe) on one Modal GPU. Saves final + intermediate checkpoints to a Modal Volume. Run once.
2. **`sweep/`** — batch executor: full grid × seeds as **vmap batches on a single GPU** (batched over cells and seeds; chunk if memory-bound). Outputs one record per rollout: `{config, seed, ttf_or_censored, failure_flag, diverged_flag, margin_features, summary_states}`. No Rerun logging in sweep mode.
3. **`probe/`** — live executor: one config (+ fixed seed by default; "reroll seed" option) → single rollout **with full state logging** → `.rrd` written to Volume/CDN path → returns `{rrd_url, outcome, timings, cost}`. JAX persistent compilation cache **baked into the image**.
4. **`reduce/`** — local-or-Modal step: manifest JSON (per-cell survival estimates via censoring-aware estimator, IQR, diverged fraction, margin stats); selects boundary-adjacent cells (~15) + 2–3 near-boundary recoveries + 1 divergence; invokes `probe/` in instrumented mode for their `.rrd`s; renders composites (batch).

4b. **`render/`** — offline video production from state logs (invoked by `reduce/`; MuJoCo renderer, batch on Modal): failure-centered clip extraction (t_push − 0.5 s → t_fail + 1 s, slow-mo segment), overlay/ghost renders (multi-rollout superposition — seeds; checkpoints), tiled composites, mobile-fallback clips per selected cell. Never runs in the live probe path.

5. **`app/`** — Modal `@asgi_app()` (FastAPI): serves static frontend, manifest, cached `.rrd`s; `POST /probe` (validated, clamped, cached, rate-limited); `GET /health`, `GET /receipt`. Static mirror job exports map + replays to GitHub Pages (permanence fallback; live probes remain Modal-only).
6. **`web/`** — static HTML/JS/CSS: heatmap (canvas/SVG from manifest), sliders + direction dial + timing slider, **2D SVG schematic preview** (top-down robot; arrow = push vector scaled/rotated live; friction shading), embedded **pinned** Rerun web viewer (vendored wasm/JS, never CDN), status line (cold start / compile / sim / cost), receipt block, degradation banner, URL-param encode/decode. Mobile: reduced mode per §4.5 outcome.
7. **`analytics/`** — minimal: probe count, configs, cache hit rate, cap events, timings. (Feeds the "when HN showed up" devlog; also the v1.1 crowd-layer data source — design the probe log schema so v1.1 only adds a read path.)

## 4. Week-1 Gates (BLOCKING, in order — no artifact code before all pass)

1. **G1 Modal verification (from live docs, not memory):** current ASGI/web-endpoint limits; max request duration (probe path budget); scale-to-zero billing for the app; Volume/CDN patterns for serving `.rrd`; `.map`/batching API shape. Record findings in `docs/modal-verification.md`.
2. **G2 Checkpoint:** train Go1 joystick policy per Playground recipe; save intermediates; verify nominal locomotion. Record training config + DR ranges (needed for §8 honesty).
3. **G3 Harness equivalence (the critical gate):** at zero perturbation, our evaluation env must reproduce the policy's nominal Playground eval performance within noise. Perturbations enter **through physics only** (body forces via `xfrc_applied`-equivalent in MJX; model friction params), never through the observation/action contract. Failing G3 invalidates everything downstream.
4. **G4 The real sweep:** run full grid × 16 seeds. Real data now drives: slider ranges, color scales, boundary-cell selection, which result framing (§8) the post leads with, and go/no-go on drama (if the DR-trained policy is boring in sane ranges: widen axes, feature the no-DR twin, or lead with checkpoint comparison — decide from data).
5. **G5 Mobile/wasm test:** pinned Rerun viewer on iOS Safari + Android Chrome (memory, touch, load time). Outcome selects mobile mode: full viewer / clip-fallback ("full 3D on desktop") per cell.

## 5. Physics & Metric Definitions (locked; code must match post prose exactly)

- **Rollout:** 5 s @ env dt; joystick command fixed by default (constant forward velocity; value recorded). If command presets ship (§2 stretch), each preset is a documented constant command vector and part of the config/cache key; the heatmap's command stays the default and is stated on the figure.
- **Failure:** torso height < threshold OR |roll|/|pitch| > threshold, whichever first; TTF = timestamp. Thresholds set from G4 nominal-rollout distributions and then frozen.
- **Censoring:** survival past 5 s → right-censored (TTF > 5 s), never TTF = 5 s. Per-cell estimates via Kaplan–Meier (e.g., `lifelines`); report median survival where defined + survival-at-5s.
- **Diverged:** NaN/inf in state → distinct category; hatched on map; live-path response is a real result screen ("physics diverged at t=…"), never an error.
- **Push:** constant lateral force on torso body, window 0.5 s, start-time slider (default t = 2 s), direction dial (angle in horizontal plane relative to heading). Primary axis unit: **% bodyweight** (model mass recorded and displayed); Newtons secondary.
- **Margin features (logged per rollout for v1 map + future taxonomy):** peak CoM lateral excursion, first contact-loss limb + time, recovery-step attempted flag, slip accumulation, time-to-renominal-gait (if recovered).

## 6. Statistics Requirements

- 16 seeds/cell minimum; seed = env PRNG only (document all stochasticity sources).
- Heatmap cell value: censoring-aware central estimate; companion figure: per-cell IQR/variance.
- **Solver-sensitivity slice (required for post):** re-run one boundary-crossing grid slice at 2–3 timestep/solver-iteration settings; report boundary displacement. Large displacement is itself a finding — report either way.
- Reproducibility statement: pinned image reproduces exactly; cross-hardware claim is statistical (surface within tolerance), not bitwise — say so.

## 7. Live-Path Requirements

- Validation: clamp all params server-side to published ranges (also the abuse guard).
- Caching: key = config rounded to slider resolution; cache hit → instant, $0.
- Rate limiting: per-IP; global concurrency ceiling; **hard daily spend cap** (value set pre-launch).
- Degradation rule: any limit → read-only mode, banner with honest numbers; map + cached replays always functional. The page never errors in front of traffic.
- Telemetry to UI: cold start, compile (should read "cached"), sim time, per-run cost.
- Probe latency budget (from G1/G3): target < 10 s warm, < 20 s cold; if unachievable → nearest-cached-neighbor mode, honestly labeled.

## 8. Post-Facing Requirements (the code must serve these claims)

- Pre-registered result framings (sharp cliff / smooth degradation / structured surface) — G4 data selects; no result numbers drafted before G4.
- DR-context statement: whether swept ranges sit inside or outside the policy's training randomization (from G2 records).
- Sim-scope humility: boundary is "in the simulator the policy was trained for transfer in."
- Attribution: Menagerie Go1 model license, Playground, Brax, Rerun, Modal — repo + post footer.
- Receipt: measured only. CI check idea: grep drafts for numbers lacking a `measured:` source tag.

## 9. Frontend Spec (v1)

- Heatmap: cells colored by survival estimate; diverged hatching; click cell → cached replay; click between cells → populates sliders (off-grid probe flow).
- Schematic: top-down SVG robot; push arrow (angle/magnitude live), friction shading, timing marker on a mini-timeline. Shared-URL loads reconstruct schematic before outcome reveal.
- Viewer: vendored pinned Rerun wasm; timeline event markers (push start, failure); default camera preset + failure-moment jump control.
- Compare mode + in-viewer seed-ghosts: **stretch**, only if v1 timeline allows; componentize the viewer embed to keep them cheap later. (Rendered ghost-overlay *videos* from `render/` are v1 regardless — they don't depend on viewer work.)
- Stretch controls (per §2): command-preset selector; checkpoint dropdown (labels framed as training age, e.g. "5 min old / fully trained").

## 10. Repo & Reuse Requirements

- Layout: `train/ sweep/ probe/ reduce/ app/ web/ analytics/ docs/` with config-schema module shared by sweep and probe (single source of truth for perturbation semantics).
- One-command reproduce: `make sweep` (or equivalent) from pinned image → manifest byte-comparable modulo documented nondeterminism.
- No premature framework extraction: build this project concretely; the pattern library is extracted after the drone project by diffing (thesis doc §7.4).
- Analytics/probe-log schema designed for v1.1 crowd layer (append-only records: config, outcome, timestamp; no identity).

## 11. Launch Kit (deliverables alongside code)

- Hero GIF/MP4: shove-arrow composite (cross-boundary 3×3 tiles, synchronized to push, slow-mo).
- 16-seed montage (one boundary cell).
- Show HN draft + thread skeleton (link post; artifact does the talking).
- Devlog stubs: training run, first cold-start replay, sweep receipt, launch-day traffic (from analytics).

## 12. Milestones

- **M0 Gates:** §4 G1→G5 in order. Deliverables: verification doc, checkpoint(s), equivalence report, sweep dataset, mobile decision.
- **M1 Data layer:** reduce/ (manifest, statistics incl. censoring, boundary selection, solver-sensitivity slice), instrumented replays; render/ (failure clips, ghost overlays, composites, mobile-fallback clips).
- **M2 Serving:** app/ endpoints, caching, limits, degradation mode, receipt instrumentation, static mirror export.
- **M3 Frontend:** heatmap, sliders/dial/timing, schematic, pinned viewer embed, URL sharing, mobile mode, status line.
- **M4 Hardening:** latency budget verification, spend-cap drill (simulate cap hit), cross-device pass, analytics, license/attribution audit.
- **M5 Launch kit + post assembly:** composites final, receipt final (measured), draft claims audit (§8), static mirror live.
- **M6 (v1.1, post-launch):** crowd layer — persistence read path, dot overlay, "closest call" line, second announcement beat.

Acceptance for v1 = the two definitions of done: a stranger on mobile can probe a world and see cost; every number measured; degradation drill passed.

## 13. Risk Register (consult before architecture decisions)

| Risk | Mitigation |
|---|---|
| Harness bug masquerading as boundary | G3 equivalence gate; physics-only injection; DR-range documentation |
| JIT tax on probe path | compilation cache in image; measured budget; fallback min-warm during launch week (disclosed in receipt) |
| Boundary is integrator artifact | §6 solver-sensitivity slice, reported |
| Boring physics (DR policy too robust in sane ranges) | G4 before build; widen axes / no-DR twin / checkpoint-comparison lead |
| Rerun wasm on mobile | G5; clip-fallback mode designed in |
| `.rrd`/viewer version skew | pin SDK + vendored viewer; version asserted in CI |
| Launch traffic breaks page or budget | clamps, caps, read-only degradation, cap drill in M4 |
| Link rot | static mirror; app kept deployed (scale-to-zero); MP4 fallbacks embedded in post |
| Censoring mishandled → biased surface | KM estimators; unit tests on synthetic censored data |
| Scope creep re-tripling v1 | §2 non-goals are binding; additions require moving something out |

## 14. Open Decisions (make during M0, record in docs/)

1. Probe seed policy: fixed default seed (deterministic shareable worlds) vs. random with reroll — leaning fixed + reroll button.
2. `.rrd` delivery: file-from-Volume/CDN (default) vs. streaming — decide from G1 only.
3. Sweep hardware + chunking: single-GPU batch size vs. memory; measured in G4.
4. Spend cap value + launch-day min-warm setting.
5. Mobile mode selection (from G5).
6. Exact grid ranges (from G4 pilot pass; boundary must sit comfortably inside the map).
