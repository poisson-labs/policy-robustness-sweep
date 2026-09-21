# docs/

| Path | What it is |
|---|---|
| [`spec-one-policy-400-worlds.md`](spec-one-policy-400-worlds.md) | The technical specification the build followed: goals, architecture, the five gates, the definitions of failure and censoring, the statistics, and the live-path requirements. |
| [`modal-verification.md`](modal-verification.md) | Modal's limits and pricing as read from its documentation on 2026-08-13 (gate G1). |
| [`equivalence-report.md`](equivalence-report.md) | Gate G3: the sweep environment reproduces the policy's own evaluation, and the run-to-run drift found at identical seeds. |
| [`dr-record-go1-training.json`](dr-record-go1-training.json), [`dr-record-go1-training-v1.json`](dr-record-go1-training-v1.json) | The domain-randomisation source and configuration each policy was trained with, captured verbatim. They embed Apache-2.0 source text; see [`../NOTICE`](../NOTICE). |
| [`measurements/`](measurements/) | Measured outputs: sweep manifests, reports, the v0 → v1 comparison and the figures. |
| [`drafts/`](drafts/) | Where post drafts are kept. `ci/check_measured_numbers.py` scans it. |
| [`repo-hygiene.md`](repo-hygiene.md) | The Poisson Labs repository standard this repository follows. |

## Reading the references in these documents

**Milestone labels.** Commit messages and documents cite labels such as `[M0-06]` and `[M1.5]`. `G1` to `G5` are
the five gates of spec §4. `M0` is those gates, `M1` the data layer (survival statistics, replay selection, clips),
`M1.5` the retrain → re-map → diff loop, `M1.6` the boundary statistics, `M2` serving, `M3` the frontend, `M4`
hardening, `M5` the launch kit and `M6` the crowd layer planned for after launch. The number after the dash is a
task within the milestone. Source comments that cite "spec §12" mean this milestone list, which the spec no longer
carries: section 12 there is now the v1 acceptance line.

**References that do not resolve.** Some documents and source comments cite "DEVLOG Session N", "kickoff §N",
`NOTES.md` and `TASKS.md`. They refer to the author's working log, to the build brief the project started from,
and to the notes and the task list kept during the build. None of them is in the tree.

**`MEASURED_TBD`.** A number written `MEASURED_TBD` had not been measured when the document was written.
`ci/check_measured_numbers.py` requires every number in a post draft to carry a `measured:` source tag or that
marker. Several cost lines under `measurements/` still carry the marker.

## The two sweeps and the tests on them

The v0 and v1 sweeps ran the same 32 seeds in every cell. The seed scheme is
`PRNGKey(1000000 + rollout_index); rollout_index = (mu_idx*20 + push_idx)*32 + seed_idx`, as the `grid.seed_scheme`
field of each sweep manifest says (`sweep/full_sweep.py` computes it for any number of seeds per cell;
`sweep/seed_scheme.py` holds the 16-seed form), and the `rollout` number is equal in all 12,800 pairs of records.

The test behind "60 cells safer, none worse" is Fisher's exact test per cell, corrected across all 400 cells
(Benjamini–Hochberg, q < 0.05). It compares the two sweeps as separate samples and does not use the matching
seeds. A paired test on the same seeds, an exact McNemar test per cell with the same correction, gives 55 cells
safer and none worse. This repository's code computes only the Fisher test; the McNemar figure was recomputed from
the two committed sweep manifests, which hold every rollout with its cell and seed.

## Regenerated files

`uv run python -m reduce.compare` on the two committed 32-seed manifests writes a file identical, byte for byte,
to `measurements/2026-08-18-v0-v1-diff.json` on the machine it was checked on (macOS on arm64, Python 3.12, the
versions in `uv.lock`). Floating-point output can differ in the last digits on other platforms.

`uv run python -m render.figures` reproduces the three `km-surface-*.png` files byte for byte on that machine. The
three `km-surface-*.svg` files differ from the committed ones only in an embedded timestamp and in the element
ids matplotlib generates.
