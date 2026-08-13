"""Verbatim capture of the training-time domain-randomization config (G2 requirement).

Spec §8's "inside vs. outside the training distribution" claim will be checked against
this record, so it must be the actual randomization source **verbatim** (full file text +
hash + package version), never a paraphrase. Capture runs inside the training container
against the installed `mujoco_playground` package; the record is written next to the
checkpoints and later committed under docs/.

The files captured for Go1 joystick training (paths verified against the live repo,
2026-08-13):
- mujoco_playground/_src/locomotion/go1/randomize.py  (the DR function itself)
- mujoco_playground/config/locomotion_params.py       (the PPO recipe incl. env overrides)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

GO1_DR_SOURCE_FILES = [
    "_src/locomotion/go1/randomize.py",
    "config/locomotion_params.py",
    # The env task file: holds default_config incl. pert_config (enable=False by default
    # — decides §8's "every shove is out-of-distribution" claim, so it must be in the
    # verbatim record, not just cited from upstream).
    "_src/locomotion/go1/joystick.py",
]


def capture_sources(
    package_root: Path,
    relative_paths: list[str],
    package_versions: dict[str, str],
) -> dict[str, Any]:
    """Build a verbatim source record for the given files under an installed package.

    Raises FileNotFoundError if any expected source file is missing — a missing DR file
    means the capture list is stale for this package version, and the record must never
    silently omit it.
    """
    files: list[dict[str, str]] = []
    for relative in relative_paths:
        path = package_root / relative
        if not path.exists():
            raise FileNotFoundError(
                f"expected DR/config source missing from installed package: {path} — "
                "update GO1_DR_SOURCE_FILES for this package version"
            )
        text = path.read_text()
        files.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "content": text,
            }
        )
    return {
        "package_root": str(package_root),
        "package_versions": dict(sorted(package_versions.items())),
        "files": files,
    }


def capture_go1_dr_record() -> dict[str, Any]:
    """Capture the Go1 DR/config sources from the installed mujoco_playground.

    Container-only (imports the training deps); the pure logic lives in
    ``capture_sources`` and is unit-tested locally.
    """
    from importlib import metadata

    import mujoco_playground

    package_root = Path(mujoco_playground.__file__).parent
    versions = {
        name: metadata.version(name)
        for name in ("playground", "brax", "jax", "mujoco", "mujoco-mjx")
    }
    return capture_sources(package_root, GO1_DR_SOURCE_FILES, versions)
