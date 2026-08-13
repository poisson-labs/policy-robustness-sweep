"""CI check: Rerun SDK pin must match the vendored web-viewer version (spec §13 risk:
".rrd/viewer version skew").

Sources compared:
- SDK pin: the ``rerun-sdk`` package version locked in ``uv.lock``.
- Vendored viewer: ``web/vendor/rerun/VERSION`` (a plain text file containing the version,
  written when the viewer wasm is vendored in M3/G5).

Rules:
- Neither exists yet (pre-G5 state): pass.
- SDK pinned but no vendored viewer yet: pass (viewer arrives with frontend work).
- Vendored viewer without an SDK pin: FAIL — a viewer must never be vendored unpinned.
- Both exist and differ: FAIL.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

VIEWER_VERSION_FILE = Path("web/vendor/rerun/VERSION")
UV_LOCK = Path("uv.lock")
SDK_PACKAGE_NAME = "rerun-sdk"


def locked_sdk_version(repo_root: Path) -> str | None:
    lock_path = repo_root / UV_LOCK
    if not lock_path.exists():
        return None
    lock = tomllib.loads(lock_path.read_text())
    for package in lock.get("package", []):
        if package.get("name") == SDK_PACKAGE_NAME:
            version = package.get("version")
            return str(version) if version is not None else None
    return None


def vendored_viewer_version(repo_root: Path) -> str | None:
    version_path = repo_root / VIEWER_VERSION_FILE
    if not version_path.exists():
        return None
    return version_path.read_text().strip()


def check(repo_root: Path) -> tuple[bool, str]:
    """Return (ok, message)."""
    sdk = locked_sdk_version(repo_root)
    viewer = vendored_viewer_version(repo_root)

    if viewer is None and sdk is None:
        return True, "rerun-versions: no SDK pin and no vendored viewer yet — nothing to check"
    if viewer is None:
        return True, f"rerun-versions: SDK pinned at {sdk}, viewer not vendored yet — ok"
    if sdk is None:
        return False, (
            f"rerun-versions: viewer vendored at {viewer} but rerun-sdk is not in uv.lock — "
            "the viewer must never be vendored without a matching SDK pin"
        )
    if sdk != viewer:
        return False, f"rerun-versions: MISMATCH — rerun-sdk {sdk} != vendored viewer {viewer}"
    return True, f"rerun-versions: rerun-sdk {sdk} == vendored viewer {viewer} — ok"


def main() -> int:
    ok, message = check(Path.cwd())
    print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
