"""The server's viewer pin must match the repo's VERSION file (single source of truth)."""

from pathlib import Path

from probe.g5_server import RERUN_VIEWER_TGZ_SHA256, RERUN_VIEWER_VERSION


def test_viewer_version_matches_version_file() -> None:
    version_file = Path(__file__).parent.parent / "web/vendor/rerun/VERSION"
    assert version_file.read_text().strip() == RERUN_VIEWER_VERSION


def test_tarball_hash_matches_sha256sums_file() -> None:
    sums = Path(__file__).parent.parent / "web/vendor/rerun/SHA256SUMS"
    assert RERUN_VIEWER_TGZ_SHA256 in sums.read_text()
