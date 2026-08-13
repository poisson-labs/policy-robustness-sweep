"""Unit tests for the two custom CI checks, against synthetic fixture trees."""

from pathlib import Path

from ci import check_measured_numbers, check_rerun_versions

UV_LOCK_WITH_RERUN = """\
version = 1

[[package]]
name = "somelib"
version = "2.0.0"

[[package]]
name = "rerun-sdk"
version = "0.20.1"
"""

UV_LOCK_WITHOUT_RERUN = """\
version = 1

[[package]]
name = "somelib"
version = "2.0.0"
"""


def write_viewer_version(root: Path, version: str) -> None:
    viewer_dir = root / "web" / "vendor" / "rerun"
    viewer_dir.mkdir(parents=True)
    (viewer_dir / "VERSION").write_text(f"{version}\n")


class TestRerunVersionCheck:
    def test_passes_when_nothing_exists(self, tmp_path: Path) -> None:
        ok, _ = check_rerun_versions.check(tmp_path)
        assert ok

    def test_passes_when_sdk_pinned_but_viewer_not_vendored(self, tmp_path: Path) -> None:
        (tmp_path / "uv.lock").write_text(UV_LOCK_WITH_RERUN)
        ok, _ = check_rerun_versions.check(tmp_path)
        assert ok

    def test_fails_when_viewer_vendored_without_sdk_pin(self, tmp_path: Path) -> None:
        (tmp_path / "uv.lock").write_text(UV_LOCK_WITHOUT_RERUN)
        write_viewer_version(tmp_path, "0.20.1")
        ok, message = check_rerun_versions.check(tmp_path)
        assert not ok
        assert "not in uv.lock" in message

    def test_fails_on_version_mismatch(self, tmp_path: Path) -> None:
        (tmp_path / "uv.lock").write_text(UV_LOCK_WITH_RERUN)
        write_viewer_version(tmp_path, "0.21.0")
        ok, message = check_rerun_versions.check(tmp_path)
        assert not ok
        assert "MISMATCH" in message

    def test_passes_on_version_match(self, tmp_path: Path) -> None:
        (tmp_path / "uv.lock").write_text(UV_LOCK_WITH_RERUN)
        write_viewer_version(tmp_path, "0.20.1")
        ok, _ = check_rerun_versions.check(tmp_path)
        assert ok


class TestMeasuredNumbersCheck:
    def write_draft(self, root: Path, content: str) -> None:
        drafts = root / "docs" / "drafts"
        drafts.mkdir(parents=True, exist_ok=True)
        (drafts / "post.md").write_text(content)

    def test_passes_with_no_drafts_dir(self, tmp_path: Path) -> None:
        ok, violations = check_measured_numbers.check(tmp_path)
        assert ok
        assert violations == []

    def test_passes_on_prose_without_numbers(self, tmp_path: Path) -> None:
        self.write_draft(tmp_path, "The robot walks.\nThen it gets shoved.\n")
        ok, _ = check_measured_numbers.check(tmp_path)
        assert ok

    def test_flags_bare_number(self, tmp_path: Path) -> None:
        self.write_draft(tmp_path, "The sweep cost $4.20 total.\n")
        ok, violations = check_measured_numbers.check(tmp_path)
        assert not ok
        assert len(violations) == 1
        assert "post.md:1" in violations[0]

    def test_allows_measured_tag(self, tmp_path: Path) -> None:
        self.write_draft(
            tmp_path, "The sweep cost $4.20 total (measured: modal bill 2026-08-20).\n"
        )
        ok, _ = check_measured_numbers.check(tmp_path)
        assert ok

    def test_allows_placeholder(self, tmp_path: Path) -> None:
        self.write_draft(tmp_path, "The sweep cost MEASURED_TBD total.\n")
        ok, _ = check_measured_numbers.check(tmp_path)
        assert ok

    def test_allows_explicit_no_number_marker(self, tmp_path: Path) -> None:
        self.write_draft(tmp_path, "The Go1 robot walks. <!-- no-number -->\n")
        ok, _ = check_measured_numbers.check(tmp_path)
        assert ok

    def test_reports_every_violation_with_location(self, tmp_path: Path) -> None:
        self.write_draft(tmp_path, "Costs $1.\nFine line.\nTook 20 s.\n")
        ok, violations = check_measured_numbers.check(tmp_path)
        assert not ok
        assert len(violations) == 2
