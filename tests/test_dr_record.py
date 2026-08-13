"""Verbatim DR-source capture against a synthetic package tree (train/dr_record.py)."""

import hashlib
from pathlib import Path

import pytest

from train.dr_record import capture_sources

RANDOMIZE_SRC = "FLOOR_FRICTION = (0.4, 1.0)\n\ndef domain_randomize(model, rng):\n    ...\n"


def make_fake_package(root: Path) -> Path:
    pkg = root / "mujoco_playground"
    (pkg / "_src/locomotion/go1").mkdir(parents=True)
    (pkg / "config").mkdir(parents=True)
    (pkg / "_src/locomotion/go1/randomize.py").write_text(RANDOMIZE_SRC)
    (pkg / "config/locomotion_params.py").write_text("def brax_ppo_config(name):\n    ...\n")
    return pkg


def test_captures_verbatim_content_and_hash(tmp_path: Path) -> None:
    pkg = make_fake_package(tmp_path)
    record = capture_sources(
        pkg,
        ["_src/locomotion/go1/randomize.py"],
        {"playground": "0.2.0"},
    )
    (captured,) = record["files"]
    assert captured["content"] == RANDOMIZE_SRC  # verbatim, not a paraphrase
    assert captured["sha256"] == hashlib.sha256(RANDOMIZE_SRC.encode()).hexdigest()
    assert record["package_versions"] == {"playground": "0.2.0"}


def test_missing_source_file_raises(tmp_path: Path) -> None:
    pkg = make_fake_package(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"randomize_v2\.py"):
        capture_sources(pkg, ["_src/locomotion/go1/randomize_v2.py"], {})


def test_multiple_files_all_captured_in_order(tmp_path: Path) -> None:
    pkg = make_fake_package(tmp_path)
    record = capture_sources(
        pkg,
        ["_src/locomotion/go1/randomize.py", "config/locomotion_params.py"],
        {"playground": "0.2.0", "brax": "0.14.2"},
    )
    assert [f["path"] for f in record["files"]] == [
        "_src/locomotion/go1/randomize.py",
        "config/locomotion_params.py",
    ]
