"""G3 equivalence as an executable test (spec §4 G3: "an executable pytest test, not a
notebook"). Validates the committed measured report — if the report is regenerated and
the gate no longer holds, CI fails."""

import json
from pathlib import Path
from typing import Any

REPORT = Path(__file__).parent.parent / "docs/measurements/2026-08-13-g3-equivalence-report.json"


def load_report() -> dict[str, Any]:
    assert REPORT.exists(), "committed G3 equivalence report is missing"
    report: dict[str, Any] = json.loads(REPORT.read_text())
    return report


def test_gate_passed() -> None:
    assert load_report()["passed"] is True


def test_replay_check_is_bitwise() -> None:
    replay = load_report()["replay_check"]
    assert replay["passed"] is True
    assert replay["max_qpos_diff"] <= replay["tolerance"]
    assert replay["max_qvel_diff"] <= replay["tolerance"]
    assert replay["steps_compared"] >= 100  # a trivially short replay proves nothing


def test_statistical_check_within_gate() -> None:
    stat = load_report()["statistical_check"]
    assert stat["passed"] is True
    assert stat["abs_diff"] <= stat["gate"]
    assert len(stat["our_means"]) >= 3  # repeats actually happened


def test_protocol_matches_training_eval() -> None:
    protocol = load_report()["protocol"]
    assert protocol["stochastic_policy"] is True
    assert protocol["num_eval_envs"] == 128
    assert protocol["episode_length"] == 1000


def test_nominal_world_is_identity() -> None:
    world = load_report()["nominal_world"]
    assert world["friction"] is None
    assert world["push_magnitude_pct_bw"] == 0.0


def test_diagnostic_is_not_gated() -> None:
    diagnostic = load_report()["same_seed_evaluator_diagnostic"]
    assert diagnostic["gated"] is False  # flaky-by-construction check must never gate
