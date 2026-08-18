"""API contract tests for app/core (kickoff §4 required suite: validation, cache-hit
path, degradation mode) — plus rate limits, spend cap, analytics log, hostile input."""

import json
from pathlib import Path
from typing import Any

from app.core import Limits, ProbeService, RateLimiter, SpendLedger


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, world_json: str) -> dict[str, Any]:
        self.calls.append(world_json)
        return {"outcome": "failed", "ttf_s": 2.7, "timings_s": {"sim_s": 0.5}}


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.day = "2026-08-18"

    def now(self) -> float:
        return self.t

    def today(self) -> str:
        return self.day


def make(
    tmp_path: Path,
    cached: set[str] | None = None,
    limits: Limits | None = None,
) -> tuple[ProbeService, FakeBackend, Clock, set[str]]:
    cache: set[str] = cached if cached is not None else set()
    backend = FakeBackend()
    clock = Clock()
    svc = ProbeService(
        limits=limits or Limits(per_ip_per_minute=3, global_concurrency=2, probe_cost_usd=0.01),
        backend=backend,
        rrd_exists=lambda k: k in cache,
        rrd_url=lambda k: f"/rrd/{ProbeService.safe_key(k)}.rrd",
        analytics_path=tmp_path / "analytics.jsonl",
        now=clock.now,
        today=clock.today,
    )
    return svc, backend, clock, cache


class TestValidationAndClamping:
    def test_out_of_range_is_clamped_not_rejected(self, tmp_path: Path) -> None:
        svc, backend, _, _ = make(tmp_path)
        status, body = svc.handle({"friction": 99, "push_magnitude_pct_bw": -50}, ip="1.1.1.1")
        assert status == 200
        assert body["world"]["friction"] == 2.0  # RANGES upper bound
        assert body["world"]["push_magnitude_pct_bw"] == 0.0
        assert body["clamped"] is True
        assert len(backend.calls) == 1

    def test_hostile_input_never_raises(self, tmp_path: Path) -> None:
        svc, _, _, _ = make(tmp_path)
        hostile: list[dict[str, Any]] = [
            {"friction": "abc", "push_magnitude_pct_bw": "NaN", "seed": "x"},
            {"friction": float("inf")},
            {"push_direction_deg": "1e999"},
            {"seed": None, "push_start_s": []},
            {},
        ]
        for raw in hostile:
            status, body = svc.handle(raw, ip="2.2.2.2")
            assert status in (200, 429)
            assert "cache_key" in body

    def test_direction_wraps(self, tmp_path: Path) -> None:
        svc, _, _, _ = make(tmp_path)
        _, body = svc.handle({"push_direction_deg": 450}, ip="3.3.3.3")
        assert body["world"]["push_direction_deg"] == 90.0
        assert body["clamped"] is False  # wrap is not a clamp from the user's view

    def test_unparseable_friction_means_default_not_ice(self, tmp_path: Path) -> None:
        svc, _, _, _ = make(tmp_path)
        _, body = svc.handle({"friction": "lol"}, ip="5.5.5.5")
        assert body["world"]["friction"] is None  # model default, not clamped to 0.05

    def test_in_range_not_flagged_clamped(self, tmp_path: Path) -> None:
        svc, _, _, _ = make(tmp_path)
        _, body = svc.handle({"friction": 0.5, "push_magnitude_pct_bw": 90}, ip="4.4.4.4")
        assert body["clamped"] is False


class TestCacheHitPath:
    def test_hit_is_instant_free_and_skips_backend(self, tmp_path: Path) -> None:
        svc, backend, _, cache = make(tmp_path)
        _, first = svc.handle({"friction": 0.5, "push_magnitude_pct_bw": 90}, ip="a")
        cache.add(first["cache_key"])  # canonical .rrd now exists
        status, body = svc.handle({"friction": 0.5, "push_magnitude_pct_bw": 90}, ip="a")
        assert status == 200
        assert body["status"] == "cached"
        assert body["telemetry"] == {"cache_hit": True, "cost_usd": 0.0}
        assert len(backend.calls) == 1  # no second GPU call
        assert body["rrd_url"].endswith(".rrd")

    def test_sub_resolution_configs_share_cache(self, tmp_path: Path) -> None:
        svc, backend, _, cache = make(tmp_path)
        _, a = svc.handle({"friction": 0.701, "push_magnitude_pct_bw": 90.2}, ip="a")
        cache.add(a["cache_key"])
        _, b = svc.handle({"friction": 0.699, "push_magnitude_pct_bw": 90.4}, ip="a")
        assert b["status"] == "cached"
        assert len(backend.calls) == 1

    def test_cache_hit_works_in_read_only_mode(self, tmp_path: Path) -> None:
        svc, backend, _, cache = make(tmp_path)
        _, a = svc.handle({"friction": 0.5}, ip="a")
        cache.add(a["cache_key"])
        svc.forced_mode = "read_only"
        _, b = svc.handle({"friction": 0.5}, ip="a")
        assert b["status"] == "cached"
        assert b["banner"] is not None and b["banner"]["mode"] == "read_only"
        assert len(backend.calls) == 1


class TestDegradationMode:
    def test_spend_cap_flips_to_read_only_with_honest_banner(self, tmp_path: Path) -> None:
        limits = Limits(
            per_ip_per_minute=100,
            global_concurrency=10,
            daily_spend_cap_usd=0.03,
            probe_cost_usd=0.01,
        )
        svc, backend, _, _ = make(tmp_path, limits=limits)
        for i in range(3):
            status, body = svc.handle({"seed": i}, ip="a")
            assert body["status"] == "probed"
        # 3 x $0.01 = cap → next uncached probe is read-only, HTTP 200, banner, no GPU
        status, body = svc.handle({"seed": 99}, ip="a")
        assert status == 200
        assert body["status"] == "read_only"
        assert body["banner"]["reason"] == "daily spend cap reached"
        assert body["banner"]["spent_today_usd"] == 0.03
        assert body["banner"]["cap_usd"] == 0.03
        assert len(backend.calls) == 3

    def test_new_day_resets_cap(self, tmp_path: Path) -> None:
        limits = Limits(
            per_ip_per_minute=100,
            global_concurrency=10,
            daily_spend_cap_usd=0.01,
            probe_cost_usd=0.01,
        )
        svc, _, clock, _ = make(tmp_path, limits=limits)
        svc.handle({"seed": 1}, ip="a")
        assert svc.mode() == "read_only"
        clock.day = "2026-08-19"
        assert svc.mode() == "live"
        _, body = svc.handle({"seed": 2}, ip="a")
        assert body["status"] == "probed"

    def test_unknown_cost_never_invents_a_dollar_figure(self, tmp_path: Path) -> None:
        limits = Limits(per_ip_per_minute=100, global_concurrency=10, probe_cost_usd=None)
        svc, _, _, _ = make(tmp_path, limits=limits)
        _, body = svc.handle({"seed": 5}, ip="a")
        assert body["telemetry"]["cost_usd"] is None
        assert svc.ledger.spent_today() is None
        assert svc.mode() == "live"  # cannot cap on a number we don't have

    def test_drill_hook_forces_read_only(self, tmp_path: Path) -> None:
        svc, backend, _, _ = make(tmp_path)
        svc.forced_mode = "read_only"
        status, body = svc.handle({"seed": 1}, ip="a")
        assert status == 200 and body["status"] == "read_only"
        assert body["banner"]["reason"] == "read-only drill"
        assert backend.calls == []


class TestRateLimits:
    def test_per_ip_window(self, tmp_path: Path) -> None:
        svc, _, clock, _ = make(tmp_path)  # 3/min
        for i in range(3):
            assert svc.handle({"seed": i}, ip="x")[0] == 200
        status, body = svc.handle({"seed": 9}, ip="x")
        assert status == 429 and body["reason"] == "per_ip"
        assert svc.handle({"seed": 9}, ip="y")[0] == 200  # other IP unaffected
        clock.t += 61
        assert svc.handle({"seed": 10}, ip="x")[0] == 200  # window slid

    def test_global_concurrency_ceiling(self, tmp_path: Path) -> None:
        svc, _, _, _ = make(tmp_path)  # global 2
        svc.limiter.acquire()
        svc.limiter.acquire()  # two in flight
        status, body = svc.handle({"seed": 1}, ip="z")
        assert status == 429 and body["reason"] == "global_concurrency"
        svc.limiter.release()
        assert svc.handle({"seed": 1}, ip="z")[0] == 200


class TestAnalyticsLog:
    def test_append_only_no_identity(self, tmp_path: Path) -> None:
        svc, _, _, cache = make(tmp_path)
        _, a = svc.handle({"friction": 0.5}, ip="203.0.113.7")
        cache.add(a["cache_key"])
        svc.handle({"friction": 0.5}, ip="203.0.113.7")
        lines = [
            json.loads(line) for line in (tmp_path / "analytics.jsonl").read_text().splitlines()
        ]
        assert len(lines) == 2
        assert lines[0]["cache_hit"] is False and lines[1]["cache_hit"] is True
        for rec in lines:
            assert "203.0.113.7" not in json.dumps(rec)  # no identity, ever
            assert set(rec) >= {"t", "key", "cache_hit", "mode"}


class TestUnits:
    def test_rate_limiter_standalone(self) -> None:
        t = [0.0]
        rl = RateLimiter(2, 1, now=lambda: t[0])
        assert rl.allow("a") == (True, None)
        assert rl.allow("a") == (True, None)
        assert rl.allow("a") == (False, "per_ip")

    def test_ledger_counts_without_cost(self) -> None:
        led = SpendLedger(cap_usd=1.0, probe_cost_usd=None, today=lambda: "d")
        led.record()
        assert led.probes_today() == 1 and led.spent_today() is None and not led.cap_hit()
