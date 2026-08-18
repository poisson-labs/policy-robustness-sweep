"""app/ core — the pure, testable logic behind the serving endpoints (M2-02).

Everything here is plain Python under pyright strict with contract tests
(tests/test_app_core.py). The Modal/FastAPI shell (app/server.py) is a thin wrapper:
parse request → `ProbeService.handle` → response.

Spec §7 requirements implemented here:
- validation + server-side clamping (WorldConfig.clamped IS the security boundary;
  every request is hostile input; out-of-range values clamp, never error);
- caching by rounded config (cache_key) — hit → instant, $0; durable-canonical
  invariant: a served .rrd is never evicted (DEVLOG Session 8);
- rate limiting: per-IP window + global concurrency ceiling;
- hard daily spend cap (§14 #4: $75/day launch week, $25/day steady) → read-only
  DEGRADATION: map + cached replays keep working; new probes return an honest banner
  payload, never an error; the page never errors in front of traffic;
- telemetry to UI: cold/warm, sim time, per-run cost — MEASURED per call, never
  projected; the receipt endpoint serves only measured/billed numbers;
- append-only analytics log (config, outcome, timestamp, cache-hit; no identity) —
  v1.1 crowd layer only adds a read path.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from configs.world import WorldConfig

Mode = Literal["live", "read_only"]


@dataclass(frozen=True)
class Limits:
    per_ip_per_minute: int = 6
    global_concurrency: int = 4
    daily_spend_cap_usd: float = 75.0  # §14 #4 launch week; steady-state config: 25.0
    probe_cost_usd: float | None = None  # MEASURED per-probe cost once billed; None → unknown


class ProbeBackend(Protocol):
    """The GPU probe. Real: Modal `Probe().probe.remote(world_json)`. Tests: fake."""

    def __call__(self, world_json: str) -> dict[str, Any]: ...


@dataclass
class RateLimiter:
    """Sliding-window per-IP limiter + global in-flight counter. Pure; injectable clock."""

    per_ip_per_minute: int
    global_concurrency: int
    now: Callable[[], float] = time.monotonic
    _hits: dict[str, deque[float]] = field(default_factory=lambda: {})
    _in_flight: int = 0

    def allow(self, ip: str) -> tuple[bool, str | None]:
        t = self.now()
        q = self._hits.setdefault(ip, deque())
        while q and t - q[0] > 60.0:
            q.popleft()
        if len(q) >= self.per_ip_per_minute:
            return False, "per_ip"
        if self._in_flight >= self.global_concurrency:
            return False, "global_concurrency"
        q.append(t)
        return True, None

    def acquire(self) -> None:
        self._in_flight += 1

    def release(self) -> None:
        self._in_flight = max(0, self._in_flight - 1)


@dataclass
class SpendLedger:
    """Daily spend accounting from MEASURED per-probe cost. Unknown cost → count probes
    and cap on count only when a cost is configured (never invent a dollar figure)."""

    cap_usd: float
    probe_cost_usd: float | None
    today: Callable[[], str]
    _spent: dict[str, float] = field(default_factory=lambda: {})
    _count: dict[str, int] = field(default_factory=lambda: {})

    def record(self) -> None:
        d = self.today()
        self._count[d] = self._count.get(d, 0) + 1
        if self.probe_cost_usd is not None:
            self._spent[d] = self._spent.get(d, 0.0) + self.probe_cost_usd

    def spent_today(self) -> float | None:
        return None if self.probe_cost_usd is None else self._spent.get(self.today(), 0.0)

    def probes_today(self) -> int:
        return self._count.get(self.today(), 0)

    def cap_hit(self) -> bool:
        s = self.spent_today()
        return s is not None and s >= self.cap_usd


@dataclass
class ProbeService:
    """The endpoint brain. `rrd_exists`/`rrd_url` abstract the Volume; `backend` the GPU."""

    limits: Limits
    backend: ProbeBackend
    rrd_exists: Callable[[str], bool]
    rrd_url: Callable[[str], str]
    analytics_path: Path | None = None
    now: Callable[[], float] = time.monotonic
    today: Callable[[], str] = lambda: time.strftime("%Y-%m-%d", time.gmtime())
    limiter: RateLimiter = field(init=False)
    ledger: SpendLedger = field(init=False)
    forced_mode: Mode | None = None  # M4 spend-cap DRILL hook

    def __post_init__(self) -> None:
        self.limiter = RateLimiter(
            self.limits.per_ip_per_minute, self.limits.global_concurrency, self.now
        )
        self.ledger = SpendLedger(
            self.limits.daily_spend_cap_usd, self.limits.probe_cost_usd, self.today
        )

    # ---- helpers -----------------------------------------------------------------
    @staticmethod
    def safe_key(cache_key: str) -> str:
        return cache_key.replace("|", "_").replace("=", "-")

    def mode(self) -> Mode:
        if self.forced_mode is not None:
            return self.forced_mode
        return "read_only" if self.ledger.cap_hit() else "live"

    def banner(self) -> dict[str, Any] | None:
        """Honest degradation banner payload (None when live)."""
        if self.mode() == "live":
            return None
        return {
            "mode": "read_only",
            "reason": "daily spend cap reached" if self.ledger.cap_hit() else "read-only drill",
            "cap_usd": self.limits.daily_spend_cap_usd,
            "spent_today_usd": self.ledger.spent_today(),
            "probes_today": self.ledger.probes_today(),
            "message": (
                "Live probes are paused for today; the map and every cached replay "
                "still work. New worlds return tomorrow."
            ),
        }

    def _log(self, record: dict[str, Any]) -> None:
        if self.analytics_path is None:
            return
        self.analytics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.analytics_path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    # ---- the request path -------------------------------------------------------
    def handle(self, raw: dict[str, Any], ip: str) -> tuple[int, dict[str, Any]]:
        """Returns (http_status, json_body). Never raises on user input."""
        world = WorldConfig.clamped(
            friction=_num_or_none(raw.get("friction")),
            push_magnitude_pct_bw=_num(raw.get("push_magnitude_pct_bw"), 0.0),
            push_direction_deg=_num(raw.get("push_direction_deg"), 90.0),
            push_start_s=_num(raw.get("push_start_s"), 2.0),
            seed=int(_num(raw.get("seed"), 0.0)),
        )
        key = world.cache_key()
        base: dict[str, Any] = {
            "world": json.loads(world.model_dump_json()),
            "cache_key": key,
            "clamped": json.loads(world.model_dump_json()) != _echo(raw, world),
        }
        # 1) cache hit: instant, $0, works in every mode (durable-canonical replay)
        if self.rrd_exists(key):
            self._log({"t": self.today(), "key": key, "cache_hit": True, "mode": self.mode()})
            return 200, base | {
                "status": "cached",
                "rrd_url": self.rrd_url(key),
                "telemetry": {"cache_hit": True, "cost_usd": 0.0},
                "banner": self.banner(),
            }
        # 2) read-only mode: honest banner, never an error
        if self.mode() == "read_only":
            self._log({"t": self.today(), "key": key, "cache_hit": False, "mode": "read_only"})
            return 200, base | {"status": "read_only", "banner": self.banner()}
        # 3) rate limits
        ok, why = self.limiter.allow(ip)
        if not ok:
            return 429, base | {
                "status": "rate_limited",
                "reason": why,
                "message": (
                    "Too many probes right now — cached replays still work; try again shortly."
                ),
            }
        # 4) live probe
        self.limiter.acquire()
        t0 = self.now()
        try:
            result = self.backend(world.model_dump_json())
        finally:
            self.limiter.release()
        wall = self.now() - t0
        self.ledger.record()
        telemetry = {
            "cache_hit": False,
            "wall_s": round(wall, 3),
            "timings_s": result.get("timings_s", {}),
            "cost_usd": self.limits.probe_cost_usd,  # measured per-probe cost or None
        }
        self._log(
            {
                "t": self.today(),
                "key": key,
                "cache_hit": False,
                "mode": "live",
                "outcome": result.get("outcome"),
                "wall_s": telemetry["wall_s"],
            }
        )
        return 200, base | {
            "status": "probed",
            "rrd_url": self.rrd_url(key),
            "outcome": result.get("outcome"),
            "ttf_s": result.get("ttf_s"),
            "telemetry": telemetry,
            "banner": self.banner(),
        }


def _num(v: Any, default: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f or f in (float("inf"), float("-inf")) else f  # NaN/inf → default


def _num_or_none(v: Any) -> float | None:
    """Friction: unparseable/missing → None = model default (nominal), NOT 0.0 → ice
    (first deploy coerced "lol" to 0.0 → clamped to the ice floor; wrong semantics)."""
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _echo(raw: dict[str, Any], world: WorldConfig) -> dict[str, Any]:
    """What the client sent, coerced the same way, for the 'clamped' flag."""
    return {
        "friction": _num_or_none(raw.get("friction")),
        "push_magnitude_pct_bw": _num(raw.get("push_magnitude_pct_bw"), 0.0),
        "push_direction_deg": _num(raw.get("push_direction_deg"), 90.0) % 360.0,
        "push_start_s": _num(raw.get("push_start_s"), 2.0),
        "seed": int(_num(raw.get("seed"), 0.0)),
    }
