"""World configuration — THE single source of truth for perturbation semantics.

`sweep/` and `probe/` both import from here (kickoff §3); the app's server-side clamping
(spec §7, the security boundary) uses `WorldConfig.clamped`, and the probe cache key
(spec §7: "key = config rounded to slider resolution") comes from `cache_key`.

Physics semantics are locked by spec §5:
- rollout: 5 s at env dt; push window 0.5 s; push start default t = 2 s
- push: constant lateral force on the torso, magnitude in % bodyweight (primary unit),
  direction dial in the horizontal plane relative to heading
- friction: global floor friction μ; `None` means "leave the model's default untouched"
  (the G3 nominal world) — the model default is read and recorded at G3, not hardcoded

Ranges and slider resolutions below are PROVISIONAL until G4 fixes grid ranges
(§14 #6) — the mechanism (clamp + round + canonical key) is final, the numbers are not.
Changing RESOLUTIONS invalidates the probe cache by design (the key encodes values at
those resolutions); bump CACHE_KEY_VERSION when semantics change.
"""

from __future__ import annotations

from typing import ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field

PUSH_DURATION_S: Final = 0.5
ROLLOUT_S: Final = 5.0
DEFAULT_PUSH_START_S: Final = 2.0
DEFAULT_SEED: Final = 0  # §14 #1: fixed default seed (deterministic shareable URLs)

CACHE_KEY_VERSION: Final = 1


class WorldConfig(BaseModel):
    """One world = one probe/sweep cell configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    friction: float | None = Field(
        default=None,
        description="Floor friction μ; None = model default (nominal, untouched physics)",
    )
    push_magnitude_pct_bw: float = Field(default=0.0, description="Push force, % bodyweight")
    push_direction_deg: float = Field(
        default=90.0, description="Horizontal-plane angle relative to heading; 90 = lateral"
    )
    push_start_s: float = Field(default=DEFAULT_PUSH_START_S)
    seed: int = Field(default=DEFAULT_SEED)

    # PROVISIONAL clamp ranges (finalized from G4; server-side clamping is the security
    # boundary — out-of-range requests are clamped, never rejected with an error).
    RANGES: ClassVar[dict[str, tuple[float, float]]] = {
        "friction": (0.05, 2.0),
        "push_magnitude_pct_bw": (0.0, 300.0),
        "push_direction_deg": (0.0, 360.0),
        "push_start_s": (0.5, 4.0),
    }
    # PROVISIONAL slider resolutions (cache-key rounding; finalized with the frontend).
    RESOLUTIONS: ClassVar[dict[str, float]] = {
        "friction": 0.01,
        "push_magnitude_pct_bw": 1.0,
        "push_direction_deg": 5.0,
        "push_start_s": 0.1,
    }

    @classmethod
    def clamped(
        cls,
        *,
        friction: float | None = None,
        push_magnitude_pct_bw: float = 0.0,
        push_direction_deg: float = 90.0,
        push_start_s: float = DEFAULT_PUSH_START_S,
        seed: int = DEFAULT_SEED,
    ) -> WorldConfig:
        """Build a config with every field clamped into its published range.

        Direction wraps modulo 360 (a dial, not a bounded slider). Seed is passed
        through: it is not attacker-leverageable (any int selects a PRNG stream).
        """

        def clamp(name: str, value: float) -> float:
            lo, hi = cls.RANGES[name]
            return min(max(value, lo), hi)

        return cls(
            friction=None if friction is None else clamp("friction", friction),
            push_magnitude_pct_bw=clamp("push_magnitude_pct_bw", push_magnitude_pct_bw),
            push_direction_deg=push_direction_deg % 360.0,
            push_start_s=clamp("push_start_s", push_start_s),
            seed=seed,
        )

    def rounded(self) -> WorldConfig:
        """Snap every field to its slider resolution (cache identity)."""

        def snap(name: str, value: float) -> float:
            resolution = self.RESOLUTIONS[name]
            return round(round(value / resolution) * resolution, 10)

        return WorldConfig(
            friction=None if self.friction is None else snap("friction", self.friction),
            push_magnitude_pct_bw=snap("push_magnitude_pct_bw", self.push_magnitude_pct_bw),
            push_direction_deg=snap("push_direction_deg", self.push_direction_deg) % 360.0,
            push_start_s=snap("push_start_s", self.push_start_s),
            seed=self.seed,
        )

    def cache_key(self) -> str:
        """Canonical, stable key for the rounded config (probe cache + shared URLs)."""
        r = self.rounded()
        friction = "default" if r.friction is None else f"{r.friction:.4f}"
        return (
            f"v{CACHE_KEY_VERSION}"
            f"|mu={friction}"
            f"|push={r.push_magnitude_pct_bw:.2f}"
            f"|dir={r.push_direction_deg:.2f}"
            f"|t0={r.push_start_s:.2f}"
            f"|seed={r.seed}"
        )
