"""M1.5-01: retrain with ONE change — the training distribution (Taylor, 2026-08-15).

v1 = the exact G2 recipe (200M steps, same PPO hyperparams, same network, same seeds
scheme, same 20 evals) with only the randomization changed:
  1. env `pert_config.enable = True` (the env's own velocity-kick perturbations —
     shipped disabled in Playground; the map showed every shove is out-of-distribution);
  2. floor friction DR widened U(0.4, 1.0) → U(0.1, 1.0) — floor ~0.1, NOT the map floor
     0.05 (training at physically-lost worlds risks teaching a conservative crouch;
     the re-sweep reports what it bought below 0.1 regardless).
Everything else — kicks' own default ranges included — is Playground's untouched
default. The config delta is recorded VERBATIM (source text + hashes + the two
overrides) as a second DR record, same treatment as the original.

    uv run modal run train/retrain_v1.py --smoke true
    uv run modal run train/retrain_v1.py
"""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import UTC, datetime
from typing import Any

import modal

from train.modal_app import (
    ENV_NAME,
    NUM_EVALS,
    SMOKE_NUM_TIMESTEPS,
    VOLUME_MOUNT,
    checkpoints,
    train_image,
)

FRICTION_MIN_V1 = 0.1  # was 0.4
FRICTION_MAX_V1 = 1.0  # unchanged

app = modal.App("opw-train-v1")


def make_v1_randomizer() -> Any:
    """Playground's Go1 domain_randomize with ONLY the friction floor changed.

    Returns (fn, source_of_original, source_of_ours) — the two texts go into the DR
    delta record so the change is auditable line-by-line.
    """
    import jax
    from mujoco_playground._src.locomotion.go1 import randomize as go1_rand

    original_src = inspect.getsource(go1_rand.domain_randomize)
    # Copy the original body verbatim and swap the single friction line.
    ours_src = original_src.replace(
        "jax.random.uniform(key, minval=0.4, maxval=1.0)",
        f"jax.random.uniform(key, minval={FRICTION_MIN_V1}, maxval={FRICTION_MAX_V1})",
    )
    assert ours_src != original_src, "friction line not found in Playground's randomize.py"
    assert ours_src.count(f"minval={FRICTION_MIN_V1}") == 1
    namespace: dict[str, Any] = {
        "jax": jax,
        "mjx": go1_rand.mjx,
        "FLOOR_GEOM_ID": go1_rand.FLOOR_GEOM_ID,
        "TORSO_BODY_ID": go1_rand.TORSO_BODY_ID,
    }
    exec(compile(ours_src, "<v1_randomize>", "exec"), namespace)
    return namespace["domain_randomize"], original_src, ours_src


@app.function(
    image=train_image,
    gpu="A100-80GB",
    timeout=4 * 3600,
    volumes={VOLUME_MOUNT: checkpoints},
)
def train_v1(smoke: bool = False) -> dict[str, Any]:
    from pathlib import Path

    import jax
    from brax.io import model as brax_model
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params

    from train.dr_record import capture_go1_dr_record

    platforms = {d.platform for d in jax.devices()}
    assert "gpu" in platforms, f"jax sees no GPU: {jax.devices()}"

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-v1" + ("-smoke" if smoke else "")
    run_dir = Path(VOLUME_MOUNT) / "runs" / run_id
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- the ONLY two changes vs G2 -------------------------------------------------
    env_cfg = registry.get_default_config(ENV_NAME)
    pert_before = json.loads(json.dumps(env_cfg.pert_config.to_dict()))
    env_cfg.pert_config.enable = True
    randomizer, rand_original_src, rand_v1_src = make_v1_randomizer()
    # ---------------------------------------------------------------------------------

    dr_record = capture_go1_dr_record()
    dr_record["v1_delta"] = {
        "description": (
            "v1 = G2 recipe with ONLY the training distribution changed: "
            "pert_config.enable=True (env-native velocity kicks, Playground defaults for "
            "their ranges) and floor-friction DR floor 0.4 -> 0.1 (max unchanged 1.0)."
        ),
        "pert_config_before": pert_before,
        "pert_config_after": json.loads(json.dumps(env_cfg.pert_config.to_dict())),
        "friction_dr_before": [0.4, 1.0],
        "friction_dr_after": [FRICTION_MIN_V1, FRICTION_MAX_V1],
        "randomize_original_src_sha256": hashlib.sha256(rand_original_src.encode()).hexdigest(),
        "randomize_v1_src": rand_v1_src,
        "randomize_v1_src_sha256": hashlib.sha256(rand_v1_src.encode()).hexdigest(),
    }
    (run_dir / "dr-record-v1.json").write_text(json.dumps(dr_record, indent=2))

    env = registry.load(ENV_NAME, config=env_cfg)
    eval_env = registry.load(ENV_NAME, config=env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(ENV_NAME)
    ppo_params.num_evals = NUM_EVALS
    if smoke:
        ppo_params.num_timesteps = SMOKE_NUM_TIMESTEPS
        ppo_params.num_evals = 2

    training_params = dict(ppo_params)
    network_config = training_params.pop("network_factory", None)
    del training_params["num_evals"]

    def network_factory(*args: Any, **kwargs: Any) -> Any:
        merged = dict(network_config or {})
        merged.update(kwargs)
        return ppo_networks.make_ppo_networks(*args, **merged)

    metrics_path = run_dir / "metrics.jsonl"

    def progress(step: int, metrics: dict[str, Any]) -> None:
        record = {"step": int(step)} | {k: float(v) for k, v in metrics.items()}
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        checkpoints.commit()

    def save_policy(step: int, make_policy: Any, params: Any) -> None:
        brax_model.save_params(str(ckpt_dir / f"step_{int(step):012d}"), params)
        checkpoints.commit()

    train_fn_kwargs = dict(training_params)
    train_fn_kwargs["network_factory"] = network_factory
    train_fn_kwargs["randomization_fn"] = randomizer  # ← v1 randomizer
    train_fn_kwargs["wrap_env_fn"] = wrapper.wrap_for_brax_training

    start = datetime.now(UTC)
    _, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        num_evals=ppo_params.num_evals,
        progress_fn=progress,
        policy_params_fn=save_policy,
        **train_fn_kwargs,
    )
    wall_seconds = (datetime.now(UTC) - start).total_seconds()
    brax_model.save_params(str(ckpt_dir / "converged"), params)

    summary = {
        "run_id": run_id,
        "variant": "v1",
        "env_name": ENV_NAME,
        "smoke": smoke,
        "wall_seconds": wall_seconds,
        "num_timesteps": int(ppo_params.num_timesteps),
        "num_evals": int(ppo_params.num_evals),
        "gpu": "A100-80GB",
        "changes_vs_v0": ["pert_config.enable=True", f"friction DR floor 0.4->{FRICTION_MIN_V1}"],
        "checkpoints": sorted(p.name for p in ckpt_dir.iterdir()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    checkpoints.commit()
    return summary


@app.local_entrypoint()
def main(smoke: bool = False) -> None:
    print(json.dumps(train_v1.remote(smoke=smoke), indent=2))
