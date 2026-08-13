"""G3 harness equivalence (spec §4 gate 3 — the critical gate).

Two GATES plus one diagnostic, run on Modal against the G2 checkpoint:

1. REPLAY (physics equivalence, bitwise): a native rollout records (state, action);
   our PushInjectionWrapper at identity is stepped from each NATIVE state with the SAME
   action. Per-step state diffs isolate single-step physics from closed-loop chaos —
   measured 0.0 (bitwise identical).

2. STATISTICAL (the spec's "within noise" criterion): our wrapped harness with fresh
   keys, `repeats` independent evaluations, compared to the training run's own final
   eval — gate: |pooled_mean - training_final| <= 3 * sqrt(SE_train^2 + SE_ours^2).

Diagnostic (NOT gated): same-seed Evaluator comparison native-vs-injected — measures GPU
XLA-reduction nondeterminism amplified by chaotic dynamics (native-vs-native varies
comparably), recorded in the report and NOTES.

    uv run modal run sweep/equivalence.py --run-id 20260813T155006Z

Writes equivalence-report.json to the run's Volume dir; the committed copy under
docs/measurements/ is validated by tests/test_equivalence_report.py in CI.
"""

from __future__ import annotations

import json
from typing import Any

import modal

from train.modal_app import ENV_NAME, VOLUME_MOUNT, base_image, checkpoints

# Gate structure (measured basis, 2026-08-13 diagnostic runs, DEVLOG Session 7):
# - REPLAY check = the physics-equivalence gate: wrapper is BITWISE identical per step
#   (max state diff 0.0 over 500 steps, consistent across runs).
# - STATISTICAL check = the spec's "within noise" gate vs the training-time eval.
# - The same-seed evaluator comparison is a DIAGNOSTIC ONLY: GPU/XLA reduction
#   nondeterminism amplified by 1000 chaotic steps gives it a broad run-to-run
#   distribution (observed 2.3e-4..3.7e-3 relative across identical-seed run pairs,
#   including native-vs-native) — gating on it would be a flaky test by construction.
REPLAY_TOLERANCE = 1e-9  # per-step max state diff; measured 0.0
STAT_GATE_SIGMA = 3.0

eval_image = base_image.uv_pip_install("pydantic==2.13.4").add_local_python_source(
    "train", "sweep", "configs"
)

app = modal.App("opw-equivalence")


@app.function(
    image=eval_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={VOLUME_MOUNT: checkpoints},
)
def run_equivalence(
    run_id: str,
    checkpoint: str = "converged",
    num_eval_envs: int = 128,
    repeats: int = 5,
    exact_seed: int = 1234,
    stat_seed: int = 5678,
) -> dict[str, Any]:
    from pathlib import Path

    import jax
    from brax.training import acting
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import locomotion_params

    from configs.world import WorldConfig
    from sweep.injected_env import PushInjectionWrapper, find_body_id
    from train.modal_app import ENV_NAME as env_name

    run_dir = Path(VOLUME_MOUNT) / "runs" / run_id
    params_path = run_dir / "checkpoints" / checkpoint
    assert params_path.exists(), f"missing checkpoint {params_path}"

    from brax.io import model as brax_model

    params = brax_model.load_params(str(params_path))

    env_cfg = registry.get_default_config(env_name)
    episode_length = int(env_cfg.episode_length)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    network_config = dict(ppo_params.network_factory)

    def build_policy_fn(sample_env: Any) -> Any:
        normalize = lambda x, y: x  # noqa: E731
        if ppo_params.get("normalize_observations", False):
            normalize = running_statistics.normalize
        ppo_network = ppo_networks.make_ppo_networks(
            sample_env.observation_size,
            sample_env.action_size,
            preprocess_observations_fn=normalize,
            **network_config,
        )
        make_policy = ppo_networks.make_inference_fn(ppo_network)
        # Stochastic policy: matches brax ppo.train's eval (deterministic_eval defaults
        # False and the recipe does not override it).
        return lambda p: make_policy(p, deterministic=False)

    def native_env() -> Any:
        return wrapper.wrap_for_brax_training(
            registry.load(env_name, config=env_cfg), episode_length=episode_length
        )

    identity = WorldConfig()  # friction None, push 0 — the nominal world
    nominal_meta: dict[str, Any] = {}

    def injected_env() -> Any:
        inner = registry.load(env_name, config=env_cfg)
        torso_id = find_body_id(inner.mj_model)
        mass = float(inner.mj_model.body_subtreemass[torso_id])
        nominal_meta.update(
            torso_body=inner.mj_model.body(torso_id).name,
            robot_mass_kg=mass,
            gravity_m_s2=float(abs(inner.mj_model.opt.gravity[2])),
        )
        injected = PushInjectionWrapper(inner, identity, torso_id, mass)
        return wrapper.wrap_for_brax_training(injected, episode_length=episode_length)

    def evaluate(env: Any, key: jax.Array) -> dict[str, float]:
        policy_fn = build_policy_fn(env)
        evaluator = acting.Evaluator(
            env,
            policy_fn,
            num_eval_envs=num_eval_envs,
            episode_length=episode_length,
            action_repeat=1,
            key=key,
        )
        metrics = evaluator.run_evaluation(params, training_metrics={})
        return {k: float(v) for k, v in metrics.items()}

    # ---- Check 0: per-step replay divergence (physics-equivalence isolation) ----
    # Native single-env rollout records (state, action); our wrapper then steps from
    # each NATIVE state with the SAME action. Per-step state diffs isolate single-step
    # physics equivalence from closed-loop chaos amplification (the exact-metric check
    # below compares different XLA programs whose rounding may differ at the ulp level).
    import jax.numpy as jnp

    raw_native = registry.load(env_name, config=env_cfg)
    raw_injected_inner = registry.load(env_name, config=env_cfg)
    torso_id0 = find_body_id(raw_injected_inner.mj_model)
    mass0 = float(raw_injected_inner.mj_model.body_subtreemass[torso_id0])
    raw_injected = PushInjectionWrapper(raw_injected_inner, identity, torso_id0, mass0)

    policy_fn_raw = build_policy_fn(raw_native)(params)
    policy_raw = jax.jit(policy_fn_raw)
    native_step = jax.jit(raw_native.step)
    injected_step = jax.jit(raw_injected.step)

    replay_key = jax.random.PRNGKey(exact_seed + 1)
    replay_key, reset_key = jax.random.split(replay_key)
    state = raw_native.reset(reset_key)
    max_qpos_diff, max_qvel_diff, first_step_qpos_diff = 0.0, 0.0, None
    steps_compared = 0
    replay_steps = 500
    for _ in range(replay_steps):
        steps_compared += 1
        replay_key, act_key = jax.random.split(replay_key)
        action, _ = policy_raw(state.obs, act_key)
        native_next = native_step(state, action)
        replay_state = state.replace(
            info=dict(state.info) | {"push_yaw": jnp.zeros(()), "push_yaw_captured": jnp.zeros(())}
        )
        injected_next = injected_step(replay_state, action)
        qpos_diff = float(jnp.max(jnp.abs(native_next.data.qpos - injected_next.data.qpos)))
        qvel_diff = float(jnp.max(jnp.abs(native_next.data.qvel - injected_next.data.qvel)))
        if first_step_qpos_diff is None:
            first_step_qpos_diff = qpos_diff
        max_qpos_diff = max(max_qpos_diff, qpos_diff)
        max_qvel_diff = max(max_qvel_diff, qvel_diff)
        state = native_next
        if bool(state.done):
            break

    replay_check = {
        "steps_compared": steps_compared,
        "first_step_max_qpos_diff": first_step_qpos_diff,
        "max_qpos_diff": max_qpos_diff,
        "max_qvel_diff": max_qvel_diff,
        "tolerance": REPLAY_TOLERANCE,
        "passed": max(max_qpos_diff, max_qvel_diff) <= REPLAY_TOLERANCE,
        "note": "wrapper stepped from NATIVE states with identical actions; diffs are "
        "single-step physics deltas, not closed-loop divergence",
    }

    # ---- Diagnostic: same-seed evaluator comparison (NOT gated; see header) ----
    exact_key = jax.random.PRNGKey(exact_seed)
    native_metrics = evaluate(native_env(), exact_key)
    injected_metrics = evaluate(injected_env(), exact_key)
    reward_native = native_metrics["eval/episode_reward"]
    reward_injected = injected_metrics["eval/episode_reward"]
    exact_rel_diff = abs(reward_injected - reward_native) / max(abs(reward_native), 1e-9)

    # ---- Check 2: statistical vs training-time final eval ----
    train_final = json.loads(
        [line for line in (run_dir / "metrics.jsonl").read_text().splitlines() if line][-1]
    )
    train_mean = float(train_final["eval/episode_reward"])
    train_std = float(train_final["eval/episode_reward_std"])
    train_n = num_eval_envs  # training eval used num_eval_envs=128 (brax default; recipe)

    stat_key = jax.random.PRNGKey(stat_seed)
    our_means: list[float] = []
    our_stds: list[float] = []
    for _ in range(repeats):
        stat_key, sub = jax.random.split(stat_key)
        metrics = evaluate(injected_env(), sub)
        our_means.append(metrics["eval/episode_reward"])
        our_stds.append(metrics["eval/episode_reward_std"])
    pooled_mean = sum(our_means) / repeats
    mean_std = sum(our_stds) / repeats
    se_train = train_std / (train_n**0.5)
    se_ours = mean_std / ((num_eval_envs * repeats) ** 0.5)
    gate = STAT_GATE_SIGMA * (se_train**2 + se_ours**2) ** 0.5
    stat_diff = abs(pooled_mean - train_mean)
    stat_pass = stat_diff <= gate

    report = {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "env_name": ENV_NAME,
        "protocol": {
            "num_eval_envs": num_eval_envs,
            "episode_length": episode_length,
            "stochastic_policy": True,
            "repeats": repeats,
            "exact_seed": exact_seed,
            "stat_seed": stat_seed,
        },
        "nominal_world": json.loads(identity.model_dump_json()),
        "model": nominal_meta,
        "replay_check": replay_check,
        "same_seed_evaluator_diagnostic": {
            "reward_native": reward_native,
            "reward_injected_identity": reward_injected,
            "relative_diff": exact_rel_diff,
            "gated": False,
            "note": "diagnostic only — measures GPU-nondeterminism amplification, not "
            "harness difference (native-vs-native varies comparably across runs; see "
            "equivalence-report.md); physics equivalence is the replay check",
            "native_avg_episode_length": native_metrics["eval/avg_episode_length"],
            "injected_avg_episode_length": injected_metrics["eval/avg_episode_length"],
        },
        "statistical_check": {
            "training_final_mean": train_mean,
            "training_final_std": train_std,
            "our_means": our_means,
            "pooled_mean": pooled_mean,
            "abs_diff": stat_diff,
            "gate": gate,
            "gate_sigma": STAT_GATE_SIGMA,
            "passed": stat_pass,
        },
        "passed": bool(replay_check["passed"]) and stat_pass,
    }
    (run_dir / "equivalence-report.json").write_text(json.dumps(report, indent=2))
    checkpoints.commit()
    return report


@app.local_entrypoint()
def main(run_id: str, checkpoint: str = "converged", repeats: int = 5) -> None:
    result = run_equivalence.remote(run_id=run_id, checkpoint=checkpoint, repeats=repeats)
    print(json.dumps(result, indent=2))
