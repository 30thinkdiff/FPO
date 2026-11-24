"""Verify the pinned simulator and JAX learner independently of the decoder weights."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.reproduce_libero_4090 import (
    Variant, atomic_json, fingerprint, gpu_reading, init_state_loader, ram_reading, validate_config,
)


def run(config, output):
    import gym
    import jax
    from flax import serialization
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.policies import policy_config  # Check the pinned decoder's imports.
    from examples.train_utils_fpo import _obs_dict, add_online_data_to_buffer
    from examples.train_utils_sim import obs_to_pi_zero_input
    from jaxrl2.agents.pixel_fpo.pixel_fpo_learner import PixelFPOLearner
    from jaxrl2.data.replay_buffer_fpo import FPOReplayBuffer

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("GPU probe requires exactly one JAX GPU.")
    module_path = Path(policy_config.__file__).resolve()
    expected_path = Path(__file__).resolve().parents[1] / "openpi/src"
    if not module_path.is_relative_to(expected_path):
        raise RuntimeError("Wrong OpenPI installation is on PYTHONPATH.")

    started = time.monotonic()
    task_id = config["task_id"]
    suite = benchmark.get_benchmark_dict()[config["suite"]]()
    task = suite.get_task(task_id)
    states = init_state_loader(Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    resolution = config["observation_resolution"]
    variant = Variant(task.language, config["resize_image"], config["query_freq"], config["agent"]["discount"])
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution,
                             control_freq=config["controller_hz"])
    try:
        state = states[config["train_init_ids"][0]]
        seed = config["train_seed"]

        def collect():
            env.seed(seed)
            env.reset()
            obs = env.set_init_state(state)
            first = dict(
                state=fingerprint(env.get_sim_state()), camera=fingerprint(obs["agentview_image"]),
                wrist=fingerprint(obs["robot0_eye_in_hand_image"]),
            )
            observations = [_obs_dict(obs, variant)]
            for step in range(8):
                # Small deterministic motions solely to exercise real sensors and the physics interface.
                command = np.array([0.08, (-1) ** step * 0.08, 0, 0, 0, 0, -1], dtype=np.float32)
                obs, _, done, _ = env.step(command)
                if done:
                    raise RuntimeError("Simulator unexpectedly terminated during interface probe.")
                observations.append(_obs_dict(obs, variant))
            return first, observations, obs_to_pi_zero_input(obs, variant)

        first, observations, base_input = collect()
        repeated, _, _ = collect()
        if first != repeated:
            raise RuntimeError("Loading the same initial state changed the first observation.")
    finally:
        env.close()

    observation_space = gym.spaces.Dict({
        key: gym.spaces.Box(
            0 if value.dtype == np.uint8 else -np.inf, 255 if value.dtype == np.uint8 else np.inf,
            value.shape[1:], value.dtype,
        )
        for key, value in observations[0].items()
    })
    latent_space = gym.spaces.Box(-2.0, 2.0, (1, 32), np.float32)
    init_start = time.monotonic()
    agent = PixelFPOLearner(config["train_seed"], observations[0], np.zeros((1, 1, 32), np.float32),
                            **config["agent"])
    replay = FPOReplayBuffer(observation_space, latent_space, 8, agent.num_fpo_samples, config["buffer_window"])
    agent.sync_old_actor()
    actions, stats = zip(*(agent.sample_actions_with_info(obs) for obs in observations[:-1]))
    trajectory = dict(
        observations=observations, actions=[a.reshape(1, 32) for a in actions],
        rewards=np.full(8, -1.0), masks=np.ones(8),
        fpo_info={k: np.stack([d[k] for d in stats]) for k in stats[0]},
    )
    add_online_data_to_buffer(variant, trajectory, replay)
    replay.compute_gae(agent.values, agent.fresh_values, agent.discount, agent.gae_lambda,
                       chunk_size=config["gae_chunk_size"])
    startup_seconds = time.monotonic() - init_start
    idx = np.arange(config["batch_size"]) % len(replay)
    batch = replay.sample(config["batch_size"], indx=idx)
    rows = []
    for _ in range(3):
        begin_update = time.monotonic()
        before = {key: serialization.to_bytes(agent._save_dict[key].params) for key in ("actor", "critic")}
        metrics = {key: float(value) for key, value in jax.device_get(agent.update(batch)).items()
                   if np.ndim(value) == 0}
        if not np.isfinite(list(metrics.values())).all():
            raise RuntimeError("Nonfinite gradient or update metric.")
        changed = {key: serialization.to_bytes(agent._save_dict[key].params) != original
                   for key, original in before.items()}
        if not all(changed.values()):
            raise RuntimeError(f"An optimizer did not update its parameters: {changed}")
        rows.append(dict(seconds=time.monotonic() - begin_update, metrics=metrics, changed=changed, gpu=gpu_reading()))
    encoded = serialization.to_bytes(agent._save_dict)
    save_path = output / "agent.msgpack"
    save_path.write_bytes(encoded)
    restored = serialization.from_bytes(agent._save_dict, save_path.read_bytes())
    if serialization.to_bytes(restored) != encoded:
        raise RuntimeError("Optimizer/learner state changed after checkpoint reload.")
    key = jax.random.PRNGKey(123)
    expected = agent.sample_actions(observations[0], seed=key)
    agent._actor = restored["actor"]
    np.testing.assert_array_equal(agent.sample_actions(observations[0], seed=key), expected)
    atomic_json(output / "report.json", dict(
        status="completed", scope="simulator and learner interface probe using scripted actions; no pi0 decoder",
        suite=config["suite"], task_id=task_id, task=task.language, matching_initial_signatures=first,
        shapes={k: tuple(v.shape) for k, v in base_input.items() if isinstance(v, np.ndarray)},
        scripted_env_steps=16, replay_samples=len(replay), optimizer_updates=3, optimizer_reload_verified=True,
        learner_setup_and_gae_seconds=startup_seconds, elapsed_seconds=time.monotonic() - started,
        cgroup_ram_bytes=ram_reading(), updates=rows,
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    args.output.mkdir(parents=True, exist_ok=False)
    run(config, args.output)


if __name__ == "__main__":
    main()
