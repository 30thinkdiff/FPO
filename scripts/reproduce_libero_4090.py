"""Bounded single-GPU LIBERO FPO experiment using the upstream learner."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


@dataclass(frozen=True)
class Variant:
    task_description: str
    resize_image: int
    query_freq: int
    discount: float
    env: str = "libero"
    add_states: bool = True


def collection_mode(transitions, warmup_threshold):
    return "gaussian_latent" if transitions < warmup_threshold else "actor"


def validate_config(config):
    train = config["train_init_ids"]
    evaluation = config["eval_init_ids"]
    if not train or not evaluation or min(train + evaluation) < 0:
        raise ValueError("Training and evaluation require explicit nonnegative initial-state IDs.")
    if len(set(train + evaluation)) != len(train + evaluation):
        raise ValueError("Training and evaluation initial-state IDs must be disjoint and unique.")
    integers = (
        "controller_hz", "query_freq", "max_timesteps", "batch_size", "start_online_updates",
        "max_grad_steps", "multi_grad_step", "max_train_episodes", "min_actor_episodes", "gae_chunk_size",
    )
    if any(type(config[k]) is not int or config[k] < 1 for k in integers):
        raise ValueError("Control and training budgets must be positive integers.")
    if config["batch_size"] < 2 or config["query_freq"] > 50 or config["max_grad_steps"] < 2:
        raise ValueError("Batch >= 2, query <= 50, and at least two updates are required.")


def atomic_json(path, document):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_identity(root, receipt):
    inventory = json.loads(receipt.read_text())
    if inventory["checkpoint"] != "pi0_libero":
        raise ValueError("Expected the official pi0_libero checkpoint.")
    files = inventory["assets"]
    for name, spec in files.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Invalid checkpoint receipt path: {name}")
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"Base checkpoint checksum mismatch: {name}")
    expected = set(files)
    actual = {
        str(p.relative_to(root)) for p in root.rglob("*")
        if p.is_file() and (p.relative_to(root).parts[0] in {"assets", "params"})
    }
    if expected != actual or "params/_METADATA" not in actual:
        raise ValueError("Base checkpoint receipt must bind every weight and normalization file.")
    return fingerprint(np.frombuffer(receipt.read_bytes(), dtype=np.uint8))


def flattened(arrays, size):
    for name, value in arrays.items():
        if isinstance(value, dict):
            for leaf, array in value.items():
                yield f"{name}/{leaf}", array[:size]
        else:
            yield name, value[:size]


def gpu_reading():
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    index, used, total, util = map(int, output.strip().split(","))
    return dict(index=index, used_mib=used, total_mib=total, utilization_pct=util)


def ram_reading():
    cgroup = Path("/sys/fs/cgroup/memory.current")
    if cgroup.is_file():
        return int(cgroup.read_text())
    status = Path("/proc/self/status").read_text().splitlines()
    return int(next(line for line in status if line.startswith("VmRSS:")).split()[1]) * 1024


def init_state_loader(path):
    import torch

    # NumPy ndarray metadata in the pinned LIBERO distributions is used by torch.save.
    # Keep the allowlist local. No pickle fallback for arbitrary Python objects.
    with torch.serialization.safe_globals([
        (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
        np.ndarray, np.dtype, np.dtypes.Float32DType, np.dtypes.Float64DType,
    ]):
        data = torch.load(path, map_location="cpu", weights_only=True)
    return np.asarray(data)


def rollout(env, state, seed, mode, config, variant, agent, base, directory=None):
    import jax
    from examples.train_utils_fpo import PI0_HORIZON, _obs_dict, latent_to_pi0_noise
    from examples.train_utils_sim import obs_to_pi_zero_input

    started = time.monotonic()
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(state)
    for _ in range(config["settle_steps"]):
        obs, _, done, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
        if done:
            raise RuntimeError("LIBERO terminated during initial state settling.")
    initial_state_sha = fingerprint(env.get_sim_state())
    initial_input_sha = {
        "camera": fingerprint(obs["agentview_image"]),
        "wrist": fingerprint(obs["robot0_eye_in_hand_image"]),
        "proprio": fingerprint(obs_to_pi_zero_input(obs, variant)["observation/state"]),
    }

    actions = []
    observations = []
    latents = []
    draws = []
    decisions = []
    frames = []
    timings = []
    track = mode in {"gaussian_latent", "actor"}
    rng = jax.random.PRNGKey(seed)
    success = done = False
    for step in range(config["max_timesteps"]):
        action_index = step % config["query_freq"]
        if action_index == 0:
            rng, key = jax.random.split(rng)
            obs_dict = _obs_dict(obs, variant)
            decision_start = time.monotonic()
            if mode == "B0":
                noise = np.asarray(jax.random.normal(key, (1, PI0_HORIZON, 32)))
                latent = noise[:, :1]
            elif track:
                supplied = None
                if mode == "gaussian_latent":
                    supplied = np.clip(
                        np.asarray(jax.random.normal(key, (1, 32))), -agent.latent_clip, agent.latent_clip
                    )
                latent, info = agent.sample_actions_with_info(obs_dict, latent=supplied)
                draws.append(info)
            else:
                latent = agent.sample_actions(obs_dict, seed=key)
            if mode != "B0":
                noise = latent_to_pi0_noise(latent, (1, 32))
            base_start = time.monotonic()
            base_input = obs_to_pi_zero_input(obs, variant)
            chunk = np.asarray(base.infer(base_input, noise=noise)["actions"])
            if chunk.shape != (PI0_HORIZON, 7) or not np.isfinite(chunk).all():
                raise ValueError(f"Invalid pi0 output: {chunk.shape}")
            chunk = np.clip(chunk, -1.0, 1.0)
            timings.append([time.monotonic() - decision_start, time.monotonic() - base_start])
            latents.append(np.reshape(latent, (1, 32)))
            observations.append(obs_dict)
            decisions.append(dict(step=step, noise_sha256=fingerprint(noise), latent=latents[-1][0].tolist()))
        if directory is not None:
            frames.append(obs["agentview_image"][::-1, ::-1].copy())
        command = chunk[action_index]
        obs, _, done, _ = env.step(command)
        actions.append(command.tolist())
        success = bool(env.check_success())
        if success or done:
            break

    row = dict(
        seed=seed, mode=mode, success=success, terminated=bool(success or done), env_steps=len(actions),
        decisions=len(latents), first_state_sha256=initial_state_sha, first_input_sha256=initial_input_sha,
        inference_seconds=[x[0] for x in timings], pi0_seconds=[x[1] for x in timings],
        rollout_seconds=time.monotonic() - started,
    )
    if directory is not None:
        import cv2

        temporary = directory.with_name(directory.name + ".partial")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        path = temporary / "video.mp4"
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), config["controller_hz"],
                                 (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not open the rollout video encoder.")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        atomic_json(temporary / "trace.json", dict(
            actions=actions, decisions=decisions, frame_binding="frame i precedes executed action i"
        ))
        atomic_json(temporary / "result.json", row)
        temporary.replace(directory)

    trajectory = None
    if track:
        n = len(latents)
        rewards, masks = np.full(n, -1.0), np.ones(n)
        if success:
            rewards[-1] = 0.0
        if success or done:
            masks[-1] = 0.0
        trajectory = dict(
            actions=latents, observations=observations + [_obs_dict(obs, variant)],
            rewards=rewards, masks=masks, fpo_info={k: np.stack([d[k] for d in draws]) for k in draws[0]},
        )
    return row, trajectory


def commit_training(output, agent, replay, cursor, rng):
    from flax import serialization

    serial = cursor["serial"] + 1
    while (output / f"state_{serial:05d}").exists():
        serial += 1  # A complete directory may exist even if the pointer rename never happened.
    name = f"state_{serial:05d}"
    final = output / name
    temporary = output / f"{name}.partial"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    (temporary / "agent.msgpack").write_bytes(serialization.to_bytes(agent._save_dict))
    np.savez(temporary / "replay.npz", **dict(flattened(replay.data, replay.size)))
    atomic_json(temporary / "cursor.json", dict(
        cursor, serial=serial, agent_rng=np.asarray(agent._rng).tolist(),
        sample_rng=rng.bit_generator.state, bounds=list(replay.traj_bounds.values()),
    ))
    temporary.replace(final)
    atomic_json(output / "resume.json", {"state": name})
    # Keep the latest complete generation and its predecessor.
    generations = sorted(output.glob("state_[0-9][0-9][0-9][0-9][0-9]"))
    for stale in generations[:-2]:
        if stale.is_dir() and not stale.is_symlink():
            shutil.rmtree(stale)
    cursor["serial"] = serial


def restore_training(path, agent, replay, rng):
    from flax import serialization
    import jax.numpy as jnp

    cursor = json.loads((path / "cursor.json").read_text())
    state = serialization.from_bytes(agent._save_dict, (path / "agent.msgpack").read_bytes())
    agent._actor, agent._critic = state["actor"], state["critic"]
    agent._target_critic_params, agent._old_actor_params = state["target_critic_params"], state["old_actor_params"]
    agent._rng = jnp.asarray(cursor.pop("agent_rng"), dtype=jnp.uint32)
    rng.bit_generator.state = cursor.pop("sample_rng")
    bounds = cursor.pop("bounds")
    replay.traj_bounds = dict(enumerate(map(tuple, bounds)))
    replay._traj_counter = len(bounds)
    replay._start = replay.size = bounds[-1][1] if bounds else 0
    if replay.size > replay.capacity:
        raise RuntimeError("Saved replay size exceeds configured capacity.")
    with np.load(path / "replay.npz", allow_pickle=False) as data:
        for key, destination in flattened(replay.data, replay.size):
            destination[:] = data[key]
    return cursor


def finite_stats(agent, metrics, actor_before, critic_before, tree_delta):
    import jax
    result = {k: float(v) for k, v in jax.device_get(metrics).items() if np.ndim(v) == 0}
    result["actor_update_l2"] = float(tree_delta(actor_before, agent._actor.params))
    result["critic_update_l2"] = float(tree_delta(critic_before, agent._critic.params))
    if not all(np.isfinite(list(result.values()))) or result.get("skipped_nan", 0):
        raise RuntimeError("Updater rejected nonfinite inputs or returned a nonfinite metric.")
    return result


def run(args):
    import gym
    import jax
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config
    from examples.train_utils_fpo import add_online_data_to_buffer
    from jaxrl2.agents.pixel_fpo.pixel_fpo_learner import PixelFPOLearner
    from jaxrl2.data.replay_buffer_fpo import FPOReplayBuffer

    begun = time.monotonic()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("This protocol requires one visible JAX GPU.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    known = {"STOP", ".lock", "status.json"}
    if not (output / "manifest.json").exists() and any(p.name not in known for p in output.iterdir()):
        raise RuntimeError("Use a fresh output directory for an initial run.")
    lock = output / ".lock"
    # flock releases on process termination. Keep the descriptor for this invocation.
    import fcntl
    fd = lock.open("a")
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        execute(args, config, output, begun, gym, jax, benchmark, get_libero_path, OffScreenRenderEnv,
                policy_config, openpi_config, PixelFPOLearner, FPOReplayBuffer, add_online_data_to_buffer)
    except Exception as error:
        status_path = output / "status.json"
        old = json.loads(status_path.read_text()) if status_path.exists() else {}
        atomic_json(status_path, dict(old, status="error", error=f"{type(error).__name__}: {error}"))
        raise
    finally:
        fd.close()


def execute(args, config, output, begun, gym, jax, benchmark, get_libero_path, env_class,
            policy_config, openpi_config, learner_class, replay_class, insert_trajectory):
    import jax.numpy as jnp

    @jax.jit
    def tree_delta(before, after):
        squared = jax.tree.map(lambda x, y: jnp.vdot(x - y, x - y), before, after)
        return jnp.sqrt(sum(jax.tree.leaves(squared)))

    def status(phase, **values):
        reading = dict(status=phase, wall_seconds=time.monotonic() - begun, gpu=gpu_reading(),
                       cgroup_ram_bytes=ram_reading(), **values)
        atomic_json(output / "status.json", reading)
        print(json.dumps(reading, allow_nan=False), flush=True)

    def stop_reason():
        if (output / "STOP").exists():
            return "STOP"
        if time.monotonic() - begun >= args.max_seconds:
            return "wall_budget"
        if shutil.disk_usage(output).free < args.min_free_gib * 1024**3:
            return "disk_budget"
        return None

    def eval_path(arm, init_id):
        return output / f"{arm}_init_{init_id:03d}"

    def eval_arm(arm):
        rows = []
        for init_id in config["eval_init_ids"]:
            directory = eval_path(arm, init_id)
            already_saved = directory.exists()
            if already_saved:
                row = json.loads((directory / "result.json").read_text())
            else:
                reason = stop_reason()
                if reason:
                    return rows, reason
                row, _ = rollout(env, states[init_id], config["eval_seed"] + init_id, arm, config, variant,
                                 agent, base, directory)
            if arm != "B0":
                control = json.loads((eval_path("B0", init_id) / "result.json").read_text())
                if any(row[k] != control[k] for k in ("first_state_sha256", "first_input_sha256")):
                    raise RuntimeError(f"Pairing mismatch: {arm} initial state {init_id}")
            rows.append(dict(row, init_id=init_id, artifact=directory.name))
            if not already_saved:
                status("evaluating", arm=arm, init_id=init_id, success=row["success"], env_steps=row["env_steps"])
        return rows, None

    checkpoint_sha = checkpoint_identity(args.checkpoint, args.receipt)
    suite = benchmark.get_benchmark_dict()[config["suite"]]()
    task_id = config["task_id"]
    task = suite.get_task(task_id)
    if config["suite"] == "libero_90" and task_id == 56 and "butter" not in task.language.lower():
        raise RuntimeError(f"Unexpected task at LIBERO-90/56: {task.language}")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    states = init_state_loader(init_path)
    ids = config["train_init_ids"] + config["eval_init_ids"]
    if max(ids) >= len(states) or any(not np.isfinite(states[i]).all() for i in ids):
        raise ValueError("Selected initial states are absent or invalid.")
    metadata = dict(
        config=config, source_revision=args.source_revision, checkpoint_receipt_sha256=checkpoint_sha,
        task=task.language, bddl_sha256=file_sha256(bddl),
        init_state_sha256={str(i): fingerprint(states[i]) for i in ids},
        jax_version=jax.__version__, jax_device=str(jax.devices()[0]), python_version=sys.version,
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text()) != metadata:
            raise RuntimeError("Existing run requires --resume and the identical manifest.")
    else:
        if args.resume:
            raise RuntimeError("Cannot resume without an existing manifest.")
        atomic_json(manifest_path, metadata)
    reason = stop_reason()
    if reason:
        status("paused", reason=reason)
        return

    resolution = config["observation_resolution"]
    env = env_class(bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution,
                    control_freq=config["controller_hz"])
    try:
        status("loading_pi0")
        load_start = time.monotonic()
        base = policy_config.create_trained_policy(openpi_config.get_config("pi0_libero"), args.checkpoint)
        status("initializing_learner", load_seconds=time.monotonic() - load_start)
        variant = Variant(task.language, config["resize_image"], config["query_freq"], config["agent"]["discount"])
        resolution = config["resize_image"]
        observation_space = gym.spaces.Dict({
            "pixels": gym.spaces.Box(0, 255, (resolution, resolution, 3, 1), np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, (8, 1), np.float32),
        })
        action_space = gym.spaces.Box(-2.0, 2.0, (1, 32), np.float32)
        agent = learner_class(
            config["train_seed"], {k: np.zeros((1, *s.shape), s.dtype) for k, s in observation_space.items()},
            np.zeros((1, 1, 32), np.float32), **config["agent"],
        )
        capacity = config["max_train_episodes"] * ((config["max_timesteps"] - 1) // config["query_freq"] + 1)
        replay = replay_class(observation_space, action_space, capacity,
                              agent.num_fpo_samples, config["buffer_window"])
        rng = np.random.default_rng(config["train_seed"])
        cursor = dict(serial=0, episodes=0, gradient_steps=0, env_steps=0, actor_episodes=0,
                      updates_remaining=0, first_actor_transition=None, records=[], updates=[])
        # Construct B1 from the exact initial actor on every invocation before state restoration.
        arms = {}
        for arm in ("B0", "B1"):
            arms[arm], reason = eval_arm(arm)
            if reason:
                status("paused", reason=reason)
                return
        if (output / "resume.json").exists():
            pointer = json.loads((output / "resume.json").read_text())["state"]
            if not pointer.startswith("state_") or not pointer.removeprefix("state_").isdigit():
                raise RuntimeError("Invalid checkpoint pointer.")
            cursor = restore_training(output / pointer, agent, replay, rng)
        while True:
            reason = stop_reason()
            if reason:
                status("paused", reason=reason, episodes=cursor["episodes"], env_steps=cursor["env_steps"],
                       transitions=len(replay), gradient_steps=cursor["gradient_steps"])
                return

            if cursor["updates_remaining"]:
                update_start = time.monotonic()
                sample = rng.integers(replay.window_start(), len(replay), size=config["batch_size"])
                actor_before, critic_before = agent._actor.params, agent._critic.params
                metrics = finite_stats(agent, agent.update(replay.sample(config["batch_size"], indx=sample)),
                                       actor_before, critic_before, tree_delta)
                cursor["gradient_steps"] += 1
                cursor["updates_remaining"] -= 1
                first_actor = cursor["first_actor_transition"]
                cursor["updates"].append(dict(
                    step=cursor["gradient_steps"], seconds=time.monotonic() - update_start,
                    actor_generated_samples=0 if first_actor is None else int(np.count_nonzero(sample >= first_actor)),
                    **metrics,
                ))
                if not cursor["updates_remaining"] or stop_reason():
                    commit_training(output, agent, replay, cursor, rng)
                status("updating", episodes=cursor["episodes"], env_steps=cursor["env_steps"],
                       transitions=len(replay), gradient_steps=cursor["gradient_steps"], metrics=metrics)
                continue

            if cursor["gradient_steps"] == config["max_grad_steps"] and cursor["actor_episodes"] >= config["min_actor_episodes"]:
                break
            episode = cursor["episodes"]
            if episode >= config["max_train_episodes"]:
                raise RuntimeError("Episode limit reached before completing online update criteria.")
            mode = collection_mode(len(replay), config["start_online_updates"])
            agent.sync_old_actor()
            train_ids = config["train_init_ids"]
            init_id = train_ids[episode % len(train_ids)]
            seed = config["train_seed"] + episode
            directory = output / f"train_episode_{episode:03d}"
            if directory.exists():
                # An interruption before state commit may leave a complete orphan rollout.
                # Recollect from the exact saved model and seed, in its own output directory.
                suffix = 1
                while directory.with_name(f"{directory.name}_retry{suffix}").exists():
                    suffix += 1
                directory = directory.with_name(f"{directory.name}_retry{suffix}")
            row, traj = rollout(env, states[init_id], seed, mode, config, variant, agent, base, directory)
            actor_rollout = mode == "actor" and cursor["gradient_steps"] > 0
            if actor_rollout and cursor["first_actor_transition"] is None:
                cursor["first_actor_transition"] = len(replay)
            insert_trajectory(variant, traj, replay)
            cursor["episodes"] += 1
            cursor["env_steps"] += row["env_steps"]
            cursor["actor_episodes"] += int(actor_rollout)
            cursor["records"].append(dict(row, init_id=init_id, artifact=directory.name))
            if len(replay) >= config["start_online_updates"]:
                remaining_grad = config["max_grad_steps"] - cursor["gradient_steps"]
                # Reserve at least one gradient update for data from a trained actor.
                if not cursor["actor_episodes"]:
                    remaining_grad = min(remaining_grad, config["max_grad_steps"] - 1 - cursor["gradient_steps"])
                cursor["updates_remaining"] = min(remaining_grad, row["decisions"] * config["multi_grad_step"])
                if cursor["updates_remaining"]:
                    replay.compute_gae(agent.values, agent.fresh_values, agent.discount, agent.gae_lambda,
                                       chunk_size=config["gae_chunk_size"])
            commit_training(output, agent, replay, cursor, rng)
            status("collected", episodes=cursor["episodes"], env_steps=cursor["env_steps"], transitions=len(replay),
                   gradient_steps=cursor["gradient_steps"], policy=mode, success=row["success"])

        if not all(any(u[key] > 0 for u in cursor["updates"]) for key in ("actor_update_l2", "critic_update_l2")):
            raise RuntimeError("The actor and critic must both show a finite nonzero parameter update.")
        if not any(u["actor_generated_samples"] for u in cursor["updates"]):
            raise RuntimeError("No minibatch included a transition collected by the updated actor.")
        from flax import serialization
        sample_before = agent.sample_actions(
            replay.sample(1, indx=np.array([0]))["observations"], seed=jax.random.PRNGKey(7)
        )
        trained_state = serialization.to_bytes(agent._save_dict)
        last_state = output / f"state_{cursor['serial']:05d}"
        cursor = restore_training(last_state, agent, replay, rng)
        sample_after = agent.sample_actions(
            replay.sample(1, indx=np.array([0]))["observations"], seed=jax.random.PRNGKey(7)
        )
        np.testing.assert_array_equal(sample_before, sample_after)
        if trained_state != serialization.to_bytes(agent._save_dict):
            raise RuntimeError("Learner state changed after disk reload.")
        arms["B2"], reason = eval_arm("B2")
        if reason:
            status("paused", reason=reason, gradient_steps=cursor["gradient_steps"])
            return
        summary = {
            arm: dict(successes=sum(row["success"] for row in rows), trials=len(rows),
                      env_steps=sum(row["env_steps"] for row in rows),
                      decisions=sum(row["decisions"] for row in rows),
                      rollout_seconds=sum(row["rollout_seconds"] for row in rows),
                      median_pi0_seconds=float(np.median([s for row in rows for s in row["pi0_seconds"]])))
            for arm, rows in arms.items()
        }
        paired = {}
        for control in ("B0", "B1"):
            comparisons = list(zip(arms[control], arms["B2"]))
            paired[control] = dict(
                gains=sum(not prior["success"] and new["success"] for prior, new in comparisons),
                losses=sum(prior["success"] and not new["success"] for prior, new in comparisons),
            )
        atomic_json(output / "summary.json", dict(
            arms=summary, evaluation_trials=arms, b2_paired_flips=paired,
            training=cursor, replay_transitions=len(replay),
            optimizer_reload_verified=True,
            paired_init_ids=config["eval_init_ids"],
        ))
        status("completed", arms=summary, gradient_steps=cursor["gradient_steps"])
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=int, default=9000)
    parser.add_argument("--min-free-gib", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 10800 or args.min_free_gib < 1:
        parser.error("This first-stage run is capped at three hours and requires a positive disk reserve.")
    run(args)


if __name__ == "__main__":
    main()
