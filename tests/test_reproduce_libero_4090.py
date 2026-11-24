import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from scripts.reproduce_libero_4090 import (
    checkpoint_identity, collection_mode, commit_training, restore_training, validate_config,
)


class TestReproductionProtocol(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "configs/libero_4090_p1.json"
        self.config = json.loads(path.read_text())

    def test_warmup_uses_transitions_even_when_gradient_count_is_zero(self):
        threshold = self.config["start_online_updates"]
        self.assertEqual(collection_mode(threshold - 1, threshold), "gaussian_latent")
        self.assertEqual(collection_mode(threshold, threshold), "actor")

    def test_reject_train_evaluation_overlap(self):
        validate_config(self.config)
        self.config["eval_init_ids"].append(self.config["train_init_ids"][0])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            validate_config(self.config)

    def test_reject_control_window_larger_than_decoder_horizon(self):
        self.config["query_freq"] = 51
        with self.assertRaisesRegex(ValueError, "50"):
            validate_config(self.config)

    def test_checkpoint_receipt_detects_bit_change_and_extra_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata = root / "params/_METADATA"
            metadata.parent.mkdir()
            metadata.write_bytes(b"official metadata")
            receipt = root / "checkpoint_receipt.json"
            receipt.write_text(json.dumps({
                "checkpoint": "pi0_libero",
                "assets": {"params/_METADATA": {"sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()}},
            }))
            self.assertEqual(len(checkpoint_identity(root, receipt)), 64)
            (root / "params/extra").touch()
            with self.assertRaisesRegex(ValueError, "every weight"):
                checkpoint_identity(root, receipt)
            (root / "params/extra").unlink()
            metadata.write_bytes(b"changed metadata")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                checkpoint_identity(root, receipt)

    def test_checkpoint_receipt_rejects_symlink_outside_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            external = folder / "external"
            external.write_text("outside")
            root = folder / "model"
            (root / "params").mkdir(parents=True)
            (root / "params/_METADATA").symlink_to(external)
            receipt = root / "checkpoint_receipt.json"
            receipt.write_text(json.dumps({
                "checkpoint": "pi0_libero",
                "assets": {"params/_METADATA": {"sha256": hashlib.sha256(external.read_bytes()).hexdigest()}},
            }))
            with self.assertRaisesRegex(ValueError, "receipt path"):
                checkpoint_identity(root, receipt)

    @unittest.skipUnless(find_spec("jax") and find_spec("flax"), "requires JAX and Flax")
    def test_resume_restores_replay_cursor_and_rng(self):
        class Learner:
            _actor = {"params": np.array([0.25]), "opt_state": np.array([12])}
            _critic = {"params": np.array([0.5]), "opt_state": np.array([12])}
            _old_actor_params = np.array([0.2])
            _target_critic_params = np.array([0.4])
            _rng = np.array([17, 23], dtype=np.uint32)

            @property
            def _save_dict(self):
                return dict(actor=self._actor, critic=self._critic, old_actor_params=self._old_actor_params,
                            target_critic_params=self._target_critic_params)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent = Learner()
            replay = SimpleNamespace(
                data={"observations": {"pixels": np.arange(5, dtype=np.uint8)}, "rewards": np.arange(5)},
                size=3, capacity=5, traj_bounds={0: (0, 2), 1: (2, 3)},
            )
            cursor = dict(serial=0, gradient_steps=12, updates_remaining=2)
            rng = np.random.default_rng(42)
            commit_training(root, agent, replay, cursor, rng)
            expected_draw = rng.integers(10000)
            agent._rng = np.zeros(2, dtype=np.uint32)
            agent._actor = {"params": np.zeros(1), "opt_state": np.zeros(1, dtype=np.int64)}
            replay.data["rewards"][:] = 0
            path = root / json.loads((root / "resume.json").read_text())["state"]
            got = restore_training(path, agent, replay, rng)
            self.assertEqual(got, cursor)
            self.assertEqual(rng.integers(10000), expected_draw)
            np.testing.assert_array_equal(agent._rng, [17, 23])
            np.testing.assert_array_equal(agent._actor["opt_state"], [12])
            np.testing.assert_array_equal(replay.data["rewards"][:replay.size], [0, 1, 2])
            self.assertEqual(replay._traj_counter, 2)
