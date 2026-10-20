"""Independently audit generated experimental artifacts, including model replay.

Run after training, from the repository root: python tests/validate_outputs.py
This reads artifacts and never retrains or changes the experimental protocol.
"""
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import statistics
import sys
import unittest
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reading5_sim2real_ppo import Joint  # noqa: E402


def read_csv(name):
    with (ROOT / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def group(rows, fields):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


class OutputAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = read_csv("evaluation_episodes.csv")
        cls.steps = read_csv("evaluation_steps.csv")
        cls.metrics = read_csv("metrics.csv")
        cls.training = read_csv("training_rewards.csv")
        cls.record = json.loads((ROOT / "experiment.json").read_text())
        cls.by_episode = group(cls.steps, ("policy", "environment", "seed"))

    def close_number(self, actual, expected, tolerance=1e-9):
        self.assertTrue(math.isfinite(float(actual)), f"Nonfinite result: {actual}")
        self.assertAlmostEqual(float(actual), float(expected), delta=tolerance)

    def test_complete_paired_design_and_finite_records(self):
        expected = set(itertools.product(("fixed", "random"), ("fixed", "real"),
                                         map(str, range(1000, 1030))))
        keys = [(row["policy"], row["environment"], row["seed"]) for row in self.episodes]
        self.assertEqual(len(self.episodes), 120)
        self.assertEqual(len(set(keys)), len(keys), "Duplicate episode rows")
        self.assertEqual(set(keys), expected)
        self.assertEqual(set(self.by_episode), expected)
        self.assertEqual(len(self.steps), 12000)
        for data in (self.episodes, self.steps, self.metrics, self.training):
            for row in data:
                for field, value in row.items():
                    if field not in ("policy", "environment"):
                        self.assertTrue(math.isfinite(float(value)), (field, value))

    def test_raw_steps_reproduce_episode_reward_rmse_and_success(self):
        for episode in self.episodes:
            key = tuple(episode[field] for field in ("policy", "environment", "seed"))
            with self.subTest(key=key):
                rows = self.by_episode[key]
                self.assertEqual(len(rows), 100)
                self.assertEqual(int(episode["steps"]), 100)
                errors = []
                rewards = []
                for step, row in enumerate(rows, start=1):
                    self.close_number(row["time_s"], step * 0.05, 1e-12)
                    q, reference = float(row["q_rad"]), float(row["reference_rad"])
                    velocity, torque = float(row["velocity_rad_s"]), float(row["torque_nm"])
                    self.assertLess(abs(q), 2.0)
                    self.assertLess(abs(velocity), 12.0)
                    self.assertLessEqual(abs(float(row["action_rad"])), 1.0)
                    self.assertLessEqual(abs(torque), 12.0)
                    error = q - reference
                    expected_reward = 1.0 - 5.0 * error * error - 0.005 * velocity * velocity - 0.001 * torque * torque
                    self.close_number(row["reward"], expected_reward)
                    errors.append(error)
                    rewards.append(float(row["reward"]))
                rmse = math.sqrt(statistics.mean(error * error for error in errors))
                self.close_number(episode["episode_reward"], math.fsum(rewards))
                self.close_number(episode["rmse_rad"], rmse)
                self.assertEqual(int(episode["success"]), int(rmse < 0.32))

    def test_metrics_recomputed_without_production_summary_function(self):
        grouped = group(self.episodes, ("policy", "environment"))
        self.assertEqual(len(self.metrics), 4)
        seen = set()
        for metric in self.metrics:
            key = (metric["policy"], metric["environment"])
            self.assertNotIn(key, seen)
            seen.add(key)
            rows = grouped[key]
            n = len(rows)
            self.assertEqual(n, 30)
            self.assertEqual(int(metric["episodes"]), n)
            rewards = [float(row["episode_reward"]) for row in rows]
            rmses = [float(row["rmse_rad"]) for row in rows]
            self.close_number(metric["reward_mean"], statistics.mean(rewards))
            self.close_number(metric["reward_std"], statistics.stdev(rewards))
            self.close_number(metric["rmse_mean_rad"], statistics.mean(rmses))
            self.close_number(metric["rmse_std_rad"], statistics.stdev(rmses))
            p = sum(int(row["success"]) for row in rows) / n
            self.close_number(metric["success_pct"], 100.0 * p)
            # Wilson endpoints independently obtained as roots of the score equation:
            # (p - endpoint)^2 = z^2 * endpoint * (1 - endpoint) / n.
            a, b, c = 1 + 1.96**2 / n, -2 * p - 1.96**2 / n, p * p
            discriminant = math.sqrt(b * b - 4 * a * c)
            low, high = (-b - discriminant) / (2 * a), (-b + discriminant) / (2 * a)
            self.close_number(metric["success_ci_low_pct"], low * 100)
            self.close_number(metric["success_ci_high_pct"], high * 100)
        self.assertEqual(seen, set(grouped))

    def test_matched_reference_trajectories_and_physical_conditions(self):
        by_seed = group(self.episodes, ("seed",))
        episode_map = {tuple(row[field] for field in ("policy", "environment", "seed")): row
                       for row in self.episodes}
        physics_fields = ("mass_scale", "friction", "delay_steps", "position_noise_rad", "velocity_noise_rad_s")
        reference_fields = ("reference_amplitude", "reference_omega", "reference_phase")
        for seed, episodes in by_seed.items():
            for field in reference_fields:
                self.assertEqual(len({row[field] for row in episodes}), 1)
            for domain in ("fixed", "real"):
                fixed = episode_map[("fixed", domain, seed[0])]
                randomized = episode_map[("random", domain, seed[0])]
                for field in physics_fields:
                    self.close_number(fixed[field], randomized[field])
                expected_physics = (1, 0.2, 0, 0.005, 0.02) if domain == "fixed" else (1.95, 0.75, 4, 0.045, 0.09)
                for field, value in zip(physics_fields, expected_physics):
                    self.close_number(fixed[field], value)
                first = self.by_episode[("fixed", domain, seed[0])]
                second = self.by_episode[("random", domain, seed[0])]
                for row1, row2 in zip(first, second):
                    self.close_number(row1["reference_rad"], row2["reference_rad"])
                    reference = float(fixed["reference_amplitude"]) * math.sin(
                        float(fixed["reference_omega"]) * float(row1["time_s"]) + float(fixed["reference_phase"]))
                    self.close_number(row1["reference_rad"], reference)

    def test_real_training_records_and_source_provenance(self):
        grouped = group(self.training, ("policy",))
        self.assertEqual(set(grouped), {("fixed",), ("random",)})
        self.assertEqual(len(self.training), 244)
        for rows in grouped.values():
            self.assertEqual([int(row["timestep"]) for row in rows], list(range(100, 12201, 100)))
        self.assertEqual(self.record["seed"], 42)
        self.assertEqual(self.record["timesteps_per_policy"], 12288)
        self.assertEqual(self.record["episode_steps"], 100)
        self.assertEqual(self.record["evaluation_seeds"], list(range(1000, 1030)))
        self.close_number(self.record["success_rmse_rad"], 0.32)
        self.close_number(self.record["dt_seconds"], 0.05)
        self.assertEqual(self.record["torch_threads"], 1)
        digest = hashlib.sha256((ROOT / "reading5_sim2real_ppo.py").read_bytes()).hexdigest()
        self.assertEqual(digest, self.record["source_sha256"])
        self.assertGreater(self.record["training_seconds"], 0)
        self.assertGreater(self.record["runtime_seconds"], self.record["training_seconds"])
        for package, version in self.record["versions"].items():
            self.assertEqual(importlib.metadata.version(package), version)

    def test_three_valid_high_resolution_figures(self):
        for name in ("fig1_training_rewards.png", "fig2_real_rollouts.png", "fig3_success_rates.png"):
            with self.subTest(figure=name):
                with Image.open(ROOT / name) as figure:
                    self.assertEqual(figure.format, "PNG")
                    self.assertGreaterEqual(min(figure.info["dpi"]), 200)
                    self.assertGreaterEqual(figure.width, 1800)
                    self.assertGreaterEqual(figure.height, 900)
                    figure.verify()

    def test_saved_models_replay_logged_actions_and_rollouts(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        models = {policy: PPO.load(ROOT / f"policy_{policy}.zip", device="cpu") for policy in ("fixed", "random")}
        for model in models.values():
            self.assertEqual(model.num_timesteps, 12288)
            self.assertEqual(model.device.type, "cpu")
            self.assertEqual(model.policy.net_arch, [32, 32])
            self.assertEqual(model.n_steps, 256)
            self.assertEqual(model.batch_size, 64)
            self.assertEqual(model.n_epochs, 5)
            self.close_number(model.learning_rate, 8e-4)
            self.close_number(model.gamma, 0.98)
        for policy, model in models.items():
            for domain in ("fixed", "real"):
                with self.subTest(policy=policy, environment=domain):
                    env = Joint(domain)
                    observation, _ = env.reset(seed=1000)
                    rows = self.by_episode[(policy, domain, "1000")]
                    for index, row in enumerate(rows, start=1):
                        action, _ = model.predict(observation, deterministic=True)
                        self.assertTrue(np.isfinite(action).all())
                        self.assertTrue(env.action_space.contains(action))
                        self.close_number(action[0], row["action_rad"], 1e-7)
                        observation, reward, terminated, truncated, info = env.step(action)
                        self.close_number(env.q, row["q_rad"], 1e-8)
                        self.close_number(env.dq, row["velocity_rad_s"], 1e-8)
                        self.close_number(reward, row["reward"], 1e-8)
                        self.close_number(info["torque"], row["torque_nm"], 1e-8)
                        self.assertFalse(terminated)
                        self.assertEqual(truncated, index == 100)
                    env.close()


if __name__ == "__main__":
    print("Auditing 120 evaluation episodes / 12,000 control steps, recorded training, figures, and saved-model replay.", flush=True)
    unittest.main(verbosity=2)
