"""Regression checks for the joint model, pairing, delay, and reward.

Run from the repository root: python -m unittest discover -s tests -v
"""
import unittest

import numpy as np
from stable_baselines3.common.env_checker import check_env

from reading5_sim2real_ppo import DT, EP_LEN, SUCCESS_RMSE, Joint


class JointTests(unittest.TestCase):
    def test_gymnasium_api(self):
        for domain in ("fixed", "random", "real"):
            with self.subTest(domain=domain):
                env = Joint(domain)
                check_env(env, warn=True)
                observation, _ = env.reset(seed=31)
                self.assertTrue(env.observation_space.contains(observation))
                self.assertTrue(env.action_space.contains(np.zeros(1, dtype=np.float32)))
                np.testing.assert_allclose(env.action_space.low, -1.0)
                np.testing.assert_allclose(env.action_space.high, 1.0)
                env.close()

    def test_seed_reproduces_observations_physics_and_steps(self):
        for domain in ("fixed", "random", "real"):
            with self.subTest(domain=domain):
                first, second = Joint(domain), Joint(domain)
                obs1, _ = first.reset(seed=101)
                obs2, _ = second.reset(seed=101)
                np.testing.assert_array_equal(obs1, obs2)
                self.assertEqual((first.mass, first.friction, first.delay),
                                 (second.mass, second.friction, second.delay))
                for target in np.linspace(-0.7, 0.7, 15):
                    action = np.array([target], dtype=np.float32)
                    result1, result2 = first.step(action), second.step(action)
                    np.testing.assert_array_equal(result1[0], result2[0])
                    self.assertEqual(result1[1:], result2[1:])
                first.close()
                second.close()

    def test_reference_sequence_is_matched_across_domains(self):
        for seed in (0, 42, 1000, 1001):
            envs = [Joint(domain) for domain in ("fixed", "random", "real")]
            for env in envs:
                env.reset(seed=seed)
            for env in envs[1:]:
                np.testing.assert_allclose(
                    [env.amp, env.omega, env.phase, env.q, env.dq],
                    [envs[0].amp, envs[0].omega, envs[0].phase,
                     envs[0].q, envs[0].dq], rtol=0, atol=0)
                np.testing.assert_allclose(
                    [env._ref(t) for t in range(EP_LEN + 1)],
                    [envs[0]._ref(t) for t in range(EP_LEN + 1)], rtol=0, atol=0)
            for env in envs:
                env.close()

    def test_randomized_parameters_and_held_out_physics(self):
        samples = []
        env = Joint("random")
        for seed in range(40):
            env.reset(seed=seed)
            self.assertGreaterEqual(env.mass, 0.7)
            self.assertLessEqual(env.mass, 1.8)
            self.assertGreaterEqual(env.friction, 0.1)
            self.assertLessEqual(env.friction, 0.65)
            self.assertIn(env.delay, (0, 1, 2, 3))
            samples.append((env.mass, env.friction, env.delay))
        self.assertGreater(len(set(samples)), 1)
        held_out = Joint("real")
        held_out.reset(seed=42)
        self.assertGreater(held_out.mass, 1.8)
        self.assertGreater(held_out.friction, 0.65)
        self.assertEqual(held_out.delay, 4)
        self.assertGreater(held_out.noise[0], 0)
        env.close()
        held_out.close()

    def test_delay_is_exactly_four_control_intervals(self):
        env = Joint("real")
        env.reset(seed=9)
        env.noise = np.zeros(2)
        env.q = env.dq = 0.0
        env.sensed = np.zeros(2)
        action = np.ones(1, dtype=np.float32)
        for _ in range(4):
            _, _, terminated, _, info = env.step(action)
            self.assertFalse(terminated)
            self.assertAlmostEqual(info["torque"], 0.0)
            self.assertAlmostEqual(env.q, 0.0)
        _, _, _, _, info = env.step(action)
        self.assertAlmostEqual(info["torque"], 12.0)
        self.assertGreater(env.q, 0)
        env.close()

    def test_pd_semi_implicit_euler_and_reward_equation(self):
        env = Joint("fixed")
        env.reset(seed=12)
        env.noise = np.zeros(2)
        env.q, env.dq = 0.1, 0.2
        env.sensed = np.array([env.q, env.dq])
        action = np.array([0.4], dtype=np.float32)
        torque = np.clip(12 * (float(action[0]) - env.q) - 2 * env.dq, -12, 12)
        expected_dq = env.dq + DT * (torque - env.friction * env.dq) / env.mass
        expected_q = env.q + DT * expected_dq
        _, reward, terminated, truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(env.dq, expected_dq)
        self.assertAlmostEqual(env.q, expected_q)
        self.assertAlmostEqual(info["error"], expected_q - env._ref(env.t))
        expected_reward = 1 - 5 * info["error"] ** 2 - 0.005 * expected_dq ** 2 - 0.001 * torque ** 2
        self.assertAlmostEqual(reward, expected_reward)
        self.assertLessEqual(abs(info["torque"]), 12)
        env.close()

    def test_episode_horizon_and_predeclared_success_threshold(self):
        env = Joint("fixed")
        env.reset(seed=24)
        for step in range(EP_LEN):
            _, reward, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(np.isfinite(info["error"]))
            self.assertFalse(terminated)
            self.assertEqual(truncated, step + 1 == EP_LEN)
        self.assertEqual(env.t, 100)
        self.assertEqual(SUCCESS_RMSE, 0.32)
        env.close()

    def test_safety_limit_terminates_episode(self):
        env = Joint("fixed")
        env.reset(seed=2)
        env.q, env.dq = 2.1, 0.0
        env.sensed = np.array([env.q, env.dq])
        _, _, terminated, _, _ = env.step(np.zeros(1, dtype=np.float32))
        self.assertTrue(terminated)
        env.close()


if __name__ == "__main__":
    unittest.main()
