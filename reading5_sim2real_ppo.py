"""ENGR 6010: paired, CPU-only SB3 PPO joint-tracking experiment; see README.md.
Policy targets -> delayed actuator -> saturated PD -> one-joint dynamics.
This educational simulation uses neither robot hardware nor MuJoCo.
"""
from collections import deque
from pathlib import Path
from time import perf_counter
import hashlib
import importlib.metadata as metadata
import json
import platform

START = perf_counter()  # Include third-party imports in the reported program runtime.
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env

OUT = Path(__file__).resolve().parent
SEED, STEPS, EP_LEN, DT, SUCCESS_RMSE = 42, 12288, 100, .05, .32
LABELS = {"fixed": "Fixed physics", "random": "Domain randomized"}
DOMAINS = {"fixed": "Nominal simulation", "real": "Held-out simulation"}


class Joint(gym.Env):
    """Semi-implicit Euler: update velocity, then angle using that new velocity.
    I = mass * 1 kg m^2; viscous friction in N m s/rad. Noise affects policy AND PD.
    Reward = 1 - 5*(q-reference)^2 - .005*velocity^2 - .001*torque^2.
    """
    metadata = {"render_modes": []}

    def __init__(self, domain="fixed"):
        super().__init__()
        if domain not in ("fixed", "random", "real"):
            raise ValueError(domain)
        self.domain = domain
        self.action_space = spaces.Box(-1., 1., (1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([-np.inf, -np.inf, -1, -1, -1], np.float32),
            np.array([np.inf, np.inf, 1, 1, 1], np.float32))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # Independent streams preserve identical references despite physics/noise draws.
        streams = np.random.SeedSequence(int(self.np_random.integers(2**32))).spawn(3)
        ref_rng, phys_rng, self.sensor_rng = [np.random.default_rng(s) for s in streams]
        self.amp, self.omega, self.phase = ref_rng.uniform(.5, .75), ref_rng.uniform(1.2, 1.8), ref_rng.uniform(-np.pi, np.pi)
        self.q, self.dq, self.t, self.last_target = float(ref_rng.uniform(-.05, .05)), 0., 0, 0.
        self.mass, self.friction, self.delay = 1., .2, 0
        if self.domain == "random":
            self.mass, self.friction = phys_rng.uniform(.7, 1.8), phys_rng.uniform(.1, .65)
            self.delay = int(phys_rng.integers(0, 4))
        elif self.domain == "real":
            self.mass, self.friction, self.delay = 1.95, .75, 4
        self.noise = np.array([.045, .09]) if self.domain == "real" else np.array([.005, .02])
        self.queue = deque([0.] * self.delay)
        return self._obs(), {}

    def _ref(self, t):
        return self.amp * np.sin(self.omega * t * DT + self.phase)

    def _obs(self):
        self.sensed = np.array([self.q, self.dq]) + self.sensor_rng.normal(size=2) * self.noise
        return np.array([*self.sensed, self._ref(self.t), self._ref(self.t + 3), self.last_target], np.float32)

    def step(self, action):
        self.last_target = float(np.clip(action[0], -1, 1))
        self.queue.append(self.last_target)
        command = self.queue.popleft()  # Exactly delay steps; zero means immediate.
        torque = float(np.clip(12 * (command - self.sensed[0]) - 2 * self.sensed[1], -12, 12))
        self.dq += DT * (torque - self.friction * self.dq) / self.mass
        self.q += DT * self.dq
        self.t += 1
        error = self.q - self._ref(self.t)
        reward = 1 - 5 * error**2 - .005 * self.dq**2 - .001 * torque**2
        terminated = bool(abs(self.q) > 2 or abs(self.dq) > 12)
        return self._obs(), float(reward), terminated, self.t >= EP_LEN, {"error": error, "torque": torque}


class Rewards(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rows, self.total = [], 0.

    def _on_step(self):
        self.total += float(self.locals["rewards"][0])
        if self.locals["dones"][0]:
            self.rows.append({"timestep": self.num_timesteps, "episode_reward": self.total})
            self.total = 0.
        return True


def train(domain):
    callback = Rewards()
    model = PPO("MlpPolicy", Joint(domain), seed=SEED, device="cpu", verbose=0,
                policy_kwargs={"net_arch": [32, 32]}, n_steps=256, batch_size=64,
                n_epochs=5, learning_rate=8e-4, gamma=.98)
    model.learn(total_timesteps=STEPS, callback=callback)
    assert model.num_timesteps == STEPS
    model.save(OUT / f"policy_{domain}.zip")
    model.get_env().close()
    return model, pd.DataFrame(callback.rows).assign(policy=domain)


def roll(model, policy, domain, seed):
    env = Joint(domain)
    obs, _ = env.reset(seed=seed)
    rows = []
    for _ in range(EP_LEN):
        action, _ = model.predict(obs, deterministic=True)
        assert np.isfinite(action).all() and env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(dict(policy=policy, environment=domain, seed=seed, time_s=env.t * DT,
                         q_rad=env.q, reference_rad=env._ref(env.t), velocity_rad_s=env.dq,
                         action_rad=float(action[0]), torque_nm=info["torque"], reward=reward))
        if terminated or truncated:
            break
    data = pd.DataFrame(rows)
    rmse = float(np.sqrt(np.mean((data.q_rad - data.reference_rad)**2)))
    episode = dict(policy=policy, environment=domain, seed=seed, steps=env.t,
                   episode_reward=float(data.reward.sum()), rmse_rad=rmse,
                   success=int(not terminated and env.t == EP_LEN and rmse < SUCCESS_RMSE),
                   mass_scale=env.mass, friction=env.friction, delay_steps=env.delay,
                   position_noise_rad=env.noise[0], velocity_noise_rad_s=env.noise[1],
                   reference_amplitude=env.amp, reference_omega=env.omega, reference_phase=env.phase)
    env.close()
    return episode, data


def summarize(episodes):
    rows = []
    for (policy, domain), g in episodes.groupby(["policy", "environment"], sort=False):
        n, p, z = len(g), g.success.mean(), 1.96  # Wilson binomial 95% interval.
        center = (p + z*z/(2*n)) / (1 + z*z/n)
        half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        rows.append(dict(policy=policy, environment=domain, episodes=n,
                         reward_mean=g.episode_reward.mean(), reward_std=g.episode_reward.std(ddof=1),
                         rmse_mean_rad=g.rmse_rad.mean(), rmse_std_rad=g.rmse_rad.std(ddof=1),
                         success_pct=100*p, success_ci_low_pct=100*(center-half), success_ci_high_pct=100*(center+half)))
    return pd.DataFrame(rows)


def figures(training, trajectories, metrics):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"fixed": "#0072B2", "random": "#D55E00"}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for policy, g in training.groupby("policy", sort=False):
        ax.plot(g.timestep, g.episode_reward, color=colors[policy], alpha=.18, lw=.8)
        ax.plot(g.timestep, g.episode_reward.rolling(10, min_periods=1).mean(), color=colors[policy], label=LABELS[policy])
    ax.set(title="PPO learning: joint-tracking rewards", xlabel="Training timesteps per policy", ylabel="Episode reward (higher is better)")
    ax.text(.02, .03, "Faint: recorded episodes; bold: trailing 10-episode mean", transform=ax.transAxes, fontsize=9)
    ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(OUT / "fig1_training_rewards.png", dpi=250); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, policy in zip(axes, LABELS):
        g = trajectories.query("policy == @policy and environment == 'real' and seed == 1000")
        rmse = np.sqrt(np.mean((g.q_rad - g.reference_rad)**2))
        ax.plot(g.time_s, g.reference_rad, "k--", label="Desired angle")
        ax.plot(g.time_s, g.q_rad, color=colors[policy], label="Actual angle")
        ax.set(title=f"{LABELS[policy]} | RMSE {rmse:.3f} rad", xlabel="Time (s)")
        ax.grid(alpha=.2); ax.legend()
    axes[0].set_ylabel("Joint angle (rad)")
    fig.suptitle("Held-out simulated reality: paired trial, seed 1000\nInertia 1.95×; friction 0.75; delay 0.20 s; position noise 0.045 rad", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / "fig2_real_rollouts.png", dpi=250); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for j, policy in enumerate(LABELS):
        g = metrics.query("policy == @policy").set_index("environment").loc[list(DOMAINS)]
        errors = np.maximum(0, np.array([g.success_pct-g.success_ci_low_pct, g.success_ci_high_pct-g.success_pct]))
        bars = ax.bar(np.arange(2) + (j-.5)*.36, g.success_pct, .36, yerr=errors, capsize=5, color=colors[policy], label=LABELS[policy])
        ax.bar_label(bars, labels=[f"{p:.1f}%" for p in g.success_pct], padding=5, fontsize=10)
    ax.set(xticks=np.arange(2), xticklabels=list(DOMAINS.values()), ylim=(-3, 120), yticks=np.arange(0, 101, 20),
           ylabel="Successful episodes (%)", title="Sim-to-real approximation: joint-tracking success\n30 paired trials per condition; RMSE < 0.32 rad; 95% Wilson intervals")
    ax.legend(loc="upper center", ncol=2); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "fig3_success_rates.png", dpi=250); plt.close(fig)


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for domain in ("fixed", "random", "real"):
        env = Joint(domain)
        check_env(env, warn=True)
        env.close()
    print("check_env: PASS (all three physics modes)", flush=True)
    training_start = perf_counter()
    models, logs = {}, []
    for policy in LABELS:
        models[policy], log = train(policy)
        logs.append(log)
        print(f"Trained {policy}: {STEPS} timesteps, {len(log)} complete episodes", flush=True)
    training_seconds = perf_counter() - training_start
    training = pd.concat(logs, ignore_index=True)
    runs = [roll(model, policy, domain, seed) for policy, model in models.items() for domain in DOMAINS for seed in range(1000, 1030)]
    episodes, trajectories = pd.DataFrame([r[0] for r in runs]), pd.concat([r[1] for r in runs], ignore_index=True)
    metrics = summarize(episodes)
    for name, data in [("training_rewards", training), ("evaluation_episodes", episodes), ("evaluation_steps", trajectories), ("metrics", metrics)]:
        data.to_csv(OUT / f"{name}.csv", index=False)
    figures(training, trajectories, metrics)
    record = dict(seed=SEED, timesteps_per_policy=STEPS, episode_steps=EP_LEN, dt_seconds=DT,
                  success_rmse_rad=SUCCESS_RMSE, evaluation_seeds=list(range(1000, 1030)),
                  training_seconds=training_seconds, runtime_seconds=perf_counter()-START,
                  python=platform.python_version(), platform=platform.platform(), torch_threads=torch.get_num_threads(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions={n: metadata.version(n) for n in ("torch", "stable-baselines3", "gymnasium", "numpy", "pandas", "matplotlib")})
    (OUT / "experiment.json").write_text(json.dumps(record, indent=2) + "\n")
    print(metrics.to_string(index=False))
    print(f"Runtime: {record['runtime_seconds']:.2f} s; training: {training_seconds:.2f} s. Outputs: {OUT}")


if __name__ == "__main__":
    main()
