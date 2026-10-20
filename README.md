# ENGR 6010 Reading 5: Sim-to-Real Joint Tracking with PPO

This experiment asks whether randomizing training physics improves transfer
to an unfamiliar **simulated** joint. Two Stable-Baselines3 PPO policies choose
target angles for the same one-degree-of-freedom joint and inner PD controller.
It is an educational analogy to one joint of a Unitree Go2 or ANYmal, not a
quadrupedal locomotion controller or a physical robot experiment. No robot,
GPU, MuJoCo installation, or external service is required.

Classroom materials are in [classroom/README_CLASSROOM.md](classroom/README_CLASSROOM.md):
an editable six-slide PowerPoint with speaker notes, a short Word/PDF presenter
script with six Q&As, and a saved-model demonstration that preserves the
original experiment outputs.

## Install and run

Use Python 3.11 or 3.12. From this project directory on Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python reading5_sim2real_ppo.py
python -m unittest discover -s tests -v
python tests/validate_outputs.py
```

On Windows, use `py -3.12 -m venv .venv`, then
`.venv\Scripts\Activate.ps1` in PowerShell; the remaining `python` commands
are the same. Alternatively invoke `.venv\Scripts\python.exe` directly.
On macOS, create and activate the environment as above, omit the separate
CPU-wheel command, and install `requirements.txt` from PyPI. The script still
uses `device="cpu"`. The `+cpu` Linux/Windows Torch build satisfies the
`torch==2.7.1` requirement.

In this prepared cloud environment, the interpreter is
`/workspace/.venvs/reading5/bin/python`; use it instead of creating another
environment. First-time downloads are separate from experiment runtime.

For VS Code, open this project folder, install the Microsoft Python extension,
and choose **Python: Select Interpreter** → the environment's Python executable.
Run `python reading5_sim2real_ppo.py` in the integrated terminal. The program
saves files beside the script, so its output directory does not depend on the
terminal's current directory. No display window is needed: Matplotlib uses
the noninteractive Agg backend. A rerun replaces the generated results and
models; copy any run you want to retain first.
The supplied `.vscode/launch.json` also provides a debug launch after selecting
the interpreter. The terminal command was executed here; the VS Code GUI was
not opened in this headless cloud environment.

## Joint, controller, and reward

The action is a target angle in **[-1, 1] rad**, rather than motor torque.
A FIFO queue delays that target by exactly the configured number of 0.05-second
control intervals. The PD controller then computes

`tau = clip(12 * (delayed_target - measured_q) - 2 * measured_velocity, -12, 12)`.

Torque is in N·m. Gaussian sensor noise affects both the policy observation
and PD feedback. The observation contains measured angle, measured velocity,
current desired reference, its value 150 ms into the future, and the latest
policy target. The reference preview assumes the prescribed trajectory is
known. The delayed-action queue is not fully observed; this remains a small
partially observed control problem.

Dynamics are `I * acceleration = tau - friction * velocity`, with
`I = inertia_scale * 1 kg·m²` and viscous friction in N·m·s/rad. The variable
named `mass` in the code is this **inertia multiplier**, not a measured robot
mass. Semi-implicit Euler first updates velocity from acceleration, then
updates angle using the new velocity. It does not artificially clip velocity.
An episode terminates for `|q| > 2 rad` or `|velocity| > 12 rad/s`, and is
otherwise truncated at 100 steps (5 seconds).

For each step, using the true simulated state after integration:

`reward = 1 - 5 * tracking_error² - 0.005 * velocity² - 0.001 * tau²`.

The tracking term encourages accuracy; the velocity term discourages fast,
oscillatory motion; the torque term discourages excessive control effort.
The constant sets a readable reward scale without changing which action is
best in a fixed-length episode. Episode reward is the sum of these rewards.
Ground-truth tracking error is available for training and evaluation in this
simulator; the policy receives noisy measurements.

## Matched experiment

The design and success threshold are recorded in [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
before training. Both policies use seed 42, one CPU thread, MLP [32, 32],
12,288 timesteps, 256-step PPO rollouts, batch size 64, 5 update epochs,
learning rate 0.0008, gamma 0.98, and otherwise SB3 2.7.1 defaults.
12,288 is exactly 48 complete rollouts; it avoids SB3 silently rounding a
12,000-step request up to another rollout.

| Condition | Inertia scale | Viscous friction | Target latency | Position / velocity noise SD |
|---|---:|---:|---:|---:|
| Fixed training / nominal evaluation | 1.0 | 0.2 | 0 ms | 0.005 rad / 0.02 rad/s |
| Randomized training, sampled each reset | U(0.7, 1.8) | U(0.1, 0.65) | Uniform {0, 50, 100, 150} ms | 0.005 rad / 0.02 rad/s |
| Held-out simulated reality | 1.95 | 0.75 | 200 ms | 0.045 rad / 0.09 rad/s |

Each reset samples a sinusoidal reference: amplitude 0.5–0.75 rad, angular
frequency 1.2–1.8 rad/s, phase -π–π, and initial angle -0.05–0.05 rad.
Independent RNG streams keep reference and initial-state sequences matched
between physics modes; randomizing physics cannot consume reference draws.
Inertia and friction stay constant within an episode.

Both deterministic trained policies are evaluated with seeds 1000–1029 in
each of the two evaluation environments: **120 episodes total**. Matched seeds
give matching reference trajectories, initial states, and underlying sensor
noise draws. Held-out noise has a larger scale. Nominal trials therefore vary
meaningfully rather than repeating an identical deterministic trajectory.

**Success means a complete, safely finished 5-second episode with tracking
RMSE strictly below 0.32 rad** (about 18.3°). This threshold is retained from
the uploaded program; it is an educational tolerance, not a robot safety or
industrial accuracy standard. A safety termination counts as failure.

## Verified results

<!-- VERIFIED_RESULTS_START -->
The actual CPU run used Python 3.12.14, Torch 2.7.1+cpu, and SB3 2.7.1.
Both policies completed 12,288 training steps without exceptions. The full
final program run took **27.72 seconds**, including imports, training,
evaluation, and figure generation; training itself took **20.45 seconds**.
The first run took 31.59 seconds (21.78 seconds training). After correcting
only a chart legend, an identical-seed repeat reproduced all four raw/summary
CSVs byte for byte; see `reproducibility.json`. These are cloud measurements,
not a promise of identical laptop runtime.

| Policy | Evaluation environment | Mean reward ± SD | Mean episode RMSE ± SD (rad) | Success | 95% Wilson interval |
|---|---|---:|---:|---:|---:|
| Fixed physics | Nominal simulation | 93.419 ± 2.645 | 0.1022 ± 0.0225 | 30/30 (100%) | 88.6–100% |
| Domain randomized | Nominal simulation | 93.666 ± 2.621 | 0.1037 ± 0.0235 | 30/30 (100%) | 88.6–100% |
| Fixed physics | Held-out simulation | 34.082 ± 26.052 | 0.3299 ± 0.0734 | 14/30 (46.7%) | 30.2–63.9% |
| Domain randomized | Held-out simulation | 70.970 ± 12.653 | 0.2250 ± 0.0505 | 29/30 (96.7%) | 83.3–99.4% |

**In this single-seed experiment, domain randomization improved held-out
tracking and success:** mean RMSE fell from 0.3299 to 0.2250 rad, and success
rose from 14/30 to 29/30. Nominal success was 100% for both; the randomized
policy's nominal RMSE was slightly higher. These results support an advantage
for this trained policy pair under this synthetic stress condition, not a
general guarantee of improvement or a demonstration of hardware transfer.

`check_env()` passed in all three physics modes, and eight environment
regression tests passed. Finite bounded actions were checked throughout the
120 evaluation episodes. Seven independent artifact audits also passed,
including recomputing statistics and reloading/replaying both saved models.
All three figures and the numerical audit files were generated by the actual
run. Full precision is retained in `metrics.csv`. See `environment_tests.txt`
and `verification.txt` for executed checks.
<!-- VERIFIED_RESULTS_END -->

## Figures and audit data

- `fig1_training_rewards.png`: actual completed-episode training rewards
  (faint lines) and a trailing 10-episode mean (bold lines). Higher is better.
  Because randomized training exposes harder conditions, raw returns between
  training modes are not an identical-physics evaluation.
- `fig2_real_rollouts.png`: both policies on held-out evaluation seed 1000,
  selected before viewing results. The horizontal axis begins at the first
  post-integration sample, 0.05 s. Higher inertia reduces acceleration for a
  given torque, friction damps motion, and latency makes corrections arrive
  late; their combination can produce tracking lag or overshoot. The plot
  alone does not isolate each parameter's causal contribution.
- `fig3_success_rates.png`: actual success percentages from the 30 matched
  trials per policy and environment, with 95% Wilson binomial intervals.
  All figures are saved at **250 DPI**.
- `metrics.csv`: four aggregate rows with policy/environment labels, episode
  counts, mean reward, reward sample SD, mean episode RMSE, RMSE sample SD,
  success percentage, and success confidence bounds. SD uses `ddof=1`.
- `evaluation_episodes.csv` and `evaluation_steps.csv`: episode outcomes,
  physical settings, reference parameters, and step-level angles, targets,
  torques, velocities, and rewards for independently recomputing the metrics.
- `training_rewards.csv`: recorded completed-episode rewards and timesteps.
- `experiment.json`: actual runtime, training time, versions, seeds, and
  source SHA-256. `policy_fixed.zip` and `policy_random.zip` are SB3 models.

The main executable includes logging, statistics, plots, and provenance in
about 220 lines; these additional checks make it longer than the approximate
150-line target while keeping the complete experiment in one file.

## Interpretation, limitations, and source connections

Short CPU training is a classroom runtime compromise, not evidence of
convergence. Only one training seed per policy is used. Evaluation uncertainty
describes these trained policies on sampled trials; it does not capture
variation across independent training runs. Do not generalize any observed
advantage to all seeds, physics ranges, or real robots. Domain randomization
may help, have little effect, or hurt nominal accuracy. Broader randomization
can make learning harder, and a held-out condition can exceed its coverage.

This model omits gravity, contact, full rigid-body coupling, foot placement,
balance, transmissions, thermal limits, and motor electrical dynamics. No
parameter was calibrated to a physical robot. Transfer here means transfer
between two versions of a simple simulation. Reproducibility is strongest
with the recorded dependency versions and platform; exact numerical equality
across operating systems and Torch builds is not guaranteed.

1. [Tan et al. (2018), *Sim-to-Real: Learning Agile Locomotion for Quadruped Robots*](https://arxiv.org/abs/1804.10332)
   motivates learning control policies with attention to actuation and
   randomized simulation physics. This example borrows that engineering
   pattern without reproducing their locomotion system or hardware results.
2. [Zhao et al., *Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: A Survey*](https://arxiv.org/abs/2009.13303)
   provides the reality-gap and domain-randomization context. Randomization
   is a transfer strategy, not a guarantee of real-world success.
3. [Sutton and Barto, *Reinforcement Learning: An Introduction*, Chapter 13](http://incompleteideas.net/book/the-book-2nd.html)
   explains policy-gradient and actor–critic foundations. **PPO is a specific,
   later algorithm**, here implemented by SB3 using a clipped surrogate
   objective and learned value function; it is not synonymous with all
   policy-gradient methods. See [Schulman et al. (2017), *Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347).
4. [MuJoCo](https://mujoco.org/) supplies richer multibody dynamics, contacts,
   constraints, and actuator models for robotics research. This transparent
   one-joint Euler model illustrates a small control loop that could be
   embedded in such a simulator; it does not provide MuJoCo's physics fidelity.

The uploaded ZIP's `validation_run.txt` identifies its previous results as
custom PyTorch PPO, not Stable-Baselines3. Its report also contradicts its own
metrics about held-out success. Those older plots and claims are not evidence
for this revised experiment; use the newly generated data and report summary.
