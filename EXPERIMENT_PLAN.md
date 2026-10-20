# Experiment specified before the first revised training run

Retain the uploaded source's success threshold: **RMSE < 0.32 rad**, measured
over a full 100-step (5-second) episode with no safety termination. The strict
inequality and threshold will not change in response to the results.

Both SB3 PPO policies use seed 42, CPU with one Torch thread, MLP [32, 32],
12,288 timesteps (exactly 48 rollouts of 256), batch size 64, 5 epochs,
learning rate 0.0008, gamma 0.98, and otherwise SB3 2.7.1 defaults.
Reward is `1 - 5*tracking_error^2 - 0.005*velocity^2 - 0.001*torque^2`.

Nominal inertia scale, viscous friction, and target-command delay are 1.0,
0.2, and 0 steps. At every randomized-training reset, sample inertia
uniformly in [0.7, 1.8], friction uniformly in [0.1, 0.65], and delay uniformly
from {0, 1, 2, 3} steps (0–150 ms). Both training modes have position noise
0.005 rad and velocity noise 0.02 rad/s (independent Gaussian standard deviations).

Held-out simulated reality uses 1.95 inertia scale, 0.75 friction, 4-step
(200 ms) delay, 0.045 rad position noise, and 0.09 rad/s velocity noise.
All three physical parameters are outside the training ranges; the noisy
feedback affects both the policy and the PD controller. This is a simulated
stress condition, not a hardware trial or a calibrated real robot.

Reference is A*sin(omega*t + phase), with A in [0.5, 0.75] rad,
omega in [1.2, 1.8] rad/s, phase in [-pi, pi], and initial angle in
[-0.05, 0.05] rad. Sample at resets using an independent RNG stream so
the reference/initial-state sequences match across training modes. The
observation includes the reference 150 ms into the future; the reference
is assumed known, as it would be in a prescribed joint trajectory.

Evaluate deterministic policies on seeds 1000–1029, matched across both
policies and environments. Report all 120 episode records, mean return,
sample SD of returns and RMSE, success percentage, and Wilson 95% success
intervals. Plot seed 1000, selected in advance rather than for visual effect.
One training seed per policy limits inference to this run; evaluation
intervals do not estimate variability across independent training runs.

No tuning or additional seed selection based on the observed comparison.
Runtime target: full executable experiment under five minutes on CPU,
excluding first-time package installation. Measure rather than assume it.
