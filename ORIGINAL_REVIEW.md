# Review of supplied material

The standalone uploaded source was inspected before modification. Its ZIP copy
is byte-identical. Original attachments are preserved outside this checkout;
the generated results in this checkout belong to the revised SB3 experiment.

The original program already had a one-joint environment, bounded angle-target
actions, an inner saturated PD controller, episode-level randomization, and
two SB3 PPO training calls. Those design choices are retained.

Significant corrections:

- Run SB3 `check_env()` before training; add explicit position/velocity safety
  termination and remove artificial velocity clipping. The numerical method is
  documented as semi-implicit Euler.
- Use independent reference, physics, and sensor random streams. Both policies
  see the same sequence of varied training trajectories and matched evaluation
  conditions. Original deterministic nominal episodes were identical repeats.
- Add an explicit velocity penalty. Replace the original extra command-to-future-
  reference penalty with the three requested penalties: error, speed, torque.
- Apply measured-state noise to both the policy observation and PD feedback;
  keep the true state for dynamics and scoring. Document radians, inertia
  scaling, viscous friction, command delay, and Gaussian noise.
- Retain the original strict RMSE < 0.32 rad success threshold. Require a full
  five-second episode without a safety termination. Specify the protocol in
  EXPERIMENT_PLAN.md before revised training.
- Use 12,288 timesteps (48 full PPO rollouts) rather than requesting 12,000,
  which SB3 would round up to 12,032. All hyperparameters and seeds remain
  identical between the two revised policies.
- Evaluate once per policy/condition/seed; save all raw trajectories and episode
  records, means, sample SDs, and Wilson 95% success intervals.
- Record real training returns and runtime; save trained SB3 models. Use the
  actual 0.05–5.00 s rollout timestamps and export figures at 250 DPI.

The ZIP's `validation_run.txt` explicitly identifies its bundled results as
**custom PyTorch PPO**, not Stable-Baselines3 execution. Its metrics/table and
written interpretation also disagree about held-out success. None of those
figures, numbers, or claims are reused as evidence for the revised experiment.

The self-contained executable is slightly longer than the suggested 150 lines
because it includes raw-data export, auditable statistics, three complete
figures, and runtime/version recording. Tests and explanation remain separate.
