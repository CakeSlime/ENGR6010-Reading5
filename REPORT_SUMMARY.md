# Verified Reading 5 results

Two Stable-Baselines3 PPO policies were trained for 12,288 timesteps each to
select joint-angle targets for a saturated PD controller in a one-degree-of-
freedom joint simulation. The fixed policy used nominal physics; the second
sampled inertia, viscous friction, and command delay at every episode reset.
Architecture, PPO hyperparameters, reference-trajectory sequences, reward,
and training seed were matched. Success was specified before training as a
full five-second episode with tracking RMSE strictly below 0.32 rad and no
safety termination. Each policy was evaluated on 30 matched seeds in each
condition, producing 120 episodes in total.

| Policy | Condition | Episode reward, mean ± SD | Episode RMSE, mean ± SD (rad) | Success | Success 95% Wilson interval |
|---|---|---:|---:|---:|---:|
| Fixed physics | Nominal simulation | 93.419 ± 2.645 | 0.1022 ± 0.0225 | 30/30 (100%) | 88.6–100% |
| Domain randomized | Nominal simulation | 93.666 ± 2.621 | 0.1037 ± 0.0235 | 30/30 (100%) | 88.6–100% |
| Fixed physics | Held-out simulated reality | 34.082 ± 26.052 | 0.3299 ± 0.0734 | 14/30 (46.7%) | 30.2–63.9% |
| Domain randomized | Held-out simulated reality | 70.970 ± 12.653 | 0.2250 ± 0.0505 | 29/30 (96.7%) | 83.3–99.4% |

**Paragraph for the written report:** In this educational sim-to-real
approximation, domain randomization improved transfer to a held-out joint
simulation with greater inertia, friction, actuator delay, and sensor noise.
Across 30 matched evaluation trials, the fixed-physics PPO policy achieved
46.7% success and 0.3299 rad mean tracking RMSE, while the randomized policy
achieved 96.7% success and 0.2250 rad mean RMSE. Both achieved 100% nominal
success, although randomized training produced a slightly higher nominal
RMSE (0.1037 versus 0.1022 rad). These results support improved robustness
for this policy pair and synthetic test condition. They do not establish
physical robot transfer, full quadrupedal locomotion, or improvement across
independent training seeds. PPO is the specific clipped-surrogate
policy-gradient/actor–critic method used here; the demonstration connects
Tan et al.'s randomized-simulation motivation and Zhao et al.'s reality-gap
survey with the foundations in Sutton and Barto Chapter 13. Its simple
Euler-integrated joint omits the multibody dynamics and contacts available
in MuJoCo.

The final complete CPU program took **27.72 seconds**, including imports,
training, evaluation, and figure generation; training took **20.45 seconds**.
An initial run took 31.59 seconds. The repeat, after only adjusting a plot
legend, reproduced the training and evaluation CSVs exactly. Installation
time is excluded. These timings were measured on the cloud machine, not a
student laptop; the short training budget does not establish convergence.

Validation: Gymnasium/SB3 `check_env()` passed for all physics modes;
8 environment regression tests and 7 independent artifact audits passed.
The audits recomputed all statistics from the 12,000 evaluation steps,
verified bounded actions/torques and matched conditions, checked three
250-DPI PNGs, and reloaded/replayed both saved policies. The terminal run
command and VS Code launch configuration are provided; the VS Code GUI
was not opened in the cloud. No robot hardware is needed.

The revised source adds explicit velocity cost, safety checks, matched
varied trajectories with independent RNG streams, full raw-data export,
sample SDs and confidence intervals, actual runtime/version records, and
correct rollout timestamps. The source is 218 lines including all plotting
and audit exports, rather than forcing those features into 150 dense lines.
The original ZIP's custom-PPO figures and conflicting interpretation are
not used as evidence; see ORIGINAL_REVIEW.md.
