"""Repeat saved PPO joint controllers in class without retraining or overwriting results.

Run from the project folder: python classroom/demo.py --trials 3 --rounds 2
Use --trials 30 --rounds 1 to replay the full paired evaluation design.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=3, help="paired seeds per policy and condition (default: 3; minimum: 2)")
    parser.add_argument("--rounds", type=int, default=2, help="repeat the SAME seeds this many times (default: 2)")
    parser.add_argument("--seed", type=int, default=1000, help="first evaluation seed (default: 1000)")
    parser.add_argument("--open-figure", action="store_true", help="open the final saved tracking PNG with the local image viewer")
    args = parser.parse_args()
    if args.trials < 2 or args.rounds < 1 or args.seed < 0:
        parser.error("--trials must be >=2, --rounds >=1, and --seed >=0")
    return args


def main():
    started = perf_counter()
    args = arguments()
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import pandas as pd
    import torch
    from stable_baselines3 import PPO
    from reading5_sim2real_ppo import LABELS, DOMAINS, SUCCESS_RMSE, roll, summarize
    import matplotlib.pyplot as plt  # The experiment selects Agg before this import.

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    for policy in LABELS:
        path = ROOT / f"policy_{policy}.zip"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved model: {path}. Extract the complete project package.")
    models = {policy: PPO.load(ROOT / f"policy_{policy}.zip", device="cpu") for policy in LABELS}
    seeds = list(range(args.seed, args.seed + args.trials))
    directory = ROOT / "classroom" / "demo_runs" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
    directory.mkdir(parents=True, exist_ok=False)
    print("CLASSROOM DEMO: SAVED-MODEL REPLAY, NOT RETRAINING", flush=True)
    print(f"CPU inference; {args.trials} paired episodes per policy/condition; seeds {seeds[0]}-{seeds[-1]}.")
    print(f"Success: complete five-second episode, RMSE < {SUCCESS_RMSE:.2f} rad, no safety termination.")
    print("Repeated rounds reuse the SAME seeds; they are not independent evidence or additional trials.")
    print("The 3-trial quick demo does not replace the project's original 30-trial statistics.\n", flush=True)
    first_episodes, first_steps, first_metrics = None, None, None
    for round_number in range(1, args.rounds + 1):
        replay_started = perf_counter()
        runs = [roll(model, policy, domain, seed)
                for policy, model in models.items() for domain in DOMAINS for seed in seeds]
        episodes = pd.DataFrame([run[0] for run in runs])
        steps = pd.concat([run[1] for run in runs], ignore_index=True)
        metrics = summarize(episodes)
        if first_episodes is None:
            first_episodes, first_steps, first_metrics = episodes, steps, metrics
        else:
            pd.testing.assert_frame_equal(episodes, first_episodes, check_exact=True)
            pd.testing.assert_frame_equal(steps, first_steps, check_exact=True)
            pd.testing.assert_frame_equal(metrics, first_metrics, check_exact=True)
        prefix = f"round_{round_number:02d}"
        episodes.to_csv(directory / f"{prefix}_episodes.csv", index=False)
        steps.to_csv(directory / f"{prefix}_steps.csv", index=False)
        metrics.to_csv(directory / f"{prefix}_metrics.csv", index=False)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
        for axis, policy, color in zip(axes, LABELS, ("#0072B2", "#D55E00")):
            trial = steps.query("policy == @policy and environment == 'real' and seed == @args.seed")
            rmse = float(np.sqrt(np.mean((trial.q_rad - trial.reference_rad) ** 2)))
            axis.plot(trial.time_s, trial.reference_rad, "k--", label="Desired angle")
            axis.plot(trial.time_s, trial.q_rad, color=color, label="Actual angle")
            axis.set(title=f"{LABELS[policy]} | RMSE {rmse:.3f} rad", xlabel="Time (s)")
            axis.legend(); axis.grid(alpha=.2)
        axes[0].set_ylabel("Joint angle (rad)")
        figure.suptitle(f"Saved-policy classroom replay | held-out SIMULATION | seed {args.seed}\nSame physical conditions and reference; no policy learning during this demo")
        figure.tight_layout()
        final_figure = directory / f"{prefix}_heldout_tracking.png"
        figure.savefig(final_figure, dpi=250)
        plt.close(figure)
        display = metrics[["policy", "environment", "reward_mean", "rmse_mean_rad", "success_pct"]].copy()
        display["policy"] = display.policy.map(LABELS)
        display["environment"] = display.environment.map(DOMAINS)
        print(f"Round {round_number}/{args.rounds} ({perf_counter()-replay_started:.2f} s):")
        print(display.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        if round_number > 1:
            print("Repeatability check: PASS (all episodes, control steps, and metrics match round 1 exactly).")
        print(flush=True)
    record = dict(mode="deterministic saved-model replay; no retraining", trials_per_condition=args.trials,
                  rounds=args.rounds, seeds=seeds, same_seeds_reused_between_rounds=True,
                  repeatability_checked=args.rounds > 1, success_rmse_rad=SUCCESS_RMSE,
                  runtime_seconds=perf_counter()-started,
                  saved_model_sha256={policy: hashlib.sha256((ROOT / f"policy_{policy}.zip").read_bytes()).hexdigest() for policy in LABELS})
    (directory / "demo_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Total runtime (including imports): {record['runtime_seconds']:.2f} s")
    print(f"New demo files: {directory}")
    print(f"Open tracking figure: {final_figure}")
    print("Original experiment CSVs, figures, and saved models were not overwritten.")
    if args.open_figure:
        try:
            if os.name == "nt":
                os.startfile(str(final_figure))
            elif not webbrowser.open(final_figure.as_uri()):
                print("No image viewer opened; use the figure path above.")
        except OSError as error:
            print(f"Could not open a viewer ({error}); open the saved PNG manually.")


if __name__ == "__main__":
    main()
