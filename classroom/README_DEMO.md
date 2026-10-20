# Repeatable classroom demonstration

This helper **reloads the two already-trained SB3 PPO models** and performs
fresh deterministic evaluations. It does not call `learn()` or change the
saved policies. Keep the complete extracted project folder, including
`policy_fixed.zip` and `policy_random.zip`.

Use the Python 3.12 project environment from the main README. No network or
physical robot is needed once dependencies are installed.

## Quick live demonstration

From the project folder on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe classroom\demo.py --trials 3 --rounds 2 --open-figure
```

Or double-click `classroom\run_demo.cmd`. That launcher selects the project's
`.venv\Scripts\python.exe` explicitly, prints the results, opens the final
tracking PNG using the local viewer, and keeps the terminal open. The Windows
launcher is supplied for local use; it cannot be executed in the Linux cloud.

On a shell with the correct environment already active:

```bash
python classroom/demo.py --trials 3 --rounds 2
```

The quick demo evaluates both policies on nominal and held-out simulation,
with seeds 1000–1002. Each round contains 12 five-second simulated episodes;
the simulated time runs faster than wall-clock time. The second round uses
the **same** seeds and checks that every episode and control step matches
the first round exactly. Do not report those repeated rounds as new evidence
or combine them into a larger sample size. Three-trial quick-demo success
rates can differ from the original 30-trial rates on the results slides.

Suggested spoken explanation:

> “I am reloading saved policies, not retraining them. Both controllers face
> the same reference, initial state, and physical test condition. I repeat
> the same seeds to show reproducible execution. For the scientific
> comparison, I use the original 30 distinct matched trials per condition.”

## Replay the complete evaluation design

```powershell
.\.venv\Scripts\python.exe classroom\demo.py --trials 30 --rounds 1
```

This evaluates seeds 1000–1029: 120 episodes total, with the unchanged
full-episode RMSE < 0.32 rad success criterion. The helper saves actual
episode data, step records, metrics, and a 250-DPI paired held-out tracking
plot. Numerical equality across different operating systems and Torch
builds is not guaranteed.

All helper outputs go into a new timestamped `classroom/demo_runs/` folder.
The original experiment metrics, figures, and trained models remain intact.

## Verified helper behavior

Actual Linux cloud tests used the project interpreter and one CPU thread:

- `--trials 3 --rounds 2` completed in **7.06 s**, including imports. Both
  rounds matched exactly at the episode, control-step, and metric levels.
  For those three seeds, held-out success was 1/3 fixed and 2/3 randomized;
  these are quick-demo results, not the 14/30 and 29/30 scientific results.
- `--trials 30 --rounds 1` completed in **9.32 s**, including imports. Its
  episode, control-step, and metric CSVs matched the original evaluation
  records byte for byte. The generated PNGs passed integrity checks at
  250 DPI. Original source, models, CSVs, and figures stayed unchanged.

Wall-clock times vary by machine. Automatic image opening and the `.cmd`
launcher require a local desktop and were not exercised in the Linux cloud.

## Retraining is a separate demonstration

The original command remains:

```powershell
.\.venv\Scripts\python.exe reading5_sim2real_ppo.py
```

That command runs environment checks, retrains both PPO agents, evaluates
them, and **replaces the root experiment outputs**. Run it in a disposable
copy of the complete project if preserving the prepared report data. The
cloud validated full run took 27.72 seconds; local runtimes vary. Avoid
promising that elapsed time during a live presentation.

If the classroom machine cannot run Python, use the prepared figures and
actual logs already included in the package and clearly label that segment
as a walkthrough of previously executed results.
