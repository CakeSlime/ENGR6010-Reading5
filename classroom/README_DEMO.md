# Repeat the classroom demonstration

The slides use the full original experiment. Run it from the project folder:

    .\.venv\Scripts\python.exe reading5_sim2real_ppo.py

That command retrains both policies, evaluates them, prints metrics, and
regenerates the three figures. It replaces root experiment outputs.

## Full evaluation replay without retraining

    .\.venv\Scripts\python.exe classroom\demo.py --trials 30 --rounds 1 --open-figure

Or double-click classroom\run_demo.cmd. Both reload the saved models and
run seeds 1000–1029: 120 episodes total. They do not call learn(). Each launch
writes CSVs, a tracking PNG, and metadata into a new timestamped
classroom/demo_runs/ folder. Original data and models remain unchanged.
The full replay took 9.32 s on the cloud CPU and produced byte-identical
metrics, episode, and step CSVs to the original evaluation. Local timing
and numerical equality across platforms are not guaranteed.

The launcher selects the project's .venv\Scripts\python.exe explicitly.
Keep the entire extracted project, including policy_fixed.zip and
policy_random.zip. Install dependencies before class; subsequent replay
needs no Internet or physical robot. The Windows launcher and automatic
image-viewer opening were not executed on Linux; Python replay was tested.

## Optional short repeatability check

    .\.venv\Scripts\python.exe classroom\demo.py --trials 3 --rounds 2

This additional option checks that the same three seeds give identical
results in two rounds. These repeated seeds are not independent trials.
The three-seed subset has held-out success 1/3 fixed versus 2/3 randomized
in the tested run; do not substitute those counts for the slides' full
14/30 versus 29/30 comparison. The initial quick check took 7.06 s, and
an import-order cleanup was rechecked in 5.82 s. Tests are recorded in
demo_validation.json.

If a run fails, show the previously generated graphs and describe them
as saved results. Use Reading5_Presenter_Guide.docx for the short script.
