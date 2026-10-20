# Short Reading 5 classroom demonstration

Use these current files:

- **Reading5_Classroom_Presentation.pptx** — six editable slides, short speaking cues, and full speaker notes.
- **Reading5_Classroom_Presentation.pdf** — portable slide copy.
- **Reading5_Presenter_Guide.docx / .pdf** — short read-aloud scripts, the run sequence, and six likely Q&As.

The presentation is approximately **5–6 minutes**, including running the code.
The six slides cover the basic idea, live execution, tracking graph, numerical
results, the four readings, and a quadruped-joint application.

## Run the code in class

Open the complete project folder in VS Code and use its working Python 3.12
.venv interpreter. From the project folder:

    .\.venv\Scripts\python.exe reading5_sim2real_ppo.py

This trains both PPO agents, evaluates 30 matched trials per policy and
condition, prints results, and generates all three original PNG graphs.
Briefly point to Joint.step(), train(), and main(). Open the training graph
from the generated files; slides 3–4 provide readable tracking and success
graphs. The verified cloud run took 27.72 s; local timings vary. Rerunning
this command replaces its generated experiment files and saved models.

## Optional quick replay for repeated demonstrations

If you want to evaluate the saved models without retraining:

    .\.venv\Scripts\python.exe classroom\demo.py --trials 30 --rounds 1 --open-figure

Or double-click classroom\run_demo.cmd, which now uses this full 30-seed
replay. The VS Code **Reading 5: classroom replay (saved models)** launch
configuration uses the same arguments. Outputs go to a new timestamped
classroom/demo_runs/ folder, preserving the original results. Say that this
is saved-model evaluation, rather than training. The full Python replay
was tested in 9.32 s and matched the original evaluation CSVs exactly.
The Windows launcher/image viewer were not executed on the Linux cloud.

## Keep the explanation accurate

The slide results are from the verified cloud experiment: held-out success
14/30 fixed versus 29/30 randomized; mean RMSE 0.3299 versus 0.2250 rad.
Both policies achieve 100% nominal success and both pass the particular
seed-1000 trial shown in the tracking graph. The result concerns one joint,
one training seed, and simulated transfer; it does not establish performance
on a physical Go2 or ANYmal. Sources are linked in slide notes and the guide.

If a live run fails, use the saved graphs and identify them as previously
executed results. See README_DEMO.md for additional replay options and the
root README.md for installation. If you already have a working project,
copy this classroom folder into it to reuse the installed .venv.

## Rebuild the presentation and guide

    .\.venv\Scripts\python.exe -m pip install -r classroom\requirements_presentation.txt
    .\.venv\Scripts\python.exe classroom\build_materials.py

Add --pdf when LibreOffice's soffice command is on PATH; otherwise export
from PowerPoint and Word. The builder reads actual CSV data and uses
presenter_content.json for the six scripts and Q&A. If rebuilding after a
new experiment, also review the narrative numbers before presenting.
