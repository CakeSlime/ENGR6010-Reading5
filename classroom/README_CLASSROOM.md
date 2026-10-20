# Reading 5 classroom presentation

Open these files directly in PowerPoint and Word, or use the PDF copies:

| File | Use |
|---|---|
| `Reading5_Classroom_Presentation.pptx` | 18 editable widescreen slides; short SAY cues on every slide and full scripts in speaker notes |
| `Reading5_Classroom_Presentation.pdf` | Portable slide handout / projector fallback |
| `Reading5_Presenter_Guide.docx` | Read-aloud slide scripts, stage directions, transitions, Windows setup, demo runbook, and 20 graduate-level Q&As |
| `Reading5_Presenter_Guide.pdf` | Printable presenter copy |

The core 16-slide talk is timed at approximately **14 minutes**, including a
short live replay. Slides 17–18 provide references and backup numerical data.
Allow additional time for discussion. The student voice is a suggested script;
adapt the wording and introduce yourself by name when presenting.

The deck uses the verified SB3 experiment in the project CSVs. It reports
14/30 fixed versus 29/30 randomized held-out successes and explicitly limits
that finding to this single trained policy pair and synthetic test condition.
The illustrated seed-1000 tracking trial passes for **both** policies.

## Repeat the code demonstration

Keep this `classroom` directory inside the complete project, next to the root
source, `policy_fixed.zip`, and `policy_random.zip`. If your original local
project already runs, copy this `classroom` folder into that project to reuse
its installed `.venv`. The PowerPoint and guide can be opened independently.

From the project folder in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe classroom\demo.py --trials 3 --rounds 2 --open-figure
```

Or double-click **`classroom\run_demo.cmd`**. In VS Code, open the whole project
folder, select its Python 3.12 `.venv` interpreter, then choose **Reading 5:
classroom replay (saved models)** in Run and Debug.

The helper reloads the trained models, evaluates seeds 1000–1002 twice, and
checks that the two rounds match exactly. It saves each launch to a **new
timestamped `classroom/demo_runs/` folder**, preserving the original experiment.
It performs inference/evaluation; it never calls PPO learning. After installing
dependencies, no Internet connection, GPU, or robot hardware is needed.

Say: “This is saved-model replay. Repeated seeds demonstrate repeatable
execution; they do not add independent experimental trials.”

The short subset produces held-out success of **1/3 fixed and 2/3 randomized**
in the verified demo. These three-trial counts are distinct from the original
30-trial results shown on the slides. To replay the full evaluation design:

```powershell
.\.venv\Scripts\python.exe classroom\demo.py --trials 30 --rounds 1
```

See [README_DEMO.md](README_DEMO.md) for full instructions and the separate
retraining command. The original experiment command retrains models and
replaces root experiment outputs; use a disposable project copy if presenting
that operation. If a live run fails, use the embedded figures and identify
them honestly as saved outputs of the verified run.

## Validation and timing

The Python helper was actually executed on the cloud CPU. The initial quick
demo took 7.06 s, a post-cleanup quick replay took 5.82 s, and full evaluation
replay took 9.32 s, including imports. Full replay produced byte-identical
episode, step, and aggregate CSVs to the original evaluation. Timings vary
on other machines. The Windows launcher and desktop image-viewer opening
were provided but were not executed on this Linux cloud machine.

`demo_validation.json` records the tests, timings, hashes, and PNG checks.
`materials_validation.json` records document structure and rendering checks.
The deck and guide were exported through LibreOffice and visually inspected.
The existing experiment's 8 environment tests and 7 artifact audits remain
documented in the project root; the main experiment source and models were
not changed to create these materials.

## Rebuild the editable materials

Optional build dependencies are separate from the experiment requirements:

```powershell
.\.venv\Scripts\python.exe -m pip install -r classroom\requirements_presentation.txt
.\.venv\Scripts\python.exe classroom\build_materials.py
```

`presenter_content.json` holds the speaking text and Q&A. The builder reads
the actual project CSVs, draws presentation-sized versions of the three
figures, writes the PowerPoint with notes, and writes the Word guide.
It does not train or alter the experiment. Add `--pdf` when LibreOffice's
`soffice` command is installed and on PATH; otherwise export PDFs using
PowerPoint and Word. If rebuilding from different experiment data, review
the presenter text and hard-coded narrative values before presenting:
the current narrative intentionally describes the verified supplied run.
