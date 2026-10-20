"""Build the classroom PowerPoint and presenter guide from verified project data.

Run from any directory: python classroom/build_materials.py [--pdf]
PDF export additionally needs LibreOffice/soffice; editable files need no Office.
This builder reads the experiment artifacts and never trains or changes them.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt
from docx import Document
from docx.shared import Inches as DocInches, Pt as DocPt, RGBColor as DocColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"
NAVY, TEAL, ORANGE = "14283F", "00746C", "C65924"
PALE, GRAY, WHITE = "F0F5F8", "526477", "FFFFFF"
BLUE = "0072B2"
DECK = HERE / "Reading5_Classroom_Presentation.pptx"
GUIDE = HERE / "Reading5_Presenter_Guide.docx"


def rgb(value):
    return RGBColor.from_string(value)


def box(slide, x, y, w, h, fill=PALE, radius=False, line=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.fill.background() if line is None else None
    if line:
        shape.line.color.rgb = rgb(line)
    return shape


def text(slide, value, x, y, w, h, size=22, color=NAVY, bold=False, align=PP_ALIGN.LEFT, font="Arial"):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(.025)
    tf.margin_top = tf.margin_bottom = Inches(.015)
    for i, line in enumerate(str(value).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text, p.alignment = line, align
        p.space_after = Pt(7)
        p.font.name, p.font.size = font, Pt(size)
        p.font.bold, p.font.color.rgb = bold, rgb(color)
    return shape


def arrow(slide, x1, y1, x2, y2, color=TEAL):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb, line.line.width = rgb(color), Pt(2.5)
    # A small filled triangle provides a portable arrowhead in PowerPoint.
    tri = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x2-.07), Inches(y2-.08), Inches(.15), Inches(.16))
    tri.fill.solid(); tri.fill.fore_color.rgb = rgb(color); tri.line.fill.background()
    return line


def card(slide, x, y, w, h, title, body, accent=TEAL, title_size=23, body_size=20):
    box(slide, x, y, w, h, PALE, True)
    box(slide, x, y, .07, h, accent)
    text(slide, title, x+.2, y+.17, w-.4, .55, title_size, accent, True)
    text(slide, body, x+.2, y+.85, w-.4, h-.95, body_size)


def table(slide, headers, rows, x, y, widths, row_height=.58, font_size=17):
    for r, values in enumerate([headers] + rows):
        start = x
        for j, value in enumerate(values):
            color = NAVY if r == 0 else (PALE if r % 2 else WHITE)
            box(slide, start, y+r*row_height, widths[j], row_height, color)
            text(slide, value, start+.10, y+r*row_height+.10, widths[j]-.18, row_height-.14,
                 font_size, WHITE if r == 0 else NAVY, r == 0)
            start += widths[j]


def picture(slide, name, x=.58, y=1.55, w=12.15):
    slide.shapes.add_picture(str(ASSETS / name), Inches(x), Inches(y), width=Inches(w))


def base(prs, item, title=None, dark=False):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(NAVY if dark else WHITE)
    text(slide, "ENGR 6010  /  AI IN ROBOTICS", .58, .20, 10.7, .25, 11, "93D9D3" if dark else TEAL, True)
    if title:
        text(slide, title, .58, .62, 12.15, .69, 30, WHITE if dark else NAVY, True)
        box(slide, .60, 1.36, 12.1, .025, TEAL)
    box(slide, .58, 6.69, 12.15, .46, TEAL if dark else "E8F3F0", True)
    text(slide, 'SAY  “' + item["cue"] + '”', .73, 6.79, 11.83, .27, 14, WHITE if dark else TEAL)
    text(slide, "One-joint educational simulation  •  No physical robot trial", .60, 7.24, 11.3, .18, 9,
         "BED1DD" if dark else GRAY)
    text(slide, f'{item["number"]:02d} / 6', 12.04, 7.23, .68, .20, 10, "BED1DD" if dark else GRAY, align=PP_ALIGN.RIGHT)
    slide.notes_slide.notes_text_frame.text = (
        f'SLIDE {item["number"]}: {item["title"]}\nSuggested time: {item["seconds"]} seconds\n\n'
        f'READ-ALOUD SCRIPT\n{item["script"]}\n\nPOINT TO / DO\n{item["point_to"]}\n\n'
        f'TRANSITION\n{item["transition"]}')
    return slide


def charts(metrics):
    ASSETS.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#14283F", "text.color": "#14283F"})
    training = pd.read_csv(ROOT / "training_rewards.csv")
    fig, ax = plt.subplots(figsize=(12.6, 4.75))
    for policy, label, color in (("fixed", "Fixed physics", "#0072B2"), ("random", "Domain randomized", "#C65924")):
        g = training[training.policy == policy]
        ax.plot(g.timestep, g.episode_reward, color=color, alpha=.17, lw=1.0)
        ax.plot(g.timestep, g.episode_reward.rolling(10, min_periods=1).mean(), color=color, lw=3, label=label)
    ax.set(xlabel="Training timesteps per policy", ylabel="Episode reward")
    ax.legend(loc="lower right", frameon=False, fontsize=16)
    ax.grid(alpha=.18)
    fig.tight_layout(pad=.8); fig.savefig(ASSETS / "training_for_slides.png", dpi=220); plt.close(fig)
    steps = pd.read_csv(ROOT / "evaluation_steps.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3), sharey=True)
    for ax, policy, label, color in zip(axes, ("fixed", "random"), ("Fixed physics", "Domain randomized"), ("#0072B2", "#C65924")):
        g = steps.query("policy == @policy and environment == 'real' and seed == 1000")
        rmse = np.sqrt(np.mean((g.q_rad - g.reference_rad)**2))
        ax.plot(g.time_s, g.reference_rad, "k--", label="Desired", lw=2.2)
        ax.plot(g.time_s, g.q_rad, color=color, label="Actual", lw=2.8)
        ax.set(title=f"{label}: RMSE {rmse:.3f} rad", xlabel="Time (s)")
        ax.legend(loc="upper right", frameon=False, fontsize=14); ax.grid(alpha=.2)
    axes[0].set_ylabel("Joint angle (rad)")
    fig.tight_layout(pad=.8); fig.savefig(ASSETS / "tracking_for_slides.png", dpi=220); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12.6, 4.7))
    for i, (policy, label, color) in enumerate((("fixed", "Fixed physics", "#0072B2"), ("random", "Domain randomized", "#C65924"))):
        g = metrics[metrics.policy == policy].set_index("environment").loc[["fixed", "real"]]
        error = np.maximum(0, np.array([g.success_pct-g.success_ci_low_pct, g.success_ci_high_pct-g.success_pct]))
        bars = ax.bar(np.arange(2)+(i-.5)*.32, g.success_pct, .32, color=color, yerr=error, capsize=6, label=label)
        counts = [round(float(p)*30/100) for p in g.success_pct]
        ax.bar_label(bars, labels=[f"{p:.1f}% ({n}/30)" for p, n in zip(g.success_pct, counts)], padding=9, fontsize=16)
    ax.set(xticks=[0, 1], xticklabels=["Nominal simulation", "Held-out simulation"],
           ylabel="Successful episodes (%)", ylim=(0, 126), yticks=[0, 25, 50, 75, 100])
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=16)
    ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.tight_layout(pad=.8); fig.savefig(ASSETS / "success_for_slides.png", dpi=220); plt.close(fig)


def build_deck(content, metrics, record):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    prs.core_properties.title = "Reading 5: A Short Sim-to-Real PPO Demonstration"
    prs.core_properties.author = "ENGR 6010 student presenter"
    for item in content["slides"]:
        n = item["number"]
        s = base(prs, item, item["title"], dark=n == 6)
        if n == 1:
            text(s, "Can training with varied physics make a joint controller more robust?", .76, 1.68, 11.8, 1.0, 32, NAVY, True)
            blocks = [(1.0, "PPO POLICY", "Chooses a target angle"),
                      (5.0, "PD CONTROL", "Converts angle error to torque"),
                      (9.0, "SIMULATED JOINT", "Moves with inertia + friction")]
            for x, title, body in blocks:
                card(s, x, 3.05, 3.25, 1.93, title, body, title_size=20, body_size=20)
            arrow(s, 4.25, 4.0, 4.90, 4.0)
            arrow(s, 8.25, 4.0, 8.90, 4.0)
            text(s, "Compare two PPO agents: fixed training physics versus randomized training physics.", .85, 5.50, 11.6, .72, 24, TEAL, True)
        elif n == 2:
            text(s, "Run the program from the working project folder:", .80, 1.72, 11.8, .55, 26, NAVY, True)
            box(s, .78, 2.50, 11.8, .95, NAVY, True)
            text(s, r".\.venv\Scripts\python.exe reading5_sim2real_ppo.py", 1.02, 2.80, 11.3, .43, 23, WHITE, font="Consolas")
            card(s, .82, 3.88, 3.72, 1.99, "1  TRAIN", "Same PPO recipe.\nDifferent training physics.", title_size=22, body_size=20)
            card(s, 4.80, 3.88, 3.72, 1.99, "2  TEST", "Nominal + held-out.\n30 matched trials each.", BLUE, title_size=22, body_size=20)
            card(s, 8.78, 3.88, 3.72, 1.99, "3  VIEW", "Print evaluation metrics.\nSave three PNG graphs.", ORANGE, title_size=22, body_size=20)
            text(s, "Code tour: Joint.step() = physics + reward; train() = PPO; main() = experiment + graphs.", .85, 6.20, 11.6, .28, 15, GRAY)
        elif n == 3:
            picture(s, "tracking_for_slides.png", y=1.72)
            text(s, "Dashed = desired angle  •  Colored = actual angle  •  Same held-out trial for both policies", .83, 5.99, 11.75, .35, 18, TEAL, True)
            text(s, "Seed 1000: both pass RMSE < 0.32 rad. Overall success uses all 30 trials, shown next.", .83, 6.35, 11.75, .22, 14, GRAY)
        elif n == 4:
            picture(s, "success_for_slides.png", y=1.52)
            text(s, "Held-out mean RMSE: 0.3299 rad fixed → 0.2250 rad randomized. Both reach 100% nominal success.", .78, 6.16, 11.83, .40, 17, TEAL, True)
        elif n == 5:
            readings = [
                (.78, 1.76, "TAN ET AL. (2018)", "Learned control, actuation, and\nrandomized simulation physics.", "https://arxiv.org/abs/1804.10332", TEAL),
                (6.84, 1.76, "ZHAO ET AL. · SURVEY", "The reality gap: simulation and\nreal hardware behave differently.", "https://arxiv.org/abs/2009.13303", BLUE),
                (.78, 4.08, "SUTTON & BARTO · CH. 13", "Policy gradients are the foundation.\nPPO: a clipped actor–critic method.", "http://incompleteideas.net/book/the-book-2nd.html", TEAL),
                (6.84, 4.08, "MUJOCO", "Richer multibody and contact physics.\nThis example uses a simpler joint model.", "https://mujoco.org/", ORANGE),
            ]
            for x, y, title, body, url, color in readings:
                card(s, x, y, 5.70, 1.98, title, body, color, title_size=20, body_size=19)
                s.notes_slide.notes_text_frame.text += "\n" + title + ": " + url
        elif n == 6:
            text(s, "Application: a quadruped joint must track commands\nas its payload, friction, and communication delay change.", .80, 1.79, 11.7, 1.31, 29, WHITE, True)
            for x, title, body in [( .83, "PAYLOAD", "Changes joint inertia"),
                                  (4.83, "WEAR", "Changes friction"),
                                  (8.83, "ACTUATION", "Introduces command delay")]:
                box(s, x, 3.46, 3.67, 1.37, "203B53", True)
                text(s, title, x+.17, 3.67, 3.32, .32, 20, "93D9D3", True)
                text(s, body, x+.17, 4.16, 3.32, .40, 18, WHITE)
            text(s, "This run: randomized training improved held-out tracking and success.", .86, 5.25, 11.6, .62, 26, WHITE, True)
            text(s, "Go2 / ANYmal inspiration • one joint • one training seed • simulation only", .86, 6.14, 11.6, .33, 18, "BED1DD")
    prs.save(DECK)


def para(doc, value, style=None):
    return doc.add_paragraph(value, style)


def build_guide(content, metrics, record):
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = sec.bottom_margin = DocInches(.65)
    sec.left_margin = sec.right_margin = DocInches(.75)
    doc.styles["Normal"].font.name = "Arial"
    doc.styles["Normal"].font.size = DocPt(12)
    doc.styles["Normal"].paragraph_format.space_after = DocPt(6)
    for name in ("Title", "Heading 1", "Heading 2"):
        doc.styles[name].font.name = "Arial"
        doc.styles[name].font.color.rgb = DocColor.from_string(TEAL)
    sec.header.paragraphs[0].text = "ENGR 6010  |  SHORT CLASSROOM DEMONSTRATION"
    footer = sec.footer.paragraphs[0]
    footer.alignment = 2
    footer.add_run("Presenter copy • ")
    field = OxmlElement('w:fldSimple'); field.set(qn('w:instr'), 'PAGE'); footer._p.append(field)
    doc.add_heading("Short presenter script + Q&A", 0)
    para(doc, "Six slides • approximately 5–6 minutes including the live code run", "Subtitle")
    doc.add_heading("The simple plan", 1)
    para(doc, "Explain the joint and two training choices. Run the program. Show the terminal results and open its graphs. Interpret the measured difference, then connect the idea to the readings and a quadruped joint.")
    doc.add_heading("1. Run your working code", 2)
    para(doc, r".\.venv\Scripts\python.exe reading5_sim2real_ppo.py")
    para(doc, "Use the project folder and Python 3.12 environment that already work in VS Code. Briefly point to Joint.step(), train(), and main(). The program trains both PPO agents, evaluates both conditions, prints statistics, and saves three PNGs. The verified cloud run took 27.72 seconds; local timing varies. Rerunning this command replaces its generated results and models.")
    doc.add_heading("2. Show the results", 2)
    para(doc, "Point out the fixed/random policy rows and fixed/real environment labels in the terminal. Here, real means held-out simulation. Open fig1_training_rewards.png briefly: recorded rewards show learning. Use slides 3–4 for readable tracking and success graphs, or open fig2_real_rollouts.png and fig3_success_rates.png directly.")
    para(doc, "Reported held-out results: fixed 14/30 successes and 0.3299 rad mean RMSE; randomized 29/30 and 0.2250 rad. Both achieve 30/30 nominal success. Success requires a full five-second episode with RMSE < 0.32 rad and no safety termination.")
    doc.add_heading("Optional: repeat quickly with saved models", 2)
    para(doc, r".\.venv\Scripts\python.exe classroom\demo.py --trials 30 --rounds 1 --open-figure")
    para(doc, "Or double-click classroom\\run_demo.cmd. This replays the same full evaluation without retraining, writes a new timestamped folder, and preserves the original outputs. Say clearly that this is saved-policy evaluation. If a live run fails, show the saved figures and identify them as previously executed results.")
    for start, end in ((0, 3), (3, 6)):
        doc.add_page_break()
        doc.add_heading(f"Speaking script: slides {start+1}–{end}", 1)
        for item in content["slides"][start:end]:
            doc.add_heading(f'{item["number"]}. {item["title"]}', 2)
            para(doc, "SAY: " + item["script"])
            para(doc, "DO: " + item["point_to"])
            para(doc, "NEXT: " + item["transition"])
        if start == 3:
            doc.add_heading("Reading links", 2)
            sources = ["Tan et al. (2018): https://arxiv.org/abs/1804.10332",
                       "Zhao et al. survey: https://arxiv.org/abs/2009.13303",
                       "Sutton & Barto, Ch. 13: http://incompleteideas.net/book/the-book-2nd.html",
                       "MuJoCo: https://mujoco.org/",
                       "Specific PPO method: https://arxiv.org/abs/1707.06347"]
            for source in sources:
                p = para(doc, source)
                p.paragraph_format.space_after = DocPt(3)
                for run in p.runs:
                    run.font.size = DocPt(9)
    doc.add_page_break()
    doc.add_heading("Quick Q&A", 1)
    for i, qa in enumerate(content["qa"], 1):
        doc.add_heading(f'{i}. {qa["question"]}', 2)
        p = para(doc, qa["answer"])
        for run in p.runs:
            run.font.size = DocPt(11)
    doc.save(GUIDE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true", help="also export PDFs with installed LibreOffice")
    args = parser.parse_args()
    content = json.loads((HERE / "presenter_content.json").read_text())
    metrics = pd.read_csv(ROOT / "metrics.csv")
    record = json.loads((ROOT / "experiment.json").read_text())
    assert [s["number"] for s in content["slides"]] == list(range(1, 7))
    charts(metrics)
    build_deck(content, metrics, record)
    build_guide(content, metrics, record)
    print("Created:", DECK, "\nCreated:", GUIDE)
    if args.pdf:
        office = shutil.which("soffice") or shutil.which("libreoffice")
        if office is None:
            raise SystemExit("Editable files created. Export PDFs from PowerPoint/Word or install LibreOffice.")
        profile = Path(tempfile.mkdtemp(prefix="reading5_short_office_"))
        subprocess.run([office, f"-env:UserInstallation={profile.as_uri()}", "--headless", "--convert-to", "pdf",
                        "--outdir", str(HERE), str(DECK), str(GUIDE)], check=True, timeout=90)
        print("PDF export completed.")


if __name__ == "__main__":
    main()
