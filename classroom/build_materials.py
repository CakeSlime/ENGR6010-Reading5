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
    text(slide, f'{item["number"]:02d} / 18', 12.04, 7.23, .68, .20, 10, "BED1DD" if dark else GRAY, align=PP_ALIGN.RIGHT)
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
    prs.core_properties.title = "ENGR 6010: Sim-to-Real Joint Tracking with PPO"
    prs.core_properties.subject = "Mechatronics student classroom demonstration and verified results"
    prs.core_properties.author = "ENGR 6010 student presenter"
    slides = content["slides"]
    for item in slides:
        n = item["number"]
        s = base(prs, item, None if n == 1 else item["title"], dark=n in (1, 16))
        if n == 1:
            text(s, "Sim-to-Real\nJoint Tracking\nwith PPO", .65, 1.05, 7.8, 2.8, 44, WHITE, True)
            text(s, "Can varied training physics improve\nperformance in an unfamiliar simulation?", .68, 4.02, 7.6, 1.03, 24, "D3E5EE")
            text(s, "Mechatronics engineering student demonstration", .70, 5.35, 10.8, .4, 19, "93D9D3")
            box(s, 9.1, 1.55, 3.5, 4.3, "203B53", True)
            # Schematic leg: motor joint, two rigid links, angle target motif.
            pivot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.1), Inches(2.16), Inches(1.1), Inches(1.1))
            pivot.fill.solid(); pivot.fill.fore_color.rgb=rgb(TEAL); pivot.line.fill.background()
            text(s, "q", 10.40, 2.34, .5, .5, 26, WHITE, True)
            for a, b, c, d in ((10.65, 3.2, 11.5, 4.1), (11.5, 4.1, 10.0, 4.8)):
                line=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(a), Inches(b), Inches(c), Inches(d))
                line.line.color.rgb=rgb("93D9D3"); line.line.width=Pt(13)
            text(s, "1 joint • 2 policies\n120 evaluation episodes", 9.30, 5.08, 3.1, .58, 17, WHITE, align=PP_ALIGN.CENTER)
        elif n == 2:
            text(s, "A robot controller meets more than an ideal equation.", .67, 1.65, 11.95, .65, 27, NAVY, True)
            card(s, .68, 2.65, 3.82, 2.75, "MECHANICS", "Inertia changes acceleration.\nFriction changes damping.")
            card(s, 4.75, 2.65, 3.82, 2.75, "ACTUATION", "Delayed commands arrive late.\nTorque has physical limits.", ORANGE)
            card(s, 8.83, 2.65, 3.82, 2.75, "SENSING", "Measurements contain noise.\nFeedback quality affects control.", BLUE)
            text(s, "Inspired by Go2 / ANYmal applications; this model isolates one joint.", .70, 5.85, 11.9, .45, 21, GRAY)
        elif n == 3:
            blocks=[(.72, 2.05, "OBSERVATION", "q, ω (noisy)\nr, r (preview)\nLast target", 2.35),
                    (3.47, 2.05, "PPO", "Target angle\n−1 to +1 rad", 2.05),
                    (5.92, 2.05, "DELAY + PD", "Queued target\nTorque ±12 N·m", 2.53),
                    (8.85, 2.05, "JOINT MODEL", "Inertia + friction\nAngle + velocity", 3.05)]
            for x,y,title,body,w in blocks:
                card(s,x,y,w,2.05,title,body, title_size=19,body_size=18)
            for x1,x2 in ((3.08,3.38),(5.53,5.83),(8.46,8.77)):
                arrow(s,x1,3.05,x2,3.05)
            text(s, "Noisy state returns to both the policy and PD feedback.", .78, 4.47, 11.8, .5, 22, TEAL, True)
            box(s,.74,5.14,11.85,.82,PALE,True)
            text(s,"τ = clip[12(qtarget, delayed − qmeasured) − 2ωmeasured, −12, 12]",.92,5.38,11.45,.4,23)
            text(s,"Both policy and PD updates run at 20 Hz in this simplified implementation.",.78,6.14,11.8,.3,15,GRAY)
        elif n == 4:
            card(s,.72,1.75,6.0,3.55,"TRANSPARENT DYNAMICS", "I · angular acceleration = τ − b · ω\n\nSemi-implicit Euler, Δt = 0.05 s:\nωnext = ω + Δt(τ − bω) / I\nqnext = q + Δt · ωnext",body_size=22)
            card(s,7.0,1.75,5.6,3.55,"GYMNASIUM CONTRACT", "5 continuous observations\n1 bounded angle-target action\n100 steps = 5 simulated seconds\nSafety: |q| > 2 rad or |ω| > 12 rad/s",accent=BLUE,body_size=21)
            text(s,"reset(seed) → observation, info       step(action) → observation, reward, terminated, truncated, info",.78,5.70,11.82,.65,17,font="Consolas")
        elif n == 5:
            box(s,.74,1.78,11.85,1.08,PALE,True)
            text(s,"reward = 1 − 5e² − 0.005ω² − 0.001τ²",1.05,2.06,11.2,.5,32,TEAL,True,PP_ALIGN.CENTER)
            card(s,.76,3.10,3.77,1.68,"TRACKING ERROR", "Follow the desired angle.",title_size=20,body_size=19)
            card(s,4.78,3.10,3.77,1.68,"ANGULAR VELOCITY", "Discourage fast oscillations.",BLUE,title_size=20,body_size=19)
            card(s,8.80,3.10,3.77,1.68,"TORQUE", "Limit excessive effort.",ORANGE,title_size=20,body_size=19)
            text(s,"Success: full 5 s episode, no safety termination, RMSE < 0.32 rad",.84,5.14,11.7,.63,25,NAVY,True)
            text(s,"Threshold retained before training; 0.32 rad ≈ 18.3° is an educational tolerance.",.84,5.96,11.6,.4,18,GRAY)
        elif n == 6:
            card(s,.73,1.77,5.92,3.95,"MATCHED PPO RECIPE", "SB3 PPO • MLP [32, 32] • CPU\n12,288 timesteps per policy\nRollout 256 • batch 64 • 5 epochs\nLearning rate 0.0008 • γ = 0.98\nSame reward, references, seed 42",body_size=22)
            card(s,6.94,1.77,5.65,3.95,"POLICY-GRADIENT CONNECTION", "Sutton & Barto: general foundations\n\nPPO: a specific actor–critic method\nwith a clipped surrogate objective\n\nTraining-ratio clipping ≠ torque saturation",BLUE,body_size=20)
            text(s,"Experimental change: the distribution of training physics.",.82,6.03,11.7,.42,23,TEAL,True)
        elif n == 7:
            table(s,["Condition","Inertia scale","Friction b","Target delay"],
                  [["Fixed / nominal","1.00","0.20","0 ms"],
                   ["Randomized training","U(0.70, 1.80)","U(0.10, 0.65)","0, 50, 100, 150 ms"],
                   ["Held-out simulation","1.95","0.75","200 ms"]],.74,1.92,[3.25,2.55,2.35,3.66],.72,20)
            text(s,"Randomize at every reset; keep each episode’s physics constant.",.84,5.04,11.8,.45,23,TEAL,True)
            text(s,"Sensor SD: training / nominal = 0.005 rad, 0.02 rad/s\nHeld-out = 0.045 rad, 0.09 rad/s; noise affects policy and PD.",.84,5.63,11.7,.72,19)
        elif n == 8:
            table(s,["Code or artifact","Purpose"],
                  [["Joint / check_env()","Physical model, API, and environment validation"],
                   ["train() + policy_*.zip","Two matched PPO training runs and saved controllers"],
                   ["roll() / summarize()","Paired evaluations and reproducible statistics"],
                   ["figures() + three PNGs","Recorded learning, tracking, and success"],
                   ["CSVs + tests + README","Audit trail, installation, and limitations"]],.74,1.77,[4.15,7.66],.70,19)
            text(s,"The classroom helper reuses saved models and writes into classroom/demo_runs/.",.84,6.13,11.7,.30,16,TEAL)
        elif n == 9:
            picture(s,"training_for_slides.png",y=1.57)
            text(s,"Faint: actual episode returns  •  Bold: trailing 10-episode mean  •  Different training physics",.83,6.26,11.8,.25,14,GRAY)
        elif n == 10:
            card(s,.76,1.85,3.78,3.12,"30 MATCHED SEEDS", "1000–1029\nSame reference + initial state\nSame underlying noise draws",body_size=21)
            card(s,4.78,1.85,3.78,3.12,"120 EPISODES", "2 policies × 2 environments\n× 30 episodes\nDeterministic evaluation",BLUE,body_size=21)
            card(s,8.80,1.85,3.78,3.12,"AUDITABLE METRICS", "Reward and RMSE: mean ± SD\nSuccess: percentage\n95% Wilson intervals",ORANGE,body_size=20)
            text(s,"The 30 evaluation seeds vary trials; there is only one training seed per policy.",.82,5.49,11.85,.75,24,NAVY,True)
        elif n == 11:
            picture(s,"tracking_for_slides.png",y=1.68)
            text(s,"Held-out seed 1000: both policies pass 0.32 rad. This one trial illustrates tracking, not the overall success rate.",.77,6.08,11.9,.45,16,TEAL,True)
        elif n == 12:
            picture(s,"success_for_slides.png",y=1.52)
            text(s,"Held-out mean episode RMSE: 0.3299 → 0.2250 rad  •  Intervals describe evaluation uncertainty",.79,6.19,11.85,.34,17,TEAL,True)
        elif n == 13:
            text(s,"Replay the trained models; repeat the same matched seeds.",.79,1.74,11.8,.62,27,NAVY,True)
            box(s,.75,2.59,11.85,1.04,NAVY,True)
            text(s,r".\.venv\Scripts\python.exe classroom\demo.py --trials 3 --rounds 2",.97,2.91,11.43,.43,19,WHITE,font="Consolas")
            card(s,.78,3.98,3.76,1.88,"1  RUN", "Load saved policies.\nWatch both rounds finish.",title_size=21,body_size=20)
            card(s,4.79,3.98,3.76,1.88,"2  COMPARE", "Open the new tracking PNG.\nCheck identical rounds.",BLUE,title_size=21,body_size=19)
            card(s,8.80,3.98,3.76,1.88,"3  REPEAT", "Run again: a fresh folder.\nOr double-click run_demo.cmd.",ORANGE,title_size=21,body_size=18)
            text(s,"Quick demo = 3 seeds; original results = 30. Repeated rounds add no independent evidence.",.82,6.20,11.77,.28,15,TEAL,True)
        elif n == 14:
            card(s,.77,1.83,3.76,2.68,"ENVIRONMENT", "check_env(): all 3 modes\n8 regression tests passed\nBounds, delay, dynamics",title_size=22,body_size=20)
            card(s,4.79,1.83,3.76,2.68,"EXPERIMENT", "7 artifact audits passed\n12,000 evaluation steps\nSaved-model replay",BLUE,title_size=22,body_size=20)
            card(s,8.81,1.83,3.76,2.68,"CLASSROOM", "Two rounds match exactly\nFull replay matches CSVs\nRoot results preserved",ORANGE,title_size=22,body_size=20)
            text(s,"Measured cloud runtimes",.82,4.80,11.8,.5,26,NAVY,True)
            text(s,f"Full experiment: {record['runtime_seconds']:.2f} s  •  Training: {record['training_seconds']:.2f} s\nQuick replay: 5.82–7.06 s  •  Full replay: 9.32 s\nMeasured on the cloud CPU; laptop timing varies.",.82,5.38,11.7,1.17,20)
        elif n == 15:
            card(s,.76,1.80,5.78,3.87,"WHAT THIS RUN SUPPORTS", "A controlled comparison of training physics\nObserved robustness in one held-out condition\nAn inspectable mechatronics control loop",body_size=22)
            card(s,6.80,1.80,5.78,3.87,"NEXT ENGINEERING TESTS", "Multiple independent training seeds\nPlain PD baseline and parameter ablations\nTimestep sensitivity and richer dynamics\nCalibrated hardware with safety systems",ORANGE,body_size=21)
            text(s,"No full quadruped, contacts, balance, calibrated hardware, or proof of stability is claimed.",.82,6.05,11.75,.4,18,GRAY)
        elif n == 16:
            text(s,"For this policy pair, varied physics\nimproved held-out robustness.",.78,1.85,11.8,1.44,37,WHITE,True)
            for x, number, label in ((.90,"14 → 29","Successful held-out trials / 30"),(5.22,"0.3299 → 0.2250","Held-out mean RMSE (rad)"),(10.0,"100%","Nominal success, both")):
                width=4.4 if x<9 else 2.5
                text(s,number,x,3.90,width,.64,30 if x<9 else 34,"93D9D3",True)
                text(s,label,x,4.70,width,.8,18,"D3E5EE")
            text(s,"Engineering takeaway: test the reality gap, retain the evidence, and qualify the claim.",.82,5.93,11.77,.44,21,WHITE)
        elif n == 17:
            refs=[("TAN ET AL. (2018)","Sim-to-Real: Learning Agile Locomotion for Quadruped Robots","https://arxiv.org/abs/1804.10332","Actuation and randomized-simulation motivation"),
                  ("ZHAO ET AL.","Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: A Survey","https://arxiv.org/abs/2009.13303","Reality-gap context and transfer strategies"),
                  ("SUTTON & BARTO · CHAPTER 13","Reinforcement Learning: An Introduction","http://incompleteideas.net/book/the-book-2nd.html","Policy-gradient / actor–critic foundations"),
                  ("MUJOCO","Full multibody physics, contacts, constraints, and actuators","https://mujoco.org/","A richer simulator; not used in this one-joint code")]
            for j,(author,title,url,connection) in enumerate(refs):
                y=1.66+j*1.08
                text(s,author,.80,y,11.8,.24,14,TEAL,True)
                text(s,title,.80,y+.28,11.8,.32,18,NAVY,True)
                sh=text(s,url+"  |  "+connection,.80,y+.67,11.8,.26,12,GRAY)
                sh.text_frame.paragraphs[0].runs[0].hyperlink.address=url
            text(s,"Specific PPO method: Schulman et al. (2017), https://arxiv.org/abs/1707.06347",.80,6.18,11.8,.25,13,GRAY)
        elif n == 18:
            rows=[]
            for policy,domain in (("fixed","fixed"),("random","fixed"),("fixed","real"),("random","real")):
                g=metrics[(metrics.policy==policy)&(metrics.environment==domain)].iloc[0]
                rows.append(["Fixed" if policy=="fixed" else "Randomized", "Nominal" if domain=="fixed" else "Held-out",
                             f"{g.reward_mean:.2f} ± {g.reward_std:.2f}", f"{g.rmse_mean_rad:.4f} ± {g.rmse_std_rad:.4f}",
                             f"{g.success_pct:.1f}%", f"{g.success_ci_low_pct:.1f}–{g.success_ci_high_pct:.1f}%"])
            table(s,["Policy","Condition","Reward ± SD","RMSE ± SD (rad)","Success","95% interval"],rows,.74,1.82,[1.75,1.52,2.26,2.64,1.50,2.14],.67,16)
            text(s,"Replay all 30 seeds without retraining:",.83,5.41,11.7,.42,22,TEAL,True)
            text(s,r".\.venv\Scripts\python.exe classroom\demo.py --trials 30 --rounds 1",.83,5.99,11.74,.43,18,font="Consolas")
    prs.save(DECK)


def para(doc, value, style=None):
    return doc.add_paragraph(value, style)


def doc_table(doc, headers, rows):
    t=doc.add_table(rows=1, cols=len(headers)); t.style="Light Shading Accent 1"
    for cell,value in zip(t.rows[0].cells,headers):
        cell.text=value
    for values in rows:
        for cell,value in zip(t.add_row().cells,values):
            cell.text=str(value)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size=DocPt(9)
    return t


def build_guide(content, metrics, record):
    doc=Document()
    section=doc.sections[0]
    section.top_margin=section.bottom_margin=DocInches(.65)
    section.left_margin=section.right_margin=DocInches(.75)
    for name in ("Normal","Body Text"):
        style=doc.styles[name]
        style.font.name="Arial"; style.font.size=DocPt(12)
        style.paragraph_format.space_after=DocPt(6)
    for name in ("Title","Heading 1","Heading 2","Heading 3"):
        doc.styles[name].font.name="Arial"
        doc.styles[name].font.color.rgb=DocColor.from_string(TEAL)
    section.header.paragraphs[0].text="ENGR 6010  |  AI IN ROBOTICS  |  PRESENTER COPY"
    footer=section.footer.paragraphs[0]
    footer.alignment=2
    footer.add_run("Presenter guide  •  ")
    field=OxmlElement('w:fldSimple'); field.set(qn('w:instr'),'PAGE'); footer._p.append(field)
    doc.add_heading("Sim-to-Real Joint Tracking with PPO",0)
    para(doc,"Presenter script, repeatable demo checklist, and Q&A", "Subtitle")
    para(doc,"Voice: a mechatronics engineering student presenting a measured classroom experiment. Suggested delivery: approximately 15 minutes plus discussion; slide 17 is the source list and slide 18 is a backup. Use PowerPoint Presenter View for the same slide-by-slide notes.")
    doc.add_heading("The central claim I can defend",1)
    para(doc,"For this pair of PPO policies, domain randomization improved performance in one held-out simulated condition: success increased from 14/30 to 29/30, and mean episode RMSE decreased from 0.3299 to 0.2250 rad. Both achieved 30/30 nominal success. This is an educational simulation study, not a physical robot trial or full quadrupedal locomotion result.")
    doc.add_heading("What to keep separate",1)
    for value in ["Original evidence: 30 distinct matched evaluation seeds per policy and condition; one training seed per policy.",
                  "Quick classroom demo: 3 seeds per condition, repeated twice. Expected held-out counts for these 3 seeds: fixed 1/3, randomized 2/3. This subset does not replace the 30-trial results.",
                  "Repeated rounds demonstrate deterministic replay. They are not independent replications and do not increase the scientific sample size.",
                  "Full retraining is a separate command. The quick replay only loads saved models and performs inference/evaluation.",
                  "Slides show the verified cloud results. Locally regenerated values may differ slightly across platforms; identify any replacement data honestly."]:
        para(doc,value,"List Bullet")
    doc.add_heading("Presentation plan",1)
    para(doc,"Slides 1–8: question, control loop, model, and experiment. Slides 9–12: evidence. Slide 13: live replay. Slides 14–16: validation, limitations, and conclusion. Slide 17: references. Slide 18: detailed results for questions.")
    doc.add_page_break()
    doc.add_heading("Before class: Windows / VS Code checklist",1)
    for value in ["Extract the entire classroom package. Keep main source, requirements, both policy ZIPs, and classroom/ together.",
                  "In VS Code, use File → Open Folder and select the folder containing reading5_sim2real_ppo.py. Opening only the Python file can prevent interpreter paths from resolving.",
                  "Use Python 3.12 and the project's .venv. Python 3.14 is not the tested interpreter for these pinned requirements.",
                  "Use Python: Select Interpreter and browse to the full path of .venv\\Scripts\\python.exe. The status bar should show Python 3.12.",
                  "Rehearse the replay with the projector connected. Open the deck and generated tracking PNG once before class. Installed dependencies and saved policies allow an offline demo."]:
        para(doc,value,"List Number")
    para(doc,"One-time setup, from the project folder:")
    commands=[r"py -3.12 -m venv .venv",r".\.venv\Scripts\python.exe -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu",r".\.venv\Scripts\python.exe -m pip install -r requirements.txt"]
    for value in commands:
        p=para(doc,value); p.runs[0].font.name="Consolas"; p.runs[0].font.size=DocPt(9)
    para(doc,"If Python 3.12 is absent: winget install --exact --id Python.Python.3.12, then restart VS Code. Package installation needs Internet access; the prepared demo does not.")
    doc.add_heading("Live run: default short demonstration",2)
    para(doc,r".\.venv\Scripts\python.exe classroom\demo.py --trials 3 --rounds 2 --open-figure")
    para(doc,"Alternatively, double-click classroom\\run_demo.cmd. It uses the project interpreter explicitly. The supplied Windows launcher and image-viewer opening were not executed on the Linux cloud; the Python replay itself was executed successfully.")
    doc.add_heading("If something interrupts the demonstration",2)
    para(doc,"Missing gymnasium or torch: check the selected interpreter and install requirements into that same .venv. Missing policy ZIP: extract the complete package. No image window: open the PNG at the printed path. For a persistent machine problem, return to the embedded result slides and say, ‘These are saved outputs of the verified run; I am switching to a walkthrough of those results.’ Do not describe cached figures as a new live run.")
    doc.add_page_break()
    doc.add_heading("Slide-by-slide presenter script",1)
    para(doc,"Read the script as needed, point to the indicated visual, and use the transition to keep the story connected. The short SAY phrase on each slide is a speaking anchor, not additional experimental evidence.")
    for i,item in enumerate(content["slides"]):
        if i and i%2==0:
            doc.add_page_break()
        doc.add_heading(f'{item["number"]:02d}. {item["title"]}',2)
        para(doc,f'Suggested time: {item["seconds"]} seconds.  Speaking anchor: “{item["cue"]}”')
        para(doc,"READ: "+item["script"])
        para(doc,"POINT TO / DO: "+item["point_to"])
        para(doc,"TRANSITION: "+item["transition"])
    doc.add_page_break()
    doc.add_heading("Live-demo runbook",1)
    doc_table(doc,["Mode","Command after the project interpreter","Meaning"],
              [["Quick replay",r"classroom\demo.py --trials 3 --rounds 2","12 episodes per round; same 3 paired seeds; two identical rounds"],
               ["Full evaluation replay",r"classroom\demo.py --trials 30 --rounds 1","120 episodes, original 30-seed evaluation design; no training"],
               ["Full retraining","reading5_sim2real_ppo.py","Runs check_env, trains both models, evaluates, replaces root experiment outputs"]])
    for label,value in [
        ("Before pressing Enter","Say: ‘These policies are already trained. I am testing the saved controllers, not teaching them again.’"),
        ("During execution","Point to policy/environment labels and the repeatability PASS message. Each simulated episode lasts five seconds, but simulation can run faster than wall-clock time."),
        ("When the plot opens","Identify dashed desired angle and actual angle. The first held-out example is seed 1000; both policies pass the threshold on this particular trial."),
        ("When repeating","Run the command again. A new timestamped folder prevents previous results being overwritten. Same-seed repeats demonstrate execution repeatability only."),
        ("When showing numerical results","The quick subset gives fixed 1/3 and randomized 2/3 held-out successes in the verified run. Refer back to the original 30-trial slide for 14/30 versus 29/30."),
        ("If you choose to retrain","Use a disposable copy of the complete project to preserve the prepared data. Report whatever that run produces; do not change the success threshold or select favorable seeds.")]:
        doc.add_heading(label,2); para(doc,value)
    para(doc,f"Measured cloud reference: full experiment {record['runtime_seconds']:.2f} s, training {record['training_seconds']:.2f} s. The tested short replay took 7.06 s, and full evaluation replay took 9.32 s, including imports. These timings are separate operations and vary across machines.")
    doc.add_page_break()
    doc.add_heading("Questions and answers",1)
    para(doc,"Start with the short answer. Use the longer answer if the instructor asks for mechanics, learning, or experimental details.")
    for i,qa in enumerate(content["qa"]):
        if i and i%4==0:
            doc.add_page_break()
        doc.add_heading(f'{i+1}. {qa["question"]}',2)
        for label,value,keep in [("SHORT ANSWER: ",qa["short_answer"],True),
                                 ("",qa["answer"],True),("Relevant slides: ",qa["slides"],False)]:
            p=para(doc,label+value)
            p.paragraph_format.keep_with_next=keep
            p.paragraph_format.space_after=DocPt(4)
            for run in p.runs:
                run.font.size=DocPt(11)
    doc.add_page_break()
    doc.add_heading("Exact-results reference",1)
    rows=[]
    for _,g in metrics.iterrows():
        rows.append(["Fixed" if g.policy=="fixed" else "Randomized","Nominal" if g.environment=="fixed" else "Held-out",
                     f"{g.reward_mean:.3f} ± {g.reward_std:.3f}",f"{g.rmse_mean_rad:.4f} ± {g.rmse_std_rad:.4f}",
                     f"{g.success_pct:.1f}%",f"{g.success_ci_low_pct:.1f}–{g.success_ci_high_pct:.1f}%"])
    doc_table(doc,["Policy","Condition","Reward ± SD","RMSE ± SD (rad)","Success","95% Wilson"],rows)
    para(doc,"SD is the sample standard deviation of the 30 episode outcomes. RMSE is first computed separately per episode and then averaged. Wilson intervals describe evaluation success for fixed trained policies, not uncertainty across training seeds. The success rule also requires a full episode without safety termination.")
    doc.add_heading("Sources and how I connect them",2)
    for value in [
        "Tan et al. (2018), Sim-to-Real: Learning Agile Locomotion for Quadruped Robots — https://arxiv.org/abs/1804.10332. Motivation: learned joint commands, actuation modeling, and randomized simulation.",
        "Zhao et al., Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: A Survey — https://arxiv.org/abs/2009.13303. Motivation: the reality gap and limits of transfer strategies.",
        "Sutton and Barto, Reinforcement Learning: An Introduction, Chapter 13 — http://incompleteideas.net/book/the-book-2nd.html. General policy-gradient and actor–critic foundations.",
        "MuJoCo — https://mujoco.org/. A richer multibody/contact simulator; not a dependency of this example.",
        "Schulman et al. (2017), Proximal Policy Optimization Algorithms — https://arxiv.org/abs/1707.06347. The specific clipped-surrogate PPO method, distinct from policy gradients in general."]:
        para(doc,value)
    para(doc,"Evidence files: metrics.csv, evaluation_episodes.csv, evaluation_steps.csv, training_rewards.csv, experiment.json, verification.txt, environment_tests.txt, and classroom/demo_validation.json. The older uploaded ZIP's custom-PPO results are not the source of this deck's SB3 claims.")
    doc.save(GUIDE)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf",action="store_true",help="also export PDFs with installed LibreOffice")
    args=parser.parse_args()
    content=json.loads((HERE / "presenter_content.json").read_text())
    metrics=pd.read_csv(ROOT / "metrics.csv")
    record=json.loads((ROOT / "experiment.json").read_text())
    assert [s["number"] for s in content["slides"]]==list(range(1,19))
    charts(metrics); build_deck(content,metrics,record); build_guide(content,metrics,record)
    print("Created:",DECK,"\nCreated:",GUIDE)
    if args.pdf:
        office=shutil.which("soffice") or shutil.which("libreoffice")
        if office is None:
            raise SystemExit("Editable files created. Install LibreOffice or export PDFs from PowerPoint and Word.")
        profile=Path(tempfile.mkdtemp(prefix="reading5_office_"))
        subprocess.run([office,f"-env:UserInstallation={profile.as_uri()}","--headless","--convert-to","pdf",
                        "--outdir",str(HERE),str(DECK),str(GUIDE)],check=True,timeout=90)
        print("PDF export completed.")


if __name__=="__main__":
    main()
