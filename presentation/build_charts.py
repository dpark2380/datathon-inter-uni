"""Generate the presentation chart/diagram PNGs from the project's real numbers.

All figures use only values already verified in REPORT.md / slide_plan.md / the
fairness audit -- nothing here is decorative or invented. Run with:

    .venv/bin/python3 presentation/build_charts.py

Outputs land in presentation/charts/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrow, FancyBboxPatch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "charts")
os.makedirs(OUT, exist_ok=True)

# Palette from slide_plan.md
CANVAS = "#F5F5F7"
SURFACE = "#FFFFFF"
TEXT = "#1D1D1F"
SECOND = "#6E6E73"
BLUE = "#0071E3"
DARKBLUE = "#0057B8"
PALEBLUE = "#E8F2FF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": TEXT,
    "axes.edgecolor": SECOND,
    "axes.labelcolor": TEXT,
    "xtick.color": SECOND,
    "ytick.color": SECOND,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, transparent=True,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def box_chain(ax, labels, y=0.5, box_h=0.36, gap=0.06, fontsize=13, fill=SURFACE, edge=SECOND, textcolor=TEXT):
    n = len(labels)
    w = (1.0 - gap * (n - 1)) / n
    x = 0.0
    centers = []
    for lab in labels:
        box = FancyBboxPatch((x, y - box_h / 2), w, box_h,
                              boxstyle="round,pad=0.01,rounding_size=0.03",
                              linewidth=1.2, edgecolor=edge, facecolor=fill)
        ax.add_patch(box)
        cx = x + w / 2
        ax.text(cx, y, lab, ha="center", va="center", fontsize=fontsize,
                 color=textcolor, wrap=True)
        centers.append((x, cx, x + w))
        x += w + gap
    for i in range(n - 1):
        x0 = centers[i][2]
        x1 = centers[i + 1][0]
        ax.annotate("", xy=(x1 - 0.005, y), xytext=(x0 + 0.005, y),
                     arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1)
    ax.axis("off")


# ---------------------------------------------------------------------------
# 1. End-to-end pipeline (slide 1)
fig, ax = plt.subplots(figsize=(11, 1.6))
box_chain(ax, ["Raw data", "Cleaning", "81 features", "3 models", "Weighted\naverage", "Prediction"],
          fontsize=12.5)
save(fig, "pipeline.png")

# ---------------------------------------------------------------------------
# 2. Raw-to-engineered data flow (slide 2)
fig, ax = plt.subplots(figsize=(9, 1.5))
box_chain(ax, ["23 raw columns", "Minimal, targeted cleaning", "81 model features"], fontsize=13)
save(fig, "cleaning_flow.png")

# ---------------------------------------------------------------------------
# 3a. Monthly account-flow / spending-proxy diagram (slide 3)
fig, ax = plt.subplots(figsize=(9.5, 2.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

def term(x, label, sub=None):
    box = FancyBboxPatch((x, 1.0), 2.0, 1.1, boxstyle="round,pad=0.02,rounding_size=0.06",
                          linewidth=1.2, edgecolor=SECOND, facecolor=SURFACE)
    ax.add_patch(box)
    ax.text(x + 1.0, 1.68, label, ha="center", va="center", fontsize=12.5, color=TEXT)
    if sub:
        ax.text(x + 1.0, 1.28, sub, ha="center", va="center", fontsize=9.5, color=SECOND)

term(0.0, "Current bill", "known")
ax.text(2.3, 1.55, "−", ha="center", va="center", fontsize=20, color=SECOND)
term(2.6, "Previous bill", "known")
ax.text(4.9, 1.55, "+", ha="center", va="center", fontsize=20, color=SECOND)
term(5.2, "Payment", "known")
ax.text(7.5, 1.55, "=", ha="center", va="center", fontsize=20, color=SECOND)

box = FancyBboxPatch((7.8, 0.85), 2.1, 1.4, boxstyle="round,pad=0.02,rounding_size=0.06",
                      linewidth=1.6, edgecolor=BLUE, facecolor=PALEBLUE)
ax.add_patch(box)
ax.text(8.85, 1.75, "New activity", ha="center", va="center", fontsize=12.5, color=DARKBLUE, fontweight="bold")
ax.text(8.85, 1.32, "spending / net\naccount-activity proxy", ha="center", va="center", fontsize=9, color=DARKBLUE)
save(fig, "spend_proxy.png")

# 3b. Feature-family improvement comparison (real OOF deltas, REPORT.md sec1.3)
fig, ax = plt.subplots(figsize=(7.2, 3.2))
families = ["Spend\ndecomposition", "Behavioural\nsignatures", "Spend as\nsequence",
            "Interaction\nterms", "Peer-relative\nlimit", "Velocity /\nacceleration",
            "Target\nencoding"]
deltas = [0.00046, 0.00006, 0.00002, 0.00000, -0.00011, -0.00024, -0.00052]
colors = [BLUE if d == max(deltas) else "#C7C7CC" for d in deltas]
bars = ax.bar(families, [d * 1000 for d in deltas], color=colors, width=0.6)
ax.axhline(0.5, color=SECOND, linestyle="--", linewidth=1)
ax.text(6.5, 0.55, "noise threshold (0.0005)", ha="right", va="bottom", fontsize=9, color=SECOND)
ax.set_ylabel("OOF log loss improvement (×10⁻³)", fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", labelsize=8.5)
for b, d in zip(bars, deltas):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.05 if d >= 0 else -0.05),
             f"{d:+.5f}", ha="center", va="bottom" if d >= 0 else "top", fontsize=8, color=TEXT)
ax.set_ylim(-0.75, 0.62)
save(fig, "feature_family_gains.png")

# ---------------------------------------------------------------------------
# 4. Hidden-test experiment progression (slide 4)
fig, ax = plt.subplots(figsize=(9.5, 4.2))
steps = [
    ("LightGBM + GRU\nbaseline", 0.41260),
    ("Spending features\n+ tuned GRU", 0.41038),
    ("LightGBM full-\ndata refit", 0.41032),
    ("TabPFN added", 0.40995),
    ("Final six-partition\nTabPFN ensemble", 0.40982),
]
labels = [s[0] for s in steps]
values = [s[1] for s in steps]
y = range(len(values))
colors = [DARKBLUE if i == len(values) - 1 else BLUE for i in range(len(values))]
bars = ax.barh(list(y), values, color=colors, height=0.55)
ax.invert_yaxis()
ax.set_yticks(list(y))
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlim(0.4085, 0.4130)
ax.set_xlabel("Hidden-test log loss  -  lower is better", fontsize=11, color=TEXT)
ax.spines[["top", "right"]].set_visible(False)
for b, v in zip(bars, values):
    ax.text(v + 0.00015, b.get_y() + b.get_height() / 2, f"{v:.5f}", va="center", fontsize=10, color=TEXT)
save(fig, "experiment_progression.png")

# ---------------------------------------------------------------------------
# 5. Three-branch ensemble diagram (slide 5)
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis("off")

branches = [
    ("LightGBM  -  45%", "81 aggregate features", 3.2),
    ("GRU  -  30%", "6-month sequence", 2.0),
    ("TabPFN  -  25%", "pretrained tabular prior", 0.8),
]
for label, sub, y0 in branches:
    box = FancyBboxPatch((0.2, y0 - 0.45), 3.6, 0.9, boxstyle="round,pad=0.02,rounding_size=0.05",
                          linewidth=1.2, edgecolor=SECOND, facecolor=SURFACE)
    ax.add_patch(box)
    ax.text(2.0, y0 + 0.1, label, ha="center", va="center", fontsize=12, color=TEXT, fontweight="bold")
    ax.text(2.0, y0 - 0.22, sub, ha="center", va="center", fontsize=9.5, color=SECOND)
    ax.annotate("", xy=(6.1, 2.0), xytext=(3.85, y0),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4,
                                  connectionstyle=f"arc3,rad={(y0-2.0)*0.12}"))

box = FancyBboxPatch((6.3, 1.55), 2.0, 0.9, boxstyle="round,pad=0.02,rounding_size=0.05",
                      linewidth=1.6, edgecolor=BLUE, facecolor=PALEBLUE)
ax.add_patch(box)
ax.text(7.3, 2.0, "Weighted\naverage", ha="center", va="center", fontsize=11.5, color=DARKBLUE, fontweight="bold")
ax.annotate("", xy=(9.6, 2.0), xytext=(8.35, 2.0),
             arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
ax.text(9.65, 2.0, "0.40982", ha="left", va="center", fontsize=15, color=TEXT, fontweight="bold")
save(fig, "ensemble_diagram.png")

# ---------------------------------------------------------------------------
# 6. Annotate the existing SHAP summary: outline the six strongest features
src_path = os.path.join(HERE, "..", "output", "analysis", "shap_summary.png")
im = Image.open(src_path).convert("RGB")
w, h = im.size
draw = ImageDraw.Draw(im)
# Six strongest feature rows sit at the top of the plot, below the title.
top = int(h * 0.075)
bottom = int(h * 0.391)
draw.rectangle([int(w * 0.005), top, int(w * 0.86), bottom], outline=(0, 113, 227), width=5)
im.save(os.path.join(OUT, "shap_summary_top6.png"))

# ---------------------------------------------------------------------------
# 7a. Decision-support continuum (slide 7)
fig, ax = plt.subplots(figsize=(10.5, 1.6))
segs = [("Low risk\nstandard process", 0.5, "#C7C7CC"),
        ("Uncertain\nmanual review", 0.25, BLUE),
        ("High risk\nverification / support", 0.25, DARKBLUE)]
x = 0
for label, width, color in segs:
    ax.barh(0, width, left=x, height=0.6, color=color)
    ax.text(x + width / 2, 0, label, ha="center", va="center", fontsize=10.5,
             color=(TEXT if color == "#C7C7CC" else "white"))
    x += width
ax.set_xlim(0, 1)
ax.set_ylim(-0.6, 0.6)
ax.axis("off")
save(fig, "decision_continuum.png")

# 7b. False-positive / false-negative consequence panel
fig, ax = plt.subplots(figsize=(10.5, 2.4))
ax.axis("off")
ax.set_xlim(0, 2)
ax.set_ylim(0, 1)
panel1 = FancyBboxPatch((0.03, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02,rounding_size=0.03",
                         linewidth=1.2, edgecolor=SECOND, facecolor=SURFACE)
panel2 = FancyBboxPatch((1.05, 0.06), 0.92, 0.88, boxstyle="round,pad=0.02,rounding_size=0.03",
                         linewidth=1.2, edgecolor=SECOND, facecolor=SURFACE)
ax.add_patch(panel1)
ax.add_patch(panel2)
ax.text(0.49, 0.72, "False positive", fontsize=13, color=DARKBLUE, fontweight="bold", ha="center")
ax.text(0.49, 0.40, "A reliable customer faces\nunnecessary review, reduced\ncredit or an adverse decision.",
        fontsize=10, color=TEXT, ha="center", va="center")
ax.text(1.51, 0.72, "False negative", fontsize=13, color=DARKBLUE, fontweight="bold", ha="center")
ax.text(1.51, 0.40, "A likely default is missed;\nlosses are underestimated and\nearly support is not offered.",
        fontsize=10, color=TEXT, ha="center", va="center")
save(fig, "fp_fn_panel.png")

# ---------------------------------------------------------------------------
# Appendix: ensemble weight-search plateau (LightGBM weight vs OOF log loss)
fig, ax = plt.subplots(figsize=(8, 4))
lgbm_w = [1.00, 0.65, 0.55, 0.45, 0.34]
oof = [0.42211, 0.42155, 0.42141, 0.42136, 0.42147]
order = sorted(range(len(lgbm_w)), key=lambda i: lgbm_w[i])
lgbm_w = [lgbm_w[i] for i in order]
oof = [oof[i] for i in order]
ax.plot(lgbm_w, oof, marker="o", color=BLUE, linewidth=2)
ax.axvspan(0.50, 0.70, color=PALEBLUE, alpha=0.7, zorder=0)
ax.text(0.60, max(oof) + 0.00004, "within 0.0001 of\nthe optimum", ha="center", fontsize=9, color=DARKBLUE)
best_i = oof.index(min(oof))
ax.scatter([lgbm_w[best_i]], [oof[best_i]], color=DARKBLUE, s=70, zorder=5)
ax.annotate("selected: 0.45 / 0.30 / 0.25", (lgbm_w[best_i], oof[best_i]),
             textcoords="offset points", xytext=(10, -18), fontsize=9.5, color=TEXT)
ax.set_xlabel("LightGBM weight in the blend", fontsize=10.5)
ax.set_ylabel("Out-of-fold log loss", fontsize=10.5)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "weight_search_plateau.png")

print("Charts written to", OUT)
