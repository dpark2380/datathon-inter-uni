"""Build presentation/finalist-presentation.pptx from slide_plan.md.

Content, numbers and speaker notes are taken verbatim from slide_plan.md,
script.md and REPORT.md -- see presentation/speaker-notes.md for the notes
text and presentation/build_charts.py for the chart sources.

Run after build_charts.py:
    .venv/bin/python3 presentation/build_deck.py
"""
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
CHARTS = os.path.join(HERE, "charts")
ANALYSIS = os.path.join(HERE, "..", "output", "analysis")

# Palette (slide_plan.md)
CANVAS = RGBColor(0xF5, 0xF5, 0xF7)
SURFACE = RGBColor(0xFF, 0xFF, 0xFF)
TEXT = RGBColor(0x1D, 0x1D, 0x1F)
SECOND = RGBColor(0x6E, 0x6E, 0x73)
BLUE = RGBColor(0x00, 0x71, 0xE3)
DARKBLUE = RGBColor(0x00, 0x57, 0xB8)
PALEBLUE = RGBColor(0xE8, 0xF2, 0xFF)

FONT = "Helvetica Neue"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN = Inches(0.7)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def add_slide():
    slide = prs.slides.add_slide(BLANK)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = CANVAS
    bg.line.fill.background()
    bg.shadow.inherit = False
    # send background to back
    spTree = slide.shapes._spTree
    spTree.remove(bg._element)
    spTree.insert(2, bg._element)
    return slide


def add_text(slide, left, top, width, height, text, size, color=TEXT, bold=False,
             align=PP_ALIGN.LEFT, font=FONT, anchor=MSO_ANCHOR.TOP, line_spacing=1.0,
             italic=False):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.name = font
        r.font.color.rgb = color
    return box


def add_bullets(slide, left, top, width, height, items, size=18, color=TEXT,
                 bullet_color=BLUE, gap_pt=10, font=FONT, bold_lead=None):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap_pt)
        p.line_spacing = 1.08
        r = p.add_run()
        r.text = f"-  {item}"
        r.font.size = Pt(size)
        r.font.name = font
        r.font.color.rgb = color
    return box


def kicker(slide, text):
    add_text(slide, MARGIN, Inches(0.55), Inches(8), Inches(0.4), text.upper(),
              13, color=BLUE, bold=True, font=FONT)


def title(slide, text, size=34, top=Inches(0.9), width=Inches(11.9)):
    add_text(slide, MARGIN, top, width, Inches(1.0), text, size, color=TEXT,
              bold=True, font=FONT)


def footer(slide, n):
    add_text(slide, SLIDE_W - Inches(1.0), SLIDE_H - Inches(0.55), Inches(0.6),
              Inches(0.4), str(n), 11, color=SECOND, align=PP_ALIGN.RIGHT)


def hairline(slide, top, left=MARGIN, width=None):
    width = width or (SLIDE_W - 2 * MARGIN)
    ln = slide.shapes.add_connector(1, left, top, left + width, top)
    ln.line.color.rgb = RGBColor(0xD2, 0xD2, 0xD7)
    ln.line.width = Pt(0.75)


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def picture_fit(slide, path, left, top, max_w, max_h):
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    ar = iw / ih
    w, h = max_w, max_w / ar
    if h > max_h:
        h = max_h
        w = max_h * ar
    x = left + (max_w - w) / 2
    y = top + (max_h - h) / 2
    slide.shapes.add_picture(path, x, y, width=int(w), height=int(h))


NOTES = {}  # filled in below, mirrors presentation/speaker-notes.md

# ===========================================================================
# Slide 1 - Overview
s = add_slide()
kicker(s, "Stream 1 · Credit Card Default")
title(s, "Overview")
add_text(s, MARGIN, Inches(1.75), Inches(5.6), Inches(0.5),
          "Predict each customer's probability of default next month.", 19, color=SECOND)
add_bullets(s, MARGIN, Inches(2.35), Inches(5.6), Inches(2.6), [
    "Binary log loss, probability quality and calibration matter, not just ranking.",
    "24,000 labelled customers, 6,000 test customers, 23 raw columns, 22.12% default rate.",
    "Final model: 0.45 LightGBM + 0.30 GRU + 0.25 TabPFN.",
    "Final submission: submission_tabpfn6.csv.",
], size=17)
# headline result panel
panel = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.7), Inches(1.75),
                             Inches(4.9), Inches(2.1))
panel.fill.solid(); panel.fill.fore_color.rgb = PALEBLUE
panel.line.color.rgb = BLUE; panel.line.width = Pt(1.25)
panel.adjustments[0] = 0.06
panel.shadow.inherit = False
add_text(s, Inches(7.7), Inches(1.95), Inches(4.9), Inches(0.4), "HIDDEN-TEST LOG LOSS",
          13, color=DARKBLUE, bold=True, align=PP_ALIGN.CENTER)
add_text(s, Inches(7.7), Inches(2.35), Inches(4.9), Inches(1.3), "0.40982",
          64, color=DARKBLUE, bold=True, align=PP_ALIGN.CENTER)
picture_fit(s, os.path.join(CHARTS, "pipeline.png"), MARGIN, Inches(4.5),
             SLIDE_W - 2 * MARGIN, Inches(2.0))
add_text(s, MARGIN, Inches(6.6), Inches(11.9), Inches(0.5),
          "The result came from combining three complementary views of each customer.",
          15, color=SECOND, italic=True)
footer(s, 1)
NOTES[1] = ("We're predicting each customer's probability of default next month, scored by "
            "binary log loss, so calibration matters as much as ranking. Twenty-four thousand "
            "labelled customers, six thousand held out, twenty-three raw columns, a 22.12 "
            "percent default rate. Our final hidden-test log loss is 0.40982, from a blend of "
            "0.45 LightGBM, 0.30 GRU, and 0.25 TabPFN.")

# ===========================================================================
# Slide 2 - Data Cleaning (four-card layout: verified against models/common.py)
s = add_slide()
kicker(s, "Preparation")
title(s, "Data Cleaning")

def data_card(x, y, w, h, num, heading, bullets, heading_color=TEXT):
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    card.fill.solid(); card.fill.fore_color.rgb = SURFACE
    card.line.color.rgb = RGBColor(0xD2, 0xD2, 0xD7); card.line.width = Pt(1)
    card.adjustments[0] = 0.04
    card.shadow.inherit = False
    pad = Inches(0.22)
    add_text(s, x + pad, y + Inches(0.14), Inches(1.0), Inches(0.5), num, 30, color=RGBColor(0xD2, 0xD2, 0xD7), bold=True)
    add_text(s, x + pad, y + Inches(0.62), w - 2 * pad, Inches(0.4), heading, 17, color=heading_color, bold=True)
    add_bullets(s, x + pad, y + Inches(1.05), w - 2 * pad, h - Inches(1.2), bullets, size=11.5, gap_pt=7)

cw, ch, gx, gy = Inches(5.7), Inches(2.35), Inches(0.3), Inches(0.25)
x1, x2 = MARGIN, MARGIN + cw + gx
y1, y2 = Inches(1.75), Inches(1.75) + ch + gy

data_card(x1, y1, cw, ch, "01", "Repayment Status", [
    "PAY_0, PAY_2 through PAY_6 show months behind on payment.",
    "-2 = no balance, -1 = paid in full, 0 = paid minimum amount.",
    "Raw values stay untouched in the cleaned dataset, no grouping here.",
    "Aggregate lateness features (pay_max, n_late, trend_pay) clip -2/-1/0 to 0 only.",
    "ever_paid_full and never_used deliberately keep the original codes.",
])
data_card(x2, y1, cw, ch, "02", "Education", [
    "1 = graduate school, 2 = university, 3 = high school, 4 = other (official codes).",
    "Undocumented codes 0, 5, 6 folded into \"other\" (4), 290 rows.",
    "Confirmed cost: \"other\" over-predicted about 61%, AUC 0.645 vs about 0.79 elsewhere.",
    "The only subgroup where the model underperforms its own base rate.",
], heading_color=RGBColor(0xCB, 0x38, 0x2D))
data_card(x1, y2, cw, ch, "03", "Marriage", [
    "1 = married, 2 = single, 3 = other (official codes).",
    "Undocumented code 0 folded into \"other\" (3), 42 rows.",
    "No confirmed fairness or accuracy issue for this group.",
])
data_card(x2, y2, cw, ch, "04", "What We Didn't Clean", [
    "Zero missing values, no malformed rows, confirmed directly across all 24,000 rows.",
    "No outlier removal: large balances and payments can be genuine risk signal.",
    "No resampling or class weighting: preserves the true 22.12% base rate for calibration.",
    "client_id dropped as a non-informative identifier.",
])
footer(s, 2)
NOTES[2] = ("The data had no missing values and no malformed rows. Cleaning stayed deliberately "
            "narrow. EDUCATION codes 0, 5, and 6 got grouped into \"other\", 290 rows; MARRIAGE "
            "code 0 grouped into \"other\", 42 rows. We kept PAY status values of minus 1 and "
            "minus 2 as features, since they mean paid-in-full and inactive account, not missing "
            "data. No outlier removal, since large balances can be genuine signal. No resampling, "
            "since changing the 22 percent base rate would hurt calibration. Twenty-three raw "
            "columns became 81 engineered features.")

# ===========================================================================
# Slide 3 - Feature Engineering
s = add_slide()
kicker(s, "The largest single gain")
title(s, "Feature Engineering")
add_text(s, MARGIN, Inches(1.85), Inches(11.9), Inches(0.55),
          "Statement balances and payments are given, monthly account activity is not.",
          17, color=SECOND)
picture_fit(s, os.path.join(CHARTS, "spend_proxy.png"), MARGIN, Inches(2.5),
             Inches(11.9), Inches(1.9))
add_text(s, MARGIN, Inches(4.55), Inches(6.2), Inches(0.9),
          "A rising balance from new spending is a different risk story than one from missed "
          "payments, this proxy tells them apart.", 15.5, color=TEXT)
picture_fit(s, os.path.join(CHARTS, "feature_family_gains.png"), Inches(6.9), Inches(4.35),
             Inches(5.7), Inches(2.75))
add_text(s, MARGIN, Inches(5.65), Inches(6.2), Inches(1.1),
          "+0.00046 OOF log loss  ·  +0.00173 hidden-test log loss.\n"
          "Six other feature families never cleared the 0.0005 noise threshold.",
          15.5, color=DARKBLUE, bold=True, line_spacing=1.2)
footer(s, 3)
NOTES[3] = ("The data gives statement balances and payments, but never states what a customer "
            "actually spent. So we worked it out ourselves. New activity equals current bill "
            "minus previous bill plus payment. That matters because a rising balance from new "
            "spending is a different risk story than a rising balance from missed payments. This "
            "one feature family improved out-of-fold log loss by 0.00046 and hidden-test log "
            "loss by 0.00173, the largest gain of the project. Six other feature families we "
            "tried never cleared our 0.0005 noise threshold.")

# ===========================================================================
# Slide 4 - Experimentation
s = add_slide()
kicker(s, "41 experiments")
title(s, "Experimentation")
picture_fit(s, os.path.join(CHARTS, "experiment_progression.png"), Inches(6.9), Inches(1.75),
             Inches(5.7), Inches(4.9))
add_bullets(s, MARGIN, Inches(1.85), Inches(6.1), Inches(2.3), [
    "10 model families · 7 candidate feature families · 135 LightGBM configurations.",
    "Switched selection from AUC to log loss.",
    "Added the GRU to preserve month-to-month order.",
    "Introduced spending decomposition, the largest gain.",
    "Added TabPFN as the final complementary model.",
], size=15.5, gap_pt=8)
add_text(s, MARGIN, Inches(4.35), Inches(6.1), Inches(0.4), "Notable failures", 15,
          color=DARKBLUE, bold=True)
add_bullets(s, MARGIN, Inches(4.75), Inches(6.1), Inches(2.0), [
    "135 LightGBM configs: no reliable gain.",
    "1D-CNN duplicated the GRU signal (0.9924 correlated).",
    "Survival model improved OOF but worsened hidden test, removed.",
    "Isotonic calibration looked best on OOF, worst under nested CV.",
], size=14, gap_pt=6, bullet_color=SECOND)
add_text(s, MARGIN, Inches(6.75), Inches(11.9), Inches(0.5),
          "Disagreement between models only helps when it contains useful signal.",
          15, color=SECOND, italic=True)
footer(s, 4)
NOTES[4] = ("We ran 41 experiments across 10 model families, 7 feature families, and 135 "
            "LightGBM configurations. Only a handful of decisions actually moved the needle. "
            "We switched from AUC to log loss, added a GRU to preserve month-to-month order, "
            "added the spending decomposition, the biggest single jump, then added TabPFN. "
            "Score fell from 0.41260 to 0.40982. Plenty of experiments went nowhere. 135 "
            "LightGBM configs found nothing reliable, a 1D-CNN just duplicated the GRU's "
            "signal, a survival model improved out-of-fold but got worse on hidden test so we "
            "dropped it, and isotonic calibration looked great until nested cross-validation "
            "revealed it was leakage. Disagreement between models only helps when it carries "
            "real signal.")

# ===========================================================================
# Slide 5 - The Models (three-column card layout: name+weight / hyperparams / how it predicts)
s = add_slide()
kicker(s, "Model selection and validation")
title(s, "The Models")

model_cols = [
    ("LightGBM", "45%", "0.42211",
     [("learning_rate", "0.03"), ("num_leaves", "12"), ("max_depth", "4"),
      ("min_child_samples", "100"), ("reg_lambda", "30.0"), ("fits", "90 + 3-seed refit")],
     "Gradient-boosted decision trees. Hundreds of shallow trees, each correcting errors left "
     "by the ones before it, reading the 81 engineered features as one aggregate customer profile."),
    ("GRU", "30%", "0.42392",
     [("architecture", "2-layer bi-GRU"), ("hidden size", "32"), ("dropout", "0.4"),
      ("optimiser", "Adam, lr 2e-3"), ("batch size", "512"), ("fits", "150 + 5-seed refit")],
     "Bidirectional recurrent network. Reads the six months in order, updating an internal "
     "memory at each step, so it can tell a recovering customer apart from a deteriorating one."),
    ("TabPFN", "25%", "0.42345",
     [("pretrained", "0 params fit"), ("n_estimators", "4"), ("balance_probabilities", "False"),
      ("device", "cpu"), ("context size", "19,200 / fold"), ("fits", "30 + 3-seed refit")],
     "Pretrained transformer. Applies a prior learned from millions of synthetic datasets; the "
     "training rows act as live context at prediction time, with no parameters fit to this data."),
]
col_w = Inches(3.78)
col_h = Inches(5.35)
col_top = Inches(1.85)
for i, (name, weight, oof, params, desc) in enumerate(model_cols):
    x = MARGIN + i * (col_w + Inches(0.18))
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, col_top, col_w, col_h)
    card.fill.solid(); card.fill.fore_color.rgb = SURFACE
    card.line.color.rgb = RGBColor(0xD2, 0xD2, 0xD7); card.line.width = Pt(1)
    card.adjustments[0] = 0.04
    card.shadow.inherit = False
    pad = Inches(0.22)
    ix = x + pad
    # segment 1: name + weight + OOF (narrow, top)
    add_text(s, ix, col_top + Inches(0.18), col_w - 2 * pad, Inches(0.4), name, 20, color=TEXT, bold=True)
    add_text(s, ix, col_top + Inches(0.62), Inches(1.6), Inches(0.2), "WEIGHT", 9.5, color=SECOND, bold=True)
    add_text(s, ix, col_top + Inches(0.82), Inches(1.6), Inches(0.35), weight, 17, color=DARKBLUE, bold=True)
    add_text(s, ix + Inches(1.9), col_top + Inches(0.62), Inches(1.6), Inches(0.2), "OOF LOG LOSS", 9.5, color=SECOND, bold=True)
    add_text(s, ix + Inches(1.9), col_top + Inches(0.82), Inches(1.6), Inches(0.35), oof, 17, color=TEXT, bold=True)
    hairline(s, col_top + Inches(1.25), left=ix, width=col_w - 2 * pad)
    # segment 2: hyperparameters
    add_text(s, ix, col_top + Inches(1.38), Inches(2.5), Inches(0.2), "HYPERPARAMETERS", 9.5, color=SECOND, bold=True)
    for j, (k, v) in enumerate(params):
        py = col_top + Inches(1.65) + Inches(0.27) * j
        add_text(s, ix, py, (col_w - 2 * pad) * 0.55, Inches(0.24), k, 12, color=TEXT)
        add_text(s, ix + (col_w - 2 * pad) * 0.5, py, (col_w - 2 * pad) * 0.5 - pad, Inches(0.24), v, 12, color=SECOND, align=PP_ALIGN.RIGHT)
    hairline(s, col_top + Inches(3.35), left=ix, width=col_w - 2 * pad)
    # segment 3: how it predicts
    add_text(s, ix, col_top + Inches(3.48), Inches(2.5), Inches(0.2), "HOW IT PREDICTS", 9.5, color=SECOND, bold=True)
    add_text(s, ix, col_top + Inches(3.74), col_w - 2 * pad, Inches(1.55), desc, 12.5, color=TEXT, line_spacing=1.18)
footer(s, 5)
NOTES[5] = ("Three models, three different views of the same customer. LightGBM carries the "
            "most weight and reads 81 engineered features as one aggregate profile. The GRU "
            "reads six months as an ordered sequence, telling recovery apart from deterioration. "
            "TabPFN applies a pretrained prior, with zero parameters fit to our data.")

# ===========================================================================
# Slide 6 - Validation & Results
s = add_slide()
kicker(s, "Model selection and validation")
title(s, "Validation & Results")
add_text(s, MARGIN, Inches(2.0), Inches(6.6), Inches(0.4), "Validation strategy", 17, color=TEXT, bold=True)
add_text(s, MARGIN, Inches(2.5), Inches(6.6), Inches(1.3),
          "Stratified 5-fold CV across 6 partitions, identical folds for every model. "
          "Differences below 0.0005 OOF log loss are noise; nested CV for anything fitted.",
          15, color=TEXT, line_spacing=1.2)
add_text(s, MARGIN, Inches(4.1), Inches(6.6), Inches(0.4), "Post-processing", 17, color=TEXT, bold=True)
add_text(s, MARGIN, Inches(4.6), Inches(6.6), Inches(1.0),
          "Plain weighted average of the three probabilities, 0.45 / 0.30 / 0.25. No "
          "calibration. Clipped to [1e-4, 1-1e-4].",
          15, color=TEXT, line_spacing=1.2)
picture_fit(s, os.path.join(CHARTS, "ensemble_diagram.png"), Inches(8.0), Inches(1.9),
             Inches(4.6), Inches(1.7))
add_text(s, Inches(8.0), Inches(3.75), Inches(2.2), Inches(0.25), "OUT-OF-FOLD", 12, color=SECOND, bold=True)
add_text(s, Inches(7.96), Inches(4.0), Inches(3.0), Inches(0.7), "0.42136", 34, color=TEXT, bold=True)
add_text(s, Inches(8.0), Inches(4.75), Inches(2.2), Inches(0.25), "HIDDEN TEST", 12, color=DARKBLUE, bold=True)
add_text(s, Inches(7.96), Inches(5.0), Inches(3.0), Inches(0.75), "0.40982", 40, color=DARKBLUE, bold=True)
footer(s, 6)
NOTES[6] = ("All three share identical cross-validation folds, so a 0.0005 gap is just noise. "
            "The blend reaches 0.42136 out-of-fold and 0.40982 on hidden test, a plain weighted "
            "average, no calibration, just a clip to the probability bounds.")

# ===========================================================================
# Slide 7 - Model Insights & Limitations
s = add_slide()
kicker(s, "What the model learned, and where it fails")
title(s, "Model Insights & Limitations")
picture_fit(s, os.path.join(CHARTS, "shap_summary_top6.png"), MARGIN, Inches(1.85),
             Inches(6.6), Inches(4.9))
add_text(s, MARGIN, Inches(1.85) + Inches(4.9) + Inches(0.05), Inches(6.6), Inches(0.35),
          "LightGBM component interpretation, 45% of ensemble", 12.5, color=SECOND, italic=True)
add_text(s, Inches(7.85), Inches(1.85), Inches(4.75), Inches(0.4), "Strongest indicators", 15,
          color=DARKBLUE, bold=True)
add_bullets(s, Inches(7.85), Inches(2.25), Inches(4.75), Inches(1.5), [
    "Recent, severe delinquency.",
    "Low available credit, low payment-to-bill coverage.",
    "High utilisation.",
], size=14.5, gap_pt=6)
add_text(s, Inches(7.85), Inches(3.75), Inches(4.75), Inches(0.9),
          "20 largest errors → all unexpected defaults, scored 2-4% risk, with clean "
          "payment histories. The driver isn't in this data.", 14, color=TEXT)
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.85), Inches(4.9), Inches(4.75), Inches(1.75))
box.fill.solid(); box.fill.fore_color.rgb = SURFACE
box.line.color.rgb = RGBColor(0xD2, 0xD2, 0xD7); box.line.width = Pt(1)
box.adjustments[0] = 0.05
box.shadow.inherit = False
add_text(s, Inches(8.1), Inches(5.05), Inches(4.3), Inches(0.4), "EDUCATION = \"other\"",
          14.5, color=DARKBLUE, bold=True)
add_text(s, Inches(8.1), Inches(5.45), Inches(4.3), Inches(1.1),
          "387 customers, 28 defaults, risk over-predicted ~61%. AUC 0.645 vs. ~0.79 "
          "elsewhere. Should not support automated adverse decisions.", 13, color=TEXT, line_spacing=1.15)
footer(s, 7)
NOTES[7] = ("The strongest predictors are all about recent, severe delinquency, low available "
            "credit, and weak payment coverage. This SHAP view explains the LightGBM component "
            "specifically, not the full ensemble. But the model has real blind spots. Our "
            "twenty worst-predicted customers all defaulted despite scoring just 2 to 4 percent "
            "risk, every one with a clean payment history, meaning the real driver isn't in "
            "this data at all. And the EDUCATION \"other\" subgroup, 387 customers, 28 "
            "defaults, gets over-predicted by about 61 percent, with an AUC of just 0.645 "
            "against roughly 0.79 elsewhere. That subgroup shouldn't drive automated decisions.")

# ===========================================================================
# Slide 8 - Real-World Application
s = add_slide()
kicker(s, "Decision support, not automation")
title(s, "Real-World Application")
add_bullets(s, MARGIN, Inches(1.85), Inches(11.9), Inches(1.5), [
    "Calibrated probabilities support manual review, early customer assistance, limit "
    "monitoring and portfolio prioritisation, not an automatic approve/decline rule.",
    "Thresholds should reflect business cost, regulation and review capacity, not the "
    "competition score alone.",
], size=16, gap_pt=8)
picture_fit(s, os.path.join(CHARTS, "decision_continuum.png"), MARGIN, Inches(3.15),
             SLIDE_W - 2 * MARGIN, Inches(1.0))
picture_fit(s, os.path.join(CHARTS, "fp_fn_panel.png"), MARGIN, Inches(4.35),
             SLIDE_W - 2 * MARGIN, Inches(1.7))
add_text(s, MARGIN, Inches(6.25), Inches(11.9), Inches(0.4),
          "Limitations: missing drivers of unexpected defaults · weak education subgroup · "
          "ensemble compute cost · TabPFN licence dependency.", 13.5, color=SECOND)
add_text(s, MARGIN, Inches(6.7), Inches(11.9), Inches(0.6),
          "Understanding the account mechanics and validating decisions carefully mattered "
          "more than adding model complexity.", 16, color=DARKBLUE, bold=True)
footer(s, 8)
NOTES[8] = ("These probabilities should support manual review, early customer assistance, "
            "limit monitoring, and portfolio prioritization, not function as an automatic "
            "approve or decline rule. A false positive means a reliable customer faces "
            "unnecessary friction; a false negative means a missed default and a missed chance "
            "to help early. Thresholds should come from business cost and review capacity, not "
            "the competition score. Some defaults aren't predictable from this data, one "
            "subgroup is unreliable, and TabPFN adds licensing and compute cost for a small "
            "gain. Understanding the account mechanics and validating carefully mattered more "
            "than adding model complexity.")

for n in range(1, 9):
    set_notes(prs.slides[n - 1], NOTES[n])

# ===========================================================================
# Appendix slides (for Q&A only) -----------------------------------------
def appendix_slide(label, heading, bullets, size=16.5, bullet_width=None):
    s = add_slide()
    add_text(s, MARGIN, Inches(0.5), Inches(6), Inches(0.4), f"APPENDIX · {label}",
              12.5, color=BLUE, bold=True)
    title(s, heading, size=28, top=Inches(0.85))
    width = bullet_width or (SLIDE_W - 2 * MARGIN)
    add_bullets(s, MARGIN, Inches(1.8), width, Inches(5.2), bullets, size=size, gap_pt=10)
    return s

appendix_slide("A1", "Model comparison and blend weights", [
    "Ten model families tested; three earned blend weight (LightGBM 0.45, GRU 0.30, TabPFN 0.25).",
    "Rejected at weight 0.00: survival/hazard (0.42260 OOF, worsened hidden test), sequence+spend "
    "channel (0.42426), 1D-CNN (0.42601, 0.9924 correlated with GRU), autoencoder (0.42685), "
    "MLP (0.42882), CatBoost, ExtraTrees, logistic regression, multi-task GRU.",
    "135 LightGBM configurations across 3 sweeps (40 coarse, 60 fine, 1 blend-scored); best "
    "challenger beat the incumbent by only 0.00004, well below the noise floor.",
    "Correlation is not the selection criterion: logistic regression was the most decorrelated "
    "model (0.948 vs. LightGBM) yet received weight 0.00. GRU earned 0.30 at a higher "
    "correlation of 0.983, what matters is whether disagreement carries signal.",
])

appendix_slide("A2", "Full experiment ledger (41 experiments)", [
    "10 model families, 7 candidate feature families, 135 LightGBM configurations, 8 "
    "ensembling/calibration schemes, grouped as successful, neutral or harmful in "
    "docs/experiment-ledger.html.",
    "Survival model: +0.00027 OOF but worse hidden test (0.41044 vs. 0.41038), removed.",
    "5 techniques from the AMEX Kaggle write-ups tested (DART, high min_data_in_leaf, "
    "feature_fraction_bynode, recency-window aggregates, within-customer ranks), none adopted.",
    "Next investigation is new information, not further tuning: e.g. whether EDUCATION "
    "\"other\" is a genuinely homogeneous group.",
])

appendix_slide("A3", "Ensemble weight search and calibration", [
    "Grid search over ~230 valid weight triples (0.05 steps) on OOF log loss; nothing is fitted.",
    "(1.00, 0.00, 0.00) 0.42211 → (0.65, 0.35, 0.00) 0.42155 → (0.55, 0.30, 0.15) 0.42141 "
    "→ (0.45, 0.30, 0.25) selected 0.42136 → (0.34, 0.33, 0.33) equal 0.42147.",
    "Broad plateau: every LightGBM weight 0.50-0.70 lands within 0.0001 of the optimum. "
    "Shrinking weights toward uniform made hidden test worse (0.41045 vs. 0.41032), the "
    "standard overfitting check the fitted weights passed.",
    "4 calibration schemes tested under nested CV, all worse than no calibration: Platt "
    "(−0.00005), per-segment (−0.00030), isotonic (−0.00330), base-rate pull "
    "(α=1.0, i.e. no adjustment selected).",
    "Post-processing: one clip to [1e-4, 1-1e-4] and nothing else. No value hand-edited.",
], size=15, bullet_width=Inches(6.6))
picture_fit(prs.slides[-1], os.path.join(CHARTS, "weight_search_plateau.png"),
             Inches(7.6), Inches(1.8), Inches(5.0), Inches(5.0))

appendix_slide("A4", "SHAP detail", [
    "Full SHAP summary (14 features) and per-customer waterfalls for the highest- and "
    "lowest-risk predictions available on request.",
    "SHAP explains the LightGBM component only (45% of the ensemble weight), not the GRU "
    "or TabPFN.",
], size=14.5)
picture_fit(prs.slides[-1], os.path.join(ANALYSIS, "shap_summary.png"), Inches(0.7), Inches(3.3), Inches(5.6), Inches(3.9))
picture_fit(prs.slides[-1], os.path.join(ANALYSIS, "shap_waterfall_highest_risk.png"), Inches(6.6), Inches(3.3), Inches(3.0), Inches(3.9))
picture_fit(prs.slides[-1], os.path.join(ANALYSIS, "shap_waterfall_lowest_risk.png"), Inches(9.8), Inches(3.3), Inches(3.0), Inches(3.9))

appendix_slide("A5", "Subgroup fairness audit", [
    "EDUCATION = \"other\": n=387, 28 defaults (7.2% observed rate).",
    "Risk over-predicted by approximately 61% relative to the observed rate.",
    "AUC 0.645 in this subgroup versus approximately 0.79 elsewhere, the only segment "
    "where the model underperforms its own base-rate guess.",
    "This is a heterogeneous, undocumented category and should route to manual review rather "
    "than support an automated adverse decision.",
])

appendix_slide("A6", "Reproducibility and disclosure", [
    "Saved OOF and test prediction vectors are committed under artifacts/; applying the "
    "0.45/0.30/0.25 weighted average and the clip reconstructs the submission to 9.7e-17, in "
    "seconds, without rerunning the ~3-hour TabPFN pass.",
    "Pretrained model: TabPFN (Prior Labs, tabpfn package v8.5.0), requires a licence and "
    "API token; performs in-context learning, fits zero parameters to this dataset.",
    "No external datasets used. AMEX winning-solution write-ups were consulted; 5 techniques "
    "tested, none adopted.",
    "Claude Code (Anthropic) used throughout for code, experiments and drafting these "
    "documents; all modelling decisions were taken on measured cross-validation results.",
    "No manual modification of predictions beyond the [1e-4, 1-1e-4] clip. No calibration applied.",
])

appendix_slide("A7", "Operational thresholds and OOF vs. hidden-test gaps", [
    "OOF and hidden-test improvements can diverge because OOF rows are only ever scored by "
    "models that didn't see them, while hidden-test rows are scored by the average of all "
    "folds, e.g. the sixth CV partition helped OOF by 0.00024 and did nothing on hidden test.",
    "Decision thresholds should be set from business cost, regulatory requirement and review "
    "capacity, not the competition log loss score alone.",
    "False positive: unnecessary review, reduced credit or an adverse decision for a reliable "
    "customer. False negative: missed default, underestimated loss, missed early support.",
])

prs.save(os.path.join(HERE, "finalist-presentation.pptx"))
print("Saved", os.path.join(HERE, "finalist-presentation.pptx"))
print("Slides:", len(prs.slides.__iter__.__self__._sldIdLst))
