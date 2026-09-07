# Finalist Presentation Slide Plan

## Presentation objective

Deliver a four-minute presentation followed by a three-minute judging-panel Q&A.
The presentation should explain the team's approach, findings and understanding
of the credit-card default problem. The main narrative is:

> Careful data understanding and disciplined validation produced more value than
> model complexity. The largest gain came from deriving missing account activity,
> while the final ensemble combined three complementary views of each customer.

Use seven short main slides. Keep detailed evidence in appendix slides for Q&A.

## Visual direction

Build the deck in Figma using any available Apple-design and presentation-design
skills. The result should feel carefully art-directed by a human rather than
generated from a generic presentation template.

### Theme and palette

- Use a light theme with generous white space.
- Use Apple-inspired restraint rather than copying a specific Apple presentation.
- Suggested palette:
  - Warm off-white canvas: `#F5F5F7`.
  - White content surface: `#FFFFFF`.
  - Near-black primary text: `#1D1D1F`.
  - Secondary grey text: `#6E6E73`.
  - Primary blue accent: `#0071E3`.
  - Darker blue for chart contrast: `#0057B8`.
  - Pale blue highlight: `#E8F2FF`.
- Use blue to direct attention, not to decorate every element.
- Keep charts to blue, near-black and neutral greys unless an additional colour
  carries a specific meaning such as a rejected or harmful result.

### Typography and layout

- Use SF Pro if it is available legally in the design environment; otherwise use
  Inter, Helvetica Neue or Arial as a close neutral fallback.
- Use large, sentence-style takeaway titles rather than generic headings.
- Keep titles at approximately 32-40 pt and body text at 20-24 pt or larger.
- Use a consistent grid, wide outer margins and disciplined left alignment.
- Limit each slide to one main visual idea and no more than three concise text groups.
- Use subtle thin rules and restrained rounded corners only where they clarify
  structure.
- Check every slide at presentation size, not only while zoomed in within Figma.

### Human-crafted presentation rules

- Build charts and diagrams from the project's actual data and results. Do not use
  decorative AI-generated imagery.
- Avoid stock photography, generic illustrations, humanoid robots, brains,
  circuit-board imagery and unrelated finance icons.
- Avoid mesh gradients, glowing objects, glassmorphism, excessive shadows,
  floating-card layouts, large collections of pills and decorative blobs.
- Avoid placing every sentence inside a separate rounded rectangle.
- Do not use icons where a direct label is clearer.
- Do not use repetitive title/subtitle/three-card layouts across every slide.
- Vary composition deliberately while keeping typography, spacing and colour
  consistent.
- Prefer specific annotations written for this project over generic labels such as
  "Key insight", "Innovation" or "Our journey".
- Use restrained transitions only if they materially help explain sequence or model
  assembly; otherwise use none.
- Perform a final manual pass for optical alignment, spacing, line breaks, chart
  labelling and awkwardly wrapped text.

## Main presentation - four minutes

### Slide 1 - Problem, approach and result · 20 seconds

Cover:

- Predict each customer's probability of default next month.
- Evaluation metric: binary log loss, so probability quality and calibration matter.
- Dataset: 24,000 labelled customers, 6,000 test customers, 23 raw columns and a
  22.12% default rate.
- Final result: **0.40982 hidden-test log loss**.
- Final submission: `submission_tabpfn6.csv`.
- Final model: **0.45 LightGBM + 0.30 GRU + 0.25 TabPFN**.

Visual:

- Display `0.40982` as the headline result.
- Include a compact pipeline:

  `Raw data -> cleaning -> 81 features -> three models -> weighted average -> prediction`

Main message: the result came from combining three complementary views of each
customer.

### Slide 2 - Data understanding and preparation · 30 seconds

Cover:

- The dataset had no missing values or malformed rows.
- Cleaning was deliberately limited to undocumented category codes:
  - `EDUCATION`: codes 0, 5 and 6 grouped into "other" - 290 rows.
  - `MARRIAGE`: code 0 grouped into "other" - 42 rows.
- Payment-status values `-1` and `-2` were retained because they represent
  paid-in-full and inactive-account behaviour rather than missingness.
- No outlier removal: large balances and payments may be genuine risk signals.
- No class resampling or weighting: changing the 22.12% base rate could damage
  probability calibration.
- Numerical scaling and one-hot encoding were applied only where required by the
  GRU; LightGBM used native tabular values and categories.
- The 23 raw columns became 81 features across repayment history, utilisation,
  payment coverage, account levels, spending behaviour and minimum-payment
  behaviour.

Visual:

- Show `23 raw columns -> minimal cleaning -> 81 model features`.
- Include only the two category corrections and the three most important feature
  families to avoid overcrowding the slide.

Main message: preprocessing preserved meaningful financial states instead of
applying generic cleaning steps.

### Slide 3 - The most important feature-engineering insight · 40 seconds

Cover:

- The dataset provides statement balances and payments but not monthly account
  activity directly.
- Derive a spending or net account-activity proxy:

  `new activity = current bill - previous bill + payment`

- Explain why it matters:
  - A balance can rise because spending increased.
  - It can also rise because repayments stopped.
  - These represent different risk behaviours.
- This feature family delivered the largest improvement:
  - **0.00046 improvement in OOF log loss**.
  - **0.00173 improvement on the hidden test set**.
- Six other feature families failed to exceed the 0.0005 noise threshold because
  they mostly recombined information already present.

Visual:

- Draw a monthly account-flow diagram containing previous balance, payment, new
  account activity and current balance.
- Add a small bar chart comparing the spending-feature improvement with the six
  unsuccessful feature families.

Important wording: describe this as a "spending or net account-activity proxy"
because fees, interest and adjustments may also contribute.

Main message: new information mattered more than producing further transformations
of existing information.

### Slide 4 - How experimentation changed the approach · 45 seconds

Cover:

- **41 experiments**, including:
  - 10 model families.
  - 7 candidate feature families.
  - 135 LightGBM configurations.
  - Calibration, ensembling and segmentation methods.
- Show the important decision sequence:
  - Changed selection from AUC to log loss.
  - Added the GRU to preserve month-to-month ordering.
  - Introduced spending decomposition - the largest gain.
  - Added TabPFN as the final complementary model.
- Include meaningful unsuccessful experiments:
  - 135 LightGBM configurations produced no reliable improvement.
  - The 1D-CNN was too similar to the GRU.
  - Logistic regression was less correlated but still received zero blend weight.
  - The survival model improved OOF slightly but worsened the hidden result, so it
    was removed.
  - Isotonic calibration initially appeared beneficial, but nested CV showed this
    was leakage or overfitting; it became the worst calibration result.
- State the lesson: disagreement between models only helps when it contains useful
  signal.

Visual:

- Create a stepped or waterfall chart of the key hidden-test submissions:
  - 0.41260 - LightGBM + GRU baseline.
  - 0.41038 - spending features + tuned GRU.
  - 0.41032 - LightGBM full-data refit.
  - 0.40995 - TabPFN added.
  - **0.40982 - final six-partition TabPFN ensemble**.
- Label the chart clearly with "lower is better".
- Add no more than three small rejected-experiment callouts. Do not display all 41
  experiments on the main slide.

Main message: experiments were used to change direction, not simply to produce a
large model list.

### Slide 5 - Final model and validation · 40 seconds

Cover:

- **LightGBM - weight 0.45**
  - Strongest individual model.
  - Reads 81 engineered features as an aggregate customer profile.
  - OOF log loss: 0.42211.
- **Bidirectional GRU - weight 0.30**
  - Reads six months x eight channels as a sequence.
  - Distinguishes recovery from recent deterioration.
  - OOF log loss: 0.42392.
- **TabPFN - weight 0.25**
  - Reads the same tabular features through a pretrained prior.
  - Adds a different source of predictive information.
  - OOF log loss: 0.42345.
- Validation:
  - Stratified five-fold CV across six fold partitions.
  - Identical folds for all three models.
  - Differences below 0.0005 OOF log loss treated as noise.
  - Nested CV used for fitted calibration and combination methods.
- Final ensemble:
  - OOF log loss: **0.42136**.
  - Hidden-test log loss: **0.40982**.
  - Plain weighted average, no calibration and clipping to `[1e-4, 1-1e-4]`.

Visual:

```text
81 aggregate features -- LightGBM 45% -+
6-month sequence ------- GRU 30%       |- weighted probability - 0.40982
pretrained tabular prior - TabPFN 25% -+
```

Main message: the models were retained because they represented customers
differently, not because each was individually stronger than LightGBM.

### Slide 6 - What the model learned and where it fails · 45 seconds

Cover:

- Strongest indicators of default:
  - Recent delinquency.
  - Severity and frequency of late payments.
  - Low available credit.
  - Low payment-to-bill coverage.
  - High utilisation.
- Highest risk occurs when late payments are both recent and severe.
- Strong repayments and greater available credit generally reduce predicted risk.
- Error analysis:
  - The 20 worst predictions were false negatives.
  - They defaulted despite being assigned only 2-4% risk.
  - All had clean payment histories, moderate utilisation and healthy limits.
  - This indicates that important drivers were absent from the supplied data.
- Subgroup finding:
  - `EDUCATION="other"` contained 387 customers and 28 defaults.
  - Risk was over-predicted by approximately 61%.
  - AUC was 0.645 versus approximately 0.79 elsewhere.
  - This heterogeneous category should not support automated adverse decisions.

Visual:

- Use `output/analysis/shap_summary.png`, cropped to the six strongest features.
- Label it **"LightGBM component interpretation - 45% of ensemble"**. Do not
  imply that SHAP explains the GRU or TabPFN components.
- Add a small callout: `20 largest errors -> all unexpected defaults with clean histories`.

Main message: observed repayment behaviour is highly informative, but it cannot
predict defaults caused by circumstances absent from the data.

### Slide 7 - Real-world use, limitations and conclusion · 20 seconds

Cover:

- Use calibrated probabilities to support:
  - Manual credit review.
  - Early customer assistance.
  - Limit monitoring.
  - Portfolio risk prioritisation.
- Do not treat the score as an automatic approval or rejection rule.
- False-positive consequence: a reliable customer may face unnecessary review,
  reduced credit or an adverse decision.
- False-negative consequence: the lender may miss a likely default, underestimate
  losses and miss an opportunity for early support.
- Decision thresholds should be selected using business costs, regulatory
  requirements and review capacity, rather than the competition score alone.
- Key limitations:
  - Important causes of default are missing from the dataset.
  - One education subgroup performs poorly.
  - LightGBM provides most of the predictive value; the other components add
    considerable computation for a relatively small gain.
  - TabPFN requires external pretrained weights, a licence and an API token.
- Close with: **Understanding the account mechanics and validating decisions
  carefully mattered more than adding model complexity.**

Visual:

- Show a decision-support continuum:

  `Low risk -> standard process | uncertain -> manual review | high risk -> verification/support`

- Include a small false-positive versus false-negative consequence panel.

## Q&A appendix slides

Prepare these slides but do not present them unless asked:

- Full model hyperparameters, component scores, seeds and fit counts.
- All 41 experiments grouped into successful, neutral and harmful changes.
- Weight-search surface showing the broad optimum around LightGBM weights of
  0.50-0.70.
- Nested-CV calibration comparison and explanation of the original isotonic error.
- Detailed SHAP beeswarm and high-/low-risk customer waterfall examples.
- Subgroup fairness table, especially `EDUCATION="other"`.
- Reproducibility pipeline and exact artifact-to-submission reconstruction.
- Required disclosure:
  - TabPFN pretrained model and licence requirement.
  - No external datasets.
  - AMEX winning-solution write-ups consulted; none of the tested techniques retained.
  - AI coding-agent use.
  - No manual prediction modification beyond clipping.
- Operational-threshold discussion covering false-positive and false-negative costs.
- Explanation for not resampling the classes.
- Explanation for retaining GRU and TabPFN despite LightGBM providing most of the score.
- Why hidden-test and OOF improvements sometimes differed.

## Visuals to produce or reuse

### Produce

- End-to-end pipeline diagram.
- Raw-to-engineered data preparation diagram.
- Monthly account-flow/spending-proxy diagram.
- Hidden-test experiment progression chart with "lower is better" shown explicitly.
- Three-branch final ensemble diagram.
- False-positive versus false-negative consequence panel.
- Decision-support continuum.
- Optional appendix weight-search surface.

### Reuse

- `output/analysis/shap_summary.png` - crop to the strongest features and identify
  it as a LightGBM-component explanation.
- `output/analysis/shap_waterfall_highest_risk.png` - appendix only.
- `output/analysis/shap_waterfall_lowest_risk.png` - appendix only.

### Regenerate before use

- Do not use `output/analysis/reliability_diagram.png` as-is. Its legend shows
  `lgbm` and `nn`, so it does not clearly document the final
  LightGBM-GRU-TabPFN ensemble. Regenerate and verify it before including it.

## Likely judging-panel questions

Prepare concise answers to:

- Why optimise log loss rather than AUC?
- Why was class resampling or weighting not used?
- Why retain GRU and TabPFN when LightGBM provides most of the predictive value?
- How were the ensemble weights selected, and could they be overfitted?
- Why were identical cross-validation folds used for all models?
- Why did OOF and hidden-test improvements differ for some changes?
- What went wrong with the initial isotonic-calibration experiment?
- Why did the survival model get removed after improving OOF?
- What information does the GRU retain that aggregate features lose?
- What does TabPFN contribute, and how is its pretrained status disclosed?
- How can the final submission be reproduced without retraining TabPFN?
- What caused the largest false-negative errors?
- How should the `EDUCATION="other"` subgroup issue affect real-world use?
- What are the relative consequences of false positives and false negatives?
- How would an operational decision threshold be selected?
- What would be investigated with additional time or data?

## Copy-ready prompt for another agent

```text
Create a polished finalist presentation for the Credit Card Default stream using
the repository at:

/Users/danielpark/Documents/Hackathons/inter-uni-datathon

Read these sources before creating anything:
- Expected Submission Materials.pdf
- REPORT.md
- Methodology.md
- docs/methodology-report.html
- docs/experiment-ledger.html
- output/analysis/fairness_audit.txt

Goal:
Produce a 16:9 presentation that can be delivered in exactly four minutes,
followed by a three-minute Q&A. The tone should be professional and evidence-led,
like a concise technical report, not personal, promotional or overly academic.

Design and Figma requirements:
- Build the editable deck in Figma.
- Before designing, explicitly invoke and follow any available Apple-design,
  Figma-design and presentation-design skills.
- Use a light, Apple-inspired visual system with generous white space, disciplined
  typography and precise alignment. Take inspiration from Apple's restraint and
  hierarchy without copying a specific Apple deck.
- Suggested colours: #F5F5F7 canvas, #FFFFFF surfaces, #1D1D1F primary text,
  #6E6E73 secondary text, #0071E3 primary blue, #0057B8 dark blue and #E8F2FF
  pale-blue highlights.
- Use SF Pro if legally available in the design environment; otherwise use Inter,
  Helvetica Neue or Arial.
- Use large sentence-style takeaway titles, 32-40 pt titles, 20-24 pt minimum body
  copy, wide margins and a consistent grid.
- Use one main visual idea and no more than three concise text groups per slide.
- Make charts primarily blue and neutral grey, with any additional colour carrying
  a specific semantic meaning.
- The presentation must look manually designed for this project, not generated from
  a generic AI slide template.
- Do not use stock photos, decorative AI imagery, robots, brains, circuit boards,
  generic finance icons, mesh gradients, glowing elements, glassmorphism, excessive
  shadows, decorative blobs, walls of rounded cards or unnecessary pills.
- Do not put every sentence in a separate box and do not repeat the same
  title-plus-three-cards composition on every slide.
- Prefer direct labels and project-specific annotations over generic headings such
  as "Key insight", "Innovation" and "Our journey".
- Manually review optical alignment, spacing, line breaks, chart labels and text
  wrapping at presentation size before exporting.

Core narrative:
Careful data understanding and disciplined validation produced more value than
model complexity. The largest gain came from deriving a missing account-activity
quantity, while the final ensemble combined three genuinely different views of
each customer.

Create seven main slides:

1. Problem and headline result - 20 seconds
   - Predict next-month default probability using binary log loss.
   - 24,000 training rows, 6,000 test rows, 23 raw columns, 22.121% default rate.
   - Final hidden-test log loss: 0.40982.
   - Final submission: submissions/submission_tabpfn6.csv.
   - Final ensemble: 0.45 LightGBM + 0.30 GRU + 0.25 TabPFN.
   - Include a compact end-to-end pipeline diagram.

2. Data understanding and preparation - 30 seconds
   - No missing or malformed rows.
   - EDUCATION codes 0/5/6 grouped into "other"; MARRIAGE code 0 grouped into
     "other".
   - Preserve PAY status values -1 and -2 because they are meaningful states.
   - No outlier removal and no resampling/class weighting; explain why.
   - Show 23 raw columns becoming 81 features across six families.

3. Key feature-engineering insight - 40 seconds
   - Explain the derived account-activity/spending proxy:
     current bill - previous bill + payment.
   - Explain why it separates increased spending from failure to repay.
   - Report the improvement: 0.00046 OOF and 0.00173 hidden-test log loss.
   - State that six other feature families failed to exceed the 0.0005 noise
     threshold.
   - Build a simple financial-flow diagram and comparison chart.
   - Call this a proxy because fees, interest and adjustments may be included.

4. Experimentation and decision-making - 45 seconds
   - 41 experiments, 10 model families, 7 candidate feature families and
     135 LightGBM configurations.
   - Show the important progression:
       0.41260 LightGBM + GRU baseline
       0.41038 spending features + tuned GRU
       0.41032 LightGBM full-data refit
       0.40995 TabPFN added
       0.40982 final six-partition TabPFN ensemble
   - Lower log loss is better.
   - Include selected failures and lessons:
       135 LightGBM settings gave no reliable gain;
       CNN duplicated the GRU signal;
       survival modelling improved OOF but worsened hidden-test performance;
       logistic regression was less correlated but still received zero weight;
       isotonic calibration initially looked useful but failed under nested CV.
   - Do not place all 41 experiments on the main slide.

5. Final model and validation - 40 seconds
   - LightGBM: weight 0.45, 81 aggregate features, OOF 0.42211.
   - Bidirectional GRU: weight 0.30, six months x eight channels, OOF 0.42392.
   - TabPFN: weight 0.25, pretrained tabular prior, OOF 0.42345.
   - Explain why the representations are complementary.
   - Validation: stratified five-fold CV over six partitions, identical folds,
     0.0005 noise threshold and nested CV for fitted post-processing.
   - Ensemble OOF 0.42136; hidden test 0.40982.
   - Plain weighted average, no calibration, clip to [1e-4, 1-1e-4].
   - Use a three-branch ensemble architecture diagram.

6. Insights, interpretation and mistakes - 45 seconds
   - Strongest indicators: recent/severe delinquency, low available credit, low
     payment coverage and high utilisation.
   - Use output/analysis/shap_summary.png, cropped to the most important features.
   - Explicitly label SHAP as an explanation of the LightGBM component, not the
     complete ensemble.
   - Show that the 20 largest errors were defaults predicted at only 2-4% risk,
     all with clean repayment histories.
   - Include the EDUCATION="other" finding: n=387, 28 defaults, 61% relative
     over-prediction and AUC 0.645 versus about 0.79 elsewhere.

7. Real-world use, limitations and conclusion - 20 seconds
   - Position predictions as decision support for manual review, early customer
     assistance, limit monitoring and portfolio prioritisation.
   - Explain false-positive harm: unnecessary restriction or adverse treatment
     of a reliable customer.
   - Explain false-negative harm: missed default loss and missed early support.
   - State that operating thresholds must reflect costs, regulation and review
     capacity.
   - Limitations: missing drivers of unexpected defaults, weak education subgroup,
     ensemble compute cost and TabPFN licensing/reproduction dependency.
   - Close with: understanding account mechanics and validating decisions carefully
     mattered more than adding complexity.

Add speaker notes that total approximately four minutes. Each slide must have one
clear takeaway title, no more than three concise text groups, large readable type
and minimal visual clutter.

Create appendix slides for Q&A:
- Full hyperparameters and fit counts.
- Complete experiment summary.
- Weight-search and calibration evidence.
- SHAP waterfalls.
- Fairness/subgroup audit.
- Reproduction pipeline and required disclosures.
- False-positive/false-negative threshold considerations.

Existing assets:
- output/analysis/shap_summary.png
- output/analysis/shap_waterfall_highest_risk.png
- output/analysis/shap_waterfall_lowest_risk.png
- output/analysis/reliability_diagram.png

Do not use reliability_diagram.png as-is: its legend shows lgbm and nn rather
than the final three-model ensemble. Regenerate and verify it before using it.
Do not imply that SHAP explains the GRU or TabPFN components.

Do not invent metrics, causal claims or business outcomes. Do not describe
TabPFN as fitted or fine-tuned on this dataset; it performs in-context prediction
using pretrained weights. Keep "lower is better" visible wherever log loss is
charted.

Deliver:
- an editable Figma deck with reusable text, colour, spacing and chart styles
- presentation/finalist-presentation.pptx
- presentation/finalist-presentation.pdf
- presentation/speaker-notes.md
- editable source and any scripts used to create charts
- a short verification note confirming timings, values and source files

The repository currently has unrelated uncommitted work. Preserve it and do not
reset, overwrite or commit files outside the presentation directory.
```

## Source material

- [`Expected Submission Materials.pdf`](../Expected%20Submission%20Materials.pdf)
- [`REPORT.md`](../REPORT.md)
- [`Methodology.md`](../Methodology.md)
- [`docs/methodology-report.html`](../docs/methodology-report.html)
- [`docs/experiment-ledger.html`](../docs/experiment-ledger.html)
- [`output/analysis/fairness_audit.txt`](../output/analysis/fairness_audit.txt)
