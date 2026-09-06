# Documentation

A three-page static site, served by GitHub Pages from this directory and
openable directly in any browser. Every page carries the same sidebar, so any
section of either document is one click away from anywhere.

| File | Covers |
|---|---|
| `index.html` | Landing page: the result as a statement of accounts, what the two documents contain, disclosure, reproduction |
| `methodology-report.html` | §1–§7. What the final model is, how each component makes a prediction, how the blend weights were chosen, how we arrived at it, its strengths and weaknesses, and what was left untried at the deadline |
| `experiment-ledger.html` | §1–§3. All 41 experiments with measured deltas against the 0.0005 noise threshold, and why each did or did not work |
| `assets/site.css` | Shared stylesheet. Light and dark, responsive, print-friendly |
| `assets/site.js` | Sidebar: marks the section in view, opens as a drawer under 1040px |
| `assets/figures/` | SHAP summary, two SHAP waterfalls, reliability diagram — copied from `../output/analysis/` |

No build step. Edit the HTML and commit.

Also published at:
- Methodology report — https://claude.ai/code/artifact/c9d283fa-29a2-4d1b-9b4e-98b82503d115
- Experiment ledger — https://claude.ai/code/artifact/071d051d-9a2c-4704-9f32-305877926cd6

## Final result

Hidden-test log loss **0.40982** from `submissions/submission_tabpfn6.csv`.
Ensemble: 0.45 LightGBM + 0.30 GRU + 0.25 TabPFN, plain weighted average of
probabilities, no calibration or post-processing beyond clipping to
[1e-4, 1-1e-4].

Reproduction instructions and saved prediction vectors are in `../artifacts/`.

## Disclosure

- **Pretrained model used:** TabPFN (Prior Labs), via the `tabpfn` package.
  Requires a licence and API token; performs in-context learning and fits no
  parameters to this dataset.
- **AI coding agent used:** Claude Code, throughout.
- **External solutions consulted:** the 1st, 2nd and 3rd place write-ups from
  the AMEX Default Prediction competition. Five techniques were tested from
  them (DART boosting, high `min_data_in_leaf`, `feature_fraction_bynode`,
  recency-window aggregates, within-customer ranks). **None were adopted** —
  all measured worse on cross-validation. See the ledger for the numbers.
- **External datasets used:** none. Only the competition-provided files.
- **Manual modification of predictions:** none beyond the clipping noted above.
