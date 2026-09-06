# Literature Survey: Credit Default Prediction — Modeling, Calibration & Responsible Use

**Date:** 2026-09-05 | **Papers Found:** 165 unique (25 curated below) | **Date Range:** last 1 year for the 5 topic queries (2025-09 to 2026-09), broadened per-query for 3 seminal-paper searches (2014+, 1998+, 2007+) | **Search Queries:** 8

> Scope note: this is the *full* academic-research-plugin survey (not the fast pass), run with the user's "broader credit-risk domain" scope — it includes credit-scoring-specific literature (feature engineering conventions, fairness/responsible-use of demographic attributes) alongside generic ML-technique papers, per the competition brief's explicit callout on calibration, interpretability, and responsible use of demographic information.
>
> Tooling note: `bibtex_utils.py`'s `fetch` command is broken in the installed plugin version — it imports `pybtex.latexenc`, a module that does not exist in any available `pybtex` release (0.21–0.26.1 all checked). `references.bib` below was therefore hand-built from the metadata `paper_search.py` already returned (title/authors/year/venue/DOI), not from the fetch script.

---

## Paper Summary Table

| # | Title | Authors | Year | Venue | Citations | Notes |
|---|-------|---------|------|-------|-----------|-------|
| 1 | [XGBoost: A Scalable Tree Boosting System](http://arxiv.org/abs/1603.02754v3) | Chen & Guestrin | 2016 | KDD | — | Seminal — the GBDT lineage `model.py`'s LightGBM belongs to |
| 2 | [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | Ke, Meng, Finley, Wang, Chen, Ma, Ye, Liu | 2017 | NeurIPS | — | Seminal — the exact library `model.py` runs |
| 3 | [Predicting Good Probabilities With Supervised Learning](https://www.semanticscholar.org/paper/eae3948747c9d051314c1f5851957b833aa83eca) | Niculescu-Mizil & Caruana | 2005 | ICML | 2231 | Seminal — the calibration methods (Platt/isotonic) the parked plan applies |
| 4 | [Probabilistic Outputs for Support Vector Machines](https://home.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf) | J. Platt | 1999 | — | — | Seminal — origin of Platt scaling |
| 5 | [Deep Learning vs. Gradient Boosting: Benchmarking state-of-the-art ML for credit scoring](http://arxiv.org/abs/2205.10535v1) | M. Schmitt | 2022 | arXiv | — | GBDT vs. DL head-to-head on credit scoring |
| 6 | [Gradient Boosting Survival Tree with Applications in Credit Scoring](http://arxiv.org/abs/1908.03385v5) | Bai, Zheng, Shen | 2019 | arXiv | — | Survival-analysis reframing of default prediction |
| 7 | [Toward interpretable credit scoring: integrating explainable AI with deep learning for credit card default prediction](https://www.semanticscholar.org/paper/e3b6199a29c7defb3e53a965aebcbf521a95fa4f) | Talaat, Aljadani, Badawy et al. | 2023 | Neural Computing & Applications | 102 | Highest-cited applied paper found |
| 8 | [Credit Risk Prediction Using Machine Learning and Deep Learning: A Study on Credit Card Customers](https://www.semanticscholar.org/paper/a94d34100f3bbc193f8b930d1cc2f17529ee6ab4) | Chang, Sivakulasingam, Wang et al. | 2024 | Risks | 84 | Recent benchmark, same problem framing |
| 9 | [Credit Card Default Prediction using Machine Learning Techniques](https://www.semanticscholar.org/paper/549d405db370095212b7a3d8d9d2b832ac5ef3d2) | Sayjadah, Hashem, Alotaibi et al. | 2018 | ICACCA | 38 | Widely-cited applied baseline |
| 10 | [The Application of Machine Learning Algorithms in Credit Card Default Prediction](https://www.semanticscholar.org/paper/4e80f5e510b016c2a492e4c6324ff6ef51aa5f69) | Yu | 2020 | CDS | 20 | Applied baseline |
| 11 | [Research on Credit Card Default Prediction for Class-Imbalanced Datasets Based on ML](https://www.semanticscholar.org/paper/95f12d805a37a9e0df7b5ee4721e9d386cf699ed) | Liu | 2024 | ICDSE | 1 | Imbalance-focused, same problem |
| 12 | [Calibration of Machine Learning Classifiers for Probability of Default Modelling](http://arxiv.org/abs/1710.08901v1) | Fonseca & Lopes | 2017 | arXiv | — | Calibration specifically for PD models |
| 13 | [Probability Calibration Trees](http://arxiv.org/abs/1808.00111v2) | Leathart, Frank, Holmes et al. | 2018 | arXiv | — | Tree-local calibration, alternative to global Platt/isotonic |
| 14 | [Evaluating outlier probabilities: assessing sharpness, refinement, and calibration](https://www.semanticscholar.org/paper/7a506c97a260a5d408af9cf6f006636d29bbfae4) | Röchner, Marques, Campello et al. | 2024 | Data Mining & Knowledge Discovery | 12 | Calibration diagnostics on imbalanced data |
| 15 | [Revisiting Deep Learning Models for Tabular Data](http://arxiv.org/abs/2106.11959v5) | Gorishniy, Rubachev, Khrulkov et al. | 2021 | NeurIPS | — | FT-Transformer / ResNet tabular baselines |
| 16 | [Revisiting Pretraining Objectives for Tabular Deep Learning](http://arxiv.org/abs/2207.03208v2) | Rubachev, Alekberov, Gorishniy et al. | 2022 | arXiv | — | Pretraining rarely helps below ~50k rows |
| 17 | [Benchmarking Optimizers for MLPs in Tabular Deep Learning](http://arxiv.org/abs/2604.15297v2) | Gorishniy, Rubachev, Feoktistov et al. | 2026 | arXiv | — | Most recent entry in this line |
| 18 | [Transfer Learning with Deep Tabular Models](http://arxiv.org/abs/2206.15306v2) | Levin, Cherepanova, Schwarzschild et al. | 2022 | arXiv | — | Cross-dataset transfer for tabular nets |
| 19 | [Classification of Imbalanced Credit Scoring Data Sets Based on Ensemble Method with Weighted-Hybrid-Sampling](http://arxiv.org/abs/2102.04721v1) | Liu, Zhang, Wang | 2021 | arXiv | — | Sampling + ensembling for class imbalance |
| 20 | [Bagging Supervised Autoencoder Classifier for Credit Scoring](http://arxiv.org/abs/2108.07800v1) | Abdoli, Akbari, Shahrabi | 2021 | arXiv | — | Autoencoder + bagging alternative to GBDT/MLP |
| 21 | [Explainable AI for Interpretable Credit Scoring](http://arxiv.org/abs/2012.03749v1) | Demajo, Vella, Dingli | 2020 | arXiv | — | SHAP/LIME on credit-scoring models |
| 22 | [Deep Generative Models for Reject Inference in Credit Scoring](http://arxiv.org/abs/1904.11376v2) | Mancisidor, Kampffmeyer, Aas et al. | 2019 | arXiv | — | Selection-bias correction (VAEs) |
| 23 | [Demographic-Reliant Algorithmic Fairness: Characterizing the Risks of Demographic Data Collection](http://arxiv.org/abs/2205.01038v2) | Andrus & Villeneuve | 2022 | arXiv | — | Direct to the brief's "responsible use of demographic information" |
| 24 | [Evaluating AI Fairness in Credit Scoring with the BRIO Tool](http://arxiv.org/abs/2406.03292v1) | Coraglia, Genco, Piantadosi et al. | 2024 | arXiv | — | Ready-made fairness-audit methodology, credit-scoring-specific |
| 25 | [Calibrated Credit Intelligence: Shift-Robust and Fair Risk Scoring with Bayesian Uncertainty and Gradient Boosting](http://arxiv.org/abs/2603.06733v1) | Nayak | 2026 | arXiv | — | Only paper found combining calibration *and* fairness on GBDT credit models |

*(165 unique papers were retrieved and deduped across all 8 searches; the table above is the representative subset actually discussed below. Full dump in `papers_raw.json`.)*

---

## Theme Clusters

### Theme 1: Gradient-boosted trees as the credit-risk default

**Summary**: XGBoost and LightGBM established GBDT as the standard for structured/tabular prediction; Schmitt (2022) and Bai et al. (2019) show GBDT continues to dominate credit-scoring benchmarks, with the survival-tree variant reframing "will they default" as "when will they default."

**Key Papers**: Chen & Guestrin (2016) · Ke et al. (2017) · Schmitt (2022) · Bai et al. (2019)

**Contribution**: Confirms `model.py`'s choice of LightGBM as the primary model is the literature-supported default for this data shape and size, not just a convenient one.

### Theme 2: Applied ML/DL benchmarks on credit-card default prediction

**Summary**: A large, mostly incremental applied literature (Talaat et al. 2023, Chang et al. 2024, Sayjadah et al. 2018, Yu 2020, Liu 2024) runs standard classifiers (logistic regression, random forest, XGBoost, small NNs) on credit-card default data — frequently the *same* UCI "Default of Credit Card Clients" dataset this competition's schema (`PAY_*`, `BILL_AMT*`, `PAY_AMT*`) is drawn from. Most report AUC/accuracy; few report log loss or calibration.

**Key Papers**: Talaat et al. (2023) · Chang et al. (2024) · Sayjadah et al. (2018) · Yu (2020) · Liu (2024)

**Contribution**: Validates the feature groupings `common.py`'s `features()` already engineers (lateness streaks, utilization, payment coverage) — this is the standard feature vocabulary for this exact data shape, not a bespoke invention.

### Theme 3: Probability calibration

**Summary**: Niculescu-Mizil & Caruana (2005) and Platt (1999) are the foundational references for Platt scaling and isotonic regression; Fonseca & Lopes (2017) apply calibration specifically to probability-of-default models, Leathart et al. (2018) propose a tree-local alternative, and Röchner et al. (2024) give diagnostics for calibration quality under class imbalance — directly relevant since this dataset's default rate is a minority class.

**Key Papers**: Niculescu-Mizil & Caruana (2005) · Platt (1999) · Fonseca & Lopes (2017) · Leathart et al. (2018) · Röchner et al. (2024)

**Contribution**: This is the literature base the already-approved (parked) plan draws on — swapping AUC-based model selection for log-loss-based selection plus a Platt/isotonic calibration check on OOF predictions is exactly the Niculescu-Mizil & Caruana playbook.

### Theme 4: Tabular deep learning vs. GBDT

**Summary**: Gorishniy et al.'s line of work (2021, 2022, 2026) and Levin et al. (2022) are the standard references for whether transformer/MLP-style nets can beat GBDT on tabular data — the consistent finding is that they're competitive only with much larger data or pretraining, and rarely win outright on datasets in the tens-of-thousands-of-rows range.

**Key Papers**: Gorishniy et al. (2021, 2022, 2026) · Levin et al. (2022)

**Contribution**: This directly corroborates `nn_model.py`'s own docstring finding ("a wider or deeper net does not help... 24k rows is too few") — the literature says this is expected, not a tuning failure, which lowers the priority of investing more effort into the MLP.

### Theme 5: Class imbalance and ensembling for credit scoring

**Summary**: Liu, Zhang & Wang (2021) and Abdoli et al. (2021) address the credit-scoring class-imbalance problem via resampling/hybrid ensembles and bagged autoencoders respectively, as alternatives or complements to the LightGBM+MLP blend already in `blend.py`.

**Key Papers**: Liu, Zhang & Wang (2021) · Abdoli et al. (2021)

**Contribution**: Suggests SMOTE/hybrid-sampling variants as *lower*-priority alternatives to the current class-weighting-free approach — but see Gap 5 below on why resampling is generally not preferred for log-loss-scored problems.

### Theme 6: Interpretability, reject inference, and responsible use of demographic data

**Summary**: Demajo et al. (2020) apply SHAP/LIME to credit-scoring models; Mancisidor et al. (2019) address reject inference (the selection bias from only observing outcomes for approved applicants); Andrus & Villeneuve (2022), Coraglia et al. (2024), and Nayak (2026) address fairness and responsible use of demographic attributes in scoring/classification systems, with Nayak (2026) being the only paper found that jointly targets calibration *and* fairness on gradient-boosted credit models.

**Key Papers**: Demajo et al. (2020) · Mancisidor et al. (2019) · Andrus & Villeneuve (2022) · Coraglia et al. (2024) · Nayak (2026)

**Contribution**: Directly answers the competition brief's explicit callout on "responsible use of demographic information" — gives a methodology (BRIO-style subgroup auditing) rather than leaving it as an unaddressed footnote.

---

## Research Gaps

1. **Calibration under blending is unstudied**: every calibration paper found (Niculescu-Mizil & Caruana, Platt, Fonseca & Lopes, Leathart et al.) calibrates a *single* classifier. None address whether averaging two already-calibrated models (as `blend.py` does with LightGBM + MLP) preserves calibration, or whether the blend itself needs re-calibrating after mixing — which is exactly what the parked plan's post-blend calibration step tests empirically rather than assuming.

2. **Calibration-fairness joint evaluation is nearly absent**: only Nayak (2026) — a very recent, single, non-peer-reviewed-looking preprint — evaluates calibration and demographic fairness together on a GBDT credit model. This is a genuine open gap, not just an under-explored one.

3. **Reject inference doesn't obviously apply here but is worth naming**: Mancisidor et al. (2019) address the standard credit-scoring problem that training labels only exist for approved applicants (a form of selection bias). This competition's 24k/6k split looks like a stratified sample of an already-observed population rather than an application-approval pipeline, so this gap is likely *not* actionable for this dataset — flagged for completeness, not as an action item.

4. **Temporal modeling of the 6-month repayment history is underused**: nearly every applied paper found (Theme 2) treats `PAY_*`/`BILL_AMT*`/`PAY_AMT*` as a flat feature vector, same as `common.py`'s `features()` does. Only Bai et al.'s survival-tree framing treats time explicitly. A recurrent/1D-convolutional architecture over the 6 monthly snapshots is essentially untried in this literature.

5. **Isotonic regression's overfitting risk at n≈24k is not directly addressed**: the classic calibration papers benchmark on larger datasets; Röchner et al. (2024) discuss calibration diagnostics but not sample-size-driven overfitting of isotonic curves specifically. This matters for the parked plan's calibration step — motivates comparing isotonic against the more conservative Platt (sigmoid) fit rather than trusting isotonic by default.

---

## Cross-Domain Findings

- **Survival analysis (medical/reliability engineering) → credit scoring**: Bai et al. (2019) transplant gradient-boosted survival trees (originally a medical/reliability-engineering tool for time-to-event data) into credit scoring, modeling *time to default* rather than binary default. `common.py`'s `features()` already computes `months_since_late` and `max_late_streak` — quantities that are natural survival-analysis covariates — without the codebase framing them that way; this is a transfer that's already half-adopted here informally.

- **Meteorological probability forecasting → log-loss calibration practice**: proper scoring rules and reliability-diagram calibration checks are decades-old standard practice in weather forecasting (predicting "70% chance of rain" and checking it rains ~70% of the time on those days). Niculescu-Mizil & Caruana's ML calibration framework is a direct descendant of that idea, and it's exactly the check the parked plan adds to `blend.py` — same logic, just applied to "70% chance of default" instead of rain.

- **Vision/NLP bias-auditing tooling → credit-scoring fairness**: the BRIO tool (Coraglia et al. 2024) is explicitly built on bias-auditing methodology that originated in vision/NLP fairness research and is retrofitted onto credit scoring. This gives a concrete, reusable playbook (per-subgroup calibration/error-rate comparison) for auditing `SEX`/`EDUCATION`/`MARRIAGE` in this exact dataset, rather than inventing an audit methodology from scratch.

---

## Innovation Proposals

Ranked by plausible log-loss impact vs. implementation cost, given ~1 day remaining and the current stack (`common.py` feature pipeline, LightGBM + MLP blend in `model.py`/`nn_model.py`/`blend.py`, 5-fold CV on ~24k rows).

### Proposal 1: Calibration-first model selection and blending (already scoped — this *is* the parked plan)

**Description**: Switch every model-selection decision (LightGBM early-stopping metric, blend weight) from AUC to log loss, then fit and compare Platt/isotonic calibration on the pooled OOF blend, keeping whichever (including "none") scores lowest OOF log loss.

**Feasibility**:
- Data: none needed beyond what's already loaded.
- Compute: negligible — reuses existing OOF vectors, no retraining beyond the existing pipeline.
- Novelty: not novel research, but directly literature-grounded (Niculescu-Mizil & Caruana 2005; Fonseca & Lopes 2017) and precisely targeted at this competition's actual scoring metric.
- Timeline: already planned as a same-day, ~3-file diff (see the parked plan at `/Users/danielpark/.claude/plans/financial-institutions-need-to-optimized-lollipop.md`'s prior version, or re-derive: `model.py` eval_metric, `blend.py` weight + calibration + clipping).

**Potential Weaknesses**: Gap 1 and Gap 5 above apply directly — calibrating *after* blending is empirically justified but not literature-proven optimal, and isotonic risks overfitting at this sample size, which is why the plan compares it against Platt and against no calibration rather than assuming isotonic wins.

**Landing Plan**: (1) `eval_metric="binary_logloss"` in `model.py`; (2) log-loss-optimal blend weight + Platt/isotonic/raw comparison in `blend.py`; (3) clip to `[1e-4, 1-1e-4]`. Success metric: OOF log loss strictly ≤ current AUC-tuned blend's log loss. Fallback: if calibration doesn't help (all three options within noise), ship the raw log-loss-tuned blend — the code already treats "none" as a valid winner.

### Proposal 2: Subgroup calibration/fairness audit on SEX, EDUCATION, MARRIAGE (new, additive, zero risk to the score)

**Description**: A standalone analysis script (BRIO-style, per Coraglia et al. 2024) that reports OOF log loss, AUC, and predicted-vs-actual default rate broken out by `SEX`, `EDUCATION`, and `MARRIAGE` subgroups, directly answering the brief's "responsible use of demographic information" callout.

**Feasibility**:
- Data: already present (`SEX`/`EDUCATION`/`MARRIAGE` in `datasets/train_clean.csv`) and the OOF predictions already saved by `model.py`/`nn_model.py` (`.output/oof_*.npy`).
- Compute: trivial — a `groupby` over existing arrays, no retraining.
- Novelty: not a new modeling technique, but a genuinely underused angle per Gap 2 (calibration-fairness joint reporting is almost absent in the literature found).
- Timeline: under an hour — one new small script, no changes to the modeling pipeline.

**Potential Weaknesses**: With only ~24k rows split across 5-6 categories per demographic column, some subgroup cells will be small enough that per-group log loss is noisy; report should note confidence caveats rather than over-interpreting small-cell differences. This is reporting/interpretability value, not a score improvement — it won't move the leaderboard number.

**Landing Plan**: (1) load `.output/oof_lgbm.npy`/`oof_nn.npy` (or the blend's OOF) alongside `train_clean.csv`'s demographic columns; (2) compute per-subgroup log loss/AUC/calibration; (3) print or plot a short table. Success metric: a readable subgroup breakdown exists to cite if the datathon's judging considers responsible-use writeups. Fallback: skip entirely if time runs out — it's additive and doesn't block the submission.

### Proposal 3: Recurrent/temporal encoding of the 6-month repayment history (deprioritized)

**Description**: Replace or augment the flat `PAY_*`/`BILL_AMT*`/`PAY_AMT*` feature engineering in `common.py` with a small sequence model (e.g. a tiny GRU or 1D-CNN over the 6 monthly snapshots) per Gap 4, feeding its output as an additional feature or a third blend member.

**Feasibility**:
- Data: sufficient rows exist (24k), but 6 timesteps is an extremely short sequence — much of what a sequence model would learn (trend, streaks, recency) is already hand-engineered in `common.py`'s `features()` (`trend_pay`, `max_late_streak`, `months_since_late`).
- Compute: modest (small net, CPU-trainable), but requires a new data-loading path parallel to `common.py`'s tabular one.
- Novelty: matches Gap 4, but Theme 4's own finding (tabular deep nets rarely beat GBDT below ~50k rows, and `nn_model.py`'s docstring already found bigger nets don't help here) suggests the marginal gain is likely small relative to the engineering cost.
- Timeline: multi-hour to build and validate properly — the highest-cost proposal here.

**Potential Weaknesses**: High risk of reinventing features `common.py` already computes by hand, for uncertain gain, with the added risk of introducing a fourth model to validate/blend/calibrate under time pressure.

**Landing Plan**: Not recommended to start given ~1 day remaining. If pursued later: (1) build a minimal GRU/1D-CNN prototype; (2) compare its solo OOF log loss against `nn_model.py`'s current 0.7830 AUC MLP baseline before deciding whether it's even worth blending. Fallback (and the actual recommendation for now): skip — spend the remaining time on Proposals 1 and 2 instead.

---

## References

See `references.bib` in this directory for complete BibTeX entries. Note the tooling caveat at the top of this report: entries were hand-built from `paper_search.py` metadata rather than `bibtex_utils.py`'s broken `fetch` command.

---

**Generated**: 2026-09-05 17:36:23 UTC | **Tool**: literature-survey skill v1.0 (via academic-research plugin; `bibtex_utils.py fetch` substituted with hand-built BibTeX due to a `pybtex.latexenc` import error in the installed plugin version)
