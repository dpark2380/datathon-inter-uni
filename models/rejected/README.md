# Tested and rejected

Nine scripts that are **not part of the final submission pipeline**. They are kept
because the negative results are part of the methodology: each was measured on the
same folds as the winning models, and each was rejected by the same weight search
that set the final weights — not by judgement.

None of these is imported by anything in the pipeline. Each adds the parent
directory to `sys.path` so it can still import `common`, `lgbm_model` and
`seq_model` when run from here:

```bash
uv run python models/rejected/survival_model.py
```

| Script | What it was | Result |
|---|---|---|
| `survival_model.py` | Discrete-time hazard reformulation of the target | OOF 0.42260 — best of the rejects. Briefly took a third of the blend on OOF (+0.00027), but the gain did not survive on the hidden test set (0.41044 against 0.41038) |
| `seq_spend_model.py` | GRU with spending supplied as an ordered 9th channel | OOF 0.42426 — +0.00002 against a matched control, inside the 0.0005 noise floor |
| `cnn_model.py` | 1D-CNN over the same 6×8 monthly panel | OOF 0.42601 — correlated 0.9924 with the GRU. Different architecture, identical signal; split the sequence weight rather than adding to it |
| `autoenc_model.py` | Supervised autoencoder | OOF 0.42685 — reconstruction loss was a straight cost against classification |
| `multitask_model.py` | Multi-task GRU with a next-month auxiliary head | Weight 0.00 |
| `attn_model.py` | Transformer encoder over the same panel | Weight 0.00; never completed a full run |
| `nn_model.py` | The original MLP over the 81 features | OOF 0.42882 — displaced entirely the moment the GRU existed |
| `pseudo_label.py` | Pseudo-labelling the test set | A recorded negative result |
| `rf_baseline.py` | The earliest RandomForest baseline (0.43615) | Superseded. Standalone: reads `datasets/train.csv` directly, uses its own feature function, and writes `rf_predictions.csv`. Retained only as the starting point of the progression |

Measured deltas for all of these, against the 0.0005 fold-noise threshold, are in
`docs/experiment-ledger.html`.

**The pattern.** Every model here that was merely a *different algorithm* over the
same view of a customer received weight 0.00. The three models that earned weight
read the data in genuinely different ways — as aggregates, as an ordered sequence,
and through a pretrained prior. New information wins; new algorithms do not.
