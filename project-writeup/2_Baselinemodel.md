# Baseline Model — Logistic Regression + LightGBM/HistGBM

Notebook: [02_baseline_model.ipynb](../notebooks/02_baseline_model.ipynb) (2 of 4 in the pipeline).

## Purpose

Before reaching for a graph neural network, build the standard tabular baseline a data scientist
would put together in an afternoon: light feature engineering, no graph, no deep dive into the
opaque Vesta `V`-columns (that's notebook 3's job). This is the number
[04_gcn_model.ipynb](../notebooks/04_gcn_model.ipynb) has to beat — and the notebook is explicit
that "GCN beats baseline" is only a meaningful claim if both models are evaluated on **exactly the
same rows**. That's why the train/valid split logic lives in `fraud_utils.time_based_split` and is
imported (not reimplemented) by both notebook 2 and notebook 3/4.

The notebook falls back from `lightgbm` to `sklearn.ensemble.HistGradientBoostingClassifier` when
`lightgbm` isn't installed — same histogram-based GBM family, just without early stopping against a
held-out eval set. In the run whose results are discussed below, that fallback path was the one
exercised (`HAS_LIGHTGBM = False`).

## 1–2. Data load and time-based split

Same merged `train`/`test` frames as notebook 1 (`load_raw_data`), then split via
`time_based_split`: sort by `TransactionDT`, hold out the last 20% as validation. This directly
operationalizes the EDA finding in [1_EDA_on_dataset.md](1_EDA_on_dataset.md) that the real Kaggle
test set sits entirely after train in time, and that the fraud rate is non-stationary over the
182-day window — a random split would leak near-future signal and overstate validation AUC.

The resulting split is not tiny: **472,432 train rows / 118,108 validation rows**, both close to
the dataset's overall ~3.5% fraud rate, so the split isn't skewing class balance on its own.

## 3. Baseline feature engineering (`engineer_baseline_features`)

Deliberately minimal — 50 features total, built from:

| Family | Treatment |
|---|---|
| `TransactionAmt` | kept raw + `log1p` transform |
| `TransactionDT` | derived `hour`, `day`, `day-of-week` (linear, **not** cyclically encoded here — see note below) |
| `card1`, `addr1`, `P_emaildomain` | frequency-encoded (count of each value in the fitted set) |
| `ProductCD`, `card4`, `card6`, `DeviceType`, `M1`–`M9` | label-encoded via a fitted category→int map |
| `C1`–`C14`, `D1`–`D15` | kept as raw numeric, missing values filled with `-999` sentinel |

Two things worth flagging against the EDA notebook's own recommendations:

- The EDA (§"Time of transaction") explicitly recommended **cyclical (sin/cos) encoding** for
  hour-of-day, since fraud rate swings ~4–5x between trough and peak hour. This notebook uses the
  raw linear `Transaction_hour` integer instead — a reasonable simplification for a first-pass
  baseline, but it means LightGBM/HistGBM (which can split on hour non-linearly) benefits from that
  feature more than LogisticRegression does, and neither gets the "wrap-around" continuity a
  cyclical encoding gives near midnight.
- Missing values are globally filled with `-999` rather than paired with the missingness-indicator
  flags the EDA recommended for the heavily-sparse `D`/`M`/`id_` families. For a tree model this is
  usually fine (a GBM can learn to split off the sentinel), but it costs LogisticRegression some
  signal, since "value is missing" collapses into "value is very negative" rather than being its
  own feature.
- No V-columns, PCA components, or `id_`-columns are included at all here — despite the EDA finding
  `id_17`/`id_01`/`id_22` as the single strongest raw correlations with `isFraud`, and PCA-on-V
  reaching 0.82 AUC on its own. That's intentional (notebook 3 owns that feature set for the GNN
  input), but it does mean this baseline is handicapped relative to what a "best tabular effort"
  could reach — worth keeping in mind when comparing its ~0.91 AUC against the GCN later, since a
  richer tabular baseline (with V/id features included) might close some of that gap on its own.

## 4–5. Models

**Model 1 — Logistic Regression**: `StandardScaler` → `LogisticRegression(class_weight="balanced")`.
A linear, calibrated-ish reference point — useful mainly as a floor, not a serious contender, given
the feature set includes essentially no interaction terms and several categoricals are only
label-encoded (which imposes an arbitrary ordinal relationship that a linear model will take at
face value).

**Model 2 — GBM** (LightGBM if available, else `HistGradientBoostingClassifier`): 2000 estimators
target with early stopping (LightGBM path) or a fixed 500 iterations (HistGBM fallback, which has
no early-stopping hook here), `scale_pos_weight`/`class_weight="balanced"` for the ~28:1 class
imbalance, shallow-ish trees (`num_leaves=63` / `max_leaf_nodes=63`) with subsampling. This is the
one the notebook calls "the number to watch."

## 6. Evaluation

ROC and precision-recall curves are plotted side by side for both models, plus a classification
report and confusion matrix for the GBM at the default 0.5 threshold. Given the class imbalance
(~3.5% fraud), PR-AUC is the more informative of the two headline numbers — ROC-AUC on this dataset
looks deceptively good even for weak models because true negatives dominate.

### Results (this run — HistGBM fallback, no LightGBM installed)

| Model | Valid AUC | Valid PR-AUC | n_train | n_valid | n_features |
|---|---|---|---|---|---|
| Logistic Regression | 0.805 | 0.213 | 472,432 | 118,108 | 50 |
| HistGradientBoosting | **0.912** | **0.488** | 472,432 | 118,108 | 50 |

(From [`results/baseline_logreg_metrics.json`](../results/baseline_logreg_metrics.json) and
[`results/baseline_lightgbm_metrics.json`](../results/baseline_lightgbm_metrics.json).)

The gap between the two models is large — GBM AUC is ~11 points higher, and PR-AUC is more than
**2x** the logistic regression's. That's the expected shape for this kind of tabular fraud data:
non-linear interactions (e.g., "unusual hour *and* high frequency card *and* mismatched `M`-flags")
matter more than any single feature's marginal effect, and label-encoded categoricals plus raw
`-999` sentinels hurt a linear model far more than a tree-based one. PR-AUC of 0.488 with a base
rate of ~3.5% means the GBM baseline is already reasonably strong — precision well above the base
rate is achievable at usable recall levels — which raises the bar for the GCN in notebook 4.

Published IEEE-CIS solutions with heavy feature engineering and ensembling reach ~0.96 AUC; this
0.912 AUC is a credible, honestly-scoped baseline (50 features, single model, no ensembling, no
V/id columns) sitting meaningfully below that ceiling — which is the right place for a "first
afternoon" baseline to land.

## 7. Saved artifacts

Metrics and validation-set predictions are persisted via `fraud_utils.save_metrics` /
`save_predictions` to `results/baseline_logreg_{metrics,preds}` and
`results/baseline_lightgbm_{metrics,preds}` (the `_lightgbm` name is kept even when the
HistGBM fallback trained, since `save_metrics` is called with that literal string). Notebook 4
loads these directly to build the final GCN-vs-baseline comparison — it does not re-run this
notebook — so **this baseline's numbers are only as fresh as the last time 02 was executed**; if
`fraud_utils.py`, the feature engineering, or the split logic changes, this notebook needs to be
re-run before the comparison in notebook 4 can be trusted.

## Takeaways for the GNN comparison

- **The bar is ~0.91 AUC / ~0.49 PR-AUC**, set by a plain histogram-GBM on 50 lightweight
  features — with real LightGBM (early-stopped, tuned) likely slightly higher still.
- The baseline deliberately excludes the graph-relevant relational structure (shared
  card/address/device/email links) that motivates the GNN in the first place, so a fair takeaway
  from notebook 4 isn't just "did the GCN beat 0.91 AUC" but "did the GCN add signal a *richer*
  tabular model (with V/id features) couldn't also capture" — worth calling out explicitly when
  writing up notebook 4's results, since this baseline alone isn't the strongest possible tabular
  opponent.
- Because both baseline and GCN reuse `fraud_utils.time_based_split` with the same seed and
  fraction, the comparison is apples-to-apples on rows — the remaining question is purely about
  which features/architecture extract more signal from them.

**Next**: [03_preprocessing_gnn_features.ipynb](../notebooks/03_preprocessing_gnn_features.ipynb)
builds the richer feature set and graph edges the GCN needs.
