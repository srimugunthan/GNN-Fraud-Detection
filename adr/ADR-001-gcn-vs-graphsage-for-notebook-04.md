# ADR-001: Use GCNConv (GCN) Instead of SAGEConv (GraphSAGE) in the Notebook-4 Model Comparison

**Status:** Accepted
**Date:** 2026-09-12
**Deciders:** srimugunthan.d@gmail.com

---

## Context

`gnn_fraud_detection.py` is the project's end-to-end production-style pipeline: it loads the raw IEEE-CIS
Kaggle CSVs, engineers features, constructs the shared-attribute transaction graph, and trains a
`FraudGNN` model built on PyTorch Geometric's `SAGEConv` (GraphSAGE), using mini-batch training via
`NeighborLoader` so it scales to the full competition-sized graph without requiring the whole graph to
fit on one device.

`notebooks/04_gcn_model.ipynb` is a separate, later-stage notebook whose job is to close out the
notebook pipeline (`01_eda` → `02_baseline_model` → `03_preprocessing_gnn_features` → `04_gcn_model`) by
training a GNN on the graph artifacts produced by notebook 3 and benchmarking it against the notebook-2
tabular baselines (logistic regression, LightGBM) on an identical, time-based validation split. For that
comparison, the graph and feature matrix already fit comfortably in memory (they are precomputed,
fixed-size `.npy` artifacts), so full-batch training over the whole graph in a single forward pass is
viable — mini-batch neighbor sampling is not required to make training tractable here.

Given a full-batch, single-graph setting, the notebook needed to pick a message-passing operator for its
`FraudGCN` model class. The natural choices were GraphSAGE (to mirror the production script exactly),
GCN (Kipf & Welling's classic spectral-style operator), or GAT (attention-based). A decision was needed
on which operator to use as the notebook's default, since the benchmark numbers and any written
conclusions ("does the GNN beat the GBM baseline?") depend on it.

---

## Decision

We will use **`GCNConv`** (Kipf & Welling GCN) as the default message-passing operator in
`notebooks/04_gcn_model.ipynb`'s `FraudGCN` model, deliberately diverging from `SAGEConv` used in
`gnn_fraud_detection.py`. The model class accepts a `conv_type` parameter (`"gcn"` / `"sage"` / `"gat"`)
so GraphSAGE or GAT can be swapped in against the same training/eval code without further changes.

---

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **GraphSAGE (`SAGEConv`)** — match the production script | Consistent with `gnn_fraud_detection.py`; inductive by design; works naturally with `NeighborLoader` mini-batching if the graph later outgrows memory | Doesn't add a distinct data point to the comparison — would just be a re-run of the same operator under different training-loop mechanics (full-batch vs. mini-batch), which is less useful for the notebook's stated goal of comparing operators/approaches |
| **GCN (`GCNConv`) — chosen** | Classic, well-understood baseline GNN operator; simple normalized-adjacency aggregation is a natural fit for full-batch training on a graph that already fits in memory; gives an architecturally distinct comparison point against the production GraphSAGE model; requested explicitly for this notebook | Transductive/full-graph assumption — normalized adjacency aggregation doesn't decompose cleanly into per-batch neighbor samples, so it doesn't scale to the full training pipeline's `NeighborLoader` mini-batching without extra work; treats all shared-attribute edges uniformly (no learned edge weighting) |
| **GAT (`GATConv`)** | Attention weights could let the model learn which shared-attribute edges (card vs. email vs. device) matter most, rather than weighting them uniformly like GCN | More parameters and slower per epoch; adds a second axis of variation (attention) on top of the operator choice, muddying a first comparison pass; left as a documented follow-up experiment (`conv_type="gat"`) rather than the default |

---

## Consequences

**Positive:**
- The notebook produces a comparison point that is architecturally distinct from the production
  `FraudGNN` (GraphSAGE), rather than a redundant re-run of the same operator.
- `FraudGCN`'s `conv_type` switch keeps GraphSAGE and GAT one line away, so re-running the same
  training/eval code with `conv_type="sage"` gives a same-notebook, apples-to-apples check against the
  production script's operator without touching the model class or training loop.
- GCN's simpler aggregation (no per-node neighbor sampling to configure) keeps the full-batch training
  loop in the notebook straightforward for a one-off benchmarking run.

**Negative / Accepted Tradeoffs:**
- The notebook's GCN model is **not** a validation of the production `gnn_fraud_detection.py` model —
  its AUC/PR-AUC numbers should not be read as "how well does the production model do," only as "how
  does GCN do on this graph, as one data point among several architectures."
- `GCNConv`'s full-graph normalized-adjacency aggregation does not have a direct mini-batch/neighbor-sampled
  equivalent the way `SAGEConv` does, so if the notebook's graph ever grows past what fits in memory on
  one device, switching back to `conv_type="sage"` (with `NeighborLoader`, as in the production script) is
  the fallback path rather than scaling `GCNConv` directly.
- Uniform edge weighting in GCN (vs. GAT's learned attention) means the model can't down-weight noisy
  shared-attribute edges (e.g., a very common `P_emaildomain` value) on its own — this is called out in
  the notebook as a reason to try `conv_type="gat"` next.

---

## Follow-up Actions

- [ ] Re-run the notebook with `conv_type="sage"` to get a same-graph, same-split GraphSAGE number to
      compare directly against `gnn_fraud_detection.py`'s production model.
- [ ] Try `conv_type="gat"` to test whether learned attention over shared-attribute edges improves on
      GCN's uniform weighting.
- [ ] If the notebook-3 graph artifacts grow beyond single-device memory, migrate notebook 4's training
      loop to `NeighborLoader` mini-batching (matching `gnn_fraud_detection.py`) before continuing to rely
      on `GCNConv`.
