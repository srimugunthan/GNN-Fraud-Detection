# ADR-004: Use RGCNConv With Relation-Typed Edges (Not Heterogeneous Node Types) in Notebook 05

**Status:** Accepted
**Date:** 2026-09-14
**Deciders:** srimugunthan.d@gmail.com

---

## Context

[ADR-001](ADR-001-gcn-vs-graphsage-for-notebook-04.md) picked `GCNConv` as the message-passing operator
for `notebooks/04_gcn_model.ipynb`, benchmarked against the notebook-2 tabular baselines on an identical
time-based validation split. That GCN scored **0.60 val AUC**, well below the **0.80** logistic-regression
baseline — a large enough gap to warrant root-causing before trying to close it with more epochs or tuning.

Root-cause analysis found two concrete, fixable issues, not a fundamental "GNNs don't help this dataset"
result:

1. **`edge_type` is computed but discarded.** `notebooks/03_preprocessing_gnn_features.ipynb` builds a
   transaction–transaction graph from 12 shared-attribute relations (`card1`–`card6`, `addr1`, `addr2`,
   `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`) and saves each edge's relation as
   `edge_type.npy`. `notebooks/04_gcn_model.ipynb` loads that array but never attaches it to the PyG
   `Data` object or passes it to the model — `GCNConv` has no way to know an edge came from a shared
   card vs. a shared device, so it applies the same symmetric-normalized weight to all of them.
2. **One relation dominates the graph.** `card1` alone accounts for ~77% of all edges (899,694 of
   1,161,398), via a capped star-hub topology for large groups (`MAX_GROUP_SIZE=50`,
   `STAR_THRESHOLD=20`). With relation-blind aggregation, this one relation's structure — including the
   arbitrary "hub" transaction's own features being propagated to every node in its star — has an
   outsized, unmoderated effect on every node's embedding, diluting the signal from smaller but likely
   more informative relations like `DeviceInfo` (160,910 edges) or `P_emaildomain`/`R_emaildomain` match.

A decision was needed on how to make the model relation-aware, and specifically **whether that
relation-awareness should be expressed as edge types on the existing transaction-only graph, or as a
fully heterogeneous graph with separate node types per identity attribute** (i.e., `card1` values,
`addr1` values, email domains, and devices each becoming their own node type, transactions connecting to
them bipartite-style) — the architecture used by the AWS `sagemaker-graph-fraud-detection` DGL reference
implementation that the target Kaggle RGCN notebook is built on.

---

## Decision

We will use **`RGCNConv`** (Schlichtkrull et al., relation-specific weight matrices, summed across
relations) in `notebooks/05_rgcn_model.ipynb`, keeping the **existing transaction-only node set** and
expressing each of the 12 shared-attribute relations as an **edge type** (`edge_type`, threaded through
`Data` and every model call) rather than promoting identity attributes to their own node types. Basis
decomposition (`num_bases=6`) is used so the rare relations (`card3`, `card4`, `card6`, `DeviceType` —
a few hundred edges each) share statistical strength with the common ones instead of each learning a full
independent 128×128 weight matrix from very little data. A full heterogeneous multi-node-type graph
(`HeteroConv` with separate `card`/`addr`/`email`/`device` node types) is explicitly deferred, not
adopted, in this notebook.

---

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Keep `GCNConv`, just wire up `edge_type`** | Smallest change; no new PyG operator to learn/debug | `GCNConv` has no mechanism to *use* `edge_type` even if it's passed — architecturally can't express "trust this relation more than that one," so this doesn't fix the root cause, only the bug of discarding the array |
| **GraphSAGE (`SAGEConv`)** | Matches the production `gnn_fraud_detection.py` model (ADR-001); separate self/neighbor transforms are more robust to noisy neighbors than GCN's symmetric normalization | Still relation-blind on a flat edge list unless split into 12 separate per-relation `SAGEConv` passes (effectively reinventing RGCN, less directly); doesn't address the `card1`-dominance problem on its own |
| **GAT / Graph Transformer (`GATConv`/`TransformerConv`)** | Learned per-edge attention could in principle discover the same "trust `DeviceInfo` over `card3`" weighting without explicit relation labels | Graph is very sparse (avg degree ~1.1 across ~1.1M nodes) — attention needs multiple neighbors per node to learn meaningful weights, and most nodes don't have enough of them; strictly more parameters/harder to train well than giving the model the relation label directly, which we already have for free |
| **RGCNConv with relation-typed edges on the existing transaction graph — chosen** | Directly targets the diagnosed root cause (relation-blind aggregation); reuses `edge_type.npy` and the existing transaction-transaction graph from notebook 03 with no graph-construction rewrite; relation-specific weights are inspectable post-hoc (Section 5 of notebook 05 prints per-relation weight norms), giving a concrete check on which relations the model actually leans on | Basis decomposition (`num_bases=6`) trades some per-relation expressiveness for regularization — a choice that itself needs validating (try `num_bases=None` if underfitting is suspected); still inherits the underlying graph's sparsity and the `card1` star-hub's "borrowed" hub features, since RGCN reweights existing edges, it doesn't add missing ones |
| **True heterogeneous graph — separate node types per identity attribute (`HeteroConv`)** | Matches the AWS/Kaggle reference architecture most closely; identity-attribute nodes carry no borrowed transaction features (unlike the current star-hub proxy), so a shared `card1` value is represented as its own node rather than one arbitrary transaction standing in for the group; likely the strongest long-term option | Requires reworking notebook 03's graph construction (bipartite transaction↔attribute edge lists, per-node-type feature tensors) and notebook 05's model/training loop (`HeteroData`, `HeteroConv`) — a materially larger change than swapping the conv operator; deferred as follow-up work rather than done now, to keep this decision isolated and testable against the GCN baseline first |

---

## Consequences

**Positive:**
- Directly fixes the diagnosed root cause: the model can now learn to down-weight `card3`/`card4`/`card6`/`DeviceType`
  (near-noise relations by edge count) and up-weight `DeviceInfo`/email-match edges, instead of averaging
  all 12 relations uniformly.
- Reuses `edge_type.npy` and the notebook-03 graph as-is — no graph-construction rewrite, so the
  comparison against `notebooks/04_gcn_model.ipynb` isolates the architecture change (GCN → RGCN) rather
  than conflating it with a graph-construction change.
- The per-relation weight-norm inspection (notebook 05, Section 5) turns "we suspect `card3` is
  low-signal" from an edge-count-based assumption into something the model itself confirms or refutes.
- Keeps the door open to the fuller heterogeneous-node-type architecture later without discarding this
  work — `edge_type` labels map directly onto canonical relation names if/when the graph is rebuilt as
  bipartite.

**Negative / Accepted Tradeoffs:**
- Does **not** fix the underlying graph sparsity (~1.1 avg degree) or the `card1` star-hub's arbitrary
  "hub" feature propagation — RGCN can reweight existing edges but can't manufacture edges that
  `MAX_GROUP_SIZE=50` capped away. If RGCN still trails the tabular baseline by a wide margin, the next
  lever is graph density/construction, not the conv operator.
- `num_bases=6` is a regularization choice, not a proven-optimal one for this data; it needs to be
  validated against `num_bases=None` (full per-relation weights) before being treated as final.
- We are explicitly *not* building the heterogeneous multi-node-type graph in this pass, so the
  "arbitrary hub transaction represents its whole shared-card group" distortion identified in notebook
  04's discussion is only partially mitigated (RGCN can learn to discount the `card1` relation overall,
  but can't recover the information lost when large groups were capped/starred).
- Adds a second GNN architecture to maintain and compare (`FraudGCN` in notebook 04, `FraudRGCN` in
  notebook 05) rather than replacing the former outright.

---

## Follow-up Actions

- [ ] Run `notebooks/05_rgcn_model.ipynb` end-to-end (GPU environment) and record val AUC/PR-AUC against
      `gcn_gcn`, `baseline_logreg`, and `baseline_lightgbm` in `results/`.
- [ ] Inspect Section 5's per-relation weight norms; confirm or refute that `card3`/`card4`/`card6`/`DeviceType`
      end up down-weighted relative to `DeviceInfo`/email-match relations.
- [ ] Ablate `num_bases` (`None` vs. `6` vs. other values) to check whether basis decomposition is helping
      or under-fitting the rare relations.
- [ ] If RGCN still trails the tabular baseline substantially, revisit `MAX_GROUP_SIZE`/`STAR_THRESHOLD` in
      `notebooks/03_preprocessing_gnn_features.ipynb` before concluding the architecture is the bottleneck.
- [ ] Scope a follow-up ADR for the full heterogeneous multi-node-type graph (`HeteroData`/`HeteroConv`,
      separate `card`/`addr`/`email`/`device` node types) if RGCN's edge-typed-only approach plateaus
      below the tabular baseline.
