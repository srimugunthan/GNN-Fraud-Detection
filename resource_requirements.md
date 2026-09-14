# Resource Requirements — Memory & Compute for Training the GNN

This repo's own [`notebooks/03_preprocessing_gnn_features.ipynb`](notebooks/03_preprocessing_gnn_features.ipynb)
has already been run once, and its saved output cells give real numbers instead of pure guesses. This doc
is the estimate built on those, plus reasoned extrapolation for the parts that weren't logged.

## Hard numbers from the executed run (notebook 3 output)

| Quantity | Value |
|---|---|
| Combined nodes (train+valid+test) | 1,097,231 |
| Engineered features | 117 |
| Total graph edges | 1,161,398 (avg degree 1.1 — sparse) |
| Raw train_transaction+identity, after downcast | 2514 MB → 1603 MB |
| Raw test_transaction+identity, after downcast | 2164 MB → 1386 MB |

Biggest single edge contributor: `card1` alone → 899,694 of the 1.16M edges (shared card numbers dominate
the graph structure); everything else (email, device, address) contributes far fewer.

## Memory — three separate numbers, not one

### 1. Final graph tensors (what actually gets fed to the GNN) — small

| Artifact | Size |
|---|---|
| `features.npy` (1,097,231 × 117, float32) | ~490 MB |
| `edge_index.npy` (2 × 1,161,398, int64) | ~18 MB |
| `edge_type.npy` | ~9 MB |
| labels/split | a few MB |
| **Total** | **~530–600 MB** |

The trained artifacts themselves are not the bottleneck at all.

### 2. Preprocessing/graph-construction peak — the actual bottleneck

Concatenating the two downcast frames (1603 + 1386 MB) plus the temporary copy `pd.concat` makes, plus
~20-30 new derived columns, plus the `feature_df` copy — realistically peaks around **6–10 GB** transiently.

This lines up with the float64-upcast bug flagged in `notebooks/03_preprocessing_gnn_features.ipynb`:
`feature_df.astype(np.float64)` alone costs `1,097,231 × 117 × 8 bytes ≈ 980 MB` versus `~490 MB` at
float32 — a concrete, quantifiable ~500 MB paid for nothing, at the exact point memory is already peaking.
That's a real, provable number, not just a guess.

### 3. Training-time memory — differs a lot by notebook vs. script

- **Notebook 4 (full-batch GCN)**: holds activations for **every node, every layer** simultaneously for
  backprop. Each `[1,097,231 × 128]` float32 hidden layer ≈ 563 MB; with 4 stored layers (input + 3 conv,
  via skip connections) that's **~2.25 GB** just for the concatenated activations, before counting
  PyTorch's autograd-saved intermediates (BN/ReLU/Dropout each save their own tensors too — typically 2–3x
  on top). Rough full-batch training peak: **on the order of 5–10 GB** in addition to the 530-600MB
  feature/graph tensors.
- **`gnn_fraud_detection.py` (mini-batch, `NeighborLoader`, batch_size=2048, fanout [15,10,5])**: only ever
  materializes one sampled subgraph at a time — typically tens of MB per batch, not GB. This is *why* the
  production script is built this way; it trades some setup complexity for a much lower and more
  predictable memory ceiling, independent of total graph size.

## Compute

- **Model itself is tiny**: ~140K params for the GCN variant (117→128, 3 conv layers, MLP head), ~190K for
  the GraphSAGE variant — well under 1 MB of weights. Model size is not your compute constraint.
- **FLOPs per epoch are dominated by the dense linear layers** (input projection + classifier head) applied
  to all 1.1M nodes, not by message passing itself (the graph is sparse — avg degree 1.1, so aggregation is
  cheap). Rough order: a few hundred GFLOPs forward + backward per epoch.
- **Wall-clock** (order-of-magnitude, hardware-dependent — treat as rough, not measured):
  - GPU (even a modest T4-class): likely **low seconds per epoch**, full training (≤60–100 epochs, early
    stopping patience 8-10) in **a few minutes**.
  - CPU-only: likely **several seconds to ~1 minute per epoch** depending on core count/BLAS, so
    **10–30+ minutes** for a full run.
- A GPU helps but isn't strictly required — this model is small enough that a decent multi-core CPU
  finishes in a reasonable time; the GPU mainly buys you wall-clock, not feasibility.

## Bottom line

| Resource | Rough requirement |
|---|---|
| RAM for preprocessing (notebook 3 / feature+graph build) | 8 GB is tight, 16 GB comfortable, less so once the float64 bug is fixed |
| RAM for full-batch training (notebook 4) | add ~5–10 GB on top during the training loop itself |
| RAM for mini-batch training (`gnn_fraud_detection.py`) | ~2–4 GB total is plenty — this is the scalable path |
| GPU | optional — speeds things up, not required for a model this small |
| Compute time | minutes (GPU) to tens of minutes (CPU) for a full training run |

This is consistent with what tends to get hit in practice: notebook 3's preprocessing step is the real
memory pressure point, not the GNN training itself — which tracks with the `n2-highmem-4` (32 GB) sizing
recommended in [`deploy/gcp_training.md`](deploy/gcp_training.md), and explains why the float64 fix and/or
`NeighborLoader` mini-batching are the higher-leverage fixes versus just throwing a GPU at it.
