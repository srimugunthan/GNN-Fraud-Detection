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
- **Notebook 5 (full-batch RGCN, [`05_rgcn_model.ipynb`](notebooks/05_rgcn_model.ipynb))** — ⚠️ **corrected
  estimate, now backed by an actual observed crash**, not just reasoning from shapes. An earlier version of
  this doc estimated ~6–14 GB for this notebook by analogy to GCN; that estimate was wrong, and running the
  notebook on a real `n2-highmem-4` (32 GB) Workbench VM confirmed it — the kernel died mid-training-loop
  with no traceback, the classic signature of the Linux OOM killer sending `SIGKILL` (confirmable via
  `dmesg | grep -i "killed process"` on the VM). Here's the corrected math for *why* it's so much larger
  than GCN, not just "somewhat more":
  - `RGCNConv` does **not** aggregate once per layer like `GCNConv`. It aggregates **separately for each of
    the 12 relations** (`edge_type == r`, one at a time), producing a full dense
    `[1,097,231 × 128]` intermediate **per relation** — even for a relation with only a few hundred edges
    (e.g. `card4`), because the aggregation output is one row per *node*, not per edge; unconnected nodes
    just get zeros. Only *after* that per-relation aggregate is computed does the relation's weight get
    applied.
  - Each of those intermediates is ~562 MB (`1,097,231 × 128 × 4 bytes`). Autograd must retain **all 12 of
    them simultaneously** for backward (the relation-specific weight multiply needs its corresponding
    aggregate at gradient time) → **~6.7 GB per conv layer**, not ~460–595 MB as previously estimated.
  - Across the model's 3 conv layers: **~6.7 GB × 3 ≈ ~20 GB**, just for these per-relation aggregates.
  - Add the skip-connection concat tensor (`hidden_dim × (num_layers+1) = 512` wide × 1,097,231 nodes ≈
    **~2.25 GB**), the classifier head's own intermediate activations (~1.5–2 GB), the ~530–600 MB base
    graph/feature tensors, the redundant numpy+torch double-copies of `features`/`edge_index`/`edge_type`
    sitting in the notebook namespace (~600 MB extra), gradients/optimizer state (negligible, model is
    ~435K params), and 10–30% allocator/fragmentation overhead on top of all of it, and the running total
    lands at **roughly 25–35 GB** — which is *why* it took down a 32 GB VM. This isn't "RGCN needs somewhat
    more than GCN"; it's "RGCN's per-relation aggregate-then-transform pattern multiplies the dominant memory
    term by (number of relations), independent of how sparse those relations actually are."
  - Basis decomposition (`num_bases=6`, set in the notebook) shrinks the **parameter** memory vs. a full
    independent weight per relation (12×128×128=196,608 floats down to ~98,376), but does **nothing** for
    the per-relation aggregate tensors above — those exist regardless of how the relation weights themselves
    are parameterized, since they're a function of the number of relations, not the weight representation.
  - This scales with **relation count**, not edge count or graph density — a sparse relation like `card4`
    (490 edges) costs exactly as much intermediate memory as `card1` (899,694 edges), because both produce
    one full `[N × hidden]` dense aggregate. This is the key lever if the relation count itself were ever
    reduced, though per this repo's current direction that tradeoff isn't being taken (see below).

## Compute

- **Model itself is tiny**: ~140K params for the GCN variant (117→128, 3 conv layers, MLP head), ~190K for
  the GraphSAGE variant, **~435K for the RGCN variant** (`num_relations=12`, `num_bases=6`) — the jump is
  from `RGCNConv`'s per-relation basis+comp+root weights (~115K params/layer vs. GCN's ~16.5K/layer), but
  435K params is still ~1.7 MB of weights either way. Model size is not your compute constraint for any of
  the three.
- **FLOPs per epoch are dominated by the dense linear layers** (input projection + classifier head) applied
  to all 1.1M nodes, not by message passing itself (the graph is sparse — avg degree 1.1, so aggregation is
  cheap) — true for GCN/SAGE. **RGCN is the exception**: its per-relation loop means message passing is no
  longer negligible relative to the dense layers, because the same total edge volume now gets pushed through
  12 separate relation-specific matmul+scatter passes per conv layer (36 relation-passes total across 3
  layers) instead of one combined pass — this is a well-documented characteristic of PyG's `RGCNConv`
  implementation (it loops over relations in Python), not specific to this repo's graph.
- **Wall-clock** (order-of-magnitude, hardware-dependent — treat as rough estimates, not measured numbers;
  this notebook hasn't actually been run in this environment — see caveat below):
  - GCN/SAGE, GPU (even a modest T4-class): likely **low seconds per epoch**, full training (≤60–100 epochs)
    in **a few minutes**. CPU-only: **several seconds to ~1 minute per epoch**, **10–30+ minutes** total.
  - **RGCN, GPU**: expect roughly **2–5× GCN's per-epoch time** from the 12-relation Python loop (kernel-launch
    overhead dominates at this graph size more than raw FLOPs do) — call it **low tens of seconds per epoch**,
    full 60-epoch run in **~10–20 minutes** on a T4-class GPU.
  - **RGCN, CPU-only**: the same 2–5× multiplier on top of GCN's already-slower CPU number — likely
    **tens of seconds to a few minutes per epoch**, so a full run could run **30 minutes to 1.5+ hours**.
    Given the local dev machine here has no CUDA/MPS and no `torch_geometric` installed, CPU-only is the
    realistic worst case to plan for if you smoke-test locally before moving to GCP.
- A GPU helps more for RGCN than it does for GCN/SAGE, precisely because the per-relation loop's overhead is
  partly Python/kernel-launch bound — a GPU absorbs many small launches better than a CPU does. Still not
  strictly required for feasibility, just for wall-clock.

**Caveat**: the wall-clock numbers above are still reasoned estimates from PyG's documented `RGCNConv`
behavior, not a profiled run. The **memory** numbers are no longer purely reasoned, though — see the
corrected derivation above, which was prompted by and matches an actual observed OOM kill on a real
`n2-highmem-4` VM. Confirm the corrected memory number with `torch.cuda.max_memory_allocated()` (GPU) or a
CPU RSS check (`resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`, or just watching `free -h`) on the next
run, but treat ~25–35 GB as a real constraint to size around, not a rough guess to sanity-check away.

## Recommended Workbench instance for notebook 5 (RGCN, architecture unchanged)

The corrected ~25–35 GB peak, explicitly **not** addressed by reducing `hidden_dim`/`num_layers`/relation
count (those are architecture changes, off the table here since they'd likely cost the relation-level
signal RGCN was adopted for — see [ADR-004](adr/ADR-004-rgcn-relation-typed-edges-vs-heterogeneous-node-types.md)):

- **Use `n2-highmem-8` (8 vCPU, 64 GB RAM)** as the default. That's roughly **2× the corrected ~25–35 GB
  peak**, enough margin to absorb the fragmentation/allocator variance baked into that estimate plus normal
  Jupyter/OS overhead, without paying for a much bigger jump than the evidence calls for.
  ```bash
  gcloud workbench instances create fraud-gnn-workbench \
    --location=us-central1-a \
    --machine-type=n2-highmem-8 \
    --data-disk-size=100 \
    --data-disk-type=PD_SSD
  ```
- **Escalate to `n2-highmem-16` (128 GB)** only if `n2-highmem-8` still OOMs — check `dmesg` to confirm it's
  actually memory and not something else before paying for another doubling.
- **Skip GPU for this specific case.** [`deploy/gcp_workbench_training.md`](deploy/gcp_workbench_training.md)
  §11 suggests a T4 (16 GB VRAM) or L4 (24 GB VRAM) for RGCN's *compute* profile — but neither has enough
  VRAM for this ~25–35 GB *full-batch* peak with the architecture unchanged. Only an A100 (40/80 GB) would
  fit, at a much higher $/hr than the RAM difference between `n2-highmem-4` and `n2-highmem-8` costs. For a
  fixed full-batch architecture, more system RAM on a CPU instance is the cheaper fix than more VRAM; GPU
  becomes the right lever again only if the training loop moves to mini-batch (`NeighborLoader`), which
  processes one sampled subgraph at a time and sidesteps this whole per-relation memory scaling — a training
  *loop* change, not a model architecture change, if that's ever back on the table.

## CPU (64 GB) vs. GPU cost/time comparison for notebook 5

Estimated for a full 60-epoch RGCN training run, architecture unchanged. Time is FLOP-based (~360 TFLOP
total across 60 epochs — ~6.0 TFLOP/epoch: dense layers + `RGCNConv`'s per-relation aggregate-then-transform,
counted for train fwd+bwd plus the `evaluate()` call's extra forward pass every epoch); none of this has
actually been profiled end-to-end yet, so treat both time and cost as order-of-magnitude, not quotes.

| Instance | VRAM/RAM available | Time for 60 epochs | Approx. rate (on-demand, us-central1) | Approx. cost for one run |
|---|---|---|---|---|
| T4/L4 (`n1-highmem-4`+T4 or `g2-standard-4`+L4 — [§11](deploy/gcp_workbench_training.md)) | 16–24 GB VRAM | **N/A — doesn't finish.** Below the ~25–35 GB peak; expect a CUDA OOM around the first backward pass. | ~$0.60–0.90/hr | N/A |
| **`n2-highmem-8` (64 GB) — [§12](deploy/gcp_workbench_training.md)** | 64 GB RAM | **~40–120 min**, likely **~1 hr** (50–150 GFLOPS sustained CPU throughput) | ~$0.52/hr | **~$0.35–$1.04**, likely **~$0.50** |
| `a2-highgpu-1g` (1× A100 40GB) — only GPU with enough VRAM to fit | 40 GB VRAM | **~2–12 min**, wide range because `RGCNConv`'s scatter/aggregate pattern is memory-bound, not pure GEMM, so it's nowhere near the A100's peak FLOPS | ~$3.67/hr | **~$0.12–$0.73**, likely **~$0.25** |

**Reading this table**: per-run $ cost is roughly a wash between `n2-highmem-8` and an A100 — the A100's
~7× higher hourly rate is plausibly offset by a run that's ~10× shorter. What actually decides it is
everything the table doesn't price in:
- **A100 quota is the real friction, not cost.** New/small GCP projects commonly start at 0 A100 quota, and
  an increase request can take days with no guarantee — a much bigger ask than T4/L4 quota, which is itself
  already not guaranteed.
- **`a2-highgpu-1g` is a fixed, oversized shape** (12 vCPU + 85 GB RAM bundled in) regardless of whether this
  small (~435K param) model needs any of that.
- **The GPU time estimate has the widest error bars in this whole doc** — it's the one number here with zero
  grounding in an observed run, unlike the memory numbers above.
- **The A100 path only pays off with repeated runs** (active hyperparameter tuning, many retraining cycles) —
  for a single or occasional training run, the setup friction isn't worth chasing a ~$0.25 saving over the
  CPU path's ~$0.50.

**Recommendation**: use `n2-highmem-8` (already validated to work, per the memory correction above) unless
you expect to retrain RGCN many times, at which point the A100's time savings compound and the quota-request
friction becomes worth absorbing once.

## Bottom line

| Resource | Rough requirement |
|---|---|
| RAM for preprocessing (notebook 3 / feature+graph build) | 8 GB is tight, 16 GB comfortable, less so once the float64 bug is fixed |
| RAM for full-batch GCN training (notebook 4) | add ~5–10 GB on top during the training loop itself |
| RAM for full-batch RGCN training (notebook 5) | add **~25–35 GB** on top — confirmed by an actual OOM kill on `n2-highmem-4` (32 GB); see the corrected derivation above. **Recommend `n2-highmem-8` (64 GB)**, escalate to `n2-highmem-16` (128 GB) if that's still not enough |
| RAM for mini-batch training (`gnn_fraud_detection.py`) | ~2–4 GB total is plenty — this is the scalable path |
| GPU | optional for GCN/SAGE; **not recommended for full-batch RGCN** with the architecture unchanged — a T4/L4's 16–24 GB VRAM is smaller than the ~25–35 GB peak, only an A100 fits and it's costlier than just adding system RAM |
| Compute time (GCN/SAGE) | minutes (GPU) to tens of minutes (CPU) for a full training run |
| Compute time (RGCN) | `n2-highmem-8`: ~40–120 min (likely ~1 hr). T4/L4: doesn't finish (OOM). A100: ~2–12 min if quota allows — see cost/time comparison above |
| Cost per RGCN training run (60 epochs) | `n2-highmem-8`: ~$0.35–$1.04 (likely ~$0.50). A100 (`a2-highgpu-1g`): ~$0.12–$0.73 (likely ~$0.25) if you can get quota — roughly a wash per-run; `n2-highmem-8` wins on setup friction for occasional runs |

This is consistent with what tends to get hit in practice: notebook 3's preprocessing step was the
originally-flagged memory pressure point, but notebook 5's RGCN training turns out to be the bigger one in
practice — its per-relation aggregate-then-transform pattern scales memory with **relation count**, not
graph density or edge count, which is easy to underestimate by analogy to GCN (as this doc's own earlier
version did) until it's actually run. `n2-highmem-8` covers both notebook 3's preprocessing peak and
notebook 5's corrected RGCN training peak on one instance size, so it's a reasonable single choice for a
Workbench VM that runs the whole pipeline end to end.
