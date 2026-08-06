# GNN-based Solution for IEEE-CIS Fraud Detection

## Problem Formulation

Instead of treating fraud detection as a standard tabular classification problem, this solution **models transactions as a graph** where:

- **Nodes** = Transactions (both train + test)
- **Edges** = Shared attributes between transactions (same card, email, device, address)
- **Task** = Semi-supervised node classification (fraud vs. legitimate)

### Why GNN for Fraud?

Fraudsters often share infrastructure — the same card numbers, email domains, devices, or addresses appear across multiple fraudulent transactions. A GNN can learn to detect these structural patterns by aggregating information from connected transactions, catching fraud that feature-based models miss.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GRAPH CONSTRUCTION                          │
│                                                                 │
│  Transaction nodes connected via shared:                        │
│  card1-6 │ addr1-2 │ P/R_emaildomain │ DeviceType/Info         │
│                                                                 │
│  Star topology for large groups (>20), full pairwise otherwise  │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      NODE FEATURES (~100+)                      │
│                                                                 │
│  TransactionAmt (log, decimal, round flag)                      │
│  Time features (hour, day, dow, cyclical encoding)              │
│  Card aggregation stats (count, amt mean/std, z-score)          │
│  Email domain features + match flag                             │
│  Top-50 V columns (by variance) + C/D/M columns                │
│  Label-encoded categoricals                                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    GraphSAGE MODEL                              │
│                                                                 │
│  Input Projection (Linear → BN → ReLU → Dropout)               │
│         │                                                       │
│         ├──────────────┐                                        │
│         ▼              │                                        │
│  SAGEConv Layer 1      │  (skip connections)                    │
│         │              │                                        │
│         ├──────────────┤                                        │
│         ▼              │                                        │
│  SAGEConv Layer 2      │                                        │
│         │              │                                        │
│         ├──────────────┤                                        │
│         ▼              │                                        │
│  SAGEConv Layer 3      │                                        │
│         │              │                                        │
│         ▼              ▼                                        │
│  Concatenate [h0 ‖ h1 ‖ h2 ‖ h3]                              │
│         │                                                       │
│         ▼                                                       │
│  MLP Classifier (3 layers with BN + Dropout)                    │
│         │                                                       │
│         ▼                                                       │
│  Sigmoid → Fraud Probability                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **GraphSAGE** (not GCN/GAT) | Inductive learning — works on unseen test nodes; scalable via neighbor sampling |
| **Skip connections** | Prevents over-smoothing across layers; preserves node-level features |
| **Star topology** for large groups | Avoids O(n²) edge explosion while maintaining connectivity |
| **NeighborLoader** mini-batching | Full graph doesn't fit in GPU memory (~600K nodes, millions of edges) |
| **BCEWithLogitsLoss + pos_weight** | Handles severe class imbalance (~3.5% fraud rate) |
| **Cosine annealing LR** | Smooth decay helps convergence on noisy graph data |
| **Gradient clipping** | Message passing can cause gradient spikes |

## Setup & Usage

### Install Dependencies

```bash
pip install torch torchvision
pip install torch-geometric
pip install pandas numpy scikit-learn matplotlib
```

### Download Data

```bash
# Using Kaggle CLI
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/
```

### Run

```bash
python gnn_fraud_detection.py
```

Without data, it runs in **demo mode** with synthetic data to validate the pipeline.

### Output

- `best_gnn_model.pt` — Saved model weights
- `submission.csv` — Kaggle submission file

## Expected Performance

| Metric | Approximate Range |
|--------|------------------|
| Validation AUC | 0.92 – 0.95 |
| Kaggle LB AUC | 0.93 – 0.96 |

> **Note**: Top solutions on this competition achieve ~0.96 AUC by ensembling GBMs with extensive feature engineering. A GNN alone typically scores in the 0.93–0.95 range but captures complementary structural signals.

## Potential Improvements

1. **Heterogeneous GNN**: Use `HeteroConv` with different relation types instead of collapsing all edge types
2. **Temporal edges**: Only connect transactions that are close in time (not just sharing attributes)
3. **Edge features**: Encode the time difference, amount difference between connected transactions
4. **Ensemble with GBM**: Use GNN node embeddings as features for LightGBM/XGBoost
5. **GAT attention**: Replace GraphSAGE with GAT to learn which neighbor connections matter most
6. **Label propagation**: Semi-supervised pre-training step before GNN
