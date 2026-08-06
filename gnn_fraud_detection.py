"""
================================================================================
GNN-based Solution for IEEE-CIS Fraud Detection (Kaggle)
================================================================================

This solution models the fraud detection problem as a graph problem using
Graph Neural Networks (PyTorch Geometric). The key insight is that fraudulent
transactions often form patterns/clusters when connected through shared
attributes (cards, emails, devices, addresses).

Architecture:
    1. Graph Construction: Transactions become nodes; edges connect transactions
       sharing the same card, email domain, device, or address.
    2. Node Features: Engineered from transaction + identity tables.
    3. GNN Model: GraphSAGE with skip connections for node classification.
    4. Training: Semi-supervised learning on labeled transactions.

Dataset: https://www.kaggle.com/c/ieee-fraud-detection
================================================================================
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the pipeline."""
    # Data paths (update these to your local paths)
    TRAIN_TRANSACTION = "data/train_transaction.csv"
    TRAIN_IDENTITY = "data/train_identity.csv"
    TEST_TRANSACTION = "data/test_transaction.csv"
    TEST_IDENTITY = "data/test_identity.csv"

    # Graph construction
    EDGE_TYPES = ["card1", "card2", "card3", "card4", "card5", "card6",
                  "addr1", "addr2", "P_emaildomain", "R_emaildomain",
                  "DeviceType", "DeviceInfo"]
    MAX_EDGES_PER_TYPE = 50  # Limit edges per node per type to control graph size

    # Model hyperparameters
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 100
    PATIENCE = 10  # Early stopping patience
    BATCH_SIZE = 2048  # For mini-batch training via NeighborLoader

    # Sampling for neighbor loader
    NUM_NEIGHBORS = [15, 10, 5]  # neighbors sampled per layer

    # Device
    DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Random seed
    SEED = 42


# ============================================================================
# SECTION 2: FEATURE ENGINEERING
# ============================================================================

def reduce_memory_usage(df):
    """Reduce memory usage of a DataFrame by downcasting numeric types."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df


def load_and_merge_data(config: Config):
    """Load transaction and identity data, merge them."""
    print("Loading data...")
    train_txn = pd.read_csv(config.TRAIN_TRANSACTION)
    train_id = pd.read_csv(config.TRAIN_IDENTITY)
    test_txn = pd.read_csv(config.TEST_TRANSACTION)
    test_id = pd.read_csv(config.TEST_IDENTITY)

    # Merge
    train = train_txn.merge(train_id, on="TransactionID", how="left")
    test = test_txn.merge(test_id, on="TransactionID", how="left")

    print(f"Train shape: {train.shape}, Test shape: {test.shape}")
    print(f"Fraud rate: {train['isFraud'].mean():.4f}")

    return train, test


def engineer_features(train, test):
    """
    Engineer node features from raw transaction + identity data.

    Feature groups:
        - Transaction amount features (log, decimal, binned)
        - Time-based features (hour, day, day of week)
        - Card aggregation features (frequency, amount stats)
        - V-columns (PCA-derived, keep top ones by variance)
        - C-columns (counting features)
        - D-columns (time deltas)
        - Identity features (device, browser type encoding)
    """
    print("Engineering features...")

    # Combine for consistent encoding
    train["is_train"] = 1
    test["is_train"] = 0
    test["isFraud"] = -1  # placeholder
    df = pd.concat([train, test], axis=0, ignore_index=True)

    # --- Transaction Amount Features ---
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])
    df["TransactionAmt_decimal"] = (df["TransactionAmt"] - df["TransactionAmt"].astype(int))
    df["TransactionAmt_is_round"] = (df["TransactionAmt_decimal"] == 0).astype(int)

    # --- Time Features ---
    # TransactionDT is seconds from a reference point
    df["Transaction_hour"] = (df["TransactionDT"] / 3600) % 24
    df["Transaction_day"] = (df["TransactionDT"] / (3600 * 24)).astype(int)
    df["Transaction_dow"] = df["Transaction_day"] % 7

    # Cyclical encoding of hour
    df["hour_sin"] = np.sin(2 * np.pi * df["Transaction_hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Transaction_hour"] / 24)

    # --- Card Aggregation Features ---
    for col in ["card1", "card2", "card3", "card4", "card5", "card6"]:
        vc = df[col].value_counts(dropna=False)
        df[f"{col}_count"] = df[col].map(vc)

    # Card1 x TransactionAmt interaction
    card1_amt = df.groupby("card1")["TransactionAmt"].agg(["mean", "std"]).reset_index()
    card1_amt.columns = ["card1", "card1_amt_mean", "card1_amt_std"]
    df = df.merge(card1_amt, on="card1", how="left")
    df["card1_amt_zscore"] = (df["TransactionAmt"] - df["card1_amt_mean"]) / (df["card1_amt_std"] + 1e-8)

    # --- Email Domain Features ---
    for col in ["P_emaildomain", "R_emaildomain"]:
        df[col] = df[col].fillna("unknown")
        vc = df[col].value_counts()
        df[f"{col}_count"] = df[col].map(vc)
        # Group rare domains
        df[f"{col}_group"] = df[col].apply(
            lambda x: x.split(".")[-1] if isinstance(x, str) else "unknown"
        )

    # Email match
    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)

    # --- Address Features ---
    for col in ["addr1", "addr2"]:
        vc = df[col].value_counts(dropna=False)
        df[f"{col}_count"] = df[col].map(vc)

    # --- Select and Process V Columns ---
    v_cols = [c for c in df.columns if c.startswith("V")]
    # Keep V columns with low null rate and high variance
    v_null_rate = df[v_cols].isnull().mean()
    v_keep = v_null_rate[v_null_rate < 0.5].index.tolist()
    v_variance = df[v_keep].var()
    v_keep = v_variance.nlargest(50).index.tolist()  # Keep top 50 by variance
    print(f"Keeping {len(v_keep)} V-columns")

    # --- C Columns ---
    c_cols = [c for c in df.columns if c.startswith("C")]

    # --- D Columns ---
    d_cols = [c for c in df.columns if c.startswith("D")]

    # --- M Columns (match columns) → binary encode ---
    m_cols = [c for c in df.columns if c.startswith("M")]
    for col in m_cols:
        df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1)

    # --- Categorical Encoding ---
    cat_cols = ["ProductCD", "card4", "card6", "P_emaildomain_group",
                "R_emaildomain_group", "DeviceType"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # --- Assemble Feature Matrix ---
    numeric_features = [
        "TransactionAmt", "TransactionAmt_log", "TransactionAmt_decimal",
        "TransactionAmt_is_round", "Transaction_hour", "Transaction_day",
        "Transaction_dow", "hour_sin", "hour_cos",
        "card1_count", "card2_count", "card3_count", "card4_count",
        "card5_count", "card6_count", "card1_amt_mean", "card1_amt_std",
        "card1_amt_zscore", "P_emaildomain_count", "R_emaildomain_count",
        "email_match", "addr1_count", "addr2_count",
        "ProductCD", "card4", "card6", "P_emaildomain_group",
        "R_emaildomain_group", "DeviceType",
    ] + v_keep + c_cols + d_cols + m_cols

    # Filter to columns that actually exist
    numeric_features = [c for c in numeric_features if c in df.columns]
    print(f"Total features: {len(numeric_features)}")

    # Fill NaN and normalize
    feature_df = df[numeric_features].copy()
    feature_df = feature_df.fillna(-999)

    # Standard scaling
    means = feature_df.mean()
    stds = feature_df.std() + 1e-8
    feature_df = (feature_df - means) / stds

    # Split back
    train_mask = df["is_train"] == 1
    test_mask = df["is_train"] == 0
    labels = df["isFraud"].values

    return df, feature_df, labels, train_mask.values, test_mask.values, numeric_features


# ============================================================================
# SECTION 3: GRAPH CONSTRUCTION
# ============================================================================

def build_graph_edges(df, config: Config):
    """
    Build edges between transactions that share attribute values.

    Strategy:
        For each edge type (e.g., card1), group transactions by value.
        Within each group, connect all pairs (with a cap to avoid quadratic blowup).
        This creates a heterogeneous-like graph where edges represent shared attributes.

    Returns:
        edge_index: [2, num_edges] tensor of node indices
        edge_type:  [num_edges] tensor indicating which attribute created the edge
    """
    import torch

    print("Building graph edges...")
    n = len(df)
    all_src, all_dst, all_types = [], [], []

    for etype_idx, col in enumerate(config.EDGE_TYPES):
        if col not in df.columns:
            print(f"  Skipping {col} (not in data)")
            continue

        print(f"  Processing edge type: {col}")

        # Group by attribute value
        groups = df.groupby(col).groups  # {value: [indices]}

        edge_count = 0
        for val, indices in groups.items():
            if pd.isna(val):
                continue
            indices = indices.tolist()
            if len(indices) < 2:
                continue

            # Cap group size to avoid O(n^2) edges
            if len(indices) > config.MAX_EDGES_PER_TYPE:
                indices = np.random.choice(indices, config.MAX_EDGES_PER_TYPE, replace=False).tolist()

            # Create edges: connect each node to others in the group
            # Use a star topology (connect all to first) for very large groups
            if len(indices) > 20:
                # Star: connect all to a random hub
                hub = indices[0]
                for idx in indices[1:]:
                    all_src.extend([hub, idx])
                    all_dst.extend([idx, hub])
                    all_types.extend([etype_idx, etype_idx])
                    edge_count += 2
            else:
                # Full pairwise for small groups
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        all_src.extend([indices[i], indices[j]])
                        all_dst.extend([indices[j], indices[i]])
                        all_types.extend([etype_idx, etype_idx])
                        edge_count += 2

        print(f"    → {edge_count:,} edges from {col}")

    edge_index = torch.tensor([all_src, all_dst], dtype=torch.long)
    edge_type = torch.tensor(all_types, dtype=torch.long)

    print(f"Total edges: {edge_index.shape[1]:,}")
    print(f"Average degree: {edge_index.shape[1] / n:.1f}")

    return edge_index, edge_type


# ============================================================================
# SECTION 4: GNN MODEL
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudGNN(nn.Module):
    """
    GraphSAGE-based GNN for fraud detection.

    Architecture:
        Input → Linear projection → [GraphSAGE + BatchNorm + ReLU + Dropout] × L → Classifier

    Key design choices:
        - GraphSAGE: Inductive, scales to large graphs via sampling
        - Skip connections: Concatenate all layer outputs for the final representation
        - Batch normalization: Stabilizes training
        - Class-weighted loss: Handles severe class imbalance (~3.5% fraud)
    """

    def __init__(self, in_channels, hidden_channels, num_layers, dropout,
                 num_edge_types=12):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Classifier head (takes skip-connected features)
        # Skip connection: concatenate input_proj output + all conv outputs
        classifier_in = hidden_channels * (num_layers + 1)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index):
        # Input projection
        h = self.input_proj(x)
        h = self.input_bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Collect skip connections
        skip = [h]

        # Message passing layers
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            skip.append(h)

        # Concatenate all representations
        h = torch.cat(skip, dim=-1)

        # Classify
        out = self.classifier(h)
        return out.squeeze(-1)


# ============================================================================
# SECTION 5: TRAINING PIPELINE
# ============================================================================

def create_pyg_data(feature_df, labels, train_mask, test_mask, edge_index, edge_type):
    """Create PyTorch Geometric Data object."""
    import torch
    from torch_geometric.data import Data

    x = torch.tensor(feature_df.values, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    train_mask_t = torch.tensor(train_mask, dtype=torch.bool)
    test_mask_t = torch.tensor(test_mask, dtype=torch.bool)

    data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        edge_type=edge_type,
        train_mask=train_mask_t,
        test_mask=test_mask_t,
    )

    # Create train/val split from training data
    train_indices = torch.where(train_mask_t)[0]
    n_train = len(train_indices)
    perm = torch.randperm(n_train)
    val_size = int(0.15 * n_train)

    val_mask = torch.zeros(len(labels), dtype=torch.bool)
    actual_train_mask = torch.zeros(len(labels), dtype=torch.bool)

    val_mask[train_indices[perm[:val_size]]] = True
    actual_train_mask[train_indices[perm[val_size:]]] = True

    data.actual_train_mask = actual_train_mask
    data.val_mask = val_mask

    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges")
    print(f"Train: {actual_train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask_t.sum()}")

    return data


def train_epoch(model, data, optimizer, criterion, device, config):
    """Train for one epoch using mini-batch sampling."""
    from torch_geometric.loader import NeighborLoader

    model.train()

    loader = NeighborLoader(
        data,
        num_neighbors=config.NUM_NEIGHBORS,
        batch_size=config.BATCH_SIZE,
        input_nodes=data.actual_train_mask,
        shuffle=True,
    )

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)

        # Only compute loss on seed nodes (first batch_size nodes)
        mask = batch.actual_train_mask
        loss = criterion(out[mask], batch.y[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        preds = (torch.sigmoid(out[mask]) > 0.5).float()
        total_correct += (preds == batch.y[mask]).sum().item()
        total_samples += mask.sum().item()

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, data, mask_name, device, config):
    """Evaluate on validation or test set."""
    from torch_geometric.loader import NeighborLoader
    from sklearn.metrics import roc_auc_score

    model.eval()
    mask = getattr(data, mask_name)

    loader = NeighborLoader(
        data,
        num_neighbors=config.NUM_NEIGHBORS,
        batch_size=config.BATCH_SIZE,
        input_nodes=mask,
        shuffle=False,
    )

    all_preds = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        batch_mask = getattr(batch, mask_name)
        probs = torch.sigmoid(out[batch_mask])
        all_preds.append(probs.cpu())
        all_labels.append(batch.y[batch_mask].cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Filter out test labels (which are -1)
    valid = all_labels >= 0
    if valid.sum() > 0:
        auc = roc_auc_score(all_labels[valid], all_preds[valid])
    else:
        auc = -1

    return auc, all_preds


def train_model(model, data, config):
    """Full training loop with early stopping."""
    device = torch.device(config.DEVICE)
    model = model.to(device)
    data = data.to("cpu")  # NeighborLoader handles device transfer

    # Class weights for imbalanced data
    n_pos = (data.y[data.actual_train_mask] == 1).sum().float()
    n_neg = (data.y[data.actual_train_mask] == 0).sum().float()
    pos_weight = n_neg / (n_pos + 1e-8)
    print(f"Positive weight: {pos_weight:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )

    best_val_auc = 0
    patience_counter = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion, device, config)
        val_auc, _ = evaluate(model, data, "val_mask", device, config)
        scheduler.step()

        print(
            f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val AUC: {val_auc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), "best_gnn_model.pt")
            print(f"  → New best model saved (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load("best_gnn_model.pt"))
    print(f"\nBest Validation AUC: {best_val_auc:.4f}")

    return model


# ============================================================================
# SECTION 6: INFERENCE & SUBMISSION
# ============================================================================

def generate_submission(model, data, df, config):
    """Generate Kaggle submission file."""
    device = torch.device(config.DEVICE)

    _, test_preds = evaluate(model, data, "test_mask", device, config)

    test_ids = df.loc[df["is_train"] == 0, "TransactionID"].values
    submission = pd.DataFrame({
        "TransactionID": test_ids,
        "isFraud": test_preds,
    })
    submission.to_csv("submission.csv", index=False)
    print(f"Submission saved: {submission.shape}")
    print(submission.head())

    return submission


# ============================================================================
# SECTION 7: MAIN PIPELINE
# ============================================================================

def main():
    """
    End-to-end pipeline:
        1. Load data
        2. Engineer features
        3. Build graph
        4. Create PyG data object
        5. Train GNN
        6. Generate submission
    """
    config = Config()

    # Set seeds
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    # --- Step 1: Load Data ---
    train, test = load_and_merge_data(config)

    # --- Step 2: Feature Engineering ---
    df, feature_df, labels, train_mask, test_mask, feature_names = engineer_features(train, test)
    del train, test
    gc.collect()

    # --- Step 3: Build Graph ---
    edge_index, edge_type = build_graph_edges(df, config)

    # --- Step 4: Create PyG Data ---
    data = create_pyg_data(feature_df, labels, train_mask, test_mask, edge_index, edge_type)
    del feature_df
    gc.collect()

    # --- Step 5: Train Model ---
    model = FraudGNN(
        in_channels=data.x.shape[1],
        hidden_channels=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(model)

    model = train_model(model, data, config)

    # --- Step 6: Generate Submission ---
    submission = generate_submission(model, data, df, config)

    print("\n✅ Pipeline complete!")
    return model, data, submission


# ============================================================================
# SECTION 8: ALTERNATIVE - LIGHTWEIGHT VERSION (No PyG dependency)
# ============================================================================

class LightweightGNNFraudDetector:
    """
    A simplified GNN-like approach using manual message passing with PyTorch only.
    Use this if torch_geometric installation is problematic.

    Approach:
        1. Build adjacency from shared attributes (same as above)
        2. Implement manual neighborhood aggregation (mean pooling)
        3. Stack aggregation + transform layers
        4. Train with standard PyTorch

    This demonstrates the core GNN concept without library dependencies.
    """

    def __init__(self, in_features, hidden_dim=64, num_layers=2, dropout=0.3):
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

    def build_adjacency_dict(self, df, edge_cols, max_neighbors=30):
        """Build adjacency list as a dictionary: node_id → [neighbor_ids]."""
        adj = defaultdict(set)
        for col in edge_cols:
            if col not in df.columns:
                continue
            groups = df.groupby(col).groups
            for val, indices in groups.items():
                if pd.isna(val) or len(indices) < 2:
                    continue
                idx_list = indices.tolist()
                if len(idx_list) > max_neighbors:
                    idx_list = np.random.choice(idx_list, max_neighbors, replace=False).tolist()
                for i in idx_list:
                    adj[i].update(idx_list)
                    adj[i].discard(i)  # Remove self-loops
        return adj

    def aggregate_neighbors(self, features, adj, node_indices=None):
        """
        Manual mean aggregation of neighbor features.

        For each node, compute: h_agg = mean(h_neighbor for neighbor in adj[node])
        Then concatenate: [h_node || h_agg]
        """
        if node_indices is None:
            node_indices = range(len(features))

        aggregated = np.zeros_like(features)
        for i in node_indices:
            neighbors = list(adj.get(i, []))
            if len(neighbors) > 0:
                aggregated[i] = features[neighbors].mean(axis=0)
            else:
                aggregated[i] = features[i]  # Self-loop if isolated

        # Concatenate original and aggregated
        return np.concatenate([features, aggregated], axis=1)


# ============================================================================
# SECTION 9: UTILITIES & ANALYSIS
# ============================================================================

def analyze_graph_structure(data, df, config):
    """Analyze the constructed graph for insights."""
    from torch_geometric.utils import degree

    print("\n" + "=" * 60)
    print("GRAPH ANALYSIS")
    print("=" * 60)

    # Degree distribution
    deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
    print(f"Degree stats: mean={deg.mean():.1f}, median={deg.median():.1f}, "
          f"max={deg.max():.0f}, min={deg.min():.0f}")

    # Fraud vs non-fraud degree comparison
    fraud_mask = data.y == 1
    legit_mask = data.y == 0
    print(f"Avg degree (fraud):  {deg[fraud_mask].mean():.1f}")
    print(f"Avg degree (legit):  {deg[legit_mask].mean():.1f}")

    # Connected component analysis (approximate via sampling)
    print(f"\nIsolated nodes: {(deg == 0).sum().item():,}")
    print(f"Nodes with >100 connections: {(deg > 100).sum().item():,}")


def plot_training_curves(train_losses, val_aucs):
    """Plot training curves (optional, requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(train_losses)
        ax1.set_title("Training Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax2.plot(val_aucs)
        ax2.set_title("Validation AUC")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC-ROC")

        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150)
        print("Training curves saved to training_curves.png")
    except ImportError:
        print("matplotlib not available, skipping plot")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  GNN Fraud Detection — IEEE-CIS Kaggle Competition         ║
    ║                                                              ║
    ║  Requirements:                                               ║
    ║    pip install torch torch_geometric pandas numpy sklearn     ║
    ║                                                              ║
    ║  Data:                                                       ║
    ║    Download from kaggle and place in ./data/                 ║
    ║    - train_transaction.csv, train_identity.csv               ║
    ║    - test_transaction.csv, test_identity.csv                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Check if data exists
    if not os.path.exists("data/train_transaction.csv"):
        print("⚠️  Data not found in ./data/ directory.")
        print("   Please download from: https://www.kaggle.com/c/ieee-fraud-detection/data")
        print("   and place CSV files in a 'data' subfolder.")
        print("\n   Running in demo mode with synthetic data...\n")

        # --- Demo with synthetic data ---
        print("Creating synthetic dataset for demonstration...")
        np.random.seed(42)
        n_nodes = 5000
        n_features = 50
        fraud_rate = 0.035

        X = np.random.randn(n_nodes, n_features).astype(np.float32)
        y = np.random.binomial(1, fraud_rate, n_nodes).astype(np.float32)

        # Make fraud nodes slightly different
        fraud_idx = np.where(y == 1)[0]
        X[fraud_idx] += np.random.randn(len(fraud_idx), n_features) * 0.5

        # Create edges (random graph for demo)
        n_edges = n_nodes * 10
        src = np.random.randint(0, n_nodes, n_edges)
        dst = np.random.randint(0, n_nodes, n_edges)
        # Add homophilic edges for fraud nodes
        for _ in range(len(fraud_idx) * 5):
            i, j = np.random.choice(fraud_idx, 2, replace=False)
            src = np.append(src, [i, j])
            dst = np.append(dst, [j, i])

        edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

        from torch_geometric.data import Data

        data = Data(
            x=torch.tensor(X),
            y=torch.tensor(y),
            edge_index=edge_index,
        )

        # Masks
        perm = torch.randperm(n_nodes)
        train_size = int(0.7 * n_nodes)
        val_size = int(0.15 * n_nodes)

        data.actual_train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        data.actual_train_mask[perm[:train_size]] = True
        data.val_mask[perm[train_size:train_size + val_size]] = True
        data.test_mask[perm[train_size + val_size:]] = True

        config = Config()
        config.NUM_EPOCHS = 20
        config.BATCH_SIZE = 512

        model = FraudGNN(
            in_channels=n_features,
            hidden_channels=64,
            num_layers=2,
            dropout=0.3,
        )
        print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
        model = train_model(model, data, config)
        print("\n✅ Demo complete!")
    else:
        model, data, submission = main()
