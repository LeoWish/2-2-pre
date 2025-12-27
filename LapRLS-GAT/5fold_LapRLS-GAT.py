import torch
import torch.nn.functional as F
from torch.nn import Linear, BCEWithLogitsLoss, LayerNorm, Dropout, Sequential, ReLU
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. GLOBAL CONFIGURATION
# Ensures reproducibility by fixing seeds for random number generators across libraries.
# =============================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(17)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 2. MODEL ARCHITECTURE (Prot-GTN)
# Prot-GTN: Prior-informed Role-oriented Transformer Graph Training Network.
# This model uses Transformer-based Graph Convolutions to process microbial
# interaction graphs, where edges are weighted by prior biological knowledge.
# =============================================================================
class ProtGTN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=6):
        super(ProtGTN, self).__init__()

        # --- GRAPH CONVOLUTIONAL BACKBONE ---
        # TransformerConv applies multi-head attention over the graph.
        # It treats edge_attr (prior ranks/labels) as a relationship strength indicator.
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads=4, edge_dim=1, concat=True)
        self.norm1 = LayerNorm(hidden_dim * 4)  # Adjust for 4 heads concatenation

        self.conv2 = TransformerConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=1, concat=False)
        self.norm2 = LayerNorm(hidden_dim)

        # --- DUAL-PATH ADAPTIVE HEADS ---
        # Fallback MLP: Used when a graph has no edges (e.g., isolated nodes).
        # It processes raw node features directly after global pooling.
        self.fallback_mlp = Sequential(
            Linear(input_dim * 2, hidden_dim),
            ReLU(),
            Dropout(0.4),
            Linear(hidden_dim, 1)
        )

        # Primary MLP: Processes features extracted by the Graph Convolutional layers.
        # Input dimension is hidden_dim * 2 because of concatenated Mean and Max pooling.
        self.mlp = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(),
            Dropout(0.4),
            Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. STRUCTURAL VALIDATION
        # If no edges exist in the batch, skip GNN and use the fallback path
        # to prevent matrix multiplication errors in convolutional layers.
        if edge_index.size(1) == 0:
            # Global Pooling: Concatenate Mean and Max pooling to capture both
            # average distribution and peak feature signals.
            x_pool = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
            return self.fallback_mlp(x_pool).squeeze(1)

        # 2. FEATURE PROPAGATION (GNN Path)
        # Apply GNN layers with ELU activation and Layer Normalization.
        h = F.elu(self.conv1(x, edge_index, edge_attr))
        h = self.norm1(h)
        h = F.elu(self.conv2(h, edge_index, edge_attr))
        h = self.norm2(h)

        # 3. GRAPH-LEVEL REPRESENTATION
        # Aggregate node features into a single graph-level vector.
        out = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=1)

        # 4. PREDICTION
        return self.mlp(out).squeeze(1)


# =============================================================================
# 3. UTILITIES & DATA MANAGEMENT
# =============================================================================
class EarlyStopping:
    """Monitors validation loss to stop training early and prevent overfitting."""

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def build_graph_edges(lb_nodes, st_nodes, strain_names, gs_dict, pred_df, min_rank, max_rank):
    """
    Constructs graph edges and assigns weights based on:
    1. Gold standard labels (if available)
    2. Normalized prior prediction ranks (0 to 1 scale)
    """
    edge_list, edge_types = [], []

    # Pre-process prediction dataframe for faster lookup
    if not pred_df.empty and 'Lb' not in pred_df.columns:
        pred_pairs = pred_df['pair'].str.split('&', expand=True)
        pred_pairs.columns = ['Lb', 'St']
        pred_df = pred_df.join(pred_pairs)

    for lb in lb_nodes:
        for st in st_nodes:
            pair_key = f"{lb}&{st}"
            src, dst = strain_names.index(lb), strain_names.index(st)

            # Scenario A: Pair exists in Gold Standard
            if pair_key in gs_dict:
                edge_list.append((src, dst))
                edge_types.append([float(gs_dict[pair_key])])
            # Scenario B: Pair exists in Prior Predictions (Map rank to a [0,1] score)
            elif not pred_df.empty:
                row_p = pred_df[(pred_df['Lb'] == lb) & (pred_df['St'] == st)]
                if not row_p.empty:
                    rank = row_p['rank'].values[0]
                    score = 1.0 - (rank - min_rank) / (max_rank - min_rank)
                    edge_list.append((src, dst))
                    edge_types.append([score])

    # Convert to undirected graph by adding reverse edges
    edge_list += [(dst, src) for src, dst in edge_list]
    edge_types += edge_types

    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
    return torch.tensor(edge_list, dtype=torch.long).T, torch.tensor(edge_types, dtype=torch.float)


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
def main():
    # --- DATA PREPARATION ---
    # Load and standardize node features
    df = pd.read_excel('data/dataset.xlsx', sheet_name="Sheet1")
    feature_cols = df.columns[7:]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # --- PRIOR KNOWLEDGE INTEGRATION ---
    try:
        xls_path = './dataset/LapRLS_Pre.xlsx'
        pred_df = pd.read_excel(xls_path, sheet_name="prediction")
        gs_df = pd.read_excel(xls_path, sheet_name="gold_standard")
        gs_dict = dict(zip(gs_df["pair"], gs_df["label"]))
        min_rank, max_rank = pred_df['rank'].min(), pred_df['rank'].max()
    except:
        # Fallback if prior files are missing
        gs_dict, pred_df, min_rank, max_rank = {}, pd.DataFrame(), 0, 0

    # --- GRAPH CONSTRUCTIONS ---
    all_graphs = []
    for idx, row in df.iterrows():
        strains = [row["Lb1"], row["Lb2"], row["St1"], row["St2"]]
        raw_feat = row[feature_cols].values.astype(float)

        # Node Role Encoding: Lb nodes assigned 0.0, St nodes assigned 1.0.
        # This helps the model distinguish species types in the interaction network.
        x = torch.tensor([np.append(raw_feat, [0.0 if i < 2 else 1.0]) for i in range(4)], dtype=torch.float)

        e_idx, e_attr = build_graph_edges(strains[:2], strains[2:], strains, gs_dict, pred_df, min_rank, max_rank)
        all_graphs.append(
            Data(x=x, edge_index=e_idx, edge_attr=e_attr, y=torch.tensor([row["Label"]], dtype=torch.float),
                 sample_id=idx))  # sample_id stores the original row index

    # --- EVALUATION: 5-FOLD CROSS-VALIDATION ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = [int(g.y.item()) for g in all_graphs]

    fold_auroc, fold_aupr = [], []
    all_fold_results = []  # Collect complete results for all samples across folds

    print(f"Starting Training on {len(all_graphs)} samples...")

    for fold, (t_idx, v_idx) in enumerate(skf.split(all_graphs, labels), 1):
        t_loader = DataLoader([all_graphs[i] for i in t_idx], batch_size=8, shuffle=True)
        v_loader = DataLoader([all_graphs[i] for i in v_idx], batch_size=8, shuffle=False)

        model = ProtGTN(input_dim=len(feature_cols) + 1).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=15)

        pos_c = sum(labels[i] for i in t_idx)
        pos_w = torch.tensor((len(t_idx) - pos_c) / max(pos_c, 1), dtype=torch.float).to(DEVICE)
        criterion = BCEWithLogitsLoss(pos_weight=pos_w)

        # --- TRAINING LOOP ---
        for epoch in range(1, 201):
            model.train()
            for data in t_loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()

            # --- VALIDATION LOOP ---
            model.eval()
            val_l = 0
            with torch.no_grad():
                for data in v_loader:
                    data = data.to(DEVICE)
                    v_out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    val_l += criterion(v_out, data.y).item()

            scheduler.step(val_l)
            early_stopping(val_l)
            if early_stopping.early_stop: break

        # --- EVALUATION (Collect results with fold and sample_index) ---
        model.eval()
        y_true_fold, y_score_fold, sample_ids_fold = [], [], []

        with torch.no_grad():
            for data in v_loader:
                data = data.to(DEVICE)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                y_true_fold.append(data.y.cpu())
                y_score_fold.append(torch.sigmoid(out).cpu())
                sample_ids_fold.append(data.sample_id.cpu())  # Original index

        y_true_fold = torch.cat(y_true_fold).numpy()
        y_score_fold = torch.cat(y_score_fold).numpy()
        sample_ids_fold = torch.cat(sample_ids_fold).numpy()

        f_auc = roc_auc_score(y_true_fold, y_score_fold)
        f_pr = average_precision_score(y_true_fold, y_score_fold)
        fold_auroc.append(f_auc)
        fold_aupr.append(f_pr)

        # Save results for the current fold
        for i in range(len(y_true_fold)):
            all_fold_results.append({
                'fold': fold,
                'sample_index': int(sample_ids_fold[i]),
                'true_label': int(y_true_fold[i]),
                'prediction_score': float(y_score_fold[i])
            })

        print(f"Fold {fold:02d} | AUROC: {f_auc:.4f} | AUPR: {f_pr:.4f}")

    # --- FINAL PERFORMANCE REPORTING ---
    res_df = pd.DataFrame(all_fold_results)

    pooled_auroc = roc_auc_score(res_df['true_label'], res_df['prediction_score'])
    pooled_aupr = average_precision_score(res_df['true_label'], res_df['prediction_score'])

    print("\n" + "=" * 40)
    print(f"Mean AUROC: {np.mean(fold_auroc):.3f} (±{np.std(fold_auroc):.3f})")
    print(f"Mean AUPR: {np.mean(fold_aupr):.3f} (±{np.std(fold_aupr):.3f})")
    print(f"Final Pooled CV-AUROC: {pooled_auroc:.5f}")
    print(f"Final Pooled CV-AUPR: {pooled_aupr:.5f}")
    print("=" * 40)

    res_df = res_df[['fold', 'sample_index', 'true_label', 'prediction_score']]
    res_df = res_df.sort_values(['fold', 'sample_index']).reset_index(drop=True)
    res_df['prediction_score'] = res_df['prediction_score'].map('{:.5f}'.format)
    res_df.to_csv('prot_gtn_cv_results.csv', index=False)

    print("Results saved to 'prot_gtn_cv_results.csv' (prediction_score fixed to 5 decimal places)")


if __name__ == "__main__":
    main()
