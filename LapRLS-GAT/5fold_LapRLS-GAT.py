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

GS_WEIGHT = 1.5  # <<< 轻度放大 Gold Standard 边权


# =============================================================================
# 2. MODEL ARCHITECTURE (Prot-GTN)
# =============================================================================
class ProtGTN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=6):
        super(ProtGTN, self).__init__()

        # --- GRAPH CONVOLUTIONAL BACKBONE ---
        self.conv1 = TransformerConv(
            input_dim,
            hidden_dim,
            heads=4,
            edge_dim=2,     # <<< edge_attr = [score, is_gold]
            concat=True
        )
        self.norm1 = LayerNorm(hidden_dim * 4)

        self.conv2 = TransformerConv(
            hidden_dim * 4,
            hidden_dim,
            heads=1,
            edge_dim=2,
            concat=False
        )
        self.norm2 = LayerNorm(hidden_dim)

        # --- FALLBACK MLP (NO EDGE CASE) ---
        self.fallback_mlp = Sequential(
            Linear(input_dim * 2, hidden_dim),
            ReLU(),
            Dropout(0.4),
            Linear(hidden_dim, 1)
        )

        # --- PRIMARY MLP ---
        self.mlp = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(),
            Dropout(0.4),
            Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_index.size(1) == 0:
            x_pool = torch.cat(
                [global_mean_pool(x, batch), global_max_pool(x, batch)],
                dim=1
            )
            return self.fallback_mlp(x_pool).squeeze(1)

        h = F.elu(self.conv1(x, edge_index, edge_attr))
        h = self.norm1(h)
        h = F.elu(self.conv2(h, edge_index, edge_attr))
        h = self.norm2(h)

        out = torch.cat(
            [global_mean_pool(h, batch), global_max_pool(h, batch)],
            dim=1
        )
        return self.mlp(out).squeeze(1)


# =============================================================================
# 3. UTILITIES
# =============================================================================
class EarlyStopping:
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
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def build_graph_edges(lb_nodes, st_nodes, strain_names,
                      gs_dict, pred_df, min_rank, max_rank):
    """
    edge_attr = [edge_score, is_gold]
    """
    edge_list, edge_types = [], []

    if not pred_df.empty and 'Lb' not in pred_df.columns:
        pred_pairs = pred_df['pair'].str.split('&', expand=True)
        pred_pairs.columns = ['Lb', 'St']
        pred_df = pred_df.join(pred_pairs)

    for lb in lb_nodes:
        for st in st_nodes:
            pair_key = f"{lb}&{st}"
            src, dst = strain_names.index(lb), strain_names.index(st)

            # --- Gold Standard Edge ---
            if pair_key in gs_dict:
                edge_list.append((src, dst))
                edge_types.append([
                    GS_WEIGHT * float(gs_dict[pair_key]),  # score
                    1.0                                     # is_gold
                ])

            # --- Prediction Edge ---
            elif not pred_df.empty:
                row_p = pred_df[(pred_df['Lb'] == lb) & (pred_df['St'] == st)]
                if not row_p.empty:
                    rank = row_p['rank'].values[0]
                    score = 1.0 - (rank - min_rank) / (max_rank - min_rank)
                    edge_list.append((src, dst))
                    edge_types.append([
                        score,
                        0.0
                    ])

    # Undirected graph
    edge_list += [(dst, src) for src, dst in edge_list]
    edge_types += edge_types

    if len(edge_list) == 0:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 2), dtype=torch.float)
        )

    return (
        torch.tensor(edge_list, dtype=torch.long).T,
        torch.tensor(edge_types, dtype=torch.float)
    )


# =============================================================================
# 4. MAIN PIPELINE
# =============================================================================
def main():
    df = pd.read_excel('data/dataset.xlsx', sheet_name="Sheet1")
    feature_cols = df.columns[7:]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    try:
        xls_path = './dataset/LapRLS_Pre.xlsx'
        pred_df = pd.read_excel(xls_path, sheet_name="prediction")
        gs_df = pd.read_excel(xls_path, sheet_name="gold_standard")
        gs_dict = dict(zip(gs_df["pair"], gs_df["label"]))
        min_rank, max_rank = pred_df['rank'].min(), pred_df['rank'].max()
    except:
        gs_dict, pred_df, min_rank, max_rank = {}, pd.DataFrame(), 0, 0

    all_graphs = []
    for idx, row in df.iterrows():
        strains = [row["Lb1"], row["Lb2"], row["St1"], row["St2"]]
        raw_feat = row[feature_cols].values.astype(float)

        x = torch.tensor(
            [np.append(raw_feat, [0.0 if i < 2 else 1.0]) for i in range(4)],
            dtype=torch.float
        )

        e_idx, e_attr = build_graph_edges(
            strains[:2], strains[2:], strains,
            gs_dict, pred_df, min_rank, max_rank
        )

        all_graphs.append(
            Data(
                x=x,
                edge_index=e_idx,
                edge_attr=e_attr,
                y=torch.tensor([row["Label"]], dtype=torch.float),
                sample_id=idx
            )
        )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = [int(g.y.item()) for g in all_graphs]

    fold_auroc, fold_aupr = [], []
    all_fold_results = []

    print(f"Starting Training on {len(all_graphs)} samples...")

    for fold, (t_idx, v_idx) in enumerate(skf.split(all_graphs, labels), 1):
        t_loader = DataLoader([all_graphs[i] for i in t_idx], batch_size=8, shuffle=True)
        v_loader = DataLoader([all_graphs[i] for i in v_idx], batch_size=8, shuffle=False)

        model = ProtGTN(input_dim=len(feature_cols) + 1).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        early_stopping = EarlyStopping(patience=15)

        pos_c = sum(labels[i] for i in t_idx)
        pos_w = torch.tensor((len(t_idx) - pos_c) / max(pos_c, 1),
                             dtype=torch.float).to(DEVICE)
        criterion = BCEWithLogitsLoss(pos_weight=pos_w)

        for epoch in range(1, 201):
            model.train()
            for data in t_loader:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()

            model.eval()
            val_l = 0
            with torch.no_grad():
                for data in v_loader:
                    data = data.to(DEVICE)
                    v_out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    val_l += criterion(v_out, data.y).item()

            scheduler.step(val_l)
            early_stopping(val_l)
            if early_stopping.early_stop:
                break

        model.eval()
        y_true_fold, y_score_fold, sample_ids_fold = [], [], []

        with torch.no_grad():
            for data in v_loader:
                data = data.to(DEVICE)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                y_true_fold.append(data.y.cpu())
                y_score_fold.append(torch.sigmoid(out).cpu())
                sample_ids_fold.append(data.sample_id.cpu())

        y_true_fold = torch.cat(y_true_fold).numpy()
        y_score_fold = torch.cat(y_score_fold).numpy()
        sample_ids_fold = torch.cat(sample_ids_fold).numpy()

        f_auc = roc_auc_score(y_true_fold, y_score_fold)
        f_pr = average_precision_score(y_true_fold, y_score_fold)

        fold_auroc.append(f_auc)
        fold_aupr.append(f_pr)

        for i in range(len(y_true_fold)):
            all_fold_results.append({
                'fold': fold,
                'sample_index': int(sample_ids_fold[i]),
                'true_label': int(y_true_fold[i]),
                'prediction_score': float(y_score_fold[i])
            })

        print(f"Fold {fold:02d} | AUROC: {f_auc:.4f} | AUPR: {f_pr:.4f}")

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
