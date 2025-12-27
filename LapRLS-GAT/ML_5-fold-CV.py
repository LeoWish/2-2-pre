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


# --- 1. Global Configuration ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(17)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 2. Early Stopping Utility ---
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
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


# --- 3. Data Loading and Preprocessing ---
dataset_path = 'data/dataset.xlsx'
xls_path = './dataset/LapRLS_Pre.xlsx'

df = pd.read_excel(dataset_path, sheet_name="Sheet1")
feature_cols = df.columns[7:]

# Fit scaler on all features once for consistency
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

try:
    pred_df = pd.read_excel(xls_path, sheet_name="prediction")
    gs_df = pd.read_excel(xls_path, sheet_name="gold_standard")
    gs_dict = dict(zip(gs_df["pair"], gs_df["label"]))
    min_rank, max_rank = pred_df['rank'].min(), pred_df['rank'].max()
except FileNotFoundError:
    gs_dict, pred_df = {}, pd.DataFrame()


# --- 4. Optimized Graph Construction ---
def build_graph_edges(lb_nodes, st_nodes, strain_names, gs_dict, pred_df):
    edge_list, edge_types = [], []
    if not pred_df.empty and 'Lb' not in pred_df.columns:
        pred_pairs = pred_df['pair'].str.split('&', expand=True)
        pred_pairs.columns = ['Lb', 'St']
        pred_df = pred_df.join(pred_pairs)

    for lb in lb_nodes:
        for st in st_nodes:
            pair_key = f"{lb}&{st}"
            src, dst = strain_names.index(lb), strain_names.index(st)
            if pair_key in gs_dict:
                edge_list.append((src, dst))
                edge_types.append([float(gs_dict[pair_key])])
            elif not pred_df.empty:
                row_p = pred_df[(pred_df['Lb'] == lb) & (pred_df['St'] == st)]
                if not row_p.empty:
                    rank = row_p['rank'].values[0]
                    score = 1.0 - (rank - min_rank) / (max_rank - min_rank)
                    edge_list.append((src, dst))
                    edge_types.append([score])

    edge_list += [(dst, src) for src, dst in edge_list]
    edge_types += edge_types

    if len(edge_list) == 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)
    return torch.tensor(edge_list, dtype=torch.long).T, torch.tensor(edge_types, dtype=torch.float)


all_graphs = []
for idx, row in df.iterrows():
    strain_names = [row["Lb1"], row["Lb2"], row["St1"], row["St2"]]
    raw_features = row[feature_cols].values.astype(float)

    # OPTIMIZATION: Node Role Encoding (Lb=0, St=1) to help model differentiate roles
    x = []
    for i in range(4):
        role_bit = 0.0 if i < 2 else 1.0
        x.append(np.append(raw_features, [role_bit]))

    x = torch.tensor(np.array(x), dtype=torch.float)
    y = torch.tensor([row["Label"]], dtype=torch.float)

    lb_nodes, st_nodes = strain_names[:2], strain_names[2:]
    edge_index, edge_attr = build_graph_edges(lb_nodes, st_nodes, strain_names, gs_dict, pred_df)

    all_graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, sample_id=idx))


# --- 5. Improved Model Architecture (Transformer-based GNN) ---
class OptimizedGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        # Using TransformerConv for better multi-head attention over edge attributes
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads=4, edge_dim=1, concat=True)
        self.norm1 = LayerNorm(hidden_dim * 4)
        self.conv2 = TransformerConv(hidden_dim * 4, hidden_dim, heads=1, edge_dim=1, concat=False)
        self.norm2 = LayerNorm(hidden_dim)

        # Post-pooling MLP with Skip Connection from raw feature aggregation
        self.mlp = Sequential(
            Linear(hidden_dim * 2, hidden_dim),  # Combining Mean and Max pooling
            ReLU(),
            Dropout(0.4),
            Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_index.size(1) == 0:
            # Fallback for isolated graphs
            x_pool = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
            return self.mlp(x_pool).squeeze(1)

        # Graph Convolutional layers
        h1 = F.elu(self.conv1(x, edge_index, edge_attr))
        h1 = self.norm1(h1)
        h2 = F.elu(self.conv2(h1, edge_index, edge_attr))
        h2 = self.norm2(h2)

        # Global Information Aggregation (Mean + Max pooling)
        out = torch.cat([global_mean_pool(h2, batch), global_max_pool(h2, batch)], dim=1)
        return self.mlp(out).squeeze(1)


# --- 6. Training and Evaluation Functions ---
def get_predictions(loader, model):
    model.eval()
    all_y, all_scores, all_ids = [], [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            all_y.append(data.y.cpu())
            all_scores.append(torch.sigmoid(out).cpu())
            all_ids.append(data.sample_id.cpu())
    return torch.cat(all_y).numpy(), torch.cat(all_scores).numpy(), torch.cat(all_ids).numpy()


# --- 7. Execution Loop with Early Stopping & LR Scheduling ---
all_labels = [int(data.y.item()) for data in all_graphs]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
detailed_results, fold_metrics = [], []

print(f"Starting Optimized CV Training...")

for fold, (train_idx, val_idx) in enumerate(skf.split(all_graphs, all_labels), 1):
    train_loader = DataLoader([all_graphs[i] for i in train_idx], batch_size=8, shuffle=True)
    val_loader = DataLoader([all_graphs[i] for i in val_idx], batch_size=8, shuffle=False)

    model = OptimizedGNN(input_dim=len(feature_cols) + 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # OPTIMIZATION: Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=12)

    num_pos = sum([all_labels[i] for i in train_idx])
    pos_weight = torch.tensor((len(train_idx) - num_pos) / max(num_pos, 1), dtype=torch.float).to(device)
    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)

    # Extended Epoch range with Early Stopping
    for epoch in range(1, 150):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation for Scheduler and Early Stopping
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                v_out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                val_loss += loss_fn(v_out, data.y).item()

        scheduler.step(val_loss)
        early_stopping(val_loss)
        if early_stopping.early_stop:
            break

    y_true, y_score, ids = get_predictions(val_loader, model)
    f_auroc, f_aupr = roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)
    fold_metrics.append((f_auroc, f_aupr))

    for i in range(len(y_true)):
        detailed_results.append({
            'fold': fold, 'sample_index': int(ids[i]),
            'true_label': int(y_true[i]), 'prediction_score': round(float(y_score[i]), 5)
        })
    print(f"Fold {fold:02d} | AUROC: {f_auroc:.4f} | AUPR: {f_aupr:.4f} | Stop Epoch: {epoch}")

# --- 8. Reporting ---
res_df = pd.DataFrame(detailed_results)
auroc_vals, aupr_vals = [m[0] for m in fold_metrics], [m[1] for m in fold_metrics]
pooled_auroc = roc_auc_score(res_df['true_label'], res_df['prediction_score'])
pooled_aupr = average_precision_score(res_df['true_label'], res_df['prediction_score'])

print("\n" + "=" * 50)
print(f"Mean AUROC: {np.mean(auroc_vals):.3f} (±{np.std(auroc_vals):.3f})")
print(f"Mean AUPR:  {np.mean(aupr_vals):.3f} (±{np.std(aupr_vals):.3f})")
print("-" * 50)
print(f"Pooled CV-AUROC: {pooled_auroc:.5f} | Pooled CV-AUPR: {pooled_aupr:.5f}")
print("=" * 50)

res_df.to_csv('optimized_cv_predictions.csv', index=False)