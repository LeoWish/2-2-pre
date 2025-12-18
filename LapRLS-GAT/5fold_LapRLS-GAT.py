import torch
import torch.nn.functional as F
from torch.nn import Linear, BCEWithLogitsLoss, LayerNorm, Dropout
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed(17)


excel_path = './dataset/data.xlsx'
xls_path = './dataset/LapRLS_Pre.xlsx'

kegg_df = pd.read_excel(excel_path, sheet_name="KEGG").rename(columns={"Unnamed: 0": "strain"}).set_index("strain")
gpa_df = pd.read_excel(excel_path, sheet_name="GPA").rename(columns={"Unnamed: 0": "strain"}).set_index("strain")
combo_df = pd.read_excel(excel_path, sheet_name="combination")

features = pd.concat([
    kegg_df.drop(columns=["type"]),
    gpa_df.drop(columns=["type"])
], axis=1)

label_map = {"interaction": 1, "non-interaction": 0}

pred_df = pd.read_excel(xls_path, sheet_name="prediction")
gs_df = pd.read_excel(xls_path, sheet_name="gold_standard")
gs_dict = dict(zip(gs_df["pair"], gs_df["label"]))

min_rank = pred_df['rank'].min()
max_rank = pred_df['rank'].max()

# --- 构图函数 ---
def build_edge_index_by_rules(lb_nodes, st_nodes, strain_names, gs_dict, pred_df):
    edge_list = []
    edge_types = []

    pred_pairs = pred_df['pair'].str.split('&', expand=True)
    pred_pairs.columns = ['Lb', 'St']
    pred_df = pred_df.join(pred_pairs)

    for lb in lb_nodes:
        for st in st_nodes:
            pair_key = f"{lb}&{st}"
            src = strain_names.index(lb)
            dst = strain_names.index(st)

            if pair_key in gs_dict:
                edge_list.append((src, dst))
                edge_types.append([float(gs_dict[pair_key])])
            else:
                row = pred_df[(pred_df['Lb'] == lb) & (pred_df['St'] == st)]
                if not row.empty:
                    rank = row['rank'].values[0]
                    score = 1.0 - (rank - min_rank) / (max_rank - min_rank)
                    edge_list.append((src, dst))
                    edge_types.append([score])

    edge_list += [(dst, src) for src, dst in edge_list]
    edge_types += edge_types

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0, 1), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        edge_type = torch.tensor(edge_types, dtype=torch.float)

    return edge_index, edge_type


train_graphs, test_graphs = [], []

for _, row in combo_df.iterrows():
    strain_names = [row["Lb1"], row["Lb2"], row["St1"], row["St2"]]
    try:
        x = torch.tensor(features.loc[strain_names].values, dtype=torch.float)
    except KeyError:
        continue
    y = torch.tensor([label_map[row["Label"]]], dtype=torch.float)

    lb_nodes = strain_names[:2]
    st_nodes = strain_names[2:]
    edge_index, edge_attr = build_edge_index_by_rules(lb_nodes, st_nodes, strain_names, gs_dict, pred_df)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    if row["Data_split"] == "train":
        train_graphs.append(data)
    else:
        test_graphs.append(data)

print(f"训练图数: {len(train_graphs)}, 测试图数: {len(test_graphs)}")


class EdgeAwareGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.lin_in = Linear(input_dim, hidden_dim * 2)
        self.gat1 = GATConv(hidden_dim * 2, hidden_dim, heads=2, concat=True, edge_dim=1)
        self.norm1 = LayerNorm(hidden_dim * 2)
        self.dropout = Dropout(0.3)

        self.gat2 = GATConv(hidden_dim * 2, hidden_dim, heads=1, concat=False, edge_dim=1)
        self.lin_residual = Linear(hidden_dim * 2, hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)

        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, edge_dim=1)
        self.norm3 = LayerNorm(hidden_dim)

        self.lin_out = Linear(hidden_dim, 1)

        self.gold_pos_boost = 2.0
        self.gold_neg_boost = -2.0

    def forward(self, x, edge_index, edge_attr, batch):
        if edge_index.size(1) == 0:
            x = self.lin_in(x)
            x = self.lin_residual(x)
            x = global_mean_pool(x, batch)
            return self.lin_out(x).squeeze(1)

        if self.training:
            gold_pos = (edge_attr == 1.0)
            gold_neg = (edge_attr == 0.0)
            edge_attr = torch.where(
                gold_pos,
                edge_attr * self.gold_pos_boost,
                torch.where(
                    gold_neg,
                    torch.full_like(edge_attr, self.gold_neg_boost),
                    edge_attr
                )
            )

        x0 = self.lin_in(x)
        x1 = F.elu(self.gat1(x0, edge_index, edge_attr))
        x = self.norm1(x0 + self.dropout(x1))

        x2 = F.elu(self.gat2(x, edge_index, edge_attr))
        x_res = self.lin_residual(x)
        x = self.norm2(x_res + self.dropout(x2))

        x3 = F.elu(self.gat3(x, edge_index, edge_attr))
        x = self.norm3(x + self.dropout(x3))

        x = global_mean_pool(x, batch)
        return self.lin_out(x).squeeze(1)

# --- 评估函数 ---
def evaluate(loader):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.edge_attr, data.batch)
            ys.append(data.y.cpu())
            preds.append(logits.cpu())

    y_true = torch.cat(ys).numpy()
    y_score = torch.sigmoid(torch.cat(preds)).numpy()
    y_pred = (y_score >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    metrics = {
        "AUC": auc, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-score": f1, "Specificity": specificity,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    }

    return metrics, y_true, y_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


all_graphs = train_graphs
all_labels = [int(data.y.item()) for data in all_graphs]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []

print(f"\n🔁 Starting 5-fold cross-validation on training data only, total samples: {len(all_graphs)}")

for fold, (train_idx, val_idx) in enumerate(skf.split(all_graphs, all_labels), 1):
    print(f"\n📂 Fold {fold} -----------------------------")

    train_dataset = [all_graphs[i] for i in train_idx]
    val_dataset = [all_graphs[i] for i in val_idx]

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = EdgeAwareGAT(input_dim=features.shape[1]).to(device)

    num_pos = sum([data.y.item() for data in train_dataset])
    num_neg = len(train_dataset) - num_pos
    pos_weight = torch.tensor(num_neg / max(num_pos, 1), dtype=torch.float).to(device)

    loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        scheduler.step()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {total_loss / len(train_loader.dataset):.4f}")

    metrics, _, _ = evaluate(val_loader)
    fold_auc = metrics["AUC"]
    fold_aucs.append(fold_auc)
    print(f"✅ Fold {fold} AUC: {fold_auc:.4f}")


print("\n📊 Cross-Validation Results:")
for i, auc in enumerate(fold_aucs, 1):
    print(f"Fold {i}: AUC = {auc:.4f}")

mean_auc = np.mean(fold_aucs)
std_auc = np.std(fold_aucs)
print(f"\n🎯 Average AUC over 5 folds (train set only): {mean_auc:.4f} ± {std_auc:.4f}")
