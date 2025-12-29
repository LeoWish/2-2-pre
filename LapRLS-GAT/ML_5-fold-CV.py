import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# --- 1. Global Configuration ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(2)

# --- 2. Data Loading and Preparation ---
dataset_path = 'data/dataset.xlsx'
df = pd.read_excel(dataset_path, sheet_name="Sheet1")

# Features start from column H (index 7)
feature_cols = df.columns[7:]
X = df[feature_cols].values
y = df['Label'].values
sample_indices = df.index.values

# --- 3. Model Definitions ---
models = {
    "Logistic_Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Support_Vector_Machine": Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(probability=True, random_state=42))
    ]),
    "Random_Forest": Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(random_state=42))
    ]),
    "K_Nearest_Neighbors": Pipeline([
        ('scaler', StandardScaler()),
        ('model', KNeighborsClassifier())
    ]),
    "Naive_Bayes": Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianNB())
    ]),
    "XGBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBClassifier(random_state=42, eval_metric='logloss'))
    ]),
    "LightGBM": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LGBMClassifier(random_state=1, verbose=-1))
    ]),
    "MLP": Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPClassifier(hidden_layer_sizes=(32, 16, 8), max_iter=1000, random_state=42))
    ]),
    "CatBoost": Pipeline([
        ('scaler', StandardScaler()),
        ('model', CatBoostClassifier(verbose=0, random_seed=42))
    ]),
    "Gradient_Boosting": Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingClassifier(random_state=42))
    ])
}

# --- 4. Cross-Validation Loop ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)  
final_summary = []
all_detailed_predictions = []

print(f"Starting comparison of {len(models)} models using 5-Fold CV...")

for model_name, pipeline in models.items():
    print(f"Evaluating: {model_name}")

    fold_aurocs = []
    fold_auprs = []
    model_predictions = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        val_orig_indices = sample_indices[val_idx]

        # Train model
        pipeline.fit(X_train, y_train)

        # Get probability for the positive class
        y_scores = pipeline.predict_proba(X_val)[:, 1]

        # Calculate fold metrics
        f_auroc = roc_auc_score(y_val, y_scores)
        f_aupr = average_precision_score(y_val, y_scores)
        fold_aurocs.append(f_auroc)
        fold_auprs.append(f_aupr)

        # Store detailed sample predictions
        for i in range(len(y_val)):
            all_detailed_predictions.append({
                'model': model_name,
                'fold': fold,
                'sample_index': int(val_orig_indices[i]),
                'true_label': int(y_val[i]),
                'prediction_score': round(float(y_scores[i]), 5)
            })

    # Calculate Pooled CV metrics for this model
    model_df = pd.DataFrame([p for p in all_detailed_predictions if p['model'] == model_name])
    pooled_auroc = roc_auc_score(model_df['true_label'], model_df['prediction_score'])
    pooled_aupr = average_precision_score(model_df['true_label'], model_df['prediction_score'])

    # Record summary statistics
    final_summary.append({
        'Model': model_name,
        'Mean_AUROC': round(np.mean(fold_aurocs), 3),
        'Std_AUROC': round(np.std(fold_aurocs), 3),
        'Mean_AUPR': round(np.mean(fold_auprs), 3),
        'Std_AUPR': round(np.std(fold_auprs), 3),
        'Pooled_CV_AUROC': round(pooled_auroc, 5),
        'Pooled_CV_AUPR': round(pooled_aupr, 5)
    })

# --- 5. Output and Save Results ---
# Save detailed predictions for all models
detailed_df = pd.DataFrame(all_detailed_predictions)
detailed_df.to_csv('all_models_sample_predictions.csv', index=False)

# Save summary performance
summary_df = pd.DataFrame(final_summary)
summary_df.to_csv('models_performance_summary.csv', index=False)

# Display Summary Table
print("\n" + "=" * 85)
print(f"{'Model Name':<25} | {'Mean AUROC':<12} | {'Mean AUPR':<12} | {'CV-AUROC':<12}")
print("-" * 85)
for item in final_summary:
    print(
        f"{item['Model']:<25} | {item['Mean_AUROC']:<12.3f} | {item['Mean_AUPR']:<12.3f} | {item['Pooled_CV_AUROC']:<12.5f}")
print("=" * 85)

print(f"\nDetailed predictions saved to 'all_models_sample_predictions.csv'")
print(f"Summary performance saved to 'models_performance_summary.csv'")
