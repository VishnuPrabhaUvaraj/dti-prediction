"""
notebooks/06_comparison.py
Compare Random Forest vs GNN
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, torch, sys, os
sys.path.append('.')
from sklearn.metrics import roc_curve, roc_auc_score
from src.fingerprints import generate_fingerprints
from src.mol_graph import smiles_to_graph
from src.gnn_model import DTI_GNN
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from tqdm import tqdm

print("="*50)
print("PHASE 4: Model Comparison")
print("="*50)

df = pd.read_csv('data/processed/egfr_clean.csv')
print(f"Dataset: {len(df)} compounds")

# ── Load RF model and get predictions ───────────────
print("\nLoading Random Forest...")
rf = joblib.load('models/rf_model.pkl')
X, valid_idx = generate_fingerprints(
    df['canonical_smiles'].tolist())
y = df['label'].iloc[valid_idx].values
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2,
    random_state=42, stratify=y)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
print(f"RF AUC: {rf_auc:.4f}")

# ── Load GNN model and get predictions ──────────────
print("\nLoading GNN...")
print("Converting molecules to graphs...")
graphs = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    g = smiles_to_graph(
        row['canonical_smiles'],
        label=int(row['label'])
    )
    if g is not None:
        graphs.append(g)

labels_all = [g.y.item() for g in graphs]
_, test_idx = train_test_split(
    range(len(graphs)), test_size=0.2,
    random_state=42, stratify=labels_all)
test_data = [graphs[i] for i in test_idx]
test_loader = DataLoader(
    test_data, batch_size=64, shuffle=False)

gnn = DTI_GNN(node_features=15, hidden=128)
gnn.load_state_dict(
    torch.load('models/gnn_best.pth'))
gnn.eval()

gnn_preds, gnn_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        out = gnn(batch.x, batch.edge_index,
                  batch.batch)
        prob = F.softmax(out, dim=1)[:, 1]
        gnn_preds.extend(prob.numpy())
        gnn_labels.extend(batch.y.numpy())

gnn_auc = roc_auc_score(gnn_labels, gnn_preds)
print(f"GNN AUC: {gnn_auc:.4f}")

# ── Figure 1: ROC curve comparison ──────────────────
print("\nGenerating comparison figures...")
os.makedirs('figures', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fpr_rf,  tpr_rf,  _ = roc_curve(y_test, rf_probs)
fpr_gnn, tpr_gnn, _ = roc_curve(gnn_labels, gnn_preds)

axes[0].plot(fpr_rf, tpr_rf,
    color='#7F77DD', linewidth=2,
    label=f'Random Forest (AUC={rf_auc:.3f})')
axes[0].plot(fpr_gnn, tpr_gnn,
    color='#1D9E75', linewidth=2,
    label=f'GNN (AUC={gnn_auc:.3f})')
axes[0].plot([0,1],[0,1],
    'k--', linewidth=1, label='Random baseline')
axes[0].fill_between(fpr_rf, tpr_rf,
    alpha=0.1, color='#7F77DD')
axes[0].fill_between(fpr_gnn, tpr_gnn,
    alpha=0.1, color='#1D9E75')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve Comparison')
axes[0].legend(loc='lower right')
axes[0].grid(True, alpha=0.3)

# ── Figure 2: Bar chart comparison ──────────────────
models = ['Random Forest', 'GNN']
aucs   = [rf_auc, gnn_auc]
colors = ['#7F77DD', '#1D9E75']
bars = axes[1].bar(models, aucs,
    color=colors, width=0.4,
    edgecolor='white', linewidth=0.5)
axes[1].set_ylim(0.7, 1.0)
axes[1].set_ylabel('ROC-AUC')
axes[1].set_title('Model AUC Comparison')
axes[1].grid(True, alpha=0.3, axis='y')
for bar, auc in zip(bars, aucs):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f'{auc:.4f}',
        ha='center', fontsize=13, fontweight='bold'
    )

plt.suptitle('DTI Prediction: RF vs GNN on EGFR',
             fontsize=14)
plt.tight_layout()
plt.savefig('figures/model_comparison.png',
    dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/model_comparison.png")

# ── Summary table ────────────────────────────────────
winner_rf  = "WINNER" if rf_auc  > gnn_auc else ""
winner_gnn = "WINNER" if gnn_auc >= rf_auc  else ""
improvement = (gnn_auc - rf_auc) / rf_auc * 100

print("\n" + "="*50)
print(f"{'Model':<20} {'AUC':>10} {'Result':>15}")
print("="*50)
print(f"{'Random Forest':<20} {rf_auc:>10.4f} {winner_rf:>15}")
print(f"{'GNN':<20} {gnn_auc:>10.4f} {winner_gnn:>15}")
print("="*50)
print(f"Difference: {improvement:+.2f}%")

# ── Save comparison CSV ──────────────────────────────
os.makedirs('logs', exist_ok=True)
comparison = pd.DataFrame([
    {
        'model': 'Random Forest',
        'auc': round(rf_auc, 4),
        'approach': 'Morgan Fingerprints',
        'winner': rf_auc > gnn_auc
    },
    {
        'model': 'GNN',
        'auc': round(gnn_auc, 4),
        'approach': 'Graph Neural Network',
        'winner': gnn_auc >= rf_auc
    }
])
comparison.to_csv('logs/comparison_results.csv',
                  index=False)
print("Saved: logs/comparison_results.csv")
print("\nComparison complete!")
