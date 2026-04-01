"""
notebooks/05_gnn_model.py
Train and evaluate the GNN model
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import sys, os
sys.path.append('.')
from src.mol_graph import smiles_to_graph
from src.gnn_model import DTI_GNN
from tqdm import tqdm

print("="*50)
print("MODEL 2: Graph Neural Network")
print("="*50)

# ── 1. Load data ────────────────────────────────────
df = pd.read_csv('data/processed/egfr_clean.csv')
print(f"Dataset: {len(df)} compounds")
print(f"Active: {df['label'].sum()}, "
      f"Inactive: {(df['label']==0).sum()}")

# ── 2. Convert molecules to graphs ──────────────────
print("\nConverting molecules to graphs...")
print("Each SMILES → graph of atoms and bonds...")
graphs = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    g = smiles_to_graph(
        row['canonical_smiles'],
        label=int(row['label'])
    )
    if g is not None:
        graphs.append(g)
print(f"Valid graphs: {len(graphs)}")

# ── 3. Split data ───────────────────────────────────
labels = [g.y.item() for g in graphs]
train_idx, test_idx = train_test_split(
    range(len(graphs)), test_size=0.2,
    random_state=42, stratify=labels
)
train_data = [graphs[i] for i in train_idx]
test_data  = [graphs[i] for i in test_idx]

train_loader = DataLoader(
    train_data, batch_size=64, shuffle=True)
test_loader  = DataLoader(
    test_data, batch_size=64, shuffle=False)
print(f"Train: {len(train_data)}, Test: {len(test_data)}")

# ── 4. Build model ──────────────────────────────────
device = torch.device('cpu')
model = DTI_GNN(node_features=15,
                hidden=128, dropout=0.3).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=20, gamma=0.5)

params = sum(p.numel() for p in model.parameters()
             if p.requires_grad)
print(f"Model parameters: {params:,}")

# ── 5. Training functions ───────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index,
                    batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds, labels_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index,
                        batch.batch)
            prob = F.softmax(out, dim=1)[:, 1]
            preds.extend(prob.cpu().numpy())
            labels_list.extend(batch.y.cpu().numpy())
    auc = roc_auc_score(labels_list, preds)
    return auc, preds, labels_list

# ── 6. Training loop ────────────────────────────────
print("\nTraining GNN for 50 epochs...")
print("Printing every 10 epochs...")
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

train_aucs, test_aucs = [], []
best_auc = 0
best_epoch = 0

for epoch in range(1, 51):
    loss = train_epoch(model, train_loader, optimizer)
    train_auc, _, _ = evaluate(model, train_loader)
    test_auc, test_preds, test_labels = \
        evaluate(model, test_loader)
    scheduler.step()
    train_aucs.append(train_auc)
    test_aucs.append(test_auc)
    if test_auc > best_auc:
        best_auc = test_auc
        best_epoch = epoch
        torch.save(model.state_dict(),
                   'models/gnn_best.pth')
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | "
              f"Train AUC: {train_auc:.4f} | "
              f"Test AUC: {test_auc:.4f}")

print(f"\nBest Test AUC: {best_auc:.4f} "
      f"at epoch {best_epoch}")

# ── 7. Training curve ───────────────────────────────
os.makedirs('figures', exist_ok=True)
plt.figure(figsize=(10, 4))
plt.plot(train_aucs, label='Train AUC',
         color='#7F77DD', linewidth=2)
plt.plot(test_aucs, label='Test AUC',
         color='#1D9E75', linewidth=2)
plt.axhline(best_auc, color='#D85A30',
    linestyle='--', label=f'Best AUC: {best_auc:.4f}')
plt.xlabel('Epoch')
plt.ylabel('ROC-AUC')
plt.title('GNN Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/gnn_training.png', dpi=150)
plt.close()
print("Saved: figures/gnn_training.png")

# ── 8. ROC curve ────────────────────────────────────
fpr, tpr, _ = roc_curve(test_labels, test_preds)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#1D9E75', linewidth=2,
    label=f'GNN (AUC = {best_auc:.3f})')
plt.plot([0,1],[0,1],'k--',linewidth=1,label='Random')
plt.fill_between(fpr, tpr, alpha=0.1, color='#1D9E75')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — GNN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_gnn.png', dpi=150)
plt.close()
print("Saved: figures/roc_gnn.png")

# ── 9. Save results ─────────────────────────────────
pd.DataFrame([{
    'model': 'GNN',
    'auc': round(best_auc, 4),
    'best_epoch': best_epoch,
    'n_train': len(train_data),
    'n_test': len(test_data),
    'n_params': params
}]).to_csv('logs/gnn_results.csv', index=False)
print("Saved: logs/gnn_results.csv")

print(f"\n{'='*50}")
print(f"FINAL RESULT: GNN AUC = {best_auc:.4f}")
print(f"RF AUC was:          0.9694")
diff = best_auc - 0.9694
print(f"Difference:          {diff:+.4f}")
print(f"{'='*50}")
print("GNN training complete!")
