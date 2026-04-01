"""
notebooks/04_random_forest.py
Random Forest model with Morgan fingerprints
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, os, sys
sys.path.append('.')

from src.fingerprints import generate_fingerprints
from sklearn.model_selection import (train_test_split,
                                     cross_val_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay)
import shap

print("="*50)
print("MODEL 1: Random Forest + Morgan Fingerprints")
print("="*50)

# ── 1. Load data ────────────────────────────────────
df = pd.read_csv('data/processed/egfr_clean.csv')
print(f"Dataset: {len(df)} compounds")
print(f"Active: {df['label'].sum()}, "
      f"Inactive: {(df['label']==0).sum()}")

# ── 2. Generate Morgan fingerprints ────────────────
print("\nGenerating Morgan fingerprints...")
print("Each molecule → 2048-bit vector...")
X, valid_idx = generate_fingerprints(
    df['canonical_smiles'].tolist()
)
y = df['label'].iloc[valid_idx].values
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ── 3. Train/test split ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"\nTrain set: {len(X_train)} compounds")
print(f"Test set:  {len(X_test)} compounds")

# ── 4. Train Random Forest ──────────────────────────
print("\nTraining Random Forest (200 trees)...")
print("This takes 2-5 minutes...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
print("Training complete!")

# ── 5. Evaluate on test set ─────────────────────────
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

print(f"\n{'='*40}")
print(f"ROC-AUC Score: {auc:.4f}")
print(f"{'='*40}")
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Inactive', 'Active']
))

# ── 6. Cross-validation ─────────────────────────────
print("Running 5-fold cross-validation...")
cv_scores = cross_val_score(
    rf, X, y, cv=5,
    scoring='roc_auc', n_jobs=-1
)
print(f"CV AUC: {cv_scores.mean():.4f} "
      f"± {cv_scores.std():.4f}")

# ── 7. Save model ───────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(rf, 'models/rf_model.pkl')
print("\nSaved: models/rf_model.pkl")

# ── 8. ROC curve ────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#7F77DD', linewidth=2,
         label=f'Random Forest (AUC = {auc:.3f})')
plt.plot([0,1],[0,1],'k--',linewidth=1,label='Random')
plt.fill_between(fpr, tpr, alpha=0.1, color='#7F77DD')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Random Forest')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_rf.png', dpi=150)
plt.close()
print("Saved: figures/roc_rf.png")

# ── 9. Confusion matrix ─────────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    cm, display_labels=['Inactive','Active'])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap='Blues')
plt.title('Confusion Matrix — Random Forest')
plt.tight_layout()
plt.savefig('figures/cm_rf.png', dpi=150)
plt.close()
print("Saved: figures/cm_rf.png")

# ── 10. SHAP values ─────────────────────────────────
print("\nCalculating SHAP values (~2 minutes)...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test[:200], check_additivity=False)

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values[1], X_test[:200],
    max_display=20, show=False,
    plot_type='bar'
)
plt.title('Top 20 Important Features (SHAP)')
plt.tight_layout()
plt.savefig('figures/shap_rf.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: figures/shap_rf.png")

# ── 11. Save all results ────────────────────────────
os.makedirs('logs', exist_ok=True)
results = {
    'model': 'Random Forest',
    'auc': round(auc, 4),
    'cv_auc_mean': round(cv_scores.mean(), 4),
    'cv_auc_std': round(cv_scores.std(), 4),
    'n_train': len(X_train),
    'n_test': len(X_test)
}
pd.DataFrame([results]).to_csv(
    'logs/rf_results.csv', index=False)
print("\nSaved: logs/rf_results.csv")
print(f"\n{'='*50}")
print(f"FINAL RESULT: AUC = {auc:.4f}")
print(f"CV AUC: {cv_scores.mean():.4f} "
      f"± {cv_scores.std():.4f}")
print(f"{'='*50}")
print("Random Forest complete!")
