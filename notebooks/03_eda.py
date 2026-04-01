"""
notebooks/03_eda.py
Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from rdkit import Chem
from rdkit.Chem import Descriptors

df = pd.read_csv('data/processed/egfr_clean.csv')
os.makedirs('figures', exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].hist(df['pIC50'], bins=50,
               color='#7F77DD', edgecolor='white')
axes[0,0].axvline(6, color='#D85A30', linestyle='--',
                  linewidth=2, label='Active threshold (6)')
axes[0,0].axvline(5, color='#1D9E75', linestyle='--',
                  linewidth=2, label='Inactive threshold (5)')
axes[0,0].set_xlabel('pIC50')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('pIC50 Distribution')
axes[0,0].legend()

counts = df['label'].value_counts().sort_index()
labels_text = ['Inactive (0)', 'Active (1)']
axes[0,1].bar(labels_text, counts.values,
              color=['#D85A30', '#1D9E75'], width=0.5)
axes[0,1].set_title('Class Balance')
axes[0,1].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0,1].text(i, v+20, str(v), ha='center')

def get_mw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol else None

print("Calculating molecular weights...")
df['MW'] = df['canonical_smiles'].apply(get_mw)

axes[1,0].hist(df[df['label']==1]['MW'].dropna(),
               bins=40, alpha=0.6,
               color='#1D9E75', label='Active')
axes[1,0].hist(df[df['label']==0]['MW'].dropna(),
               bins=40, alpha=0.6,
               color='#D85A30', label='Inactive')
axes[1,0].set_xlabel('Molecular Weight (Da)')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Molecular Weight by Class')
axes[1,0].legend()

df.boxplot(column='pIC50', by='label', ax=axes[1,1],
           boxprops=dict(color='#7F77DD'))
axes[1,1].set_title('pIC50 by Class')
axes[1,1].set_xlabel('Label (0=inactive, 1=active)')
axes[1,1].set_ylabel('pIC50')

plt.suptitle('EGFR Dataset Exploratory Analysis',
             fontsize=14)
plt.tight_layout()
plt.savefig('figures/eda_plots.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: figures/eda_plots.png")
print(f"\nDataset summary:")
print(f"Total compounds: {len(df)}")
print(f"Active: {df['label'].sum()} ({df['label'].mean():.1%})")
print(f"Mean pIC50: {df['pIC50'].mean():.2f}")
print(f"Mean MW: {df['MW'].mean():.1f} Da")
