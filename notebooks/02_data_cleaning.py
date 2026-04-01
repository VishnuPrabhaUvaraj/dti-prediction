"""
notebooks/02_data_cleaning.py
Clean ChEMBL data and create binary labels
"""
import pandas as pd
import numpy as np
from rdkit import Chem
import os

print("Loading raw data...")
df = pd.read_csv('data/raw/egfr_chembl_raw.csv')
print(f"Raw records: {len(df)}")

df = df[['molecule_chembl_id', 'canonical_smiles',
         'standard_value', 'standard_units',
         'standard_type']].copy()

df = df.dropna(subset=['canonical_smiles', 'standard_value'])
print(f"After removing NaN: {len(df)}")

df = df[df['standard_units'] == 'nM']
print(f"After filtering nM units: {len(df)}")

df['standard_value'] = pd.to_numeric(
    df['standard_value'], errors='coerce')
df = df.dropna(subset=['standard_value'])
df = df[df['standard_value'] > 0]
print(f"After numeric conversion: {len(df)}")

df['pIC50'] = -np.log10(df['standard_value'] * 1e-9)
print(f"pIC50 range: {df['pIC50'].min():.2f} to {df['pIC50'].max():.2f}")

df['label'] = np.where(df['pIC50'] >= 6, 1,
              np.where(df['pIC50'] <= 5, 0, np.nan))
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)
print(f"After labelling: {len(df)}")
print(f"Active (1):   {df['label'].sum()}")
print(f"Inactive (0): {(df['label']==0).sum()}")
print(f"Class balance: {df['label'].mean():.2%} active")

print("Validating SMILES strings...")
def is_valid_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

df['valid'] = df['canonical_smiles'].apply(is_valid_smiles)
df = df[df['valid']].drop(columns=['valid'])
print(f"After SMILES validation: {len(df)}")

df = df.drop_duplicates(subset=['canonical_smiles'])
print(f"After removing duplicates: {len(df)}")

os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/egfr_clean.csv', index=False)
print(f"\nSaved: data/processed/egfr_clean.csv")
print(f"Final dataset: {len(df)} compounds")
print(df[['molecule_chembl_id','pIC50','label']].head())
