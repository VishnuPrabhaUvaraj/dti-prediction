"""
notebooks/01_data_download.py
Download EGFR inhibitor data from ChEMBL
"""
from chembl_webresource_client.new_client import new_client
import pandas as pd
import os

print("Connecting to ChEMBL...")

target = new_client.target
activity = new_client.activity

egfr = target.search('EGFR')
egfr_df = pd.DataFrame(egfr)
print(f"Found targets: {len(egfr_df)}")

egfr_id = 'CHEMBL203'
print(f"Using EGFR target: {egfr_id}")

print("Downloading bioactivity data (this takes 2-5 minutes)...")
activities = activity.filter(
    target_chembl_id=egfr_id,
    standard_type='IC50',
    relation='=',
    assay_type='B'
)

df = pd.DataFrame(list(activities))
print(f"Total records downloaded: {len(df)}")
print(f"Columns: {list(df.columns)}")

os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/egfr_chembl_raw.csv', index=False)
print("Saved: data/raw/egfr_chembl_raw.csv")
print("\nFirst 3 rows:")
print(df[['molecule_chembl_id','canonical_smiles',
          'standard_value','standard_units']].head(3))
