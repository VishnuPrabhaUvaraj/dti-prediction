"""
src/fingerprints.py
Morgan fingerprint generation functions
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    """
    Convert SMILES string to Morgan fingerprint
    radius=2 = look 2 bonds away from each atom
    n_bits=2048 = output is 2048 zeros and ones
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )
    return np.array(fp)

def generate_fingerprints(smiles_list,
                           radius=2, n_bits=2048):
    """
    Generate fingerprints for a list of SMILES
    Returns X (feature matrix) and valid indices
    """
    fingerprints = []
    valid_indices = []
    for i, smiles in enumerate(smiles_list):
        fp = smiles_to_morgan(smiles, radius, n_bits)
        if fp is not None:
            fingerprints.append(fp)
            valid_indices.append(i)
    X = np.array(fingerprints)
    print(f"Generated {len(fingerprints)} fingerprints")
    print(f"Feature matrix shape: {X.shape}")
    return X, valid_indices
