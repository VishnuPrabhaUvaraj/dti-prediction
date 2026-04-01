"""
src/mol_graph.py
Convert SMILES strings to molecular graphs for GNN
"""
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data

def atom_features(atom):
    """
    Extract features for each atom (node features)
    Every atom becomes a vector of 15 numbers
    """
    atom_types = ['C','N','O','S','F',
                  'P','Cl','Br','I','Other']
    atom_symbol = atom.GetSymbol()
    atom_type = [1 if atom_symbol == t else 0
                 for t in atom_types[:-1]]
    atom_type.append(
        1 if atom_symbol not in atom_types[:-1] else 0)
    features = (
        atom_type +
        [atom.GetDegree() / 10.0] +
        [atom.GetFormalCharge() / 5.0] +
        [int(atom.GetIsAromatic())] +
        [int(atom.IsInRing())] +
        [atom.GetTotalNumHs() / 8.0]
    )
    return features

def bond_features(bond):
    """
    Extract features for each bond (edge features)
    Every bond becomes a vector of 4 numbers
    """
    bond_types = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    bt = bond.GetBondType()
    return [1 if bt == t else 0 for t in bond_types]

def smiles_to_graph(smiles, label=None):
    """
    Convert SMILES string to PyTorch Geometric Data object
    Returns None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_feats = [atom_features(a)
                  for a in mol.GetAtoms()]
    x = torch.tensor(node_feats, dtype=torch.float)
    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
        bf = bond_features(bond)
        edge_attrs += [bf, bf]
    if len(edge_indices) == 0:
        return None
    edge_index = torch.tensor(
        edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(
        edge_attrs, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.long) \
        if label is not None else None
    return Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr, y=y)
