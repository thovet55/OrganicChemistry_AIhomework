# -*- coding: utf-8 -*-
"""
DAE Molecular Photochemical Property Prediction Script (Version 1.0)
"""

import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

# =========================
# 1. Configuration & Constants
# =========================
HOMA_PARAMS = {
    frozenset([6, 6]): (1.388, 257.7),
    frozenset([6, 7]): (1.334, 93.52),
    frozenset([6, 8]): (1.365, 153.0),
    frozenset([6, 16]): (1.677, 41.14),
    frozenset([7, 7]): (1.309, 130.3)
}


# =========================
# 2. Helper Functions
# =========================

def try_read_csv(filepath):

    encodings = ['utf-8', 'gbk', 'gb18030', 'cp1252', 'latin1']

    for enc in encodings:
        try:
            # 尝试读取
            df = pd.read_csv(filepath, encoding=enc)
            print(f"  Successfully loaded {filepath} using encoding: {enc}")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # 如果是其他错误（如文件不存在），直接抛出
            raise e

    raise ValueError(f"[ERROR] Failed to read {filepath}. Tried encodings: {encodings}")


def to_single_bond_mol(mol):
    mw = Chem.RWMol(mol)
    for b in mw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    for a in mw.GetAtoms():
        a.SetIsAromatic(False)
    return mw.GetMol()


def get_mapping(mol_open, mol_closed):
    o_skel = to_single_bond_mol(mol_open)
    c_skel = to_single_bond_mol(mol_closed)

    params = rdFMCS.MCSParameters()
    params.BondCompareParameters.RingMatchesRingOnly = False
    params.BondCompareParameters.CompleteRingsOnly = False
    params.AtomCompareParameters.MatchValences = False

    mcs_res = rdFMCS.FindMCS([o_skel, c_skel], params)

    if mcs_res.numAtoms == 0: return None
    patt = Chem.MolFromSmarts(mcs_res.smartsString)

    match_o = o_skel.GetSubstructMatch(patt)
    match_c = c_skel.GetSubstructMatch(patt)

    if not match_o or not match_c: return None
    return {c_idx: o_idx for c_idx, o_idx in zip(match_c, match_o)}


def generate_3d(mol):
    m = Chem.AddHs(mol)
    ps = AllChem.ETKDG()
    ps.randomSeed = 42
    res = AllChem.EmbedMolecule(m, ps)
    if res == -1:
        res = AllChem.EmbedMolecule(m, useRandomCoords=True, randomSeed=42)
    if res == -1: return None
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except:
        pass
    return m


def calculate_homa_for_ring(mol, ring_indices, is_open_center=False):
    conf = mol.GetConformer()
    n = len(ring_indices)
    sum_term = 0.0
    valid_bonds = 0

    for k in range(n):
        if is_open_center and k == 0: continue
        u = ring_indices[k]
        v = ring_indices[(k + 1) % n]
        dist = (conf.GetAtomPosition(u) - conf.GetAtomPosition(v)).Length()

        a1 = mol.GetAtomWithIdx(u).GetAtomicNum()
        a2 = mol.GetAtomWithIdx(v).GetAtomicNum()
        key = frozenset([a1, a2])

        if key in HOMA_PARAMS:
            R_opt, alpha = HOMA_PARAMS[key]
            sum_term += alpha * ((R_opt - dist) ** 2)
            valid_bonds += 1
        elif key == frozenset([6, 6]):
            sum_term += 257.7 * ((1.388 - dist) ** 2)
            valid_bonds += 1

    if valid_bonds == 0: return 0.0
    return 1.0 - (sum_term / valid_bonds)


def extract_physical_features(open_smiles, closed_smiles):
    mol_o = Chem.MolFromSmiles(open_smiles)
    mol_c = Chem.MolFromSmiles(closed_smiles)
    if not mol_o or not mol_c: raise ValueError("Invalid SMILES")

    mol_o_3d = generate_3d(mol_o)
    mol_c_3d = generate_3d(mol_c)
    if not mol_o_3d or not mol_c_3d: raise ValueError("3D Generation Failed")

    mapping = get_mapping(mol_o, mol_c)
    if not mapping: raise ValueError("MCS Mapping Failed")

    break_bond = None
    reaction_atoms = None
    for bond in mol_c.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if u in mapping and v in mapping:
            u_o, v_o = mapping[u], mapping[v]
            bond_o = mol_o.GetBondBetweenAtoms(u_o, v_o)
            if bond_o is None:
                break_bond = bond
                reaction_atoms = [u, v]
                break

    if not break_bond: raise ValueError("No Break Bond Found")

    ri = mol_c.GetRingInfo()
    atom_rings = ri.AtomRings()
    center_ring = None
    for ring in atom_rings:
        if len(ring) == 6:
            if reaction_atoms[0] in ring and reaction_atoms[1] in ring:
                center_ring = list(ring)
                break
    if not center_ring: raise ValueError("No Center 6-Ring Found")

    idx0 = center_ring.index(reaction_atoms[0])
    idx1 = center_ring.index(reaction_atoms[1])
    n = len(center_ring)

    if (idx0 + 1) % n == idx1:
        center_ring = center_ring[idx0:] + center_ring[:idx0]
    elif (idx1 + 1) % n == idx0:
        center_ring = center_ring[idx1:] + center_ring[:idx1]
    else:
        if {idx0, idx1} == {0, n - 1}:
            if idx0 == 0:
                center_ring = [center_ring[0]] + center_ring[1:][::-1]
            else:
                center_ring = center_ring[n - 1:] + center_ring[:n - 1]
        else:
            raise ValueError("Reactive atoms not adjacent in ring")

    def find_attached_ring(bond_atoms):
        u, v = bond_atoms
        for ring in atom_rings:
            if u in ring and v in ring:
                if set(ring) == set(center_ring): continue
                return list(ring)
        return None

    side_A = find_attached_ring((center_ring[1], center_ring[2]))
    side_B = find_attached_ring((center_ring[3], center_ring[4]))
    side_C = find_attached_ring((center_ring[5], center_ring[0]))

    rings_to_calc = [('Center', center_ring)]
    if side_A: rings_to_calc.append(('A', side_A))
    if side_B: rings_to_calc.append(('B', side_B))
    if side_C: rings_to_calc.append(('C', side_C))

    homa_diffs = []
    for label, ring_idxs in rings_to_calc:
        try:
            ring_idxs_o = [mapping[i] for i in ring_idxs]
        except KeyError:
            continue
        h_c = calculate_homa_for_ring(mol_c_3d, ring_idxs)
        is_center = (label == 'Center')
        h_o = calculate_homa_for_ring(mol_o_3d, ring_idxs_o, is_open_center=is_center)
        homa_diffs.append(h_c - h_o)

    dhoma_feature = np.mean(homa_diffs) if homa_diffs else 0.0

    AllChem.ComputeGasteigerCharges(mol_c_3d)
    q0 = float(mol_c_3d.GetAtomWithIdx(center_ring[0]).GetProp('_GasteigerCharge'))
    q1 = float(mol_c_3d.GetAtomWithIdx(center_ring[1]).GetProp('_GasteigerCharge'))
    dq_feature = abs(q0 - q1)

    return dhoma_feature, dq_feature


# =========================
# 3. Main Workflow
# =========================

def main():
    print(" DAE Prediction Workflow (v1.1) Started")

    # 3.1 Data Loading
    input_csv = "data.csv"
    print(f" Loading data from {input_csv}...")

    try:
        df = try_read_csv(input_csv)
    except Exception as e:
        print(f"[ERROR] Fatal Error loading CSV: {e}")
        return

    # 自动处理列名
    df.columns = [c.strip() for c in df.columns]

    # 兼容处理
    if 'SMILES_c' in df.columns: df.rename(columns={'SMILES_c': 'closed_smiles'}, inplace=True)
    if 'SMILES_o' in df.columns: df.rename(columns={'SMILES_o': 'open_smiles'}, inplace=True)

    if 't_half_ms' in df.columns:
        df['t_half_ms'] = pd.to_numeric(df['t_half_ms'], errors='coerce')
        df = df.dropna(subset=['t_half_ms'])
        df['log_t12'] = np.log10(df['t_half_ms'] / 1000.0)

    required_cols = ['open_smiles', 'closed_smiles', 'log_t12']
    # 检查列是否存在
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        print(f"   Available columns: {list(df.columns)}")
        return

    df = df.dropna(subset=required_cols)
    print(f"Valid samples: {len(df)}")

    # 3.2 Feature Extraction
    print("\n  Extracting Features (This may take a while)...")
    X_morgan = []
    X_phys = []
    y = []
    failed = []

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)

    for idx, row in df.iterrows():
        try:
            mc = Chem.MolFromSmiles(row['closed_smiles'])
            fp = morgan_gen.GetFingerprintAsNumPy(mc)
            dhoma, dq = extract_physical_features(row['open_smiles'], row['closed_smiles'])

            X_morgan.append(fp)
            X_phys.append([dhoma, dq])
            y.append(row['log_t12'])

            # 打印进度 (可选)
            # print(f"  [{idx+1}] Success")

        except Exception as e:
            failed.append(row['closed_smiles'])
            print(f"  [{idx + 1}] Failed: {e}")

    if not X_morgan:
        print("No valid samples.")
        return

    X = np.hstack([np.array(X_morgan), np.array(X_phys)])
    y = np.array(y)

    phys = np.array(X_phys)
    print("\n Data Quality Report:")
    print(f"  dHOMA: Mean={phys[:, 0].mean():.3f}")
    print(f"  dQ:    Mean={phys[:, 1].mean():.3f}")
    print(f"  Failed: {len(failed)}")

    print("\n Training Random Forest (LOOCV)...")
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    for train_ix, test_ix in loo.split(X):
        rf.fit(X[train_ix], y[train_ix])
        p = rf.predict(X[test_ix])[0]
        y_true.append(y[test_ix][0])
        y_pred.append(p)

    print("\n Final Results:")
    print(f"  R2 Score: {r2_score(y_true, y_pred):.4f}")
    print(f"  MAE:      {mean_absolute_error(y_true, y_pred):.4f}")


if __name__ == "__main__":
    main()