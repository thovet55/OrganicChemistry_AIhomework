# -*- coding: utf-8 -*-
"""
DAE Molecular Photochemical Property Prediction Script (Version 2.0)
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

# RDKit
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS, AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize as Stdz

# AI / ML
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error

# Configuration
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
HOMA_PARAMS = {
    frozenset([6, 6]): (1.388, 257.7), frozenset([6, 7]): (1.334, 93.5),
    frozenset([6, 8]): (1.349, 57.2), frozenset([6, 16]): (1.719, 24.0),
    frozenset([7, 7]): (1.309, 130.3),
}


# =========================
# 1. Helper Classes & Functions
# =========================

def try_read_csv(filepath):
    encodings = ['utf-8', 'gbk', 'gb18030', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except:
            continue
    raise ValueError(f"[ERROR] Failed to read {filepath}")


class MoleculePreprocessor:
    def __init__(self):
        self.salter = SaltRemover()
        self.chooser = Stdz.LargestFragmentChooser()
        self.uncharger = Stdz.Uncharger()

    def process(self, mol):
        if not mol: return None
        try:
            mol = self.salter.StripMol(mol)
            mol = self.chooser.choose(mol)
            mol = self.uncharger.uncharge(mol)
            Chem.SanitizeMol(mol)
            return mol
        except:
            return None

    def canonicalize(self, smi):
        mol = Chem.MolFromSmiles(str(smi))
        if not mol: return None
        mol = self.process(mol)
        if not mol: return None
        return Chem.MolToSmiles(mol, canonical=True)


def to_single_bonds(m):
    if m is None: return None
    rw = Chem.RWMol()
    for a in m.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum())
        na.SetIsAromatic(False)
        rw.AddAtom(na)
    for b in m.GetBonds():
        rw.AddBond(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx()), Chem.BondType.SINGLE)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        Chem.GetSymmSSSR(out)
    except:
        return None
    return out


def shortest_dist(mol, u, v):
    try:
        return len(Chem.rdmolops.GetShortestPath(mol, int(u), int(v))) - 1
    except:
        return 999


def align_chain_atoms(atoms, u_break, v_break):
    if u_break not in atoms or v_break not in atoms: return atoms
    lst = list(atoms)
    try:
        idx_u, idx_v = lst.index(u_break), lst.index(v_break)
        n = len(lst)
        if (idx_u + 1) % n == idx_v:
            return lst[idx_v:] + lst[:idx_v]
        elif (idx_v + 1) % n == idx_u:
            return lst[idx_u:] + lst[:idx_u]
    except:
        pass
    return lst


def _calc_homa_unit(mol_3d, atoms, mode):
    if len(atoms) < 3: return 0.0
    conf = mol_3d.GetConformer()
    n = len(atoms)
    num_bonds = n if mode == 'ring' else n - 1
    term = 0.0
    cnt = 0
    for k in range(num_bonds):
        u, v = atoms[k], atoms[(k + 1) % n]
        try:
            d = (conf.GetAtomPosition(u) - conf.GetAtomPosition(v)).Length()
            key = frozenset([mol_3d.GetAtomWithIdx(u).GetAtomicNum(), mol_3d.GetAtomWithIdx(v).GetAtomicNum()])
            if key in HOMA_PARAMS:
                p = HOMA_PARAMS[key]
                term += p[1] * ((p[0] - d) ** 2)
                cnt += 1
            elif key == frozenset([6, 6]):
                term += 257.7 * ((1.388 - d) ** 2)
                cnt += 1
        except:
            continue
    return 1.0 - (term / cnt) if cnt else 0.0


# =========================
# 2. Physics Tower (Logic from Predictor.py)
# =========================

def extract_physical_features(df):
    print("\n[Step 1/3] Physics Tower: Calculating 3D HOMA (Sum) & dQ (Min)...")
    feats = []
    preprocessor = MoleculePreprocessor()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="   Physics"):
        vec = [0.0, 0.0]  # Default [dHOMA, dQ]
        try:
            # 1. Preprocess
            mol_o = preprocessor.process(Chem.MolFromSmiles(row["_SMI_O"]))
            mol_c = preprocessor.process(Chem.MolFromSmiles(row["_SMI_C"]))

            if mol_o and mol_c:
                # 2. MCS Alignment
                sk_o, sk_c = to_single_bonds(mol_o), to_single_bonds(mol_c)
                p = rdFMCS.MCSParameters()
                p.RingMatchesRingOnly = True
                mcs = rdFMCS.FindMCS([sk_o, sk_c], p)

                if mcs.numAtoms > 0:
                    patt = Chem.MolFromSmarts(mcs.smartsString)
                    match_o, match_c = sk_o.GetSubstructMatch(patt), sk_c.GetSubstructMatch(patt)

                    if match_o and match_c:
                        amap = {c: o for c, o in zip(match_c, match_o)}
                        ri = sk_c.GetRingInfo()
                        bond_rings = ri.BondRings()
                        core_atoms, broken_bond = None, None

                        # 3. Identify Core & Break Bond
                        for b in sk_c.GetBonds():
                            if not b.IsInRing(): continue
                            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                            if u not in amap or v not in amap: continue

                            # Check distance in Open form (>=2 implies break)
                            if shortest_dist(sk_o, amap[u], amap[v]) >= 2:
                                bid = b.GetIdx()
                                rings = [list(r) for r in bond_rings if bid in r]
                                rings.sort(key=len)
                                for r_bonds in rings:
                                    r_atoms = set()
                                    for rb in r_bonds:
                                        b_obj = sk_c.GetBondWithIdx(rb)
                                        r_atoms.add(b_obj.GetBeginAtomIdx())
                                        r_atoms.add(b_obj.GetEndAtomIdx())
                                    if len(r_atoms) in [5, 6]:
                                        for ar in ri.AtomRings():
                                            if set(ar) == r_atoms: core_atoms = list(ar); break
                                        broken_bond = (u, v)
                                        break
                                    if core_atoms: break
                                if core_atoms: break

                        # 4. Expand Fused Rings
                        if core_atoms:
                            core_set = set(core_atoms)
                            all_rings = ri.AtomRings()
                            fused_pool = set(core_atoms)
                            while True:
                                added = False
                                for r in all_rings:
                                    if not set(r).issubset(fused_pool) and len(set(r).intersection(fused_pool)) >= 2:
                                        fused_pool.update(r)
                                        added = True
                                if not added: break

                            # 5. Generate 3D & Calculate
                            mo3 = Chem.AddHs(mol_o)
                            mc3 = Chem.AddHs(mol_c)
                            ps = AllChem.ETKDG()
                            ps.useRandomCoords = True
                            ps.maxIterations = 200

                            if AllChem.EmbedMolecule(mo3, ps) >= 0 and AllChem.EmbedMolecule(mc3, ps) >= 0:
                                AllChem.MMFFOptimizeMolecule(mo3)
                                AllChem.MMFFOptimizeMolecule(mc3)

                                # HOMA Calculation
                                h_c_cl = _calc_homa_unit(mc3, core_atoms, 'ring')
                                core_aligned = align_chain_atoms(core_atoms, broken_bond[0], broken_bond[1])
                                h_c_op = _calc_homa_unit(mo3, [amap[x] for x in core_aligned if x in amap], 'chain')

                                h_p_cl = 0.0
                                h_p_op = 0.0
                                for ar in all_rings:
                                    if set(ar).issubset(fused_pool) and not set(ar).issubset(core_set):
                                        h_p_cl += _calc_homa_unit(mc3, list(ar), 'ring')
                                        h_p_op += _calc_homa_unit(mo3, [amap[x] for x in ar if x in amap], 'ring')

                                dHOMA = (h_c_cl + h_p_cl) - (h_c_op + h_p_op)

                                # dQ Calculation (Min Charge Difference)
                                AllChem.ComputeGasteigerCharges(mo3)
                                AllChem.ComputeGasteigerCharges(mc3)

                                qc_min = np.min(
                                    [float(mc3.GetAtomWithIdx(i).GetProp("_GasteigerCharge")) for i in core_atoms])
                                qo_min = np.min([float(mo3.GetAtomWithIdx(i).GetProp("_GasteigerCharge")) for i in
                                                 [amap[x] for x in core_aligned if x in amap]])
                                dQ = qc_min - qo_min

                                vec = [dHOMA, dQ]
        except:
            pass
        feats.append(vec)
    return np.array(feats)


# =========================
# 3. Deep Tower (ChemBERTa + PCA)
# =========================

def extract_deep_features(smiles_list, n_components=10):
    print(f"\n[Step 2/3] Deep Tower: ChemBERTa (Open SMILES) -> PCA({n_components})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:
        print(f"[ERROR] Model Error: {e}")
        sys.exit(1)

    embeddings = []
    with torch.no_grad():
        for smi in tqdm(smiles_list, desc="   Inference"):
            try:
                inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True, max_length=128)
                outputs = model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
            except:
                embeddings.append(np.zeros((1, 768)))

    X_raw = np.vstack(embeddings)

    # Standard Scaler BEFORE PCA (Crucial from Predictor.py)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"   PCA Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    return X_pca


# =========================
# 4. Main Pipeline
# =========================

def main():
    print("DAE Prediction Script Version 2.0 (Refactored Predictor)")

    # 1. Load Data
    csv_file = "data.csv"
    if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
        csv_file = sys.argv[1]
    elif "--csv" in sys.argv:
        csv_file = sys.argv[sys.argv.index("--csv") + 1]

    print(f"   Loading data from: {csv_file}")
    try:
        df = try_read_csv(csv_file)
    except Exception as e:
        print(e);
        return

    # Preprocessing
    prep = MoleculePreprocessor()
    df["_SMI_O"] = [prep.canonicalize(s) for s in df["open_smiles" if "open_smiles" in df.columns else "SMILES_o"]]
    df["_SMI_C"] = [prep.canonicalize(s) for s in df["closed_smiles" if "closed_smiles" in df.columns else "SMILES_c"]]
    df = df.dropna(subset=["_SMI_O", "_SMI_C"]).reset_index(drop=True)

    # Target Handling
    if "t_half_ms" in df.columns:
        df["Y"] = pd.to_numeric(df["t_half_ms"], errors='coerce') / 1000.0
    elif "log_t12" in df.columns:
        df["Y"] = 10 ** df["log_t12"]

    # Clean target
    df["Y"] = df["Y"].clip(1e-9, 1e18)
    y_reg = np.log10(df["Y"].values)  # Working in log scale

    print(f"   Dataset: {len(df)} samples")

    # 2. Physics Feature Extraction
    X_phys = extract_physical_features(df)

    # Physics Audit
    print("\n   Physics Audit:")
    valid_count = np.count_nonzero(np.sum(np.abs(X_phys), axis=1))
    print(f"   Calculated: {valid_count}/{len(df)} molecules")
    if valid_count > 0:
        print(pd.DataFrame(X_phys, columns=["dHOMA", "dQ"]).describe().T[["mean", "std", "min", "max"]])

    # 3. Deep Feature Extraction
    # Predictor.py used Open SMILES for BERTa
    X_deep_pca = extract_deep_features(df["_SMI_O"].tolist(), n_components=10)

    # 4. Fusion
    print("\n[Step 3/3] Fusing Features & LOOCV...")
    # Scale Physics features (Crucial from Predictor.py)
    s_phys = StandardScaler()
    X_phys_s = s_phys.fit_transform(X_phys)

    X_final = np.hstack([X_deep_pca, X_phys_s])
    print(f"   Fused Input Shape: {X_final.shape}")

    # 5. Modeling (RF & GP)
    loo = LeaveOneOut()
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gp = GaussianProcessRegressor(kernel=C(1.0) * Matern(1.0) + WhiteKernel(0.1), normalize_y=True)

    p_rf, p_gp, actuals = [], [], []

    print("   Running Cross-Validation...")
    for tr, te in tqdm(loo.split(X_final), total=len(X_final)):
        # RF
        rf.fit(X_final[tr], y_reg[tr])
        p_rf.append(rf.predict(X_final[te])[0])

        # GP
        gp.fit(X_final[tr], y_reg[tr])
        p_gp.append(gp.predict(X_final[te])[0])

        actuals.append(y_reg[te][0])

    r2_rf = r2_score(actuals, p_rf)
    mae_rf = mean_absolute_error(actuals, p_rf)
    r2_gp = r2_score(actuals, p_gp)
    mae_gp = mean_absolute_error(actuals, p_gp)

    print("\n" + "=" * 40)
    print(f"   Version 2.0 Final Results:")
    print(f"   RF R2 : {r2_rf:.4f}  |  MAE : {mae_rf:.4f}")
    print(f"   GP R2 : {r2_gp:.4f}  |  MAE : {mae_gp:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()