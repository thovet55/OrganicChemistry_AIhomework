# -*- coding: utf-8 -*-
"""
AIn19.py (Final Fix: Variable Name & Encoding)
版本：v8.1
修复内容：
1. [NameError Fix] 修正 evaluate_model 中 p_te 与 p_test 的变量名不一致问题。
2. [Encoding Fix] 结果写入时指定 encoding='utf-8'，解决 Windows 下 Emoji 报错。
3. [Features] 保持 Smart HOMA + Hybrid Kernel + Dual SVG Verification。
"""

import os, time, argparse, sys
import numpy as np
import pandas as pd
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw, AllChem, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize as Stdz
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone


# =========================
# 1. 核函数定义
# =========================
class TanimotoKernel(Kernel):
    def __init__(self):
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None: Y = X
        X_sum = np.sum(X, axis=1).reshape(-1, 1)
        Y_sum = np.sum(Y, axis=1).reshape(1, -1)
        intersection = np.dot(X, Y.T)
        union = X_sum + Y_sum - intersection
        union[union == 0] = 1.0
        K = intersection / union
        if eval_gradient: return K, np.empty((X.shape[0], X.shape[0], 0))
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False

    @property
    def hyperparameters(self):
        return []


class HybridKernel(Kernel):
    def __init__(self, n_dim_bin, rbf_kernel):
        self.n_dim_bin = n_dim_bin
        self.rbf_kernel = rbf_kernel

    @property
    def hyperparameters(self):
        return self.rbf_kernel.hyperparameters

    @property
    def theta(self):
        return self.rbf_kernel.theta

    @theta.setter
    def theta(self, theta):
        self.rbf_kernel.theta = theta

    def __call__(self, X, Y=None, eval_gradient=False):
        X_bin, X_cont = X[:, :self.n_dim_bin], X[:, self.n_dim_bin:]
        if Y is not None:
            Y_bin, Y_cont = Y[:, :self.n_dim_bin], Y[:, self.n_dim_bin:]
        else:
            Y_bin, Y_cont = None, None

        Kt = TanimotoKernel()(X_bin, Y_bin, eval_gradient=False)
        if eval_gradient:
            Kr, Kr_grad = self.rbf_kernel(X_cont, Y_cont, eval_gradient=True)
            return Kt + Kr, Kr_grad
        else:
            Kr = self.rbf_kernel(X_cont, Y_cont, eval_gradient=False)
            return Kt + Kr

    def diag(self, X):
        return TanimotoKernel().diag(X[:, :self.n_dim_bin]) + self.rbf_kernel.diag(X[:, self.n_dim_bin:])

    def is_stationary(self):
        return False

    def clone_with_theta(self, theta):
        cloned = clone(self);
        cloned.theta = theta;
        return cloned


# =========================
# 2. HOMA 参数 & 颜色
# =========================
HOMA_PARAMS = {
    frozenset([6, 6]): (1.388, 257.7), frozenset([6, 7]): (1.334, 93.5),
    frozenset([6, 8]): (1.349, 57.2), frozenset([6, 16]): (1.719, 24.0),
    frozenset([7, 7]): (1.309, 130.3),
}
COLOR_MAP = {"center": (0.9, 0.2, 0.2), "left": (0.2, 0.4, 0.9), "right": (0.2, 0.7, 0.2), "top": (0.6, 0.2, 0.8),
             "overlap": (1.0, 0.8, 0.0)}


# =========================
# 3. 数据处理
# =========================
def try_read_csv(path):
    for enc in ["utf-8", "utf-8-sig", "gbk"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            pass
    return pd.read_csv(path)


def canonical_smiles(s):
    return Chem.MolToSmiles(Chem.MolFromSmiles(str(s)), True) if s else None


def load_data(path):
    df = try_read_csv(path)
    df["_SMI_O"] = [canonical_smiles(x) for x in df["SMILES_o"]]
    df["_SMI_C"] = [canonical_smiles(x) for x in df["SMILES_c"]]
    df = df.dropna(subset=["_SMI_O", "_SMI_C"]).drop_duplicates(subset=["_SMI_O", "_SMI_C"]).reset_index(drop=True)
    if "t_half_s" in df.columns:
        df["Y"] = df["t_half_s"]
    elif "t_half_ms" in df.columns:
        df["Y"] = df["t_half_ms"] / 1000.0
    elif "log10_s" in df.columns:
        df["Y"] = 10 ** df["log10_s"]
    return df, np.log10(np.clip(df["Y"].values, 1e-12, None))


def to_single_bonds(m):
    rw = Chem.RWMol()
    for a in m.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum());
        na.SetIsAromatic(False);
        rw.AddAtom(na)
    for b in m.GetBonds():
        rw.AddBond(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx()), Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out, Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
    return out


def mcs_map_stable(mol_o, mol_c):
    o1, c1 = to_single_bonds(mol_o), to_single_bonds(mol_c)
    p = rdFMCS.MCSParameters();
    p.RingMatchesRingOnly = True;
    p.Timeout = 30
    r = rdFMCS.FindMCS([o1, c1], p)
    if r.canceled or r.numAtoms == 0:
        p.RingMatchesRingOnly = False
        r = rdFMCS.FindMCS([o1, c1], p)
    if not r.canceled and r.numAtoms > 0:
        pat = Chem.MolFromSmarts(r.smartsString)
        ho = o1.GetSubstructMatches(pat)
        hc = c1.GetSubstructMatches(pat)
        if ho and hc:
            return True, {int(c): int(o) for c, o in zip(hc[0], ho[0])}
    return False, {}


def shortest_path_len(mol, i, j):
    try:
        return len(Chem.rdmolops.GetShortestPath(mol, int(i), int(j))) - 1
    except:
        return 999


def ring_bonds_from_atoms(mol, atoms):
    out = [];
    n = len(atoms)
    for k in range(n):
        b = mol.GetBondBetweenAtoms(int(atoms[k]), int(atoms[(k + 1) % n]))
        if b: out.append(b.GetIdx())
    return out


def get_smallest_ring_containing_bond(mol, bond_idx):
    ri = mol.GetRingInfo().BondRings()
    hits = [list(r) for r in ri if bond_idx in r]
    if not hits: return None
    ring_bonds = min(hits, key=len)
    adj = defaultdict(list)
    for bid in ring_bonds:
        b = mol.GetBondWithIdx(int(bid))
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        adj[u].append(v);
        adj[v].append(u)
    start = [k for k, v in adj.items() if len(v) == 2][0]
    path = [start];
    prev = None;
    curr = start
    while True:
        nbs = adj[curr]
        nxt = nbs[0] if nbs[0] != prev else nbs[1]
        if nxt == start: break
        path.append(nxt)
        prev, curr = curr, nxt
    return path


# =========================
# 4. HOMA 逻辑 & 区域识别
# =========================
def rotate_ring_to_bond(atoms, u, v):
    if u not in atoms or v not in atoms: return atoms
    idx_u = atoms.index(u);
    idx_v = atoms.index(v)
    n = len(atoms)
    if (idx_u + 1) % n == idx_v:
        return atoms[idx_v:] + atoms[:idx_v]
    elif (idx_v + 1) % n == idx_u:
        return atoms[idx_u:] + atoms[:idx_u]
    return atoms


def identify_regions(mo, mc, amap):
    c1 = to_single_bonds(mc);
    o1 = to_single_bonds(mo)
    cands = []
    for b in c1.GetBonds():
        if not b.IsInRing(): continue
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in amap and v in amap:
            ou, ov = amap[u], amap[v]
            if not o1.GetBondBetweenAtoms(ou, ov) and shortest_path_len(o1, ou, ov) >= 2:
                cands.append(b.GetIdx())

    if not cands: return None
    best_ring = None
    for bid in cands:
        r_atoms = get_smallest_ring_containing_bond(c1, bid)
        if r_atoms:
            if not best_ring or len(r_atoms) == 6:
                best_ring = {"atoms": r_atoms, "bond": bid}
                if len(r_atoms) == 6: break
    if not best_ring: return None

    ctr_atoms = best_ring["atoms"]
    new_bond_idx = best_ring["bond"]
    b_obj = c1.GetBondWithIdx(int(new_bond_idx))
    u_new, v_new = b_obj.GetBeginAtomIdx(), b_obj.GetEndAtomIdx()

    ctr_atoms_rotated = rotate_ring_to_bond(ctr_atoms, u_new, v_new)
    ctr_edges = ring_bonds_from_atoms(c1, ctr_atoms_rotated)

    l_edge = ctr_edges[-2];
    r_edge = ctr_edges[0]
    n = len(ctr_atoms_rotated)
    t_edge = ctr_edges[(n // 2) - 1] if n >= 4 else None

    def get_side_ring(edge_idx):
        ri = c1.GetRingInfo().AtomRings()
        b = c1.GetBondWithIdx(int(edge_idx))
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        c_set = set(ctr_atoms_rotated)
        cands = [list(r) for r in ri if u in r and v in r and set(r) != c_set]
        return min(cands, key=len) if cands else []

    return {
        "c": ctr_atoms_rotated, "l": get_side_ring(l_edge),
        "r": get_side_ring(r_edge), "t": get_side_ring(t_edge) if t_edge else []
    }


def draw_verification(mol, regions, path):
    atom_cols = {}
    atom_sets = defaultdict(list)
    for tag, atoms in regions.items():
        if isinstance(atoms, list):
            for idx in atoms: atom_sets[idx].append(tag)
    for idx, tags in atom_sets.items():
        atom_cols[idx] = COLOR_MAP["overlap"] if len(tags) > 1 else COLOR_MAP.get(tags[0], (0.8, 0.8, 0.8))
    d = Draw.MolDraw2DSVG(400, 300)
    d.drawOptions().addAtomIndices = True
    d.DrawMolecule(mol, highlightAtoms=list(atom_cols.keys()), highlightAtomColors=atom_cols)
    d.FinishDrawing()
    with open(path, 'w') as f:
        f.write(d.GetDrawingText())


# =========================
# 5. 3D & 计算
# =========================
def get_3d_robust(mol):
    if not mol: return None
    m_copy = Chem.Mol(mol)
    try:
        Chem.Kekulize(m_copy, clearAromaticFlags=True)
    except:
        pass
    Chem.RemoveStereochemistry(m_copy)
    m = Chem.AddHs(m_copy)

    params = AllChem.ETKDG()
    params.useRandomCoords = True
    params.maxIterations = 5000
    params.randomSeed = 42
    params.enforceChirality = False
    params.ignoreSmoothingFailures = True

    if AllChem.EmbedMolecule(m, params) != -1:
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=500)
        except:
            pass
        return m
    return None


def calc_homa_general(mol_3d, atoms, mode='ring'):
    if not mol_3d or not atoms or len(atoms) < 3: return 0.0
    conf = mol_3d.GetConformer()
    n = len(atoms)
    num_bonds = n if mode == 'ring' else n - 1
    term = 0.0;
    cnt = 0
    for k in range(num_bonds):
        u, v = atoms[k], atoms[(k + 1) % n]
        d = (conf.GetAtomPosition(u) - conf.GetAtomPosition(v)).Length()
        key = frozenset([mol_3d.GetAtomWithIdx(u).GetAtomicNum(), mol_3d.GetAtomWithIdx(v).GetAtomicNum()])
        if key in HOMA_PARAMS:
            p = HOMA_PARAMS[key]; term += p[1] * ((p[0] - d) ** 2); cnt += 1
        elif key == frozenset([6, 6]):
            term += 257.7 * ((1.388 - d) ** 2); cnt += 1
    if cnt == 0: return 0.0
    return 1.0 - (term / cnt)


def get_elec(mol, atoms):
    vals = []
    for i in atoms:
        try:
            vals.append(float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge")))
        except:
            pass
    return {"min": np.min(vals) if vals else 0.0}


def run_pipeline(df, args):
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=args.n_bits)
    X_list = [];
    verify_rows = []
    svg_dir = os.path.join(args.out_dir, "svg_debug")
    os.makedirs(svg_dir, exist_ok=True)

    print("🚀 Running Analysis (Dual SVG + Robust HOMA)...")

    for idx, row in df.iterrows():
        so, sc = row["_SMI_O"], row["_SMI_C"]
        mo, mc = Chem.MolFromSmiles(so), Chem.MolFromSmiles(sc)

        ok, amap = mcs_map_stable(mo, mc)
        regions = identify_regions(mo, mc, amap) if ok else None

        v_data = {"id": idx, "SMILES_C": sc}
        if regions:
            # Closed Atoms
            v_data["Center_C"] = ";".join(map(str, regions['c']))
            v_data["Left_C"] = ";".join(map(str, regions['l']))
            v_data["Right_C"] = ";".join(map(str, regions['r']))
            v_data["Top_C"] = ";".join(map(str, regions['t']))

            # Open Atoms (Mapped)
            def map_to_o(lst): return [amap[x] for x in lst if x in amap]

            r_o = {
                'c': map_to_o(regions['c']),
                'l': map_to_o(regions['l']),
                'r': map_to_o(regions['r']),
                't': map_to_o(regions['t'])
            }
            v_data["Center_O"] = ";".join(map(str, r_o['c']))

            draw_verification(mc, regions, os.path.join(svg_dir, f"mol_{idx:03d}_closed.svg"))
            draw_verification(mo, r_o, os.path.join(svg_dir, f"mol_{idx:03d}_open.svg"))

        verify_rows.append(v_data)

        fp_o = fp_gen.GetFingerprintAsNumPy(mo)
        fp_c = fp_gen.GetFingerprintAsNumPy(mc)
        fp_d = ((fp_o > 0) ^ (fp_c > 0)).astype(np.float32) * 0.6
        c_vec = [0.0, 0.0]

        if regions:
            mo3 = get_3d_robust(mo);
            mc3 = get_3d_robust(mc)
            AllChem.ComputeGasteigerCharges(mo);
            AllChem.ComputeGasteigerCharges(mc)

            if mo3 and mc3:
                hc_c = calc_homa_general(mc3, regions['c'], 'ring')
                side_c = calc_homa_general(mc3, regions['l'], 'ring') + \
                         calc_homa_general(mc3, regions['r'], 'ring') + \
                         calc_homa_general(mc3, regions['t'], 'ring')

                hc_o = calc_homa_general(mo3, r_o['c'], 'chain')
                side_o = calc_homa_general(mo3, r_o['l'], 'ring') + \
                         calc_homa_general(mo3, r_o['r'], 'ring') + \
                         calc_homa_general(mo3, r_o['t'], 'ring')

                c_vec[0] = (hc_c + side_c) - (hc_o + side_o)

            qc = get_elec(mc, regions['c'])["min"]
            qo = get_elec(mo, r_o['c'])["min"]
            c_vec[1] = qc - qo

        X_list.append(np.hstack([fp_o, fp_c, fp_d, c_vec]))

    pd.DataFrame(verify_rows).to_csv(os.path.join(args.out_dir, "verification.csv"), index=False)
    return np.vstack(X_list), ["Total_HOMA_D", "Center_QMin_D"]


# =========================
# 6. 训练与报告
# =========================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    p_tr = model.predict(X_train)
    p_test = model.predict(X_test)  # Fixed variable name here
    return {
        "tr_r2": r2_score(y_train, p_tr), "tr_mae": mean_absolute_error(y_train, p_tr),
        "te_r2": r2_score(y_test, p_test), "te_mae": mean_absolute_error(y_test, p_test)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out_dir", default="runs_AIn19")
    p.add_argument("--n_bits", type=int, default=512)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df, y = load_data(args.csv)

    X, names = run_pipeline(df, args)

    print("\n🩺 Feature Stats:")
    print(pd.DataFrame(X[:, args.n_bits * 3:], columns=names).describe().T[["mean", "std", "min", "max"]])

    rf = RandomForestRegressor(n_estimators=800, max_depth=12, random_state=42)

    k = HybridKernel(args.n_bits * 3, C(1.0) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 100.0))) + \
        WhiteKernel(noise_level=0.5, noise_level_bounds=(1e-2, 10.0))
    gp = GaussianProcessRegressor(kernel=k, n_restarts_optimizer=5, random_state=42)

    print("\n⚔️  Cross Validation (5-Fold)...")
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
    res = defaultdict(list)

    for tr, te in rkf.split(X):
        # RF
        rf.fit(X[tr], y[tr])
        m_rf = evaluate_model(rf, X[tr], y[tr], X[te], y[te])
        for k, v in m_rf.items(): res[f"rf_{k}"].append(v)

        # GP
        X_b_tr, X_c_tr = X[tr, :args.n_bits * 3], X[tr, args.n_bits * 3:]
        X_b_te, X_c_te = X[te, :args.n_bits * 3], X[te, args.n_bits * 3:]
        s = StandardScaler()
        try:
            X_c_tr_s = s.fit_transform(X_c_tr); X_c_te_s = s.transform(X_c_te)
        except:
            X_c_tr_s = X_c_tr; X_c_te_s = X_c_te

        X_tr_g = np.hstack([X_b_tr, X_c_tr_s])
        X_te_g = np.hstack([X_b_te, X_c_te_s])

        g = clone(gp);
        g.fit(X_tr_g, y[tr])
        m_gp = evaluate_model(g, X_tr_g, y[tr], X_te_g, y[te])
        for k, v in m_gp.items(): res[f"gp_{k}"].append(v)

    def fmt(key):
        return f"{np.mean(res[key]):.3f} +/- {np.std(res[key]):.3f}"

    lines = []
    lines.append("\nFinal Report (Mean +/- Std):")
    lines.append("-" * 65)
    lines.append(f"{'Metric':<12} | {'Random Forest':<22} | {'Gaussian Process':<22}")
    lines.append("-" * 65)
    lines.append(f"{'Train R2':<12} | {fmt('rf_tr_r2'):<22} | {fmt('gp_tr_r2'):<22}")
    lines.append(f"{'Train MAE':<12} | {fmt('rf_tr_mae'):<22} | {fmt('gp_tr_mae'):<22}")
    lines.append(f"{'CV R2':<12} | {fmt('rf_te_r2'):<22} | {fmt('gp_te_r2'):<22}")
    lines.append(f"{'CV MAE':<12} | {fmt('rf_te_mae'):<22} | {fmt('gp_te_mae'):<22}")
    lines.append("-" * 65)

    output_str = "\n".join(lines)
    print(output_str)

    res_path = os.path.join(args.out_dir, "results.txt")
    with open(res_path, "w", encoding="utf-8") as f:
        f.write(output_str)
    print(f"\n✅ Results saved to {res_path}")


if __name__ == "__main__":
    main()