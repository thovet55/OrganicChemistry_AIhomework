# -*- coding: utf-8 -*-
"""
AIn37.py
版本：v30.0 (RF-Guided GP Residual Learning)
目标：DAE 光开关半衰期预测 (T-type only)
架构：
    1. Feature Engineering:
       - Deep Tower: ChemBERTa (v1) -> PCA (10 components)
       - Physics Tower: dHOMA (Aromaticity), dQ (Charge Transfer), Rbond (Strain)
    2. Model Architecture (Hybrid):
       - Baseline: Random Forest (trained via OOF to prevent leakage)
       - Refinement: Gaussian Process (learning the residuals)
       - Kernel: Matern 5/2 (ARD) for physical consistency
    3. Output: Prediction + Uncertainty (Sigma) + Outlier Detection

作者：Gemini (Your AI Partner)
时间：2025-12
"""

import os
import sys
import time
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

# === RDKit ===
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize as Stdz

# === Deep Learning ===
import torch
from transformers import AutoTokenizer, AutoModel

# === Sklearn ===
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import r2_score, mean_absolute_error

# === 配置与参数 ===
# 忽略 RDKit 和 Sklearn 的繁琐警告
warnings.filterwarnings("ignore")
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# 网络镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"

# HOMA 参数库 (键长 R_opt, 归一化常数 alpha)
# 来源：Krygowski et al.
HOMA_PARAMS = {
    frozenset([6, 6]): (1.388, 257.7),  # C-C (Benzene ref)
    frozenset([6, 7]): (1.334, 93.5),  # C-N
    frozenset([6, 8]): (1.349, 57.2),  # C-O
    frozenset([6, 16]): (1.719, 24.0),  # C-S
    frozenset([7, 7]): (1.309, 130.3),  # N-N
}


# ==============================================================================
# 模块 1：化学工具箱 (预处理与拓扑操作)
# ==============================================================================

class MoleculePreprocessor:
    """负责分子的清洗、去盐、标准化和规范化"""

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
    """
    [关键步骤] 将分子所有键强制转化为单键，且去除芳香性标记。
    目的：为了让 MCS 算法能忽略双键/单键的变化，只匹配骨架连接关系。
    """
    if m is None: return None
    rw = Chem.RWMol()
    # 1. 复制原子，清除芳香性
    for a in m.GetAtoms():
        na = Chem.Atom(a.GetAtomicNum())
        na.SetIsAromatic(False)
        rw.AddAtom(na)
    # 2. 复制键，强制设为 SINGLE
    for b in m.GetBonds():
        rw.AddBond(int(b.GetBeginAtomIdx()), int(b.GetEndAtomIdx()), Chem.BondType.SINGLE)

    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
        # 强制更新环信息，否则 RingInfo 可能失效
        Chem.GetSymmSSSR(out)
    except:
        return None
    return out


def shortest_dist(mol, u, v):
    """计算两原子间的最短拓扑距离"""
    try:
        return len(Chem.rdmolops.GetShortestPath(mol, int(u), int(v))) - 1
    except:
        return 999


def align_chain_atoms(atoms, u_break, v_break):
    """
    将开链原子的顺序对齐，使其与闭环时的环原子顺序对应。
    用于正确计算 Open 态的 HOMA（虽然它是开的，但我们要算对应原子的局部指标）。
    """
    if u_break not in atoms or v_break not in atoms: return atoms
    lst = list(atoms)
    try:
        idx_u, idx_v = lst.index(u_break), lst.index(v_break)
        n = len(lst)
        # 判断是顺时针还是逆时针断开
        if (idx_u + 1) % n == idx_v:
            return lst[idx_v:] + lst[:idx_v]
        elif (idx_v + 1) % n == idx_u:
            return lst[idx_u:] + lst[:idx_u]
    except:
        pass
    return lst


# ==============================================================================
# 模块 2：物理特征引擎 (Physics Engine)
# ==============================================================================

def _calc_homa_unit(mol_3d, atoms, mode):
    """
    计算给定原子列表的 HOMA 值。
    mode='ring': 计算闭合环 (N个键)
    mode='chain': 计算开链 (N-1个键)
    """
    if len(atoms) < 3: return 0.0
    conf = mol_3d.GetConformer()
    n = len(atoms)
    num_bonds = n if mode == 'ring' else n - 1

    term = 0.0
    cnt = 0

    for k in range(num_bonds):
        u = atoms[k]
        v = atoms[(k + 1) % n]

        # 获取 3D 键长
        p1 = conf.GetAtomPosition(u)
        p2 = conf.GetAtomPosition(v)
        d = (p1 - p2).Length()

        # 查找参数
        a1 = mol_3d.GetAtomWithIdx(u).GetAtomicNum()
        a2 = mol_3d.GetAtomWithIdx(v).GetAtomicNum()
        key = frozenset([a1, a2])

        if key in HOMA_PARAMS:
            opt_d, alpha = HOMA_PARAMS[key]
            term += alpha * ((opt_d - d) ** 2)
            cnt += 1
        # 默认碳碳键参数 (容错)
        elif key == frozenset([6, 6]):
            term += 257.7 * ((1.388 - d) ** 2)
            cnt += 1

    if cnt == 0: return 0.0
    homa = 1.0 - (term / cnt)
    return homa


def get_physics_features(df):
    """
    [核心函数] 提取 dHOMA, dQ, BondLength
    流程：
    1. 清洗 SMILES
    2. 生成 3D 构象并优化
    3. 转化为单键骨架，寻找 MCS
    4. 识别 Core 区域
    5. 计算物理量
    """
    print("\n[M2] Physics Engine: Calculating dHOMA, dQ, and Bond Length...")
    feats = []
    preprocessor = MoleculePreprocessor()

    # 进度条
    iterator = tqdm(df.iterrows(), total=len(df), desc="  -> Computing Physics")

    for _, row in iterator:
        # 默认值 (如果计算失败)
        vec = [0.0, 0.0, 0.0]

        try:
            # 1. 基础处理
            mol_o = preprocessor.process(Chem.MolFromSmiles(row["_SMI_O"]))
            mol_c = preprocessor.process(Chem.MolFromSmiles(row["_SMI_C"]))

            if mol_o and mol_c:
                # 2. 骨架拓扑对齐 (全单键化)
                sk_o = to_single_bonds(mol_o)
                sk_c = to_single_bonds(mol_c)

                # MCS 搜索
                p = rdFMCS.MCSParameters()
                p.RingMatchesRingOnly = True
                p.AtomCompare = rdFMCS.AtomCompare.CompareElements
                p.BondCompare = rdFMCS.BondCompare.CompareAny  # 忽略键级差异

                mcs = rdFMCS.FindMCS([sk_o, sk_c], p)

                if mcs.numAtoms > 0:
                    patt = Chem.MolFromSmarts(mcs.smartsString)
                    match_o = sk_o.GetSubstructMatch(patt)
                    match_c = sk_c.GetSubstructMatch(patt)

                    if match_o and match_c:
                        # 建立原子映射 C -> O
                        amap = {c: o for c, o in zip(match_c, match_o)}

                        # 3. 寻找闭环的那根键 (The Closing Bond)
                        # 逻辑：在 Closed 态中存在，且两端都在 MCS 内，但在 Open 态中这两端距离很远
                        ri = sk_c.GetRingInfo()
                        bond_rings = ri.BondRings()

                        core_atoms = None
                        broken_bond = None  # (u, v) in Closed

                        for b in sk_c.GetBonds():
                            if not b.IsInRing(): continue
                            u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()

                            if u not in amap or v not in amap: continue

                            # 关键判据：Open 态中距离是否断开 (>1)
                            dist_open = shortest_dist(sk_o, amap[u], amap[v])

                            if dist_open >= 2:
                                # 找到了反应键！
                                bid = b.GetIdx()
                                # 找包含这根键的最小环 (5元或6元环)
                                rings = [list(r) for r in bond_rings if bid in r]
                                rings.sort(key=len)

                                for r_bonds in rings:
                                    # 获取环上的原子 ID
                                    r_atoms = set()
                                    for rb in r_bonds:
                                        b_obj = sk_c.GetBondWithIdx(rb)
                                        r_atoms.add(b_obj.GetBeginAtomIdx())
                                        r_atoms.add(b_obj.GetEndAtomIdx())

                                    if len(r_atoms) in [5, 6]:
                                        # 必须确保环上原子顺序正确
                                        for ar in ri.AtomRings():
                                            if set(ar) == r_atoms:
                                                core_atoms = list(ar)
                                                break
                                        broken_bond = (u, v)
                                        break
                                    if core_atoms: break
                                if core_atoms: break

                        # 4. 如果找到了核心，开始计算物理量
                        if core_atoms:
                            # 扩展核心：包括与之稠合的芳香环 (侧翼)
                            core_set = set(core_atoms)
                            all_rings = ri.AtomRings()
                            fused_pool = set(core_atoms)

                            # 简单的膨胀算法：如果有2个原子共享，就吃进来
                            while True:
                                added = False
                                for r in all_rings:
                                    if not set(r).issubset(fused_pool) and len(set(r).intersection(fused_pool)) >= 2:
                                        fused_pool.update(r)
                                        added = True
                                if not added: break

                            # 生成 3D 构象 (MMFF 优化)
                            mo3 = Chem.AddHs(mol_o)
                            mc3 = Chem.AddHs(mol_c)

                            ps = AllChem.ETKDG()
                            ps.useRandomCoords = True
                            ps.maxIterations = 200

                            res_o = AllChem.EmbedMolecule(mo3, ps)
                            res_c = AllChem.EmbedMolecule(mc3, ps)

                            if res_o >= 0 and res_c >= 0:
                                try:
                                    AllChem.MMFFOptimizeMolecule(mo3)
                                    AllChem.MMFFOptimizeMolecule(mc3)

                                    # --- 计算 A. dHOMA ---
                                    # Closed 态 (全闭合)
                                    h_c_cl = _calc_homa_unit(mc3, core_atoms, 'ring')  # 中心环

                                    # Open 态 (中心环断开)
                                    core_aligned = align_chain_atoms(core_atoms, broken_bond[0], broken_bond[1])
                                    h_c_op = _calc_homa_unit(mo3, [amap[x] for x in core_aligned if x in amap], 'chain')

                                    # 侧翼环 (Periphery)
                                    h_p_cl = 0.0
                                    h_p_op = 0.0
                                    for ar in all_rings:
                                        # 是稠合环但不是中心环
                                        if set(ar).issubset(fused_pool) and not set(ar).issubset(core_set):
                                            h_p_cl += _calc_homa_unit(mc3, list(ar), 'ring')
                                            # 对应到 Open 态
                                            h_p_op += _calc_homa_unit(mo3, [amap[x] for x in ar if x in amap], 'ring')

                                    # HOMA 差值 (反映芳香性恢复驱动力)
                                    # Open 态通常 HOMA 高 (芳香性好)，Closed 态低
                                    # Delta = (Closed总) - (Open总)
                                    # 预期为负值，越负说明开环驱动力越大
                                    dHOMA = (h_c_cl + h_p_cl) - (h_c_op + h_p_op)

                                    # --- 计算 B. dQ (Charge Transfer) ---
                                    # 使用 Gasteiger-Marsili
                                    AllChem.ComputeGasteigerCharges(mo3)
                                    AllChem.ComputeGasteigerCharges(mc3)

                                    # 取中心环原子的最小电荷 (通常是电子积聚点)
                                    q_c = np.min(
                                        [float(mc3.GetAtomWithIdx(i).GetProp("_GasteigerCharge")) for i in core_atoms])
                                    q_o = np.min([float(mo3.GetAtomWithIdx(i).GetProp("_GasteigerCharge")) for i in
                                                  [amap[x] for x in core_aligned if x in amap]])

                                    dQ = q_c - q_o

                                    # --- 计算 C. Bond Length (Strain) ---
                                    conf_c = mc3.GetConformer()
                                    u_idx, v_idx = broken_bond
                                    p_u = conf_c.GetAtomPosition(u_idx)
                                    p_v = conf_c.GetAtomPosition(v_idx)
                                    r_bond = (p_u - p_v).Length()

                                    vec = [dHOMA, dQ, r_bond]

                                except Exception as e:
                                    # MMFF 或 计算出错
                                    pass
        except Exception as e:
            # 严重拓扑错误
            pass

        feats.append(vec)

    return np.array(feats)


# ==============================================================================
# 模块 3：深度特征引擎 (Deep Tower)
# ==============================================================================

def get_chemberta_pca(smiles_list, n_components=10):
    """
    提取 ChemBERTa 特征并 PCA 降维
    """
    print(f"\n[M3] Deep Tower: Extracting ChemBERTa embeddings (PCA={n_components})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    embeddings = []
    # Batch inference could be faster, but loop is safer for small N
    for smi in tqdm(smiles_list, desc="  -> Inference"):
        try:
            inputs = tokenizer(smi, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            # 取 CLS token (index 0)
            embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
        except:
            embeddings.append(np.zeros((1, 768)))

    X_raw = np.vstack(embeddings)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"  -> PCA Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    return X_pca


# ==============================================================================
# 模块 4：混合模型架构 (RF + GP Residual)
# ==============================================================================

class HybridRFGP:
    """
    RF-Guided Gaussian Process Residual Learner
    策略：
    1. RF 负责基准预测 (Baseline)
    2. GP 负责拟合残差 (Residual = True - RF_OOF)
    3. 集成核: Matern(nu=2.5) with ARD
    """

    def __init__(self, rf_estimators=100, gp_restarts=5):
        # 弱化 RF 以防过拟合 (max_depth 限制)
        self.rf = RandomForestRegressor(n_estimators=rf_estimators, max_depth=5, random_state=42)

        # GP Kernel: Constant * Matern(ARD) + WhiteNoise
        # ARD: length_scale 为数组，允许对 13 个特征赋予不同权重
        # 初始 length_scale 设为 1.0
        dims = 13  # 10 Deep + 3 Phys
        kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0] * dims, length_scale_bounds=(1e-2, 1e2), nu=2.5) \
                 + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=gp_restarts, normalize_y=True)
        self.scaler = StandardScaler()

    def fit_predict_loocv(self, X, y):
        """
        执行严格的 LOOCV 流程。
        注意：为了严谨，我们在每个 Fold 内部重新训练 RF 和 GP。
        """
        loo = LeaveOneOut()
        y_preds = []
        y_stds = []

        # 将输入标准化，有助于 GP 收敛
        X_s = self.scaler.fit_transform(X)

        iterator = tqdm(loo.split(X_s), total=len(X_s), desc="  -> Training Hybrid Model (LOOCV)")

        for train_idx, test_idx in iterator:
            X_tr, X_te = X_s[train_idx], X_s[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # --- Step A: Generate RF OOF on Training Set ---
            # 我们需要用 RF 的 OOF 残差来训练 GP，而不是 RF 的训练集残差（那是作弊）
            # 内部再做一次 KFold
            rf_oof_train = np.zeros_like(y_tr)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for k_tr, k_val in kf.split(X_tr):
                # 训练临时 RF
                rf_temp = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=42)
                rf_temp.fit(X_tr[k_tr], y_tr[k_tr])
                rf_oof_train[k_val] = rf_temp.predict(X_tr[k_val])

            # 计算训练集残差
            residuals_tr = y_tr - rf_oof_train

            # --- Step B: Train GP on Residuals ---
            self.gp.fit(X_tr, residuals_tr)

            # --- Step C: Train Final RF on full Training Set (for inference) ---
            self.rf.fit(X_tr, y_tr)

            # --- Step D: Inference on Test Sample ---
            # 1. RF Base Prediction
            base_pred = self.rf.predict(X_te)[0]

            # 2. GP Residual Prediction (Mean + Std)
            res_pred, res_std = self.gp.predict(X_te, return_std=True)

            final_pred = base_pred + res_pred[0]

            y_preds.append(final_pred)
            y_stds.append(res_std[0])

        return np.array(y_preds), np.array(y_stds)


# ==============================================================================
# 模块 5：主程序执行 (Execution)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV file path")
    args = parser.parse_args()

    # 1. 创建带时间戳的输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"Result_AIn37_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"🚀 AIn37: Starting Hybrid RF+GP Pipeline...")
    print(f"📂 Output Directory: {out_dir}")

    # 2. 读取与清洗数据
    for enc in ['utf-8', 'gbk', 'latin1']:
        try:
            df = pd.read_csv(args.csv, encoding=enc); break
        except:
            continue

    # 剔除 P-type (基于 AIn34/35 的发现)
    # indices: 34, 37, 45 (Python 0-indexed)
    drop_indices = [34, 37, 45]
    print(f"✂️  Removing known P-type outliers: {drop_indices}")
    df = df.drop(index=drop_indices, errors='ignore').reset_index(drop=True)

    prep = MoleculePreprocessor()
    df["_SMI_O"] = [prep.canonicalize(s) for s in df["SMILES_o"]]
    df["_SMI_C"] = [prep.canonicalize(s) for s in df["SMILES_c"]]
    df = df.dropna(subset=["_SMI_O", "_SMI_C"]).reset_index(drop=True)

    # 目标值处理 (Log scale)
    if "t_half_s" in df.columns:
        df["Y"] = df["t_half_s"]
    elif "t_half_ms" in df.columns:
        df["Y"] = df["t_half_ms"] / 1000.0
    elif "log10_s" in df.columns:
        df["Y"] = 10 ** df["log10_s"]

    # 限制范围，防止数值错误
    df["Y"] = df["Y"].clip(1e-9, 1e18)
    y_reg = np.log10(df["Y"].values)

    print(f"📊 Dataset Size: {len(df)} (T-type)")

    # 3. 特征工程
    # A. Deep (PCA-10)
    X_deep = get_chemberta_pca(df["_SMI_O"].tolist(), n_components=10)

    # B. Physics (HOMA, dQ, Rbond)
    X_phys = get_physics_features(df)

    # 特征审计
    phys_df = pd.DataFrame(X_phys, columns=["dHOMA", "dQ", "Rbond"])
    print("\n🔍 Physics Feature Audit:")
    print(phys_df.describe().T[["mean", "std", "min", "max"]])

    # C. 特征融合
    # 注意：我们先不做标准化，留给模型内部的 Scaler 统一处理
    X_final = np.hstack([X_deep, X_phys])
    print(f"🔗 Fused Feature Shape: {X_final.shape} (10 Deep + 3 Phys)")

    # 4. 训练与预测 (LOOCV)
    model = HybridRFGP()
    preds, stds = model.fit_predict_loocv(X_final, y_reg)

    # 5. 结果分析
    r2 = r2_score(y_reg, preds)
    mae = mean_absolute_error(y_reg, preds)

    print(f"\n🏆 Final Results (RF+GP):")
    print(f"   R2  : {r2:.4f}")
    print(f"   MAE : {mae:.4f}")

    # 6. 生成详细报告
    df["Pred_Log_s"] = preds
    df["Uncertainty_Sigma"] = stds
    df["Error_Abs"] = np.abs(df["Pred_Log_s"] - y_reg)

    # 离群检测：Error > 1.5 或 Sigma > 2*Mean_Sigma
    mean_sigma = np.mean(stds)
    df["Is_Outlier"] = (df["Error_Abs"] > 1.5) | (df["Uncertainty_Sigma"] > 2 * mean_sigma)

    # 保存结果
    res_path = os.path.join(out_dir, "prediction_results.csv")
    df.to_csv(res_path, index=False)
    print(f"\n💾 Detailed results saved to: {res_path}")

    # 打印 Top 3 异常点
    print("\n🚨 Top 3 High Uncertainty/Error Samples:")
    outliers = df.sort_values("Error_Abs", ascending=False).head(3)
    for idx, row in outliers.iterrows():
        print(f"   Idx {idx}: True={y_reg[idx]:.2f}, Pred={row['Pred_Log_s']:.2f}, "
              f"Sigma={row['Uncertainty_Sigma']:.2f}, Error={row['Error_Abs']:.2f}")


if __name__ == "__main__":
    main()