
# Intro

## 1. Environment Setup (Bash)
```bash
# Create environment
conda create -n dae_env python=3.9 -y

# Activate
conda activate dae_env

# Install dependencies (RDKit must use conda-forge)
conda install -c conda-forge rdkit scikit-learn pandas numpy -y
````

## 2. Input Data Format

**Training Data (`data.csv`)** | SMILES_o | SMILES_c | t_half_ms | | :--- | :--- | :--- | | `CC1=...` | `C12S...` | `22.6` |

**Prediction Data (`predict_csv`)** | SMILES_o | SMILES_c | | :--- | :--- | | `CC1=...` | `C12S...` |

## 3. Execution

Put `version1_predict.py`, `data.csv`, and `predict_csv` in the same folder.

Bash

```
python version1_predict.py
```

_(Results will be saved to `prediction_results.csv`)_