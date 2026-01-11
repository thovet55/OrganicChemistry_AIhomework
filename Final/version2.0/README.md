## 1. Environment Setup

```bash
# Create environment
conda create -n dae_v2 python=3.9 -y

# Activate
conda activate dae_v2

# Install Base Dependencies (RDKit)
conda install -c conda-forge rdkit scikit-learn pandas numpy -y

# Install Deep Learning Dependencies
pip install torch transformers tqdm
```

## 2. Input Data Format

**A. Training Data (`data.csv`)** - _The knowledge base_

|**open_smiles**|**closed_smiles**|**t_half_ms**|
|---|---|---|
|`CC1=...`|`C12S...`|`22.6`|

**B. Prediction Data (`target.csv`)** - _Your new molecules_

|**open_smiles**|**closed_smiles**|
|---|---|
|`CC1=...`|`C12S...`|

## 3. Execution

Place `version2.py`, `data.csv` (training set), and your target CSV in the same folder.

**Basic Usage:**

Bash

```
python version2.py --csv target.csv
```

**Advanced Usage:**

Bash

```
python version2.py --csv target.csv --train_data my_database.csv --output my_results.csv
```

> **Note:** On the first run, the script will automatically download the ChemBERTa model (~350MB). Please ensure internet access.

_(Results will be saved to `V2_Predictions.csv`)_