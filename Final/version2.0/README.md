# Version 2.0 (Experimental)

> **Status**: **Archived / Research Record Only**
> 
> **Summary**: This version represents an experimental transition from classical informatics to deep learning. We attempted to replace explicit fingerprints with **ChemBERTa Embeddings** (reduced via PCA) to capture latent chemical semantics. The pipeline also implemented a **Dual-Model Comparison (RF vs. GP)** to exhaustively evaluate feature quality.
> 
> **Outcome**: Empirical results indicated that **deep learning features did not outperform the Version 1.0 baseline** ($R^2 \approx 0.39$ vs. $0.72$), likely due to the "curse of dimensionality" on our small dataset. Consequently, **no deployment-ready prediction module is provided** for this version.

### Updates

1. **Feature Encoder Upgrade**: Utilized **ChemBERTa** on **Open-form SMILES** to capture the initial chemical semantic state, replacing the traditional Morgan Fingerprints.
    
2. **PCA Denoising**: Applied **Principal Component Analysis (PCA)** with a hard limit of **10 components** to strictly control model complexity and filter high-dimensional noise.
    
3. **Physics Tower Refinement**: Integrated **dHOMA (Sum of Core & Fused Rings)** and **dQ (Min-Charge Difference)** as explicit physical constraints.
    
4. **Dual-Model Evaluation**: Implemented a parallel training pipeline for **Random Forest (RF)** and **Gaussian Process (GP)**. This was designed to test if the continuous nature of GP could better utilize the dense embedding space compared to RF.
    

### Features

1. **ChemBERTa (Open-Form State)**
    
    - **Encoder**: `Seyonec/ChemBERTa-zinc-base-v1` (Transformer-based).
        
    - **Input**: The SMILES string of the **Open-ring isomer** is used to capture the "starting point" semantics.
        
2. **PCA Denoising & Dimensionality Reduction**
    
    - **Mechanism**: The raw $768D$ output is compressed to $10D$.
        
    - **Rationale**: Given the extremely small sample size, this bottleneck was intended to force the model to learn dominant signals. However, results suggest that critical structural nuances might have been lost during this compression or were overwhelmed by embedding noise.
        
3. Physical Features (Physics Tower)
    
    To compensate for the "black-box" nature of embeddings, we calculated:
    
    - **dHOMA (Sum)**: Aggregates aromaticity changes of the **Central Ring** and all **Fused/Side Rings**.
        
    - **dQ (Charge Diff)**: Captures the shift in electron density at the reactive core ($Q_{\text{min}}(\text{Closed}) - Q_{\text{min}}(\text{Open})$).
        

### Performance Note

Despite the theoretical advantages of Transformer models, the **LOOCV results ($R^2 \approx 0.39$) were unsatisfactory** compared to the classical approach. The ChemBERTa embeddings, even after PCA, appeared to introduce more variance than signal for this specific small-scale task. Therefore, this version serves solely as a **comparative benchmark** to demonstrate the limitations of applying large language models to micro-datasets without extensive fine-tuning.