import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.utils.data_utils import preprocess_mnpe

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_CSV = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/siimpl_800keV_train.csv")
N_BINS = 6

def analyze_summary_features():
    print(f"⚙️ Preprocessing Training Data: {TRAIN_CSV}")
    
    # 1. Run the actual preprocessor
    # We want the 'df_summary' which contains the raw physics features
    _, _, _, _, df_summary = preprocess_mnpe(TRAIN_CSV, n_bins=N_BINS)
    
    if df_summary.empty:
        print("❌ No data processed. Check your CSV path and column names.")
        return

    # 2. Define features to analyze
    # Added 'mean_depth_A' and 'max_depth_A' to check for "location leaks"
    features_to_check = ['skew_z', 'var_diff', 'asym_count_centered', 'mean_depth_A', 'max_depth_A']
    
    # 3. Calculate Separation Scores (Fisher Score)
    # Score = |mean_0 - mean_1| / sqrt(var_0 + var_1)
    print("\n📊 FEATURE SEPARATION SCORES (Higher is better)")
    print("-" * 45)
    for feat in features_to_check:
        if feat in df_summary.columns:
            g0 = df_summary[df_summary['parity'] == 0][feat]
            g1 = df_summary[df_summary['parity'] == 1][feat]
            
            if len(g0) > 1 and len(g1) > 1:
                score = abs(g0.mean() - g1.mean()) / np.sqrt(g0.var() + g1.var() + 1e-9)
                print(f"   {feat:<25} : {score:.4f}")

    # 4. Plot Distributions
    print("\n📈 Generating Distribution Plots...")
    valid_features = [f for f in features_to_check if f in df_summary.columns]
    num_feats = len(valid_features)
    
    fig, axes = plt.subplots(1, num_feats, figsize=(5 * num_feats, 5))
    if num_feats == 1: axes = [axes] # Handle single feature case
    
    for i, feat in enumerate(valid_features):
        sns.kdeplot(data=df_summary, x=feat, hue="parity", 
                    fill=True, common_norm=False, ax=axes[i], palette="viridis")
        axes[i].set_title(f"Distribution of {feat}")
        axes[i].set_xlabel("Feature Value")
    
    plt.tight_layout()
    plt.show()

    # 5. Correlation Check
    # Check if features are just repeating each other
    print("\n🔗 Feature Correlation Matrix")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_summary[valid_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Inter-Feature Correlation")
    plt.show()

if __name__ == "__main__":
    analyze_summary_features()