import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === CONFIGURATION ===
PPC_FILE = Path('/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/ppc-results3/per_track/PPC_metrics_all_tracks_20251012_145937.csv')
SAVE_DIR = PPC_FILE.parent  # same folder as the CSV
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(PPC_FILE)

# === PLOT 1: z-score distribution ===
plt.figure(figsize=(7, 4))
sns.histplot(df["z_score"], bins=20, kde=True, color="steelblue")
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Distribution of PPC z-scores (All Features)")
plt.xlabel("z-score (standardized deviation)")
plt.ylabel("Frequency")
plt.tight_layout()

save_path = SAVE_DIR / "zscore_distribution.png"
plt.savefig(save_path, dpi=300)
print(f"[SAVED] {save_path}")
plt.show()


# === PLOT 2: percent difference per feature ===
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="delta_percent", hue="feature", bins=20, kde=True)
plt.axvline(0, color="black", linestyle="--", linewidth=1)
plt.title("Posterior Predictive Error (% difference)")
plt.xlabel("Percent Difference (Predicted - Observed)")
plt.ylabel("Count")
plt.tight_layout()

save_path = SAVE_DIR / "percent_difference_distribution.png"
plt.savefig(save_path, dpi=300)
print(f"[SAVED] {save_path}")
plt.show()


# === PLOT 3: observed vs simulated ===
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="obs_val", y="mu_sim", hue="feature", s=60)
plt.plot([df["obs_val"].min(), df["obs_val"].max()],
         [df["obs_val"].min(), df["obs_val"].max()],
         "k--", label="y = x")
plt.title("Observed vs Simulated Means per Feature")
plt.xlabel("Observed (SRIM)")
plt.ylabel("Simulated (Posterior Mean)")
plt.legend()
plt.tight_layout()

save_path = SAVE_DIR / "observed_vs_simulated.png"
plt.savefig(save_path, dpi=300)
print(f"[SAVED] {save_path}")
plt.show()