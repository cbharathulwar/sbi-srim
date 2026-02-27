import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==========================================================
# CONFIG
# ==========================================================

MNPE_EVAL = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results/Nov20_mnpe_fixed_eval.csv")
ALLTRACKS_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results/Nov20Run_FixedEval")
OUTPUT_DIR = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/results")

CALIBRATION_SPLIT = 0.8
RANDOM_SEED = 42
TARGET_FP_RATE = 0.05


# ==========================================================
# ASYMMETRY
# ==========================================================

def compute_asymmetry(track_df):

    if 'x' not in track_df.columns:
        return np.nan
        
    s = track_df["x"].values
    if len(s) < 3:
        return np.nan

    s_min, s_max = s.min(), s.max()
    L = s_max - s_min
    if L <= 0:
        return np.nan

    b_end = s_min + L/3
    e_start = s_min + 2*L/3

    N_begin = np.sum((s >= s_min) & (s < b_end))
    N_end   = np.sum((s >= e_start) & (s <= s_max))

    if N_begin == 0:
        return np.nan

    return N_end / N_begin


# ==========================================================
# PAPER THRESHOLD
# ==========================================================

def find_paper_threshold(asymmetries, target_fp=0.05):

    A = np.asarray(asymmetries)
    A = A[np.isfinite(A)]

    if len(A) == 0:
        return None

    candidates = np.sort(np.unique(A))

    for T in candidates:

        selected = A > T
        if selected.sum() == 0:
            continue

        # paper definition of FP
        fp = ((A < 1) & selected).sum() / selected.sum()

        if fp <= target_fp:
            return T

    return 1.0


# ==========================================================
# LOAD DATA
# ==========================================================

print("="*70)
print("RAJENDRAN PAPER METHOD (FINAL)")
print("="*70)

df_full = pd.read_csv(MNPE_EVAL)
print("Tracks:", len(df_full))


# ==========================================================
# SPLIT
# ==========================================================

calibration_dfs = []
test_dfs = []

for energy in sorted(df_full.energy_keV.unique()):

    energy_df = df_full[df_full.energy_keV == energy]

    cal, test = train_test_split(
        energy_df,
        test_size=1-CALIBRATION_SPLIT,
        stratify=energy_df.true_parity,
        random_state=RANDOM_SEED
    )

    calibration_dfs.append(cal)
    test_dfs.append(test)

    print(f"{energy:5.1f} keV -> {len(cal)} cal | {len(test)} test")

df_cal = pd.concat(calibration_dfs)
df_test = pd.concat(test_dfs)


# ==========================================================
# CALIBRATION
# ==========================================================

print("\nCALIBRATION")

thresholds = {}

for energy in sorted(df_cal.energy_keV.unique()):

    print(f"\n[{energy:.1f} keV]")

    file = ALLTRACKS_DIR / f"{energy:.1f}keV" / f"{energy:.1f}keV_alltracks.csv"
    if not file.exists():
        print("missing file")
        continue

    tracks_df = pd.read_csv(file)
    energy_cal = df_cal[df_cal.energy_keV == energy]

    asym = []

    for ion in energy_cal.ion_number:

        ion_df = tracks_df[tracks_df.ion_number == ion]
        if len(ion_df) == 0:
            continue

        A = compute_asymmetry(ion_df)
        if np.isfinite(A):
            asym.append(A)

    if len(asym) == 0:
        print("no valid asymmetry")
        continue

    asym = np.array(asym)
    T = find_paper_threshold(asym, TARGET_FP_RATE)
    thresholds[energy] = T

    print("tracks:", len(asym))
    print("T =", round(T,3))


# ==========================================================
# TEST
# ==========================================================

print("\nTEST RESULTS")

results = []

for energy in sorted(df_test.energy_keV.unique()):

    if energy not in thresholds:
        continue

    T = thresholds[energy]
    print(f"\n[{energy:.1f} keV]  T={T:.3f}")

    file = ALLTRACKS_DIR / f"{energy:.1f}keV" / f"{energy:.1f}keV_alltracks.csv"
    tracks_df = pd.read_csv(file)

    energy_test = df_test[df_test.energy_keV == energy]

    A_list = []
    truth = []

    for _,row in energy_test.iterrows():

        ion_df = tracks_df[tracks_df.ion_number == row.ion_number]
        if len(ion_df)==0:
            continue

        A = compute_asymmetry(ion_df)
        if not np.isfinite(A):
            continue

        A_list.append(A)
        truth.append(row.true_parity)

    A = np.array(A_list)
    truth = np.array(truth)

    # ==================================================
    # PAPER CLASSIFICATION
    # ==================================================

    selected = A > T
    n_selected = selected.sum()

    efficiency = n_selected / len(A)

    if n_selected > 0:
        paper_fp = ((A < 1) & selected).sum() / n_selected
        accuracy = (truth[selected] == 1).mean()
    else:
        paper_fp = 0
        accuracy = 0

    effective_accuracy = efficiency * accuracy

    print("selected:", n_selected, "/", len(A))
    print("efficiency:", round(efficiency,3))
    print("paper FP:", round(paper_fp,3))
    print("accuracy:", round(accuracy,3))

    results.append({
        "energy":energy,
        "threshold":T,
        "efficiency":efficiency,
        "accuracy":accuracy,
        "effective_accuracy":effective_accuracy,
        "paper_fp":paper_fp,
        "n_total":len(A),
        "n_selected":n_selected
    })


# ==========================================================
# SAVE
# ==========================================================

res_df = pd.DataFrame(results)
out = OUTPUT_DIR/"rajendran_true_method.csv"
res_df.to_csv(out,index=False)

print("\nSaved:",out)
print(res_df)