# #!/usr/bin/env python3
# """
# Rerun PPC without retraining or re-running SRIM.
# Loads summarized SRIM results and runs posterior predictive check.
# """

# import torch
# import pandas as pd
# from pathlib import Path

# from src.utils.data_utils import preprocess
# # from src.utils.data_utils import plot_ppc_histograms_per_track
# from src.utils.srim_parser import summarize_all_runs

# # ---------------------------------------------------------------------
# # Load existing data
# # ---------------------------------------------------------------------

# # 1. Load x_obs from preprocessing
# x_obs, theta, grouped = preprocess(
#     "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/all_vacancies.csv"
# )

# # 2. (Optional) Define which tracks were used for x_test originally
# selected_track_ids = [
#     26559, 21550, 331, 16708, 976,
#     39544, 21135, 2575, 16075, 29780
# ]

# # 3. Load SRIM summarized results (already generated)
# summary_df = pd.read_csv(
#     "/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013/Outputs/srim_summary.csv"
# )


# # --- Drop duplicates so we only have one row per track_id ---
# unique_tracks = summary_df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)

# # --- Sanity check: shapes should match ---
# print(f"x_obs shape: {x_obs.shape}")
# print(f"unique_tracks: {unique_tracks.shape[0]} unique tracks")
# assert x_obs.shape[0] == unique_tracks.shape[0], "Mismatch: x_obs rows != unique track_ids"

# # --- Verify first few mappings ---
# print("\nVerifying first 10 rows:")
# for i in range(min(10, len(unique_tracks))):
#     tid = int(unique_tracks.loc[i, "track_id"])
#     obs = x_obs[i].tolist()
#     print(f"Index {i:02d}  →  track_id={tid:<6}  mean_x={obs[0]:.3f}, std_x={obs[1]:.3f}, num_vac={obs[2]:.1f}")



# # ---------------------------------------------------------------------
# # Run PPC
# # ---------------------------------------------------------------------

# # metrics = plot_ppc_histograms_per_track(
# #     df=df_summary,
# #     observed=observed_dict,
# #     output_dir="/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/ppc-rerun",
# #     bins=40,
# #     save_plots=True,
# #     return_metrics=True
# # )

# # print("\n✅ PPC completed successfully.")

import pandas as pd

from src.utils.analysis_utils import clean_summary_data


#  Load the summary file (path may differ depending on where summarize_all_runs saved it)
df = pd.read_csv("/Users/cbharathulwar/Documents/Research/Walsworth/SRIM-2013/Outputs/srim_summary.csv")
df_clean = clean_summary_data(df=df)


# Print shape and columns
print("\nDataFrame shape:", df_clean.shape)
print("Columns:", list(df_clean.columns))

# Print first few rows
print("\nFirst 5 rows:")
print(df_clean.head())


# Verify how many SRIM runs per track_id
track_counts = df_clean['track_id'].value_counts().sort_index()

print("\n[INFO] SRIM runs per track_id:")
print(track_counts)

print(f"\n[INFO] Total SRIM runs: {track_counts.sum()} across {len(track_counts)} tracks.")
print(f"[INFO] Average runs per track: {track_counts.mean():.2f}")