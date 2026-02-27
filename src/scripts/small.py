import pandas as pd
import numpy as np
from pathlib import Path

# SETTINGS
INPUT_PATH = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/mcpe-3d/mcpe_3d_eval.csv"
OUTPUT_PATH = "/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data/mcpe-3d/mcpe_3d_eval_10k.csv"
TARGET_TRACKS = 10000

def create_balanced_subset():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    
    # 1. Identify Unique Tracks
    # Group by the columns that define a single track
    if 'run_id' in df.columns:
        track_cols = ['run_id', 'energy_keV', 'ion_number']
    else:
        track_cols = ['energy_keV', 'ion_number']
        
    # Get a list of all unique track identifiers
    unique_tracks = df[track_cols].drop_duplicates()
    total_tracks = len(unique_tracks)
    print(f"Found {total_tracks} total unique tracks.")
    
    # 2. Stratified Sampling (Balance across Energy)
    # We want to sample equally from low, medium, and high energy to avoid bias.
    # We bin energy into 10 bins and sample evenly from each.
    unique_tracks['energy_bin'] = pd.qcut(unique_tracks['energy_keV'], q=10, labels=False)
    
    tracks_per_bin = TARGET_TRACKS // 10
    
    sampled_tracks = unique_tracks.groupby('energy_bin', group_keys=False).apply(
        lambda x: x.sample(min(len(x), tracks_per_bin), random_state=42)
    )
    
    print(f"Sampled {len(sampled_tracks)} tracks (Target: {TARGET_TRACKS}).")
    
    # 3. Filter the Original Data
    # We merge the sampled IDs back with the original dataframe to get the points
    # This keeps ALL points for the selected tracks.
    subset_df = df.merge(sampled_tracks[track_cols], on=track_cols, how='inner')
    
    # 4. Save
    print(f"Saving {len(subset_df)} rows to {OUTPUT_PATH}...")
    subset_df.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    create_balanced_subset()