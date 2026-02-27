import pandas as pd
from pathlib import Path

# ==========================================
# Config
# ==========================================
DATA_DIR = Path("data/processed_splits")
INPUT_CSV = DATA_DIR / "shared_eval.csv"
OUTPUT_CSV = DATA_DIR / "shared_eval_5k.csv"

TRACKS_PER_GROUP = 312  # 312 tracks * 8 energies * 2 parities = 4,992 tracks

# The exact discrete energies we want to evaluate
ENERGIES_TO_PLOT = [1.0, 3.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0]

def main():
    print(f"Loading full evaluation dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # 1. Isolate the unique tracks and their properties
    print("Extracting unique tracks...")
    unique_tracks = df[['energy_keV', 'ion_number', 'parity']].drop_duplicates()
    
    # --- THE FIX: Filter down to ONLY the 8 target energies ---
    unique_tracks = unique_tracks[unique_tracks['energy_keV'].isin(ENERGIES_TO_PLOT)]
    
    print(f"Total unique target tracks found: {len(unique_tracks)}")
    
    # 2. Sample evenly across Energy and Parity combinations
    print(f"\nSampling exactly {TRACKS_PER_GROUP} tracks per Energy/Parity group...")
    
    # Group by energy and parity, then sample (group_keys=False silences the Pandas warning)
    sampled_tracks = (
        unique_tracks.groupby(['energy_keV', 'parity'], group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), TRACKS_PER_GROUP), random_state=42))
        .reset_index(drop=True)
    )
    
    # Print a quick verification table
    print("\nVerification (Tracks per group):")
    print(sampled_tracks.groupby(['energy_keV', 'parity']).size())
    print(f"\nTotal sampled tracks: {len(sampled_tracks)}")
    
    # 3. Pull all the actual point-cloud data for just those selected tracks
    print("\nMerging point-cloud data for the sampled tracks...")
    df_balanced = df.merge(
        sampled_tracks[['energy_keV', 'ion_number']], 
        on=['energy_keV', 'ion_number'], 
        how='inner'
    )
    
    # 4. Save to the new file
    print(f"Saving balanced dataset to {OUTPUT_CSV}...")
    df_balanced.to_csv(OUTPUT_CSV, index=False)
    print("Done! You are ready to run the master pipeline.")

if __name__ == "__main__":
    main()