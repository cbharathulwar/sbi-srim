import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION (Mac Paths)
# ==========================================
# Using the path you provided in previous turns
BASE_PATH = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data")
FILE_TRAIN = BASE_PATH / "MCPE_TRAIN.csv"
FILE_EVAL = BASE_PATH / "MCPE_EVAL.csv"

# The batch size used in your generator (1000 is standard)
BATCH_SIZE = 1000 

def repair_ion_ids(file_path):
    print(f"Repairing {file_path.name}...")
    
    # Read the file
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"  [ERROR] File not found: {file_path}")
        return 0

    # 1. Detect where the ion number resets (e.g. 999 -> 0)
    # The '.diff()' calculates the difference between row N and row N-1.
    # Whenever this is negative (e.g. 0 - 999 = -999), a new batch has started.
    # .cumsum() counts how many times this reset has happened so far.
    df['batch_id'] = (df['ion_number'].diff() < 0).cumsum()
    
    # 2. Create the REAL unique ID
    # We add (batch_id * 1000) to the original number.
    # Batch 0: Ion 0 -> 0
    # Batch 1: Ion 0 -> 1000
    # Batch 2: Ion 0 -> 2000
    df['ion_number_fixed'] = df['ion_number'] + (df['batch_id'] * BATCH_SIZE)
    
    # 3. Check our work (Compare "Before" vs "After" unique counts)
    # We group by Energy + Ion Number to see how many distinct tracks exist.
    old_unique = df.groupby(['energy_keV', 'ion_number']).ngroups
    new_unique = df.groupby(['energy_keV', 'ion_number_fixed']).ngroups
    
    print(f"  -> Raw Merged Tracks (Before): {old_unique}")
    print(f"  -> Real Unique Tracks (After): {new_unique}")
    
    # 4. Save the Fix
    # Overwrite the old 'ion_number' with the fixed one
    df['ion_number'] = df['ion_number_fixed']
    
    # Drop helper columns to keep file clean
    df.drop(columns=['batch_id', 'ion_number_fixed'], inplace=True)
    
    # Save to a new file named "FIXED_..."
    save_path = file_path.parent / f"FIXED_{file_path.name}"
    df.to_csv(save_path, index=False)
    print(f"  -> Saved fixed data to: {save_path.name}")
    
    return new_unique

# ==========================================
# EXECUTE REPAIR
# ==========================================
print("Starting Repair Process...\n")

count_train = repair_ion_ids(FILE_TRAIN)
print("-" * 30)
count_eval = repair_ion_ids(FILE_EVAL)

print("\n" + "="*30)
print(f"TOTAL RECOVERED TRACKS: {count_train + count_eval}")
print("="*30)