import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = Path("/Users/cbharathulwar/Documents/Research/Walsworth/Code/SBI/srim-sbi/data")
FILE_TRAIN = BASE_PATH / "MCPE_TRAIN.csv"
FILE_EVAL = BASE_PATH / "MCPE_EVAL.csv"

# Output Directory
OUTPUT_DIR = BASE_PATH / "mcpe-3d"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_SPLIT = 0.85 
START_AXIS = np.array([1, 0, 0]) 

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def generate_fibonacci_sphere(n):
    if n <= 0: return np.empty((0, 3))
    if n == 1: return np.array([[1.0, 0.0, 0.0]])
    phi = np.pi * (3. - np.sqrt(5.))
    i = np.arange(n)
    y = 1 - 2 * (i + 0.5) / n
    radius = np.sqrt(np.maximum(0, 1 - y * y))
    theta = phi * i
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.column_stack((x, y, z))

def get_rotation_matrix(vec_start, vec_end):
    a = vec_start / np.linalg.norm(vec_start)
    b = vec_end / np.linalg.norm(vec_end)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    
    if s < 1e-9 and c > 0: return np.eye(3)
    if s < 1e-9 and c < 0:
        if np.abs(a[0]) < 0.9: axis = np.cross(a, np.array([1, 0, 0]))
        else: axis = np.cross(a, np.array([0, 1, 0]))
        axis = axis / np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
        
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

# ==========================================
# 2. MAIN PIPELINE
# ==========================================
print(f"[1/4] Loading and Combining Data...")

# Load separately
df1 = pd.read_csv(FILE_TRAIN)
df2 = pd.read_csv(FILE_EVAL)

# Tag them to prevent merging duplicates
df1['run_id'] = 0
df2['run_id'] = 1

df_full = pd.concat([df1, df2], ignore_index=True)

# Group by 3 keys now
grouped = df_full.groupby(["run_id", "energy_keV", "ion_number"], sort=False)

num_unique_tracks = len(grouped)
print(f"      Detected {num_unique_tracks} total unique tracks.")

# ==========================================
# 3. ROTATION LOOP
# ==========================================
print(f"[2/4] Generating Vectors & Rotating...")
target_vectors = generate_fibonacci_sphere(num_unique_tracks)
all_rotated_data = []

# FIX 1: Unpack 3 variables ((run_id, E, ion), g)
for idx, ((run_id, E, ion), g) in enumerate(grouped):
    pts = g[["x", "y", "z"]].values
    target_v = target_vectors[idx]
    
    rot_mat = get_rotation_matrix(START_AXIS, target_v)
    rotated_pts = pts @ rot_mat.T
    
    new_rows = g.copy()
    new_rows[["x", "y", "z"]] = rotated_pts
    new_rows["target_vx"] = target_v[0]
    new_rows["target_vy"] = target_v[1]
    new_rows["target_vz"] = target_v[2]
    
    all_rotated_data.append(new_rows)
    
    if idx % 50000 == 0:
        print(f"      Processed {idx} tracks...")

df_final = pd.concat(all_rotated_data, ignore_index=True)

# ==========================================
# 4. SPLIT AND SAVE
# ==========================================
print(f"[4/4] Splitting 85/15 and Saving...")

# FIX 2: Include 'run_id' in the unique check and merge keys!
unique_tracks = df_final[["run_id", "energy_keV", "ion_number"]].drop_duplicates()
train_tracks = unique_tracks.sample(frac=TRAIN_SPLIT, random_state=42)
eval_tracks = unique_tracks.drop(train_tracks.index)

# Merge on all 3 keys to ensure safety
df_train_final = df_final.merge(train_tracks, on=["run_id", "energy_keV", "ion_number"])
df_eval_final = df_final.merge(eval_tracks, on=["run_id", "energy_keV", "ion_number"])

# Optional: Clean up helper column
df_train_final = df_train_final.drop(columns=['run_id'])
df_eval_final = df_eval_final.drop(columns=['run_id'])

df_train_final.to_csv(OUTPUT_DIR / "mcpe_3d_train.csv", index=False)
df_eval_final.to_csv(OUTPUT_DIR / "mcpe_3d_eval.csv", index=False)

print(f"\n[DONE] MCPE-3D Data Ready in: {OUTPUT_DIR}")
print(f"       Final Training Tracks:   {len(train_tracks)}")
print(f"       Final Evaluation Tracks: {len(eval_tracks)}")