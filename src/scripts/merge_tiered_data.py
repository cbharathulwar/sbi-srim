"""Merge the original siimpl_train.csv / siimpl_eval_merged.csv with the new
tiered generation outputs into siimpl_train_v2.csv / siimpl_eval_v2.csv,
renumbering `ion_number` to be globally unique across the merged files.

WHY THIS IS NECESSARY: load_raw_tracks groups points into tracks via
df.groupby('ion_number'). Every generation run (original and each new tier
file) starts its own ion_number counter at 0. Naively concatenating files
would collide ion_number ranges and silently merge points from UNRELATED
tracks (different energy/direction, different source file) into one bogus
"track" -- a serious, silent correctness bug. This script assigns each
source file a running global offset before concatenating.
"""
import os
import pandas as pd

ROOT = "C:/Inverse ML/data/siimpl_rot"

TRAIN_SOURCES = [
    "siimpl_train.csv",
    "siimpl_train_extra_low.csv",
    "siimpl_train_extra_mid.csv",
    "siimpl_train_extra_high.csv",
]
EVAL_SOURCES = [
    "siimpl_eval_merged.csv",
    "siimpl_eval_extra_low.csv",
    "siimpl_eval_extra_mid.csv",
    "siimpl_eval_extra_high.csv",
]


def merge(sources, out_name):
    frames = []
    offset = 0
    total_tracks = 0
    for name in sources:
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            print(f"  [SKIP] {name} not found"); continue
        df = pd.read_csv(path)
        n_local_tracks = df['ion_number'].nunique()
        local_max = int(df['ion_number'].max())
        df['ion_number'] = df['ion_number'] + offset
        frames.append(df)
        print(f"  [{name}] {n_local_tracks:,} tracks (local max id={local_max:,}), "
              f"{len(df):,} rows, ion_number offset={offset}")
        # IMPORTANT: advance the offset past the LOCAL MAX id, not nunique().
        # ion_number ranges are not guaranteed dense (a file assembled from
        # multiple earlier generation runs can have gaps), so nunique()
        # undercounts the required offset and causes silent ID collisions
        # with unused-but-still-reserved values -- caught by the sanity
        # check below on the first attempt (47,835 corrupted groups).
        offset += local_max + 1
        total_tracks += n_local_tracks
    merged = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(ROOT, out_name)
    merged.to_csv(out_path, index=False)
    # sanity check: no collisions, every ion_number maps to exactly one (energy,vx,vy,vz)
    check = merged.groupby('ion_number')[['energy_keV', 'target_vx', 'target_vy', 'target_vz']].nunique()
    bad = (check > 1).any(axis=1).sum()
    print(f"[{out_name}] TOTAL {total_tracks:,} tracks, {len(merged):,} rows -> {out_path}")
    print(f"[{out_name}] sanity check: {bad} ion_number groups with inconsistent "
          f"energy/direction (must be 0)")
    assert bad == 0, "ion_number collision detected -- merge is unsafe, do not use this file"
    return total_tracks


if __name__ == "__main__":
    print("=== Merging train ===")
    n_train = merge(TRAIN_SOURCES, "siimpl_train_v2.csv")
    print("\n=== Merging eval ===")
    n_eval = merge(EVAL_SOURCES, "siimpl_eval_v2.csv")
    print(f"\n[DONE] train={n_train:,} tracks, eval={n_eval:,} tracks")
