# SIIMPL GVP-EGNN

Directional reconstruction of low-energy nuclear recoils in single-crystal diamond
with simulation-based inference. A GVP-EGNN encoder maps a 3-D vacancy point cloud
to a latent vector; a posterior head outputs a distribution over the recoil
**direction** (von Mises–Fisher mixture) and **energy** (log-energy Gaussian mixture).

## Layout

```
src/
  models/       egnn.py · directional_head.py · vmf_loss.py
  utils/        data_utils.py       point clouds -> padded kNN graphs
  evaluation/   eval_mcpe3d.py      posterior sampling + metrics
  scripts/      generate_data.py · prepare_data.py · train.py
notebooks/      train_a100.ipynb    Colab A100 runner
```

## Install

```bash
conda env create -f environment.yaml      # or: pip install -r requirements.txt
```

Regenerating the dataset also needs **SIIMPL** (ion-implantation Monte-Carlo);
set `SIIMPL_PYTHON` to its python package directory.

## Run

```bash
# 1. generate the dataset (needs SIIMPL)
python src/scripts/generate_data.py --only all

# 2. merge pools + build kNN caches
python src/scripts/prepare_data.py

# 3. train + evaluate
python src/scripts/train.py --directional-head --tag ROT_FINAL
```

Data and weights are not tracked — regenerate with the scripts above, or run
`notebooks/train_a100.ipynb` on a cloud A100.

## Data convention

Tracks are generated with SIIMPL `rotate_crystal_mode=True` (beam stays normal,
crystal is rotated) and stored in the crystal frame. For a unit recoil direction `g`:

```
theta = arccos(g_z)
phi   = (180 - atan2(g_y, g_x)) mod 360
R     = Rotation.from_euler("YZ", [theta, phi])   # scipy
cloud = R^-1 · cloud_lab                           # crystal frame
```
