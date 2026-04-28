# Stage 2 Colab Run — ScalarAugmentedEmbedding

## What this does
- Loads your existing `best_checkpoint.pt` (Stage 1, frozen backbone).
- Wraps the embedding with `ScalarAugmentedEmbedding`, which appends `log(n_vac+1)` and `log(lateral_spread+1e-4)` to the latent `z`.
- Builds a fresh NSF flow conditioned on `D_AUG = 386` and trains it + new aux heads for up to 60 epochs (~1–2 hrs on A100).
- Saves Stage 2 result to a **separate** file: `best_checkpoint_stage2.pt`. Stage 1 is preserved.

## Files needed in Colab
Push the latest commits, then in Colab:

```bash
!git clone https://github.com/<your-repo>.git
%cd Inverse-ML
!git pull
```

Required existing artifacts:
- `results/gvp_egnn/best_checkpoint.pt`  (Stage 1 trained backbone)
- `data/mcpe3d/mcpe_3d_train.csv`
- `data/mcpe3d/mcpe_3d_eval.csv`

If those aren't already on the Colab disk (e.g. via Drive mount), upload them first.

## Run command

```bash
!python -m src.scripts.train_gvp_egnn --stage2-only
```

Optional batch size override (auto-detect handles A100 fine):
```bash
!python -m src.scripts.train_gvp_egnn --stage2-only --batch-size 256
```

## Expected output
- `results/gvp_egnn/best_checkpoint_stage2.pt` — new flow + aux heads + scalar stats
- `results/gvp_egnn/gvp_egnn_posterior.pt` — rebuilt posterior (uses Stage 2 if present)
- Eval CSV with new metrics
- Inference timing report

## After it runs
Pull the new checkpoint + posterior back locally and regenerate the energy KDE plots — should see GVP-EGNN MAE drop from ~5.0 keV closer to ~3.5 keV.

## Rollback
If results are worse, just delete `best_checkpoint_stage2.pt`. The posterior-building code falls back to `best_checkpoint.pt` automatically.
