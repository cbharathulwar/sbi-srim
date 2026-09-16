#!/usr/bin/env bash
# =============================================================================
# a100_final -- launch the scaling ladder (small + large) on an A100 box.
#
# SCHEDULER ASSUMPTION
#   The cluster's scheduler was not specified, so this script defaults to plain
#   `python` invocations (MODE=seq / MODE=par), which work on any box you can
#   ssh into, inside a container, or from within an already-allocated SLURM
#   job (`srun --pty bash`).  MODE=slurm additionally emits and submits real
#   sbatch scripts; edit the #SBATCH header block below for your partition and
#   account before using it.
#
# MODES
#   MODE=seq    (default) one GPU, the two configs run BACK TO BACK, each
#               getting half the wall budget.
#   MODE=par    two GPUs on one node, the two configs run CONCURRENTLY, each
#               getting the FULL wall budget (CUDA_VISIBLE_DEVICES 0 and 1).
#   MODE=slurm  submit each config as its own sbatch job, full budget each.
#
# WALL BUDGET
#   TOTAL_HOURS (default 30).  In seq mode each config gets TOTAL_HOURS/2 minus
#   a small margin for data loading and the final eval; in par/slurm each gets
#   TOTAL_HOURS minus the same margin.
#
# STEP COUNT
#   TARGET_STEPS shapes the cosine LR schedule, so it must match what the job
#   can ACTUALLY complete -- if it is set too high the LR never anneals.  Rather
#   than guess A100 throughput, this script runs a short CALIBRATION job per
#   config, reads the [THROUGHPUT] line, and sizes TARGET_STEPS from the
#   measured it/s.  Set TARGET_STEPS_SMALL / TARGET_STEPS_LARGE explicitly to
#   skip calibration.
#
# DATA
#   DEFAULTS to the final v2 corpus (1,009,384 train / 101,731 eval tracks):
#     data/siimpl_rot/siimpl_train_v2.csv
#     data/siimpl_rot/siimpl_eval_v2.csv
#   Override TRAIN_CSV / EVAL_CSV to fall back to the original smaller files.
#
# USAGE
#   bash launch_a100.sh                       # seq, 30h, calibrated
#   MODE=par TOTAL_HOURS=30 bash launch_a100.sh
#   MODE=slurm bash launch_a100.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-seq}"
TOTAL_HOURS="${TOTAL_HOURS:-30}"
CALIB_STEPS="${CALIB_STEPS:-200}"
PY="${PY:-python}"

export TRAIN_CSV="${TRAIN_CSV:-data/siimpl_rot/siimpl_train_v2.csv}"
export EVAL_CSV="${EVAL_CSV:-data/siimpl_rot/siimpl_eval_v2.csv}"
export POSTERIOR_TYPE="${POSTERIOR_TYPE:-vmf_mixture}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

TRAIN_PY="smearing_resolution/architecture_experiments/a100_final/train.py"
EVAL_PY="smearing_resolution/architecture_experiments/a100_final/eval.py"

# margin for CSV load (~2-4 min for 1M tracks), periodic evals and final eval
MARGIN_H="${MARGIN_H:-1.0}"
if [ "$MODE" = "seq" ]; then
  PER_CONFIG_H=$(python -c "print(max(0.5,($TOTAL_HOURS-2*$MARGIN_H)/2))")
else
  PER_CONFIG_H=$(python -c "print(max(0.5,$TOTAL_HOURS-$MARGIN_H))")
fi
echo "[LAUNCH] MODE=$MODE  TOTAL_HOURS=$TOTAL_HOURS  per-config=${PER_CONFIG_H}h"
echo "[LAUNCH] TRAIN_CSV=$TRAIN_CSV"
echo "[LAUNCH] EVAL_CSV=$EVAL_CSV"

# ---------------------------------------------------------------- calibrate
calibrate () {   # $1 = config name -> echoes measured it/s
  local cfg="$1"
  local dir="smearing_resolution/architecture_experiments/a100_final/calib_${cfg}"
  rm -rf "$dir"
  RESUME=0 CONFIG="$cfg" RESULTS_DIR="$dir" \
    TARGET_STEPS="$CALIB_STEPS" TIME_BUDGET_HOURS=0.5 \
    CHECKPOINT_EVERY_SEC=1e9 EVAL_N_PER_BIN=50 \
    $PY "$TRAIN_PY" > "$dir.log" 2>&1 || { cat "$dir.log"; exit 1; }
  grep -oP '\[THROUGHPUT\] \K[0-9.]+' "$dir.log" | tail -1
}

steps_for () {   # $1 = config name -> echoes TARGET_STEPS
  local cfg="$1"
  local ovr
  ovr="$(eval echo "\${TARGET_STEPS_${cfg^^}:-}")"
  if [ -n "$ovr" ]; then echo "$ovr"; return; fi
  echo "[CALIB] measuring $cfg throughput over $CALIB_STEPS steps..." >&2
  local ips; ips="$(calibrate "$cfg")"
  if [ -z "$ips" ]; then
    echo "[CALIB] FAILED to measure throughput; falling back to 40000" >&2
    echo 40000; return
  fi
  # 0.92 safety factor: the calibration run excludes periodic eval overhead
  python -c "print(int($ips*3600*$PER_CONFIG_H*0.92))"
}

run_one () {     # $1 = config, $2 = steps, $3 = optional CUDA device
  local cfg="$1" steps="$2" dev="${3:-}"
  local dir="smearing_resolution/architecture_experiments/a100_final/results_${cfg}"
  mkdir -p "$dir"
  echo "[RUN] $cfg  TARGET_STEPS=$steps  budget=${PER_CONFIG_H}h  dir=$dir"
  ( [ -n "$dev" ] && export CUDA_VISIBLE_DEVICES="$dev"
    CONFIG="$cfg" RESULTS_DIR="$dir" RESUME=1 \
      TARGET_STEPS="$steps" TIME_BUDGET_HOURS="$PER_CONFIG_H" \
      $PY "$TRAIN_PY" 2>&1 | tee -a "$dir/train.log"
    CONFIG="$cfg" RESULTS_DIR="$dir" \
      CKPT="$dir/checkpoint_final.pt" \
      $PY "$EVAL_PY" 2>&1 | tee "$dir/eval.log" )
}

S_SMALL="$(steps_for small)"
S_LARGE="$(steps_for large)"
echo "[PLAN] small -> $S_SMALL steps ; large -> $S_LARGE steps"

case "$MODE" in
  seq)
    run_one small "$S_SMALL"
    run_one large "$S_LARGE"
    ;;
  par)
    run_one small "$S_SMALL" 0 &
    P1=$!
    run_one large "$S_LARGE" 1 &
    P2=$!
    wait $P1 $P2
    ;;
  slurm)
    for cfg in small large; do
      steps=$([ "$cfg" = small ] && echo "$S_SMALL" || echo "$S_LARGE")
      dir="smearing_resolution/architecture_experiments/a100_final/results_${cfg}"
      mkdir -p "$dir"
      cat > "$dir/job.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=a100f_${cfg}
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=$(python -c "print(int($TOTAL_HOURS)+1)"):00:00
#SBATCH --output=${dir}/slurm-%j.out
# EDIT: --partition / --account for your cluster
set -euo pipefail
cd "$ROOT"
export PYTHONPATH="$ROOT:\${PYTHONPATH:-}"
export TRAIN_CSV="$TRAIN_CSV" EVAL_CSV="$EVAL_CSV"
export CONFIG=$cfg RESULTS_DIR=$dir RESUME=1
export TARGET_STEPS=$steps TIME_BUDGET_HOURS=$PER_CONFIG_H
$PY $TRAIN_PY
CKPT=$dir/checkpoint_final.pt $PY $EVAL_PY
EOF
      echo "[SLURM] submitting $dir/job.sbatch"
      sbatch "$dir/job.sbatch"
    done
    ;;
  *) echo "unknown MODE=$MODE (seq|par|slurm)"; exit 2 ;;
esac

echo "[LAUNCH] done."
