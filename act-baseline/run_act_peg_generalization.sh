#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="eval_logs/peg_generalization_seed1000"
mkdir -p "$OUT_DIR"

CKPT_DIR="ckpts/act_resnet_3cam_randpose"

if [[ ! -f "$CKPT_DIR/policy_best.ckpt" ]]; then
  echo "[Error] Missing checkpoint: $CKPT_DIR/policy_best.ckpt"
  exit 1
fi

run_one () {
  local label="$1"
  local peg_color="$2"
  local peg_scale="$3"

  echo ""
  echo "============================================================"
  echo "Method: ACT-ResNet-3Cam"
  echo "Variant: ${label}"
  echo "PEG_COLOR=${peg_color} | PEG_SCALE=${peg_scale}"
  echo "============================================================"

  FACTORY_MODE=clean \
  FACTORY_SEED=0 \
  PEG_COLOR="$peg_color" \
  PEG_SCALE="$peg_scale" \
  PEG_SHAPE=original \
  PYTHONUNBUFFERED=1 \
  python3 -u imitate_episodes_metrics_fixedposes.py \
    --eval \
    --ckpt_dir "$CKPT_DIR" \
    --policy_class ACT \
    --task_name sim_insertion_scripted \
    --batch_size 1 \
    --seed 1000 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --temporal_agg \
    --eval_rollouts 50 \
    --eval_metrics \
    --measure_latency \
    --eval_pose_path eval_poses/sim_insertion_eval_seed1000_50.pkl \
    2>&1 | tee "$OUT_DIR/act_resnet_${label}_seed1000_50.log"

  if [[ -f "$CKPT_DIR/eval_metrics_policy_best.csv" ]]; then
    cp "$CKPT_DIR/eval_metrics_policy_best.csv" \
       "$OUT_DIR/act_resnet_${label}_metrics.csv"
  fi

  if [[ -f "$CKPT_DIR/eval_summary_policy_best.json" ]]; then
    cp "$CKPT_DIR/eval_summary_policy_best.json" \
       "$OUT_DIR/act_resnet_${label}_summary.json"
  fi
}

run_one "blue"    "blue"     "1.00"
run_one "green"   "green"    "1.00"
run_one "dark"    "dark"     "1.00"
run_one "size095" "original" "0.95"
run_one "size105" "original" "1.05"

echo ""
echo "============================================================"
echo "Finished: act_resnet"
echo "Logs saved to: $OUT_DIR"
echo "============================================================"
