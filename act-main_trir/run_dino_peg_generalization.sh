#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="eval_logs/peg_generalization_seed1000"
mkdir -p "$OUT_DIR"

METHOD="${1:-}"

if [[ "$METHOD" != "dino" && "$METHOD" != "trir" ]]; then
  echo "Usage:"
  echo "  bash run_dino_peg_generalization.sh dino"
  echo "  bash run_dino_peg_generalization.sh trir"
  exit 1
fi

if [[ "$METHOD" == "dino" ]]; then
  CKPT_DIR="ckpts/dinov2_randpose_last8_pool2_3cam"
  METHOD_NAME="dinov2_cache2"
else
  CKPT_DIR="ckpts/dinov2_last8_trir_weak_3cam"
  METHOD_NAME="dinov2_weaktrir_cache2"
fi

if [[ ! -f "$CKPT_DIR/policy_best.ckpt" ]]; then
  echo "[Error] Missing checkpoint:"
  echo "  $CKPT_DIR/policy_best.ckpt"
  exit 1
fi

run_one () {
  local label="$1"
  local peg_color="$2"
  local peg_scale="$3"

  echo ""
  echo "============================================================"
  echo "Method: $METHOD_NAME"
  echo "Variant: $label"
  echo "PEG_COLOR=$peg_color | PEG_SCALE=$peg_scale"
  echo "============================================================"

  FACTORY_MODE=clean \
  FACTORY_SEED=0 \
  PEG_COLOR="$peg_color" \
  PEG_SCALE="$peg_scale" \
  PEG_SHAPE=original \
  PYTHONUNBUFFERED=1 \
  python3 -u imitate_episodes_dinov2_fixedposes.py \
    --eval \
    --ckpt_dir "$CKPT_DIR" \
    --policy_class ACT \
    --task_name sim_insertion_scripted \
    --batch_size 1 \
    --seed 1000 \
    --num_epochs 2000 \
    --lr 1e-5 \
    --lr_backbone 5e-6 \
    --backbone dinov2_vits14 \
    --dinov2_train_layers 8 \
    --dinov2_pool 2 \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --enc_layers 4 \
    --dec_layers 7 \
    --nheads 8 \
    --amp \
    --amp_dtype bf16 \
    --temporal_agg \
    --feature_cache \
    --cache_interval 2 \
    --eval_rollouts 50 \
    --eval_metrics \
    --measure_latency \
    --eval_pose_path eval_poses/sim_insertion_eval_seed1000_50.pkl \
    2>&1 | tee "$OUT_DIR/${METHOD_NAME}_${label}_seed1000_50.log"

  if [[ -f "$CKPT_DIR/eval_metrics_policy_best.csv" ]]; then
    cp "$CKPT_DIR/eval_metrics_policy_best.csv" \
       "$OUT_DIR/${METHOD_NAME}_${label}_metrics.csv"
  fi

  if [[ -f "$CKPT_DIR/eval_summary_policy_best.json" ]]; then
    cp "$CKPT_DIR/eval_summary_policy_best.json" \
       "$OUT_DIR/${METHOD_NAME}_${label}_summary.json"
  fi
}

run_one "blue"    "blue"     "1.00"
run_one "green"   "green"    "1.00"
run_one "dark"    "dark"     "1.00"
run_one "size095" "original" "0.95"
run_one "size105" "original" "1.05"

echo ""
echo "============================================================"
echo "Finished: $METHOD_NAME"
echo "Logs saved to: $OUT_DIR"
echo "============================================================"
