#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

OUT_DIR="eval_logs/peg_generalization_seed1000"
mkdir -p "$OUT_DIR"

# 若当前 TRIR 目录没有原 ACT 的固定姿态评测脚本，则从原项目复制一份。
ACT_SCRIPT="imitate_episodes_metrics_fixedposes.py"
if [[ ! -f "$ACT_SCRIPT" ]]; then
  if [[ -f "../act-main/$ACT_SCRIPT" ]]; then
    cp "../act-main/$ACT_SCRIPT" "$ACT_SCRIPT"
    echo "[Setup] Copied $ACT_SCRIPT from ../act-main/"
  else
    echo "[Error] Cannot find $ACT_SCRIPT in current directory or ../act-main/"
    exit 1
  fi
fi

ACT_CKPT="${ACT_CKPT:-ckpts/act_resnet_3cam_randpose}"
DINO_CKPT="ckpts/dinov2_randpose_last8_pool2_3cam"
TRIR_CKPT="ckpts/dinov2_last8_trir_weak_3cam"

for ckpt in "$ACT_CKPT" "$DINO_CKPT" "$TRIR_CKPT"; do
  if [[ ! -f "$ckpt/policy_best.ckpt" ]]; then
    echo "[Error] Missing checkpoint: $ckpt/policy_best.ckpt"
    echo "For the ACT baseline, you may specify its actual path as:"
    echo "ACT_CKPT=/your/actual/act_checkpoint_dir bash run_peg_generalization.sh act"
    exit 1
  fi
done

archive_metrics () {
  local ckpt="$1"
  local method="$2"
  local variant="$3"

  [[ -f "$ckpt/eval_metrics_policy_best.csv" ]] && \
    cp "$ckpt/eval_metrics_policy_best.csv" \
       "$OUT_DIR/${method}_${variant}_metrics.csv"

  [[ -f "$ckpt/eval_summary_policy_best.json" ]] && \
    cp "$ckpt/eval_summary_policy_best.json" \
       "$OUT_DIR/${method}_${variant}_summary.json"
}

run_act () {
  local label="$1"
  local color="$2"
  local scale="$3"

  echo "============================================================"
  echo "[ACT-ResNet] ${label} | color=${color} | scale=${scale}"
  echo "============================================================"

  env FACTORY_MODE=clean FACTORY_SEED=0 \
      PEG_COLOR="$color" PEG_SCALE="$scale" PEG_SHAPE=original \
      PYTHONUNBUFFERED=1 \
  python3 -u "$ACT_SCRIPT" \
    --eval \
    --ckpt_dir "$ACT_CKPT" \
    --policy_class ACT \
    --task_name sim_insertion_scripted \
    --batch_size 1 \
    --seed 0 \
    --num_epochs 1 \
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
    2>&1 | tee "$OUT_DIR/act_resnet_${label}.log"

  archive_metrics "$ACT_CKPT" "act_resnet" "$label"
}

run_dino_cache2 () {
  local label="$1"
  local color="$2"
  local scale="$3"

  echo "============================================================"
  echo "[DINOv2 + Cache2] ${label} | color=${color} | scale=${scale}"
  echo "============================================================"

  env FACTORY_MODE=clean FACTORY_SEED=0 \
      PEG_COLOR="$color" PEG_SCALE="$scale" PEG_SHAPE=original \
      PYTHONUNBUFFERED=1 \
  python3 -u imitate_episodes_dinov2_fixedposes.py \
    --eval \
    --ckpt_dir "$DINO_CKPT" \
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
    2>&1 | tee "$OUT_DIR/dinov2_cache2_${label}.log"

  archive_metrics "$DINO_CKPT" "dinov2_cache2" "$label"
}

run_trir_cache2 () {
  local label="$1"
  local color="$2"
  local scale="$3"

  echo "============================================================"
  echo "[DINOv2 + weak TRIR + Cache2] ${label} | color=${color} | scale=${scale}"
  echo "============================================================"

  env FACTORY_MODE=clean FACTORY_SEED=0 \
      PEG_COLOR="$color" PEG_SCALE="$scale" PEG_SHAPE=original \
      PYTHONUNBUFFERED=1 \
  python3 -u imitate_episodes_dinov2_fixedposes.py \
    --eval \
    --ckpt_dir "$TRIR_CKPT" \
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
    2>&1 | tee "$OUT_DIR/dinov2_trir_cache2_${label}.log"

  archive_metrics "$TRIR_CKPT" "dinov2_trir_cache2" "$label"
}

VARIANTS=(
  "blue blue 1.00"
  "green green 1.00"
  "dark dark 1.00"
  "size095 original 0.95"
  "size105 original 1.05"
)

run_group () {
  local method="$1"

  for item in "${VARIANTS[@]}"; do
    read -r label color scale <<< "$item"

    case "$method" in
      act)  run_act "$label" "$color" "$scale" ;;
      dino) run_dino_cache2 "$label" "$color" "$scale" ;;
      trir) run_trir_cache2 "$label" "$color" "$scale" ;;
    esac
  done
}

case "${1:-}" in
  act)
    run_group act
    ;;
  dino)
    run_group dino
    ;;
  trir)
    run_group trir
    ;;
  all)
    run_group act
    run_group dino
    run_group trir
    ;;
  *)
    echo "Usage:"
    echo "  bash run_peg_generalization.sh act"
    echo "  bash run_peg_generalization.sh dino"
    echo "  bash run_peg_generalization.sh trir"
    echo "  bash run_peg_generalization.sh all"
    exit 1
    ;;
esac
