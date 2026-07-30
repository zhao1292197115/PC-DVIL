#!/usr/bin/env bash
set -e

# 把本脚本和 imitate_episodes_metrics_fixedposes_step_log.py 放到 act-baseline 根目录后运行：
# bash run_baseline_step_eval_seed1000.sh
# 如果你的 ckpt 路径不是 ckpts/act_resnet_3cam_randpose，只改 CKPT_DIR 即可。

CKPT_DIR="ckpts/act_resnet_3cam_randpose"
POSE_PATH="eval_poses/sim_insertion_eval_seed1000_50.pkl"
LOG_DIR="eval_logs/resnet_clean_seed1000"

python3 imitate_episodes_metrics_fixedposes_step_log.py \
  --eval \
  --ckpt_dir "${CKPT_DIR}" \
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
  --save_eval_video \
  --eval_pose_path "${POSE_PATH}" \
  --save_step_log \
  --step_log_dir "${LOG_DIR}" \
  --method_name "ACT-ResNet-3Cam" \
  --eval_env_name "clean"

echo "完成。日志位置：${LOG_DIR}/step_log_policy_best.csv"
