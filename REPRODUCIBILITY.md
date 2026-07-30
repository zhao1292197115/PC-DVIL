# Reproducibility Guide

Run each entry script with `-h` before a long experiment because retained snapshots may contain slightly different flags.

## 1. Simulation environment

```bash
cd act-main_trir
conda activate act_sim

python3 - <<'PY'
import torch
import dm_control
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

## 2. Original three-view ACT

```bash
cd act-baseline
conda activate act_sim

python3 imitate_episodes.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/act_resnet_3cam   --policy_class ACT   --batch_size 2   --num_epochs 2000   --lr 1e-5   --seed 0   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8
```

## 3. Dense-ACT

```bash
cd act-main
conda activate act_sim

python3 imitate_episodes_dinov2_fixedposes.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/dinov2_randpose_last8_pool2_3cam   --policy_class ACT   --batch_size 2   --num_epochs 2000   --lr 1e-5   --seed 0   --backbone dinov2_vits14   --lr_backbone 1e-5   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8
```

## 4. PC-DVIL perturbation-consistent training

```bash
cd act-main_trir
conda activate act_sim

python3 imitate_episodes_dinov2_fixedposes.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/dinov2_last8_trir_weak_3cam   --policy_class ACT   --batch_size 2   --num_epochs 2000   --lr 1e-5   --seed 0   --backbone dinov2_vits14   --lr_backbone 1e-5   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --use_trir   --trir_aug_prob 0.5   --trir_view_prob 0.67   --trir_aug_weight 1.0   --trir_cons_weight 0.2   --trir_noise_std 0.015   --trir_erasing_prob 0.25   --trir_max_erase_ratio 0.25
```

Use the exact flags supported by the released training snapshot when recreating a checkpoint.

## 5. Fixed-pose simulation evaluation

Required pose file:

```text
act-main_trir/eval_poses/sim_insertion_eval_seed1000_50.pkl
```

Common evaluation flags:

```text
--eval
--eval_rollouts 50
--eval_metrics
--measure_latency
--temporal_agg
--eval_pose_path eval_poses/sim_insertion_eval_seed1000_50.pkl
```

For PC-DVIL deployment evaluation, enable Cache2:

```text
--feature_cache --cache_interval 2
```

Clean condition:

```bash
unset FACTORY_MODE FACTORY_STRENGTH FACTORY_SEED
```

Perturbation 0.60:

```bash
export FACTORY_MODE=hard_noline
export FACTORY_STRENGTH=0.60
export FACTORY_SEED=1000
```

Perturbation 0.82:

```bash
export FACTORY_STRENGTH=0.82
```

## 6. Paired Clean–0.60 drift collection

Both methods must use TFC OFF.

```bash
cd act-main_trir
conda activate act_sim

python3 collect_clean_perturb_drift.py   --num_rollouts 50   --intensity 0.60   --eval_pose_path eval_poses/sim_insertion_eval_seed1000_50.pkl   --reference_controller alternate   --stop_on_success   --output_csv eval_logs/paired_drift_s060_fixed50_seed1000.csv
```

Use only rows where `valid_mask == 1` when computing mean curves and 95% confidence intervals.

## 7. Real-robot training

```bash
cd cobot_magic/aloha-devel
conda activate aloha

python3 act/train.py   --dataset_dir ../data   --task_name aloha_mobile_dummy   --ckpt_dir ../train/dinov2_trirpp_battery_50ep_6000_final   --policy_class ACT   --batch_size 4   --num_epochs 6000   --num_episodes 50   --seed 0   --lr 1e-5   --lr_backbone 5e-6   --backbone dinov2_vits14   --dinov2_repo ../dinov2_local/dinov2-main   --dinov2_weights ../dinov2_local/dinov2_vits14_pretrain.pth   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --use_trir   --trir_aug_prob 0.5   --trir_aug_weight 0.4   --trir_cons_weight 0.08   --trir_feat_cons_weight 0.02   --trir_brightness 0.25   --trir_contrast 0.25   --trir_gamma 0.20   --trir_saturation 0.15   --trir_blur_prob 0.08   --trir_shadow_prob 0.20   --trir_shadow_strength 0.20   --trir_erasing_prob 0.0   --trir_noise_std 0.01
```

## 8. Real-robot inference with Cache2

```bash
cd cobot_magic/aloha-devel
conda activate aloha
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python3 act/inference.py   --ckpt_dir ../train/dinov2_trirpp_battery_50ep_6000_final   --ckpt_name policy_best.ckpt   --ckpt_stats_name dataset_stats.pkl   --policy_class ACT   --backbone dinov2_vits14   --dinov2_repo ../dinov2_local/dinov2-main   --dinov2_weights ../dinov2_local/dinov2_vits14_pretrain.pth   --dinov2_train_layers 8   --dinov2_pool 2   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --temporal_agg True   --feature_cache   --cache_interval 2
```

A legacy checkpoint containing a stage head must be loaded with its matching compatibility flags.
