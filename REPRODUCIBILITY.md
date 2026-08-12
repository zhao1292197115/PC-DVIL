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

### Dataset split

Both the simulation and real-robot loaders use `train_ratio = 0.8`. With the released 50-episode datasets, `np.random.permutation` therefore assigns 40 episodes to training and the remaining 10 to validation. The dense simulation and PC-DVIL entry scripts apply the command seed (0 in the commands below) before constructing the loader. The retained legacy ACT baseline and real-robot training entry scripts apply seed 1 before their loader split and then apply the command seed to model training. Normalization statistics are computed over all 50 episodes by the retained loader implementation.

## 2. Original three-view ACT

```bash
cd act-baseline
conda activate act_sim

python3 imitate_episodes.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/act_resnet_3cam   --policy_class ACT   --batch_size 2   --num_epochs 5000   --lr 1e-5   --seed 0   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200
```

The baseline entry script fixes the transformer configuration internally to 4 encoder layers, 7 decoder layers, and 8 attention heads; it does not expose those three values as command-line flags.

## 3. Dense-ACT

```bash
cd act-main
conda activate act_sim

python3 imitate_episodes_dinov2_fixedposes.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/dinov2_randpose_last8_pool2_3cam   --policy_class ACT   --batch_size 2   --num_epochs 5000   --lr 1e-5   --seed 0   --backbone dinov2_vits14   --lr_backbone 5e-6   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8
```

## 4. Action-consistency configuration used for the three-seed 60.0% result

```bash
cd act-main_trir
conda activate act_sim

python3 imitate_episodes_dinov2_fixedposes.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/dinov2_last8_trir_weak_3cam   --policy_class ACT   --batch_size 2   --num_epochs 5000   --lr 1e-5   --seed 0   --backbone dinov2_vits14   --lr_backbone 5e-6   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --amp   --amp_dtype bf16   --use_trir   --trir_aug_prob 0.3   --trir_view_prob 0.67   --trir_aug_weight 0.5   --trir_cons_weight 0.05   --trir_noise_std 0.01   --trir_erasing_prob 0.0
```

This configuration uses action-chunk consistency only (`lambda_feat = 0`). Selected views use brightness factor `U(0.55, 1.15)`, contrast factor `U(0.75, 1.25)`, independent RGB-channel gains `U(0.85, 1.15)`, and Gaussian noise with standard deviation `0.01`. Gamma, saturation, shadow, blur, and random erasing are disabled. If a selected batch samples no view, one view receives fallback brightness `U(0.55, 0.90)`.

## 4.1 Dual-level PC-DVIL configuration used for the TFC-matched evaluations

```bash
cd act-main_trir
conda activate act_sim

python3 imitate_episodes_dinov2_stageaware_v7.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/dinov2_trirpp_nostage_sim_insertion_5000   --policy_class ACT   --batch_size 2   --num_epochs 5000   --lr 1e-5   --seed 0   --backbone dinov2_vits14   --lr_backbone 5e-6   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --amp   --amp_dtype bf16   --use_trir   --trir_aug_prob 0.5   --trir_view_prob 0.8   --trir_aug_weight 0.4   --trir_cons_weight 0.08   --trir_feat_cons_weight 0.02   --trir_brightness 0.25   --trir_contrast 0.25   --trir_gamma 0.20   --trir_saturation 0.15   --trir_blur_prob 0.08   --trir_shadow_prob 0.20   --trir_shadow_strength 0.20   --trir_erasing_prob 0.0   --trir_noise_std 0.01
```

All simulation models use AdamW with weight decay `1e-4`. Validation is run every five epochs, and `policy_best.ckpt` is the checkpoint with the lowest validation loss. Run the retained entry script with `-h` before training because historical snapshots may expose slightly different flag names.

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

`hard_noline` is an image-only transform. For reported strength `s`, the code sets `level = 0.95 * s`. It applies a global affine low-light/contrast shift, RGB-channel gains, gamma, vignette, vertical illumination gradient, one camera-seeded soft shadow, Gaussian sensor noise, and desaturation. A sinusoidal illumination ripple is active only when `level >= 0.70`, so it is present for `s=0.82` but not `s=0.60`. Neither reported setting reaches the blur threshold (`level=0.85`) or salt-and-pepper threshold (`level=1.0`). Physics, object geometry, robot state, actions, and rewards are unchanged.

Latency is measured on an RTX 4090 with evaluation batch size 1 over all 50 x 400 control steps. The timing code calls `torch.cuda.synchronize()` immediately before and after every policy evaluation and reports the arithmetic mean. No separate warm-up samples are discarded. Camera acquisition, image transport, environment stepping, robot communication, and actuator delay are excluded.

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

python3 act/train.py   --dataset_dir ../data   --task_name aloha_mobile_dummy   --ckpt_dir ../train/dinov2_trirpp_stageaware_battery_6000   --policy_class ACT   --batch_size 4   --num_epochs 6000   --num_episodes 50   --seed 0   --lr 1e-5   --lr_backbone 5e-6   --backbone dinov2_vits14   --dinov2_repo ../dinov2_local/dinov2-main   --dinov2_weights ../dinov2_local/dinov2_vits14_pretrain.pth   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --use_trir   --trir_aug_prob 0.5   --trir_view_prob 0.8   --trir_aug_weight 0.4   --trir_cons_weight 0.08   --trir_feat_cons_weight 0.02   --trir_brightness 0.25   --trir_contrast 0.25   --trir_gamma 0.20   --trir_saturation 0.15   --trir_blur_prob 0.08   --trir_shadow_prob 0.20   --trir_shadow_strength 0.20   --trir_erasing_prob 0.0   --trir_noise_std 0.01   --use_auto_stage_weight   --stage_weight_max 2.5   --stage_event_window 8   --stage_speed_power 1.0   --stage_acc_power 0.7   --stage_gripper_power 0.8   --stage_gripper_indices 6,13   --use_stage_pred   --stage_num 5   --stage_loss_weight 0.03   --stage_hidden_dim 128
```

The real-robot RGB inputs are three synchronized `480 x 640` streams. The inference program publishes commands at 40 Hz unless `--publish_rate` is explicitly overridden.

## 8. Real-robot inference with Cache2

```bash
cd cobot_magic/aloha-devel
conda activate aloha
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python3 act/inference.py   --ckpt_dir ../train/dinov2_trirpp_stageaware_battery_6000   --ckpt_name policy_best.ckpt   --ckpt_stats_name dataset_stats.pkl   --policy_class ACT   --backbone dinov2_vits14   --dinov2_repo ../dinov2_local/dinov2-main   --dinov2_weights ../dinov2_local/dinov2_vits14_pretrain.pth   --dinov2_train_layers 8   --dinov2_pool 2   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --temporal_agg True   --feature_cache   --cache_interval 2   --publish_rate 40
```

Stage-aware options are training-only. The inference loader uses non-strict state-dict loading, so auxiliary `stage_head` entries in a checkpoint are ignored without requiring stage flags.
