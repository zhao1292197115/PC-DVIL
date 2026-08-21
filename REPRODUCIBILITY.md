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

Both the simulation and real-robot loaders use `train_ratio = 0.8`. With the released 50-episode datasets, `np.random.permutation` therefore assigns 40 episodes to training and the remaining 10 to validation. All reported training commands use optimization seed 0. The dense simulation and PC-DVIL entry scripts apply the command seed before constructing the loader. The retained ACT baseline and real-robot entry scripts call `set_seed(1)` only before the loader split; their training functions then reset the random state to the command seed (`--seed 0`) before model initialization and optimization. Thus, the value 1 in those retained entry points is a loader-split implementation detail, not the reported optimization seed. Normalization statistics are computed over all 50 episodes by the retained loader implementation.

## 2. Original three-view ACT

```bash
cd act-baseline
conda activate act_sim

python3 imitate_episodes.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/act_resnet_3cam   --policy_class ACT   --batch_size 2   --num_epochs 2000   --lr 1e-5   --seed 0   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200
```

The baseline entry script fixes the transformer configuration internally to 4 encoder layers, 7 decoder layers, and 8 attention heads; it does not expose those three values as command-line flags.

## 3. Dense-ACT

```bash
cd act-main
conda activate act_sim

python3 imitate_episodes_dinov2_fixedposes.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/dinov2_randpose_last8_pool2_3cam   --policy_class ACT   --batch_size 2   --num_epochs 2000   --lr 1e-5   --seed 0   --backbone dinov2_vits14   --lr_backbone 5e-6   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8
```

## 4. Loss-wise perturbation-consistency variants

All loss-wise variants retain the standard Dense-ACT reconstruction and KL objectives. The labels below indicate only which additional perturbation-consistency terms are enabled. They use the same 2,000-epoch budget and perturbation distribution; TFC and Stage Aux. are disabled during training and evaluation for this ablation.

Use the following command for the full PC-DVIL loss configuration:

```bash
cd act-main_trir
conda activate act_sim

python3 imitate_episodes_dinov2_stageaware_v7.py   --task_name sim_insertion_scripted   --ckpt_dir ckpts/dinov2_trirpp_nostage_sim_insertion_2000   --policy_class ACT   --batch_size 2   --num_epochs 2000   --lr 1e-5   --seed 0   --backbone dinov2_vits14   --lr_backbone 5e-6   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --amp   --amp_dtype bf16   --use_trir   --trir_aug_prob 0.5   --trir_view_prob 0.8   --trir_aug_weight 0.4   --trir_cons_weight 0.08   --trir_feat_cons_weight 0.02   --trir_brightness 0.25   --trir_contrast 0.25   --trir_gamma 0.20   --trir_saturation 0.15   --trir_blur_prob 0.08   --trir_shadow_prob 0.20   --trir_shadow_strength 0.20   --trir_erasing_prob 0.0   --trir_noise_std 0.01
```

To reproduce the other loss-wise rows, keep every common flag above unchanged, use a distinct `--ckpt_dir`, and change only the three additional loss weights:

| Paper label | `--trir_aug_weight` | `--trir_cons_weight` | `--trir_feat_cons_weight` |
|---|---:|---:|---:|
| `Dense-ACT + L_aug` | `0.4` | `0` | `0` |
| `Dense-ACT + L_aug + L_act` | `0.4` | `0.08` | `0` |
| `Dense-ACT + L_aug + L_feat` | `0.4` | `0` | `0.02` |
| `Full PC-DVIL` | `0.4` | `0.08` | `0.02` |

The Dense-ACT reference is the configuration in Section 3 with no `--use_trir` flag. All loss-wise labels above retain the standard reconstruction and KL terms.

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

For simulation TFC evaluation with `K_c = 2`, enable feature caching as follows:

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

python3 act/train.py   --dataset_dir ../data   --task_name aloha_mobile_dummy   --ckpt_dir ../train/dinov2_trirpp_stageaware_battery_6000   --policy_class ACT   --batch_size 4   --num_epochs 6000   --num_episodes 50   --seed 0   --lr 1e-5   --lr_backbone 5e-6   --backbone dinov2_vits14   --dinov2_repo ../dinov2_local/dinov2-main   --dinov2_weights ../dinov2_local/dinov2_vits14_pretrain.pth   --dinov2_train_layers 8   --dinov2_pool 2   --kl_weight 10   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --use_trir   --trir_aug_prob 0.7   --trir_view_prob 0.8   --trir_aug_weight 0.5   --trir_cons_weight 0.10   --trir_feat_cons_weight 0.03   --trir_brightness 0.30   --trir_contrast 0.30   --trir_gamma 0.25   --trir_saturation 0.20   --trir_blur_prob 0.10   --trir_shadow_prob 0.25   --trir_shadow_strength 0.25   --trir_erasing_prob 0.0   --trir_noise_std 0.015   --use_auto_stage_weight   --stage_weight_max 3.0   --stage_event_window 8   --stage_speed_power 1.0   --stage_acc_power 0.7   --stage_gripper_power 1.0   --stage_gripper_indices 6,13   --use_stage_pred   --stage_num 5   --stage_loss_weight 0.05   --stage_hidden_dim 128
```

The real-robot RGB inputs are three synchronized `480 x 640` streams. The inference program publishes commands at 40 Hz unless `--publish_rate` is explicitly overridden.

## 8. Real-robot inference with a fixed two-step policy-query schedule

```bash
cd cobot_magic/aloha-devel
conda activate aloha
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python3 act/inference.py   --ckpt_dir ../train/dinov2_trirpp_stageaware_battery_6000   --ckpt_name policy_best.ckpt   --ckpt_stats_name dataset_stats.pkl   --policy_class ACT   --backbone dinov2_vits14   --dinov2_repo ../dinov2_local/dinov2-main   --dinov2_weights ../dinov2_local/dinov2_vits14_pretrain.pth   --dinov2_train_layers 8   --dinov2_pool 2   --chunk_size 100   --hidden_dim 512   --dim_feedforward 3200   --enc_layers 4   --dec_layers 7   --nheads 8   --temporal_agg True   --feature_cache   --cache_interval 2   --use_stage_pred   --stage_num 5   --stage_hidden_dim 128   --publish_rate 40
```

In this retained real-robot runner, `--feature_cache --cache_interval 2` are historical flag names. They set the effective policy-query interval to two publish steps and reuse the intervening command from the previously predicted action chunk through temporal aggregation. They do not activate the feature-level TFC used in the simulation study.

The deployed checkpoint contains a training-only auxiliary `stage_head`. The fixed v7 runner instantiates that head through `--use_stage_pred --stage_num 5 --stage_hidden_dim 128` solely for exact state-dict compatibility. The head is not evaluated and does not alter the real-robot action path.
