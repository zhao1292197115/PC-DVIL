import torch
import torch.nn.functional as F
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

from utils import load_data 
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy

import sys
sys.path.append("./")


def apply_real_trir_visual_perturbation(
    image_data,
    aug_prob=0.7,
    view_prob=0.8,
    noise_std=0.015,
    brightness=0.30,
    contrast=0.30,
    gamma=0.25,
    saturation=0.20,
    blur_prob=0.10,
    shadow_prob=0.25,
    shadow_strength=0.25,
    erasing_prob=0.0,
    max_erase_ratio=0.10,
):
    """
    V6 real-robot TRIR++ visual perturbation.
    image_data: [B, num_cam, C, H, W], range [0, 1].
    Training-time only. It simulates unseen illumination/exposure/background changes
    without changing qpos/action labels.
    """
    if aug_prob <= 0:
        return image_data
    if torch.rand((), device=image_data.device) > aug_prob:
        return image_data

    x = image_data.clone()
    B, V, C, H, W = x.shape
    view_mask = (torch.rand(B, V, 1, 1, 1, device=x.device) < view_prob).float()

    # brightness + contrast
    b = 1.0 + (torch.rand(B, V, 1, 1, 1, device=x.device) * 2.0 - 1.0) * brightness
    c = 1.0 + (torch.rand(B, V, 1, 1, 1, device=x.device) * 2.0 - 1.0) * contrast
    mean = x.mean(dim=(-2, -1), keepdim=True)
    y = (x - mean) * c + mean
    y = y * b

    # gamma/exposure curve. gamma_range=0.25 means approx [e^-0.25, e^0.25].
    if gamma > 0:
        g = torch.exp((torch.rand(B, V, 1, 1, 1, device=x.device) * 2.0 - 1.0) * gamma)
        y = torch.clamp(y, 1e-4, 1.0).pow(g)

    # saturation jitter; useful for real RGB camera color/exposure drift.
    if saturation > 0 and C == 3:
        sat = 1.0 + (torch.rand(B, V, 1, 1, 1, device=x.device) * 2.0 - 1.0) * saturation
        gray = y.mean(dim=2, keepdim=True)
        y = gray + (y - gray) * sat

    # Soft shadow / side illumination mask.
    if shadow_prob > 0 and shadow_strength > 0:
        yy = torch.linspace(-1.0, 1.0, H, device=x.device).view(1, 1, 1, H, 1)
        xx = torch.linspace(-1.0, 1.0, W, device=x.device).view(1, 1, 1, 1, W)
        cx = torch.rand(B, V, 1, 1, 1, device=x.device) * 2.0 - 1.0
        cy = torch.rand(B, V, 1, 1, 1, device=x.device) * 2.0 - 1.0
        sigma = 0.45 + torch.rand(B, V, 1, 1, 1, device=x.device) * 0.45
        blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
        strength = torch.rand(B, V, 1, 1, 1, device=x.device) * shadow_strength
        shadow = 1.0 - strength * blob
        smask = (torch.rand(B, V, 1, 1, 1, device=x.device) < shadow_prob).float()
        y = y * (1.0 - smask) + y * shadow * smask

    # Light sensor noise.
    if noise_std > 0:
        y = y + torch.randn_like(y) * noise_std

    y = torch.clamp(y, 0.0, 1.0)

    # Small average blur on selected views; avoids overfitting to sharp high-frequency edges.
    if blur_prob > 0:
        flat = y.reshape(B * V, C, H, W)
        blurred = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1).reshape(B, V, C, H, W)
        bmask = (torch.rand(B, V, 1, 1, 1, device=x.device) < blur_prob).float()
        y = y * (1.0 - bmask) + blurred * bmask

    x = x * (1.0 - view_mask) + y * view_mask

    # Keep disabled by default for real robot. The battery is small; erasing can hide the target.
    if erasing_prob > 0:
        for bi in range(B):
            for vi in range(V):
                if torch.rand((), device=x.device) < erasing_prob:
                    rh = max(1, int(H * max_erase_ratio * torch.rand((), device=x.device).item()))
                    rw = max(1, int(W * max_erase_ratio * torch.rand((), device=x.device).item()))
                    top = int((H - rh) * torch.rand((), device=x.device).item())
                    left = int((W - rw) * torch.rand((), device=x.device).item())
                    x[bi, vi, :, top:top+rh, left:left+rw] = x[bi, vi].mean()
    return x


def masked_prediction_l1(pred_a, pred_b, action_is_pad):
    """L1 consistency for predicted action chunks, ignoring padded steps."""
    valid = (~action_is_pad[:, :pred_a.shape[1]]).float().unsqueeze(-1)
    return (torch.abs(pred_a - pred_b) * valid).sum() / (valid.sum() * pred_a.shape[-1] + 1e-6)


def train(args):
    set_seed(1)

    DATA_DIR = os.path.expanduser(args.dataset_dir) 
    
    TASK_CONFIGS = {
        args.task_name: {
            'dataset_dir': os.path.join(DATA_DIR, args.task_name),
            'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'],
            'num_episodes': args.num_episodes
        }
    }

    task_config = TASK_CONFIGS[args.task_name]
    
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']

    # fixed parameters
    if args.policy_class == 'ACT':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,     # chunk_size
                         'camera_names': camera_names,
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'kl_weight': args.kl_weight,        # kl
                         'hidden_dim': args.hidden_dim,      # Hidden dim
                         'dim_feedforward': args.dim_feedforward,
                         'enc_layers': args.enc_layers,
                         'dec_layers': args.dec_layers,
                         'nheads': args.nheads,
                         'dropout': args.dropout,
                         'pre_norm': args.pre_norm,
                         # real-robot DINOv2 dense backbone options; ignored by ResNet
                         'dinov2_repo': args.dinov2_repo,
                         'dinov2_weights': args.dinov2_weights,
                         'dinov2_train_layers': args.dinov2_train_layers,
                         'dinov2_pool': args.dinov2_pool,
                         # V6 grasp-aware loss options; training only
                         'use_grasp_weight': args.use_grasp_weight,
                         'grasp_stage_weight': args.grasp_stage_weight,
                         'grasp_arm_dim_weight': args.grasp_arm_dim_weight,
                         'grasp_window': args.grasp_window,
                         'grasp_gripper_index': args.grasp_gripper_index,
                         'grasp_dim_start': args.grasp_dim_start,
                         'grasp_dim_end': args.grasp_dim_end,
                         # V7 generic stage-aware options; training only
                         'use_auto_stage_weight': args.use_auto_stage_weight,
                         'stage_weight_max': args.stage_weight_max,
                         'stage_event_window': args.stage_event_window,
                         'stage_speed_power': args.stage_speed_power,
                         'stage_acc_power': args.stage_acc_power,
                         'stage_gripper_power': args.stage_gripper_power,
                         'stage_gripper_indices': args.stage_gripper_indices,
                         'use_stage_pred': args.use_stage_pred,
                         'stage_num': args.stage_num,
                         'stage_loss_weight': args.stage_loss_weight,
                         'stage_hidden_dim': args.stage_hidden_dim
                         }
    elif args.policy_class == 'CNNMLP':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': 1,     # 查询
                         'camera_names': camera_names,
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'hidden_dim': args.hidden_dim
                         }
    elif args.policy_class == 'Diffusion':
        policy_config = {'lr': args.lr,
                         'lr_backbone': args.lr_backbone,
                         'backbone': args.backbone,
                         'masks': args.masks,
                         'weight_decay': args.weight_decay,
                         'dilation': args.dilation,
                         'position_embedding': args.position_embedding,
                         'loss_function': args.loss_function,
                         'chunk_size': args.chunk_size,     # 查询
                         'camera_names': camera_names,
                         'use_depth_image': args.use_depth_image,
                         'use_robot_base': args.use_robot_base,
                         'observation_horizon': args.observation_horizon,
                         'action_horizon': args.action_horizon,
                         'num_inference_timesteps': args.num_inference_timesteps,
                         'ema_power': args.ema_power,
                         'hidden_dim': args.hidden_dim
                         }
    else:
        raise NotImplementedError

    # Training-time only weak TRIR. It preserves the real-robot qpos-as-action convention.
    policy_config['use_trir'] = bool(args.use_trir and args.policy_class == 'ACT')
    policy_config['trir_aug_prob'] = args.trir_aug_prob
    policy_config['trir_view_prob'] = args.trir_view_prob
    policy_config['trir_aug_weight'] = args.trir_aug_weight
    policy_config['trir_cons_weight'] = args.trir_cons_weight
    policy_config['trir_noise_std'] = args.trir_noise_std
    policy_config['trir_brightness'] = args.trir_brightness
    policy_config['trir_contrast'] = args.trir_contrast
    policy_config['trir_gamma'] = args.trir_gamma
    policy_config['trir_saturation'] = args.trir_saturation
    policy_config['trir_blur_prob'] = args.trir_blur_prob
    policy_config['trir_shadow_prob'] = args.trir_shadow_prob
    policy_config['trir_shadow_strength'] = args.trir_shadow_strength
    policy_config['trir_erasing_prob'] = args.trir_erasing_prob
    policy_config['trir_max_erase_ratio'] = args.trir_max_erase_ratio
    policy_config['trir_feat_cons_weight'] = args.trir_feat_cons_weight

    config = {
        'num_epochs': args.num_epochs,
        'ckpt_dir': args.ckpt_dir,
        'policy_class': args.policy_class,
        'policy_config': policy_config,
        'seed': args.seed,
        'pretrain_ckpt_dir': args.pretrain_ckpt,
    }

    # data Preprocess
    train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, num_episodes, args.arm_delay_time,
                                                           args.use_depth_image, args.use_robot_base, camera_names,
                                                           args.batch_size, args.batch_size)

    # save dataset stats
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    stats_path = os.path.join(args.ckpt_dir, args.ckpt_stats_name)
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_process(train_dataloader, val_dataloader, config, stats)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config, pretrain_ckpt_dir):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            state_dict = torch.load(pretrain_ckpt_dir)
            
            loading_status = policy.deserialize(state_dict)
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.deserialize(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
        if len(pretrain_ckpt_dir) != 0:
            loading_status = policy.deserialize(torch.load(pretrain_ckpt_dir))
            if not loading_status:
                print("ckpt path not exist")
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(policy_config, data, policy):
    image_data, image_depth_data, qpos_data, action_data, action_is_pad = data
    (image_data, qpos_data, action_data, action_is_pad) = (image_data.cuda(), qpos_data.cuda(),
                                                           action_data.cuda(), action_is_pad.cuda())
    if policy_config['use_depth_image']:
        image_depth_data = image_depth_data.cuda()
    else:
        image_depth_data = None

    clean_dict, clean_result = policy(image_data, image_depth_data, qpos_data, action_data, action_is_pad)

    # TRIR++ is training-only. Validation and inference stay identical to the real-robot ACT pipeline.
    if (not policy.training) or (not policy_config.get('use_trir', False)):
        return clean_dict, clean_result

    clean_feat = getattr(getattr(policy, 'model', None), 'last_visual_feature_summary', None)
    if clean_feat is not None:
        clean_feat = clean_feat.detach()

    aug_image_data = apply_real_trir_visual_perturbation(
        image_data,
        aug_prob=policy_config.get('trir_aug_prob', 0.7),
        view_prob=policy_config.get('trir_view_prob', 0.8),
        noise_std=policy_config.get('trir_noise_std', 0.015),
        brightness=policy_config.get('trir_brightness', 0.30),
        contrast=policy_config.get('trir_contrast', 0.30),
        gamma=policy_config.get('trir_gamma', 0.25),
        saturation=policy_config.get('trir_saturation', 0.20),
        blur_prob=policy_config.get('trir_blur_prob', 0.10),
        shadow_prob=policy_config.get('trir_shadow_prob', 0.25),
        shadow_strength=policy_config.get('trir_shadow_strength', 0.25),
        erasing_prob=policy_config.get('trir_erasing_prob', 0.0),
        max_erase_ratio=policy_config.get('trir_max_erase_ratio', 0.10),
    )
    aug_dict, aug_result = policy(aug_image_data, image_depth_data, qpos_data, action_data, action_is_pad)

    aug_feat = getattr(getattr(policy, 'model', None), 'last_visual_feature_summary', None)

    trir_aug_l1 = aug_dict.get('l1', aug_dict['loss'])
    trir_cons = masked_prediction_l1(aug_result, clean_result.detach(), action_is_pad)

    feat_w = policy_config.get('trir_feat_cons_weight', 0.03)
    if feat_w > 0 and clean_feat is not None and aug_feat is not None:
        trir_feat_cons = F.smooth_l1_loss(aug_feat, clean_feat)
    else:
        trir_feat_cons = torch.zeros((), device=image_data.device)

    loss = clean_dict['loss'] \
           + policy_config.get('trir_aug_weight', 0.5) * trir_aug_l1 \
           + policy_config.get('trir_cons_weight', 0.10) * trir_cons \
           + feat_w * trir_feat_cons

    out = dict(clean_dict)
    out['loss'] = loss
    out['trir_aug_l1'] = trir_aug_l1.detach()
    out['trir_cons'] = trir_cons.detach()
    out['trir_feat_cons'] = trir_feat_cons.detach()
    return out, clean_result


def train_process(train_dataloader, val_dataloader, config, stats):
    post_process = lambda a: a * stats['qpos_std'] + stats['qpos_mean']
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    pretrain_ckpt_dir = config['pretrain_ckpt_dir']
    set_seed(seed)

    policy = make_policy(policy_class, policy_config, pretrain_ckpt_dir)

    # Resume model weights from an existing ACT checkpoint
    resume_ckpt = os.environ.get("ACT_RESUME_CKPT", "")
    start_epoch = int(os.environ.get("ACT_START_EPOCH", "0"))

    if resume_ckpt:
        print(f"[Resume] Loading checkpoint: {resume_ckpt}")
        resume_state = torch.load(resume_ckpt, map_location="cpu")
        loading_status = policy.deserialize(resume_state)
        print(f"[Resume] Loading status: {loading_status}")
        print(f"[Resume] Continuing from epoch {start_epoch}")

    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(start_epoch, num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict, result = forward_pass(policy_config, data, policy)
                # print("result:", post_process(result.cpu().detach().numpy())[0, :, 7:])
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.serialize()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict, result = forward_pass(policy_config, data, policy)
            # print("result:", post_process(result.cpu().detach().numpy())[0, :, 7:])
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[-(batch_idx + 1):])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        if len(validation_history) > 0 and key in validation_history[0]:
            val_values = [summary[key].item() for summary in validation_history]
            plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='dataset_dir', default='./dataset', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, help='num_episodes', required=True)
   
    parser.add_argument('--pretrain_ckpt', action='store', type=str, help='pretrain_ckpt', default='', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', default='aloha_mobile_dummy', required=False)
    
    parser.add_argument('--ckpt_name', action='store', type=str, help='ckpt_name', default='policy_best.ckpt', required=False)
    parser.add_argument('--ckpt_stats_name', action='store', type=str, help='ckpt_stats_name', default='dataset_stats.pkl', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize, CNNMLP, ACT, Diffusion', default='ACT', required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', default=32, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', default=0, required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', default=3000, required=False)

    parser.add_argument('--lr', action='store', type=float, help='lr', default=4e-5, required=False)
    parser.add_argument('--weight_decay', type=float, help='weight_decay', default=1e-4, required=False)
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)", required=False)
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features", required=False)
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', default=14, required=False)
    parser.add_argument('--lr_backbone', action='store', type=float, help='lr_backbone', default=4e-5, required=False)
    parser.add_argument('--backbone', action='store', type=str, help='backbone', default='resnet18', required=False)
    parser.add_argument('--loss_function', action='store', type=str, help='loss_function l1 l2 l1+l2', default='l1', required=False)
    parser.add_argument('--enc_layers', action='store', type=int, help='enc_layers', default=4, required=False)
    parser.add_argument('--dec_layers', action='store', type=int, help='dec_layers', default=7, required=False)
    parser.add_argument('--nheads', action='store', type=int, help='nheads', default=8, required=False)
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer", required=False)
    parser.add_argument('--pre_norm', action='store_true', required=False)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', default=10, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', default=32, required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', default=512, required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', default=3200, required=False)
    parser.add_argument('--temporal_agg',  action='store', type=bool, help='temporal_agg', default=True, required=False)

    # for Diffusion
    parser.add_argument('--observation_horizon', action='store', type=int, help='observation_horizon', default=1, required=False)
    parser.add_argument('--action_horizon', action='store', type=int, help='action_horizon', default=8, required=False)
    parser.add_argument('--num_inference_timesteps', action='store', type=int, help='num_inference_timesteps', default=10, required=False)
    parser.add_argument('--ema_power', action='store', type=int, help='ema_power', default=0.75, required=False)

    parser.add_argument('--use_robot_base', action='store', type=bool, help='use_robot_base', default=False, required=False)

    parser.add_argument('--arm_delay_time', action='store', type=int, help='arm_delay_time', default=0, required=False)

    parser.add_argument('--use_depth_image', action='store', type=bool, help='use_depth_image', default=False, required=False)

    # real-robot innovations: DINOv2 dense backbone + weak TRIR. TRIR is training only.
    parser.add_argument('--dinov2_repo', action='store', type=str,
                        default='/home/d510/cobot_magic/dinov2_local/dinov2-main', required=False)
    parser.add_argument('--dinov2_weights', action='store', type=str,
                        default='/home/d510/cobot_magic/dinov2_local/dinov2_vits14_pretrain.pth', required=False)
    parser.add_argument('--dinov2_train_layers', action='store', type=int, default=8, required=False)
    parser.add_argument('--dinov2_pool', action='store', type=int, default=2, required=False)
    parser.add_argument('--use_trir', action='store_true', help='enable weak TRIR for ACT training')
    parser.add_argument('--trir_aug_prob', action='store', type=float, default=0.7)
    parser.add_argument('--trir_view_prob', action='store', type=float, default=0.8)
    parser.add_argument('--trir_aug_weight', action='store', type=float, default=0.5)
    parser.add_argument('--trir_cons_weight', action='store', type=float, default=0.10)
    parser.add_argument('--trir_noise_std', action='store', type=float, default=0.015)
    parser.add_argument('--trir_brightness', action='store', type=float, default=0.30)
    parser.add_argument('--trir_contrast', action='store', type=float, default=0.30)
    parser.add_argument('--trir_gamma', action='store', type=float, default=0.25)
    parser.add_argument('--trir_saturation', action='store', type=float, default=0.20)
    parser.add_argument('--trir_blur_prob', action='store', type=float, default=0.10)
    parser.add_argument('--trir_shadow_prob', action='store', type=float, default=0.25)
    parser.add_argument('--trir_shadow_strength', action='store', type=float, default=0.25)
    parser.add_argument('--trir_feat_cons_weight', action='store', type=float, default=0.03)
    parser.add_argument('--trir_erasing_prob', action='store', type=float, default=0.0)
    parser.add_argument('--trir_max_erase_ratio', action='store', type=float, default=0.10)

    # V6 grasp-aware loss. Defaults match ALOHA-style 14D qpos: right arm dims 7:14, right gripper index 13.
    parser.add_argument('--use_grasp_weight', action='store_true', help='enable grasp-stage weighted imitation loss')
    parser.add_argument('--grasp_stage_weight', action='store', type=float, default=3.0)
    parser.add_argument('--grasp_arm_dim_weight', action='store', type=float, default=2.0)
    parser.add_argument('--grasp_window', action='store', type=int, default=8)
    parser.add_argument('--grasp_gripper_index', action='store', type=int, default=13)
    parser.add_argument('--grasp_dim_start', action='store', type=int, default=7)
    parser.add_argument('--grasp_dim_end', action='store', type=int, default=14)

    # V7 generic auto stage-aware loss. This is not tied to battery/right-arm tasks.
    parser.add_argument('--use_auto_stage_weight', action='store_true',
                        help='enable generic auto stage-aware weighting from action transitions')
    parser.add_argument('--stage_weight_max', action='store', type=float, default=3.0,
                        help='maximum time-step weight for automatically detected key stages')
    parser.add_argument('--stage_event_window', action='store', type=int, default=8,
                        help='temporal half-window around detected key transitions')
    parser.add_argument('--stage_speed_power', action='store', type=float, default=1.0)
    parser.add_argument('--stage_acc_power', action='store', type=float, default=0.7)
    parser.add_argument('--stage_gripper_power', action='store', type=float, default=1.0)
    parser.add_argument('--stage_gripper_indices', action='store', type=str, default='6,13',
                        help='comma-separated gripper dims used only as generic event cues; use none to disable')
    parser.add_argument('--use_stage_pred', action='store_true',
                        help='enable auxiliary pseudo-stage prediction head during training')
    parser.add_argument('--stage_num', action='store', type=int, default=5)
    parser.add_argument('--stage_loss_weight', action='store', type=float, default=0.05)
    parser.add_argument('--stage_hidden_dim', action='store', type=int, default=128)

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    train(args)

if __name__ == '__main__':
    main()
# python act/train.py --dataset_dir ~/data --pretrain_ckpt policy_best.ckpt --ckpt_dir ~/train_dir/ --num_episodes 20 --batch_size 10 --num_epochs 2000 