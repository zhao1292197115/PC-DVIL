import os
import pickle
import argparse
import csv
import json
import time
import warnings

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless plotting for training; does not affect remote desktop
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data
from utils import sample_box_pose, sample_insertion_pose
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE


def configure_cuda_runtime():
    """Safe speed knobs for RTX 30/40 series. They do not change model architecture."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def get_amp_dtype(dtype_name: str):
    dtype_name = str(dtype_name).lower()
    if dtype_name == 'bf16':
        if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported'):
            if not torch.cuda.is_bf16_supported():
                warnings.warn('BF16 is not reported as supported on this GPU. Falling back to FP16.')
                return torch.float16
        return torch.bfloat16
    if dtype_name == 'fp16':
        return torch.float16
    if dtype_name in ['fp32', 'none', 'false']:
        return torch.float32
    raise ValueError(f'Unknown amp dtype: {dtype_name}. Use bf16, fp16, or fp32.')


def tensor_dict_to_cpu(state_dict):
    """Avoid keeping an extra full checkpoint copy on GPU memory."""
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def main(args):
    configure_cuda_runtime()
    set_seed(args['seed'])

    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]

    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    state_dim = 14
    lr_backbone = args['lr_backbone']
    backbone = args['backbone']

    # Extra DINOv2 options are consumed by models/backbone.py through args.
    common_policy_config = {
        'lr': args['lr'],
        'lr_backbone': lr_backbone,
        'backbone': backbone,
        'camera_names': camera_names,
        'dinov2_train_layers': args['dinov2_train_layers'],
        'dinov2_pool': args['dinov2_pool'],
    }

    if policy_class == 'ACT':
        policy_config = {
            **common_policy_config,
            'num_queries': args['chunk_size'],
            'kl_weight': args['kl_weight'],
            'hidden_dim': args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'enc_layers': args['enc_layers'],
            'dec_layers': args['dec_layers'],
            'nheads': args['nheads'],
        }
    elif policy_class == 'CNNMLP':
        policy_config = {**common_policy_config, 'num_queries': 1}
    else:
        raise NotImplementedError(f'Unsupported policy_class: {policy_class}')

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'amp': args['amp'],
        'amp_dtype': args['amp_dtype'],
        'val_every': args['val_every'],
        'save_every': args['save_every'],
        'eval_rollouts': args['eval_rollouts'],
        'save_eval_video': args['save_eval_video'],
        'eval_metrics': args['eval_metrics'],
        'measure_latency': args['measure_latency'],
        'eval_pose_path': args.get('eval_pose_path', None),
    }

    os.makedirs(ckpt_dir, exist_ok=True)

    if is_eval:
        ckpt_names = ['policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=args['save_eval_video'])
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: success_rate={success_rate}, avg_return={avg_return}')
        return

    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
    )

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_epoch, min_val_loss, best_state_dict = train_bc(train_dataloader, val_dataloader, config)

    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {float(min_val_loss):.6f} @ epoch {best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        return ACTPolicy(policy_config)
    if policy_class == 'CNNMLP':
        return CNNMLPPolicy(policy_config)
    raise NotImplementedError(f'Unsupported policy_class: {policy_class}')


def make_optimizer(policy_class, policy):
    if policy_class in ['ACT', 'CNNMLP']:
        return policy.configure_optimizers()
    raise NotImplementedError(f'Unsupported policy_class: {policy_class}')


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda(non_blocking=True).unsqueeze(0)
    return curr_image


def _safe_numeric(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def first_step_at_or_above(rewards, threshold):
    for idx, reward in enumerate(rewards):
        reward_value = _safe_numeric(reward)
        if not np.isnan(reward_value) and reward_value >= threshold:
            return idx
    return -1


def compute_motion_metrics(target_qpos_list, dt, query_frequency, temporal_agg):
    """
    Compute policy-output smoothness metrics from commanded target qpos.

    These are model-side action quality metrics, not physical force/contact metrics.
    They are useful for comparing whether a method improves success by smoother
    fine manipulation or by aggressive action jumps.
    """
    metrics = {
        'mean_action_delta_norm': np.nan,
        'max_action_delta_norm': np.nan,
        'max_joint_delta_abs': np.nan,
        'mean_action_vel_norm': np.nan,
        'mean_action_acc_norm': np.nan,
        'mean_action_jerk_norm': np.nan,
        'chunk_boundary_jump_mean': np.nan,
        'chunk_boundary_jump_max': np.nan,
    }

    actions = np.asarray(target_qpos_list, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 2:
        return metrics

    delta = np.diff(actions, axis=0)
    delta_norm = np.linalg.norm(delta, axis=1)
    metrics['mean_action_delta_norm'] = float(np.mean(delta_norm))
    metrics['max_action_delta_norm'] = float(np.max(delta_norm))
    metrics['max_joint_delta_abs'] = float(np.max(np.abs(delta)))

    dt = float(dt) if dt is not None and dt > 0 else 1.0
    vel = delta / dt
    vel_norm = np.linalg.norm(vel, axis=1)
    metrics['mean_action_vel_norm'] = float(np.mean(vel_norm))

    if vel.shape[0] >= 2:
        acc = np.diff(vel, axis=0) / dt
        acc_norm = np.linalg.norm(acc, axis=1)
        metrics['mean_action_acc_norm'] = float(np.mean(acc_norm))
    else:
        acc = None

    if acc is not None and acc.shape[0] >= 2:
        jerk = np.diff(acc, axis=0) / dt
        jerk_norm = np.linalg.norm(jerk, axis=1)
        metrics['mean_action_jerk_norm'] = float(np.mean(jerk_norm))

    # For non-temporal-aggregation ACT, a new action chunk is queried every
    # num_queries steps. The boundary jump is action[t] - action[t-1] at those
    # chunk boundaries. With temporal aggregation, this is less meaningful.
    if (not temporal_agg) and query_frequency is not None and int(query_frequency) > 1:
        boundary_indices = list(range(int(query_frequency), actions.shape[0], int(query_frequency)))
        if len(boundary_indices) > 0:
            jumps = np.asarray([
                np.linalg.norm(actions[i] - actions[i - 1])
                for i in boundary_indices
            ], dtype=np.float64)
            metrics['chunk_boundary_jump_mean'] = float(np.mean(jumps))
            metrics['chunk_boundary_jump_max'] = float(np.max(jumps))

    return metrics


def compute_rollout_metrics(
    rollout_id,
    rewards,
    target_qpos_list,
    env_max_reward,
    query_frequency,
    temporal_agg,
    policy_latency_ms_list,
):
    reward_values = np.asarray([_safe_numeric(r) for r in rewards], dtype=np.float64)
    valid_rewards = reward_values[~np.isnan(reward_values)]

    episode_return = float(np.sum(valid_rewards)) if valid_rewards.size > 0 else 0.0
    highest_reward = float(np.max(valid_rewards)) if valid_rewards.size > 0 else 0.0
    success = bool(highest_reward >= env_max_reward)
    completion_step = first_step_at_or_above(rewards, env_max_reward)

    metrics = {
        'rollout_id': int(rollout_id),
        'success': int(success),
        'episode_return': episode_return,
        'highest_reward': highest_reward,
        'env_max_reward': float(env_max_reward),
        'completion_step': int(completion_step),
    }

    for r in range(int(env_max_reward) + 1):
        step = first_step_at_or_above(rewards, r)
        metrics[f'reward_ge_{r}'] = int(step >= 0)
        metrics[f'reward_ge_{r}_first_step'] = int(step)

    metrics.update(compute_motion_metrics(target_qpos_list, DT, query_frequency, temporal_agg))

    latency = np.asarray(policy_latency_ms_list, dtype=np.float64)
    if latency.size > 0:
        metrics['policy_latency_ms_mean'] = float(np.mean(latency))
        metrics['policy_latency_ms_max'] = float(np.max(latency))
        metrics['policy_fps_mean'] = float(1000.0 / max(metrics['policy_latency_ms_mean'], 1e-9))
    else:
        metrics['policy_latency_ms_mean'] = np.nan
        metrics['policy_latency_ms_max'] = np.nan
        metrics['policy_fps_mean'] = np.nan

    if torch.cuda.is_available():
        metrics['gpu_peak_memory_mb'] = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    else:
        metrics['gpu_peak_memory_mb'] = np.nan

    return metrics


def _json_sanitize(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def summarize_rollout_metrics(rollout_metrics, env_max_reward):
    summary = {}
    if len(rollout_metrics) == 0:
        return summary

    numeric_keys = []
    for key in rollout_metrics[0].keys():
        values = []
        for row in rollout_metrics:
            value = row.get(key, np.nan)
            if isinstance(value, (int, float, np.integer, np.floating)):
                values.append(float(value))
        if len(values) == len(rollout_metrics):
            numeric_keys.append(key)

    for key in numeric_keys:
        values = np.asarray([float(row.get(key, np.nan)) for row in rollout_metrics], dtype=np.float64)
        values = values[~np.isnan(values)]
        if values.size > 0:
            summary[f'{key}_mean'] = float(np.mean(values))

    summary['success_rate'] = float(np.mean([row['success'] for row in rollout_metrics]))
    summary['avg_return'] = float(np.mean([row['episode_return'] for row in rollout_metrics]))
    for r in range(int(env_max_reward) + 1):
        key = f'reward_ge_{r}'
        summary[f'{key}_rate'] = float(np.mean([row[key] for row in rollout_metrics]))

    return summary


def save_eval_metrics(ckpt_dir, ckpt_name, rollout_metrics, summary):
    metric_stem = ckpt_name.split('.')[0]
    csv_path = os.path.join(ckpt_dir, f'eval_metrics_{metric_stem}.csv')
    json_path = os.path.join(ckpt_dir, f'eval_summary_{metric_stem}.json')

    if len(rollout_metrics) > 0:
        fieldnames = []
        for row in rollout_metrics:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rollout_metrics:
                writer.writerow(_json_sanitize(row))

    with open(json_path, 'w') as f:
        json.dump(_json_sanitize(summary), f, indent=2, ensure_ascii=False)

    print(f'📄 Saved per-rollout metrics CSV: {csv_path}')
    print(f'📄 Saved summary metrics JSON: {json_path}')


def format_metric_summary_for_txt(summary):
    keys = [
        'success_rate',
        'avg_return',
        'completion_step_mean',
        'mean_action_delta_norm_mean',
        'mean_action_vel_norm_mean',
        'mean_action_acc_norm_mean',
        'mean_action_jerk_norm_mean',
        'chunk_boundary_jump_mean_mean',
        'policy_latency_ms_mean_mean',
        'policy_fps_mean_mean',
        'gpu_peak_memory_mb_mean',
    ]
    lines = ['\nExtra Eval Metrics:\n']
    for key in keys:
        if key in summary and summary[key] is not None:
            lines.append(f'{key}: {summary[key]}\n')
    return ''.join(lines)


def eval_bc(config, ckpt_name, save_episode=True):
    configure_cuda_runtime()
    set_seed(1000)

    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    if real_robot:
        from aloha_scripts.robot_utils import move_grippers
        from aloha_scripts.real_env import make_real_env
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    amp_enabled = bool(config.get('amp', False))
    amp_dtype = get_amp_dtype(config.get('amp_dtype', 'bf16'))
    amp_enabled = amp_enabled and amp_dtype != torch.float32

    num_rollouts = int(config.get('eval_rollouts', 50))
    eval_pose_list = None
    eval_pose_path = config.get('eval_pose_path', None)
    if eval_pose_path is not None:
        with open(eval_pose_path, 'rb') as f:
            eval_pose_list = pickle.load(f)
        eval_pose_list = [np.asarray(p, dtype=np.float64).copy() for p in eval_pose_list]
        if len(eval_pose_list) == 0:
            raise ValueError(f'Empty eval pose list: {eval_pose_path}')
        print(f'Using fixed eval poses from {eval_pose_path}, num={len(eval_pose_list)}')
    episode_returns = []
    highest_rewards = []
    rollout_metrics = []

    for rollout_id in range(num_rollouts):
        if eval_pose_list is not None:
            fixed_pose = eval_pose_list[rollout_id % len(eval_pose_list)]
            BOX_POSE[0] = fixed_pose.copy()
        elif 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose()
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())

        ts = env.reset()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []
        qpos_list = []
        target_qpos_list = []
        rewards = []
        policy_latency_ms_list = []

        with torch.inference_mode():
            for t in range(max_timesteps):
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})

                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda(non_blocking=True).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                if config.get('measure_latency', True) and torch.cuda.is_available():
                    torch.cuda.synchronize()
                policy_start_time = time.perf_counter()

                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    if policy_class == 'ACT':
                        if t % query_frequency == 0:
                            all_actions = policy(qpos, curr_image)
                        if temporal_agg:
                            all_time_actions[[t], t:t + num_queries] = all_actions.float()
                            actions_for_curr_step = all_time_actions[:, t]
                            actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                            actions_for_curr_step = actions_for_curr_step[actions_populated]
                            k = 0.01
                            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                            exp_weights = exp_weights / exp_weights.sum()
                            exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                            raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                        else:
                            raw_action = all_actions[:, t % query_frequency]
                    elif policy_class == 'CNNMLP':
                        raw_action = policy(qpos, curr_image)
                    else:
                        raise NotImplementedError

                if config.get('measure_latency', True) and torch.cuda.is_available():
                    torch.cuda.synchronize()
                policy_latency_ms_list.append((time.perf_counter() - policy_start_time) * 1000.0)

                raw_action = raw_action.float().squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ts = env.step(target_qpos)

                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            if onscreen_render:
                plt.close()

        if real_robot:
            move_grippers(
                [env.puppet_bot_left, env.puppet_bot_right],
                [PUPPET_GRIPPER_JOINT_OPEN] * 2,
                move_time=0.5,
            )

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        rollout_metric = compute_rollout_metrics(
            rollout_id=rollout_id,
            rewards=rewards,
            target_qpos_list=target_qpos_list,
            env_max_reward=env_max_reward,
            query_frequency=query_frequency,
            temporal_agg=temporal_agg,
            policy_latency_ms_list=policy_latency_ms_list,
        )
        rollout_metric['eval_pose_id'] = int(rollout_id % len(eval_pose_list)) if eval_pose_list is not None else -1
        rollout_metrics.append(rollout_metric)

        print(
            f'Rollout {rollout_id}\n'
            f'episode_return={episode_return}, episode_highest_reward={episode_highest_reward}, '
            f'env_max_reward={env_max_reward}, Success: {episode_highest_reward == env_max_reward}, '
            f'completion_step={rollout_metric["completion_step"]}, '
            f'mean_latency_ms={rollout_metric["policy_latency_ms_mean"]:.2f}, '
            f'mean_jerk={rollout_metric["mean_action_jerk_norm"]}'
        )

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n'
    if eval_pose_list is not None:
        summary_str += f'Fixed eval pose path: {eval_pose_path}\n'
    summary_str += '\n'
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n'

    metric_summary = summarize_rollout_metrics(rollout_metrics, env_max_reward)
    summary_str += format_metric_summary_for_txt(metric_summary)

    print(summary_str)

    if config.get('eval_metrics', False):
        save_eval_metrics(ckpt_dir, ckpt_name, rollout_metrics, metric_summary)

    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.cuda(non_blocking=True)
    qpos_data = qpos_data.cuda(non_blocking=True)
    action_data = action_data.cuda(non_blocking=True)
    is_pad = is_pad.cuda(non_blocking=True)
    return policy(qpos_data, image_data, action_data, is_pad)


def print_trainable_summary(policy):
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(
        f'\n📊 Trainable params: {trainable_params / 1e6:.2f}M / '
        f'{total_params / 1e6:.2f}M ({100.0 * trainable_params / max(total_params, 1):.2f}%)\n'
    )


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    val_every = max(1, int(config.get('val_every', 5)))
    save_every = max(1, int(config.get('save_every', 100)))

    set_seed(seed)
    os.makedirs(ckpt_dir, exist_ok=True)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    print_trainable_summary(policy)

    optimizer = make_optimizer(policy_class, policy)

    amp_enabled = bool(config.get('amp', False))
    amp_dtype = get_amp_dtype(config.get('amp_dtype', 'bf16'))
    amp_enabled = amp_enabled and amp_dtype != torch.float32
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    print(
        f'🚀 Training config: amp={amp_enabled}, amp_dtype={amp_dtype}, '
        f'val_every={val_every}, save_every={save_every}, '
        f'lr={policy_config["lr"]}, lr_backbone={policy_config["lr_backbone"]}, '
        f'dinov2_train_layers={policy_config.get("dinov2_train_layers")}, '
        f'dinov2_pool={policy_config.get("dinov2_pool")}'
    )

    train_history = []
    validation_history = []
    validation_epochs = []
    min_val_loss = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')

        do_val = (epoch % val_every == 0) or (epoch == num_epochs - 1)
        if do_val:
            with torch.inference_mode():
                policy.eval()
                epoch_dicts = []
                for _, data in enumerate(val_dataloader):
                    with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                        forward_dict = forward_pass(data, policy)
                    epoch_dicts.append(detach_dict(forward_dict))

                epoch_summary = compute_dict_mean(epoch_dicts)
                validation_history.append(epoch_summary)
                validation_epochs.append(epoch)

                epoch_val_loss = epoch_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, tensor_dict_to_cpu(policy.state_dict()))

            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''.join([f'{k}: {v.item():.3f} ' for k, v in epoch_summary.items()])
            print(summary_string)
        else:
            print(f'Val skipped. Next validation at epoch multiple of {val_every}.')

        policy.train()
        epoch_train_dicts = []
        optimizer.zero_grad(set_to_none=True)

        for _, data in enumerate(train_dataloader):
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                forward_dict = forward_pass(data, policy)
                loss = forward_dict['loss']

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            detached = detach_dict(forward_dict)
            train_history.append(detached)
            epoch_train_dicts.append(detached)

        epoch_summary = compute_dict_mean(epoch_train_dicts)
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''.join([f'{k}: {v.item():.3f} ' for k, v in epoch_summary.items()])
        print(summary_string)

        if epoch % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(tensor_dict_to_cpu(policy.state_dict()), ckpt_path)
            plot_history(train_history, validation_history, validation_epochs, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(tensor_dict_to_cpu(policy.state_dict()), ckpt_path)

    if best_ckpt_info is None:
        best_ckpt_info = (num_epochs - 1, epoch_train_loss, tensor_dict_to_cpu(policy.state_dict()))

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {float(min_val_loss):.6f} at epoch {best_epoch}')

    plot_history(train_history, validation_history, validation_epochs, num_epochs, ckpt_dir, seed)
    return best_ckpt_info


def plot_history(train_history, validation_history, validation_epochs, num_epochs, ckpt_dir, seed):
    if len(train_history) == 0:
        return

    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        train_x = np.linspace(0, max(num_epochs - 1, 0), len(train_values))
        plt.plot(train_x, train_values, label='train')

        if len(validation_history) > 0:
            val_values = [summary[key].item() for summary in validation_history]
            val_x = np.array(validation_epochs)
            plt.plot(val_x, val_values, label='validation')

        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, required=True)
    parser.add_argument('--policy_class', action='store', type=str, required=True)
    parser.add_argument('--task_name', action='store', type=str, required=True)
    parser.add_argument('--batch_size', action='store', type=int, required=True)
    parser.add_argument('--seed', action='store', type=int, required=True)
    parser.add_argument('--num_epochs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)

    # Backbone / DINOv2 controls.
    parser.add_argument('--backbone', action='store', type=str, default='dinov2_vits14')
    parser.add_argument('--lr_backbone', action='store', type=float, default=1e-5)
    parser.add_argument(
        '--dinov2_train_layers',
        action='store',
        type=int,
        default=4,
        help='0: freeze DINOv2; N>0: train last N transformer blocks + norm; -1: train all DINOv2 parameters.',
    )
    parser.add_argument(
        '--dinov2_pool',
        action='store',
        type=int,
        default=1,
        help='1: full high-resolution patch grid; 2: avg-pool tokens by 2x2 to reduce memory.',
    )

    # Training speed / memory controls.
    parser.add_argument('--amp', action='store_true', help='Enable CUDA automatic mixed precision.')
    parser.add_argument('--amp_dtype', action='store', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    parser.add_argument('--val_every', action='store', type=int, default=5)
    parser.add_argument('--save_every', action='store', type=int, default=100)

    # Eval controls.
    parser.add_argument('--eval_rollouts', action='store', type=int, default=50)
    parser.add_argument('--save_eval_video', action='store_true')
    parser.add_argument('--eval_metrics', action='store_true', help='Save eval_metrics_policy_best.csv and eval_summary_policy_best.json.')
    parser.add_argument('--measure_latency', action='store_true', help='Synchronize CUDA and measure per-step policy latency during eval.')
    parser.add_argument('--eval_pose_path', action='store', type=str, default=None, help='Optional .pkl file containing fixed BOX_POSE list for deterministic eval.')

    # ACT config.
    parser.add_argument('--kl_weight', action='store', type=int, required=False, default=10)
    parser.add_argument('--chunk_size', action='store', type=int, required=False, default=100)
    parser.add_argument('--hidden_dim', action='store', type=int, required=False, default=512)
    parser.add_argument('--dim_feedforward', action='store', type=int, required=False, default=3200)
    parser.add_argument('--enc_layers', action='store', type=int, required=False, default=4)
    parser.add_argument('--dec_layers', action='store', type=int, required=False, default=7)
    parser.add_argument('--nheads', action='store', type=int, required=False, default=8)
    parser.add_argument('--temporal_agg', action='store_true')

    parsed_args = parser.parse_args()

    # such as --amp, --dinov2_train_layers, --dinov2_pool, --val_every.

    main(vars(parsed_args))

