import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer, build_diffusion_model_and_optimizer

import IPython
e = IPython.embed


class DiffusionPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_diffusion_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

    def configure_optimizers(self):
        return self.optimizer

    def __call__(self, image, depth_image, robot_state, actions=None, action_is_pad=None):
        B = robot_state.shape[0]
        if actions is not None:
            noise, noise_pred = self.model(image, depth_image, robot_state, actions, action_is_pad)
            # L2 loss
            all_l2 = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (all_l2 * ~action_is_pad.unsqueeze(-1)).mean()

            loss_dict = {}
            loss_dict['l2_loss'] = loss
            loss_dict['loss'] = loss
            return loss_dict, (noise, noise_pred)
        else:  # inference time
            return self.model(image, depth_image, robot_state, actions, action_is_pad)

    def serialize(self):
        return self.model.serialize()

    def deserialize(self, model_dict):
        return self.model.deserialize(model_dict)


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)

        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.loss_function = args_override['loss_function']
        self.action_dim = 16 if bool(args_override.get('use_robot_base', False)) else 14

        # V6 legacy option: task-specific grasp weighting. Kept for compatibility,
        # but it is OFF unless --use_grasp_weight is explicitly passed.
        self.use_grasp_weight = bool(args_override.get('use_grasp_weight', False))
        self.grasp_stage_weight = float(args_override.get('grasp_stage_weight', 3.0))
        self.grasp_arm_dim_weight = float(args_override.get('grasp_arm_dim_weight', 2.0))
        self.grasp_window = int(args_override.get('grasp_window', 8))
        self.grasp_gripper_index = int(args_override.get('grasp_gripper_index', 13))
        self.grasp_dim_start = int(args_override.get('grasp_dim_start', 7))
        self.grasp_dim_end = int(args_override.get('grasp_dim_end', 14))

        # V7: generic auto stage-aware loss. This does not assume a specific task
        # such as right-arm battery grasping. It detects high-change moments from
        # the demonstration action sequence and upweights those transition windows.
        self.use_auto_stage_weight = bool(args_override.get('use_auto_stage_weight', False))
        self.stage_weight_max = float(args_override.get('stage_weight_max', 3.0))
        self.stage_event_window = int(args_override.get('stage_event_window', 8))
        self.stage_speed_power = float(args_override.get('stage_speed_power', 1.0))
        self.stage_acc_power = float(args_override.get('stage_acc_power', 0.7))
        self.stage_gripper_power = float(args_override.get('stage_gripper_power', 1.0))
        self.stage_min_score_eps = float(args_override.get('stage_min_score_eps', 1e-6))
        self.stage_gripper_indices = self._parse_stage_indices(args_override.get('stage_gripper_indices', '6,13'))

        # V7: optional stage-prediction auxiliary head. It is used only during
        # training. At inference, the action path is unchanged. The head is
        # attached to ACTPolicy rather than DETRVAE so older model code remains
        # minimally invasive.
        self.use_stage_pred = bool(args_override.get('use_stage_pred', False))
        self.stage_num = int(args_override.get('stage_num', 5))
        self.stage_loss_weight = float(args_override.get('stage_loss_weight', 0.05))
        self.stage_hidden_dim = int(args_override.get('stage_hidden_dim', 128))
        if self.use_stage_pred:
            self.stage_head = nn.Sequential(
                nn.Linear(self.action_dim, self.stage_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.stage_hidden_dim, self.stage_num),
            )
            # build_ACT_model_and_optimizer only sees DETRVAE parameters, so add
            # the auxiliary head parameters to the existing optimizer.
            try:
                self.optimizer.add_param_group({'params': self.stage_head.parameters(),
                                                'lr': float(args_override.get('lr', 1e-5))})
            except Exception as e:
                print('[V7][WARN] failed to add stage_head param group:', e)

        print(f'KL Weight {self.kl_weight}')
        if self.use_auto_stage_weight:
            print(f'[V7] auto stage-aware loss: max_weight={self.stage_weight_max}, '
                  f'window={self.stage_event_window}, speed={self.stage_speed_power}, '
                  f'acc={self.stage_acc_power}, gripper={self.stage_gripper_power}, '
                  f'indices={self.stage_gripper_indices}')
        if self.use_stage_pred:
            print(f'[V7] stage prediction head: stage_num={self.stage_num}, '
                  f'weight={self.stage_loss_weight}, hidden={self.stage_hidden_dim}')
        if self.use_grasp_weight:
            print(f'[V6 compatibility] grasp-aware loss: stage_weight={self.grasp_stage_weight}, '
                  f'arm_dim_weight={self.grasp_arm_dim_weight}, window={self.grasp_window}, '
                  f'gripper_index={self.grasp_gripper_index}, dims=[{self.grasp_dim_start}:{self.grasp_dim_end}]')

    @staticmethod
    def _parse_stage_indices(v):
        if v is None:
            return []
        if isinstance(v, (list, tuple)):
            return [int(x) for x in v]
        txt = str(v).strip().lower()
        if txt in ('', 'none', 'no', 'false', '-1'):
            return []
        out = []
        for item in txt.replace(';', ',').split(','):
            item = item.strip()
            if item:
                try:
                    out.append(int(item))
                except ValueError:
                    pass
        return out

    def _element_loss(self, actions, a_hat):
        if self.loss_function == 'l1':
            return F.l1_loss(actions, a_hat, reduction='none')
        elif self.loss_function == 'l2':
            return F.mse_loss(actions, a_hat, reduction='none')
        else:
            return F.smooth_l1_loss(actions, a_hat, reduction='none')

    def _masked_mean(self, all_loss, action_is_pad, extra_weights=None):
        valid = (~action_is_pad).float().unsqueeze(-1)
        if extra_weights is None:
            denom = valid.sum() * all_loss.shape[-1] + 1e-6
            return (all_loss * valid).sum() / denom
        w = valid * extra_weights
        return (all_loss * w).sum() / (w.sum() + 1e-6)

    def _smooth_time_score(self, score, window):
        # score: [B, T]
        if window <= 0 or score.shape[1] <= 1:
            return score
        k = int(window) * 2 + 1
        x = score.unsqueeze(1)
        # Max-pool keeps sharp contact/transition moments; avg-pool would dilute
        # very short gripper events.
        x = F.max_pool1d(x, kernel_size=k, stride=1, padding=window)
        return x.squeeze(1)

    def _event_score(self, actions, action_is_pad):
        """Generic event score from action geometry.
        High values indicate task-stage transitions such as grasp, release,
        arm handoff, drawer pull start, insertion contact, etc. It uses the
        demonstrated action sequence only and does not require manual labels.
        """
        B, T, D = actions.shape
        device, dtype = actions.device, actions.dtype
        if T <= 1:
            return torch.zeros((B, T), device=device, dtype=dtype)

        valid = (~action_is_pad).float()
        delta = torch.zeros((B, T, D), device=device, dtype=dtype)
        delta[:, 1:] = actions[:, 1:] - actions[:, :-1]
        speed = delta.abs().mean(dim=-1)

        acc = torch.zeros((B, T, D), device=device, dtype=dtype)
        if T > 2:
            acc[:, 2:] = delta[:, 2:] - delta[:, 1:-1]
        accel = acc.abs().mean(dim=-1)

        gripper_score = torch.zeros((B, T), device=device, dtype=dtype)
        valid_indices = []
        for idx in self.stage_gripper_indices:
            j = idx if idx >= 0 else D + idx
            if 0 <= j < D:
                valid_indices.append(j)
        if len(valid_indices) > 0:
            gripper_score = delta[:, :, valid_indices].abs().mean(dim=-1)

        score = self.stage_speed_power * speed + self.stage_acc_power * accel + self.stage_gripper_power * gripper_score
        score = score * valid
        score = self._smooth_time_score(score, self.stage_event_window)
        score = score * valid
        return score.detach()

    def _build_auto_stage_weights(self, actions, action_is_pad):
        B, T, D = actions.shape
        device, dtype = actions.device, actions.dtype
        weights = torch.ones((B, T, D), device=device, dtype=dtype)
        if (not self.use_auto_stage_weight) or T <= 1:
            return weights

        score = self._event_score(actions, action_is_pad)
        valid = (~action_is_pad).float()
        # Per-sample robust normalization. If the trajectory is almost static,
        # weights remain near 1.
        denom = score.amax(dim=1, keepdim=True).clamp_min(self.stage_min_score_eps)
        norm = (score / denom).clamp(0.0, 1.0) * valid
        time_w = 1.0 + (self.stage_weight_max - 1.0) * norm
        return time_w.unsqueeze(-1).expand(B, T, D).to(dtype)

    def _build_grasp_weights(self, actions, action_is_pad):
        B, T, D = actions.shape
        device = actions.device
        weights = torch.ones((B, T, D), device=device, dtype=actions.dtype)
        if (not self.use_grasp_weight) or T <= 1:
            return weights

        gi = self.grasp_gripper_index if self.grasp_gripper_index >= 0 else D + self.grasp_gripper_index
        if gi < 0 or gi >= D:
            return weights

        delta = torch.zeros((B, T), device=device, dtype=actions.dtype)
        delta[:, 1:] = (actions[:, 1:, gi] - actions[:, :-1, gi]).abs()
        delta = delta.masked_fill(action_is_pad, 0.0)
        center_idx = delta.argmax(dim=1)

        ds = max(0, min(D, self.grasp_dim_start))
        de = max(ds, min(D, self.grasp_dim_end))
        dim_w = torch.ones((D,), device=device, dtype=actions.dtype)
        if de > ds:
            dim_w[ds:de] = self.grasp_arm_dim_weight

        for b in range(B):
            c = int(center_idx[b].item())
            lo = max(0, c - self.grasp_window)
            hi = min(T, c + self.grasp_window + 1)
            weights[b, lo:hi, :] = self.grasp_stage_weight
        weights = weights * dim_w.view(1, 1, D)
        return weights

    def _combined_stage_weights(self, actions, action_is_pad):
        B, T, D = actions.shape
        weights = torch.ones((B, T, D), device=actions.device, dtype=actions.dtype)
        # These training weights are intentionally disabled in validation, so
        # policy_best.ckpt is selected by the real imitation loss rather than by
        # auxiliary-stage objectives.
        if not self.training:
            return weights
        if self.use_auto_stage_weight:
            weights = torch.maximum(weights, self._build_auto_stage_weights(actions, action_is_pad))
        if self.use_grasp_weight:
            weights = torch.maximum(weights, self._build_grasp_weights(actions, action_is_pad))
        return weights

    def _compute_action_losses(self, actions, a_hat, action_is_pad):
        all_loss = self._element_loss(actions, a_hat)
        raw_l = self._masked_mean(all_loss, action_is_pad)
        if (not self.training) or ((not self.use_auto_stage_weight) and (not self.use_grasp_weight)):
            return raw_l, raw_l
        stage_weights = self._combined_stage_weights(actions, action_is_pad)
        weighted_l = self._masked_mean(all_loss, action_is_pad, stage_weights)
        return weighted_l, raw_l

    def _derive_stage_labels(self, actions, action_is_pad):
        """Pseudo stage labels from cumulative event progress.
        It converts continuous event scores into ordered stages without manual
        annotations. Fallback is linear progress when no clear transition exists.
        """
        B, T, D = actions.shape
        device = actions.device
        labels = torch.zeros((B, T), dtype=torch.long, device=device)
        valid = ~action_is_pad
        if self.stage_num <= 1:
            return labels

        score = self._event_score(actions, action_is_pad)
        for b in range(B):
            n = int(valid[b].sum().item())
            if n <= 1:
                continue
            sb = score[b, :n]
            if float(sb.sum().item()) <= self.stage_min_score_eps:
                prog = torch.arange(n, device=device).float() / max(1, n - 1)
            else:
                # Add a small constant so long smooth phases still get ordered
                # labels, while high-event zones dominate the boundaries.
                sb = sb + 0.05 * (sb.mean() + self.stage_min_score_eps)
                prog = torch.cumsum(sb, dim=0) / (sb.sum() + self.stage_min_score_eps)
            lb = torch.clamp((prog * self.stage_num).long(), 0, self.stage_num - 1)
            labels[b, :n] = lb
        return labels

    def _compute_stage_prediction_loss(self, a_hat, actions, action_is_pad):
        if (not self.training) or (not self.use_stage_pred) or (not hasattr(self, 'stage_head')):
            z = torch.zeros((), device=a_hat.device)
            return z, z
        B, T, D = a_hat.shape
        if D != self.action_dim:
            # Keep the main training path safe if action_dim differs from the
            # expected ALOHA dimension.
            z = torch.zeros((), device=a_hat.device)
            return z, z
        labels = self._derive_stage_labels(actions.detach(), action_is_pad)
        valid = ~action_is_pad
        if int(valid.sum().item()) == 0:
            z = torch.zeros((), device=a_hat.device)
            return z, z
        logits = self.stage_head(a_hat)  # [B,T,stage_num]
        loss = F.cross_entropy(logits[valid], labels[valid])
        with torch.no_grad():
            acc = (logits.argmax(dim=-1)[valid] == labels[valid]).float().mean()
        return loss, acc

    def __call__(self, image, depth_image, robot_state, actions=None, action_is_pad=None):

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        depth_normalize = transforms.Normalize(mean=[0.5], std=[0.5])

        image = normalize(image)  # 图像归一化
        if depth_image is not None:
            depth_image = depth_normalize(depth_image)

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            action_is_pad = action_is_pad[:, :self.model.num_queries]

            a_hat, (mu, logvar) = self.model(image, depth_image, robot_state, actions, action_is_pad)

            loss_dict = dict()
            l1, raw_l1 = self._compute_action_losses(actions, a_hat, action_is_pad)

            loss_dict['l1'] = l1
            loss_dict['raw_l1'] = raw_l1.detach()
            if self.kl_weight != 0:
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                loss_dict['kl'] = total_kld[0]
                loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            else:
                loss_dict['loss'] = loss_dict['l1']

            # V7 stage-prediction auxiliary task is training-only and small.
            # Validation/inference stay on the same action path.
            if self.training and self.use_stage_pred:
                stage_loss, stage_acc = self._compute_stage_prediction_loss(a_hat, actions, action_is_pad)
                loss_dict['stage_loss'] = stage_loss
                loss_dict['stage_acc'] = stage_acc.detach()
                loss_dict['loss'] = loss_dict['loss'] + self.stage_loss_weight * stage_loss

            return loss_dict, a_hat
        else:  # inference time
            a_hat, (_, _) = self.model(image, depth_image, robot_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        # strict=False allows inference with the same v7 policy even when a
        # checkpoint contains train-only stage_head parameters, or when old v5/v6
        # checkpoints do not contain them.
        result = self.load_state_dict(model_dict, strict=False)
        missing = getattr(result, 'missing_keys', [])
        unexpected = getattr(result, 'unexpected_keys', [])
        if missing:
            print('[deserialize][WARN] missing keys:', missing[:8], '...' if len(missing) > 8 else '')
        if unexpected:
            print('[deserialize][WARN] unexpected keys:', unexpected[:8], '...' if len(unexpected) > 8 else '')
        return result


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer
        self.loss_function = args_override['loss_function']

    # 而 __call__ 在对象被调用时执行
    def __call__(self, image, depth_image, robot_state, actions=None,
                 action_is_pad=None):
        env_state = None  # TODO

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        depth_normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        image = normalize(image)  # 图像归一化
        if depth_image is not None:
            depth_image = depth_normalize(depth_image)
        if actions is not None:  # training time
            actions = actions[:, 0]  # 动作
            a_hat = self.model(image, depth_image, robot_state, actions, action_is_pad)
            # 均方误差
            if self.loss_function == 'l1':
                mse = F.l1_loss(actions, a_hat)
            elif self.loss_function == 'l2':
                mse = F.mse_loss(actions, a_hat)
            else:
                mse = F.smooth_l1_loss(actions, a_hat)

            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict, a_hat

        else:  # inference time
            a_hat = self.model(image, depth_image, robot_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

    def serialize(self):
        return self.state_dict()

    def deserialize(self, model_dict):
        return self.load_state_dict(model_dict)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
