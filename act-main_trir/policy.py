import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ['1', 'true', 'yes', 'y', 'on']
    return bool(v)


def _parse_indices(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    v = str(v).strip().lower()
    if v in ['', 'none', 'null', '-1']:
        return []
    return [int(x.strip()) for x in v.split(',') if x.strip() != '']


class ACTPolicy(nn.Module):
    """ACT policy with optional v7 Stage-aware TRIR++ training losses.

    Inference remains the same ACT forward pass. All TRIR++ augmentation,
    auto-stage loss weighting, and stage prediction losses are training-only.
    """
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']

        # ---- TRIR++ visual consistency options ----
        self.use_trir = _as_bool(args_override.get('use_trir', False))
        self.trir_aug_prob = float(args_override.get('trir_aug_prob', 0.0))
        self.trir_view_prob = float(args_override.get('trir_view_prob', 1.0))
        self.trir_aug_weight = float(args_override.get('trir_aug_weight', 0.0))
        self.trir_cons_weight = float(args_override.get('trir_cons_weight', 0.0))
        self.trir_feat_cons_weight = float(args_override.get('trir_feat_cons_weight', 0.0))
        self.trir_brightness = float(args_override.get('trir_brightness', 0.0))
        self.trir_contrast = float(args_override.get('trir_contrast', 0.0))
        self.trir_gamma = float(args_override.get('trir_gamma', 0.0))
        self.trir_saturation = float(args_override.get('trir_saturation', 0.0))
        self.trir_blur_prob = float(args_override.get('trir_blur_prob', 0.0))
        self.trir_shadow_prob = float(args_override.get('trir_shadow_prob', 0.0))
        self.trir_shadow_strength = float(args_override.get('trir_shadow_strength', 0.0))
        self.trir_erasing_prob = float(args_override.get('trir_erasing_prob', 0.0))
        self.trir_noise_std = float(args_override.get('trir_noise_std', 0.0))

        # ---- Auto Stage-aware options ----
        self.use_auto_stage_weight = _as_bool(args_override.get('use_auto_stage_weight', False))
        self.stage_weight_max = float(args_override.get('stage_weight_max', 1.0))
        self.stage_event_window = int(args_override.get('stage_event_window', 1))
        self.stage_speed_power = float(args_override.get('stage_speed_power', 1.0))
        self.stage_acc_power = float(args_override.get('stage_acc_power', 0.7))
        self.stage_gripper_power = float(args_override.get('stage_gripper_power', 1.0))
        self.stage_gripper_indices = _parse_indices(args_override.get('stage_gripper_indices', '6,13'))

        self.use_stage_pred = _as_bool(args_override.get('use_stage_pred', False))
        self.stage_num = int(args_override.get('stage_num', 5))
        self.stage_loss_weight = float(args_override.get('stage_loss_weight', 0.0))

        print(f'KL Weight {self.kl_weight}')
        if self.use_trir:
            print(
                '🌈 [Stage-aware TRIR++ v7] enabled: '
                f'aug_prob={self.trir_aug_prob}, cons={self.trir_cons_weight}, '
                f'feat_cons={self.trir_feat_cons_weight}'
            )
        if self.use_auto_stage_weight:
            print(
                '🧭 [Auto Stage-aware Loss v7] enabled: '
                f'weight_max={self.stage_weight_max}, window={self.stage_event_window}, '
                f'gripper_indices={self.stage_gripper_indices}'
            )
        if self.use_stage_pred:
            print(
                '🧠 [Stage Prediction Head v7] enabled: '
                f'stage_num={self.stage_num}, loss_weight={self.stage_loss_weight}'
            )

    def _normalize_image(self, image):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        return normalize(image)

    def _brightness_contrast_gamma_saturation(self, x, view_mask):
        # x: B,K,C,H,W in [0,1], view_mask: B,K,1,1,1
        B, K = x.shape[:2]
        device = x.device
        dtype = x.dtype
        y = x

        if self.trir_brightness > 0:
            factor = 1.0 + (torch.rand(B, K, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * self.trir_brightness
            y = torch.where(view_mask, y * factor, y)

        if self.trir_contrast > 0:
            mean = y.mean(dim=(-2, -1), keepdim=True)
            factor = 1.0 + (torch.rand(B, K, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * self.trir_contrast
            yc = (y - mean) * factor + mean
            y = torch.where(view_mask, yc, y)

        if self.trir_gamma > 0:
            gamma = 1.0 + (torch.rand(B, K, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * self.trir_gamma
            yg = torch.clamp(y, 1e-5, 1.0).pow(gamma)
            y = torch.where(view_mask, yg, y)

        if self.trir_saturation > 0 and y.shape[2] == 3:
            gray = (0.299 * y[:, :, 0:1] + 0.587 * y[:, :, 1:2] + 0.114 * y[:, :, 2:3])
            factor = 1.0 + (torch.rand(B, K, 1, 1, 1, device=device, dtype=dtype) * 2 - 1) * self.trir_saturation
            ys = gray + (y - gray) * factor
            y = torch.where(view_mask, ys, y)

        return torch.clamp(y, 0.0, 1.0)

    def _apply_shadow(self, x, view_mask):
        if self.trir_shadow_prob <= 0 or self.trir_shadow_strength <= 0:
            return x
        B, K, C, H, W = x.shape
        y = x.clone()
        device = x.device
        for b in range(B):
            for k in range(K):
                if not bool(view_mask[b, k, 0, 0, 0]):
                    continue
                if torch.rand((), device=device).item() > self.trir_shadow_prob:
                    continue
                h0 = int(torch.randint(0, max(1, H - H // 4), (1,), device=device).item())
                w0 = int(torch.randint(0, max(1, W - W // 4), (1,), device=device).item())
                hh = int(torch.randint(max(1, H // 5), max(2, H // 2), (1,), device=device).item())
                ww = int(torch.randint(max(1, W // 5), max(2, W // 2), (1,), device=device).item())
                h1 = min(H, h0 + hh)
                w1 = min(W, w0 + ww)
                strength = float(torch.rand((), device=device).item()) * self.trir_shadow_strength
                y[b, k, :, h0:h1, w0:w1] = y[b, k, :, h0:h1, w0:w1] * (1.0 - strength)
        return y

    def _apply_blur(self, x, view_mask):
        if self.trir_blur_prob <= 0:
            return x
        if torch.rand((), device=x.device).item() > self.trir_blur_prob:
            return x
        B, K, C, H, W = x.shape
        flat = x.reshape(B * K, C, H, W)
        blurred = F.avg_pool2d(flat, kernel_size=3, stride=1, padding=1).reshape(B, K, C, H, W)
        return torch.where(view_mask, blurred, x)

    def _apply_erasing(self, x, view_mask):
        # Default is 0.0. For small manipulation targets, keep this disabled unless doing ablations.
        if self.trir_erasing_prob <= 0:
            return x
        B, K, C, H, W = x.shape
        y = x.clone()
        device = x.device
        for b in range(B):
            for k in range(K):
                if not bool(view_mask[b, k, 0, 0, 0]):
                    continue
                if torch.rand((), device=device).item() > self.trir_erasing_prob:
                    continue
                eh = max(1, H // 12)
                ew = max(1, W // 12)
                h0 = int(torch.randint(0, max(1, H - eh), (1,), device=device).item())
                w0 = int(torch.randint(0, max(1, W - ew), (1,), device=device).item())
                y[b, k, :, h0:h0+eh, w0:w0+ew] = y[b, k].mean()
        return y

    def _augment_image_trirpp(self, image):
        # image is expected to be 0~1 before ImageNet normalization.
        x = torch.clamp(image, 0.0, 1.0)
        B, K = x.shape[:2]
        device = x.device
        view_mask = (torch.rand(B, K, 1, 1, 1, device=device) < self.trir_view_prob)
        x = self._brightness_contrast_gamma_saturation(x, view_mask)
        x = self._apply_shadow(x, view_mask)
        x = self._apply_blur(x, view_mask)
        x = self._apply_erasing(x, view_mask)
        if self.trir_noise_std > 0:
            noise = torch.randn_like(x) * self.trir_noise_std
            x = torch.where(view_mask, x + noise, x)
        return torch.clamp(x, 0.0, 1.0)

    def _compute_stage_score_weight_label(self, actions, is_pad):
        # actions: B,T,D already sliced to num_queries. is_pad: B,T bool.
        B, T, D = actions.shape
        device = actions.device
        dtype = actions.dtype
        valid = (~is_pad).float()

        delta = torch.zeros(B, T, D, device=device, dtype=dtype)
        if T > 1:
            delta[:, 1:] = actions[:, 1:] - actions[:, :-1]
        speed = delta.abs().mean(dim=-1)

        acc_vec = torch.zeros_like(delta)
        if T > 2:
            acc_vec[:, 2:] = delta[:, 2:] - delta[:, 1:-1]
        acc = acc_vec.abs().mean(dim=-1)

        grip_score = torch.zeros(B, T, device=device, dtype=dtype)
        gripper_indices = [i for i in self.stage_gripper_indices if 0 <= i < D]
        if len(gripper_indices) > 0:
            gd = torch.zeros(B, T, len(gripper_indices), device=device, dtype=dtype)
            if T > 1:
                gd[:, 1:] = actions[:, 1:, gripper_indices] - actions[:, :-1, gripper_indices]
            grip_score = gd.abs().mean(dim=-1)

        eps = 1e-6
        def norm01(v):
            v = v * valid
            vmax = v.amax(dim=1, keepdim=True).clamp_min(eps)
            return v / vmax

        speed_n = norm01(speed).clamp_min(0.0).pow(max(self.stage_speed_power, 0.0))
        acc_n = norm01(acc).clamp_min(0.0).pow(max(self.stage_acc_power, 0.0))
        grip_n = norm01(grip_score).clamp_min(0.0).pow(max(self.stage_gripper_power, 0.0))

        score = speed_n + acc_n + grip_n
        score = score * valid

        # Expand a high-event instant to a local window, because contact/switch phases
        # usually need several neighboring steps to be supervised more strongly.
        if self.stage_event_window > 1 and T > 1:
            kernel = int(self.stage_event_window)
            if kernel % 2 == 0:
                kernel += 1
            pad = kernel // 2
            score = F.max_pool1d(score.unsqueeze(1), kernel_size=kernel, stride=1, padding=pad).squeeze(1)
            score = score[:, :T]

        score = norm01(score)
        if self.use_auto_stage_weight and self.stage_weight_max > 1.0:
            weights = 1.0 + (self.stage_weight_max - 1.0) * score
        else:
            weights = torch.ones_like(score)
        weights = torch.where(valid > 0, weights, torch.zeros_like(weights))

        # Pseudo-stage labels: monotonically increasing phase id driven by accumulated
        # event evidence plus weak time progress. No task-specific arm or object name is encoded.
        weak_progress = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype).unsqueeze(0).expand(B, T)
        event_mass = score + 0.05 * weak_progress + 1e-4
        event_mass = event_mass * valid + (1.0 - valid) * 1e-4
        cum = event_mass.cumsum(dim=1)
        denom = event_mass.sum(dim=1, keepdim=True).clamp_min(eps)
        phase = (cum / denom).clamp(0.0, 0.9999)
        labels = torch.clamp((phase * self.stage_num).long(), 0, self.stage_num - 1)
        return weights.unsqueeze(-1), labels

    def _supervised_loss(self, actions, a_hat, is_pad, stage_weights=None):
        all_l1 = F.l1_loss(actions, a_hat, reduction='none')
        valid_mask = ~is_pad.unsqueeze(-1)
        if stage_weights is None:
            weight = valid_mask.float()
        else:
            weight = valid_mask.float() * stage_weights.float()
        denom = (weight.sum() * actions.shape[-1]).clamp_min(1.0)
        l1 = (all_l1 * weight).sum() / denom
        return l1, all_l1, valid_mask, weight

    def _unpack_model_output(self, out):
        if len(out) == 4:
            a_hat, is_pad_hat, latent, stage_logits = out
        else:
            a_hat, is_pad_hat, latent = out
            stage_logits = None
        mu, logvar = latent
        return a_hat, is_pad_hat, mu, logvar, stage_logits

    def __call__(self, qpos, image, actions=None, is_pad=None, return_outputs=False):
        env_state = None
        raw_image = image
        image = self._normalize_image(raw_image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            stage_weights, stage_labels = self._compute_stage_score_weight_label(actions, is_pad)

            out = self.model(qpos, image, env_state, actions, is_pad)
            a_hat, is_pad_hat, mu, logvar, stage_logits = self._unpack_model_output(out)
            clean_feat = getattr(self.model, 'last_visual_summary', None)
            clean_feat_detached = clean_feat.detach() if clean_feat is not None else None

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            l1, all_l1, valid_mask, used_weight = self._supervised_loss(actions, a_hat, is_pad, stage_weights)

            loss_dict = dict()
            loss_dict['l1'] = l1
            # Helpful scalar for checking whether stage weighting is active.
            loss_dict['stage_weight_mean'] = used_weight[valid_mask].mean() if valid_mask.any() else torch.tensor(1.0, device=actions.device)

            if self.use_auto_stage_weight:
                # Also log the clean unweighted L1 so we can compare curves with baselines.
                unweighted_l1, _, _, _ = self._supervised_loss(actions, a_hat, is_pad, None)
                loss_dict['l1_unweighted'] = unweighted_l1.detach()

            if self.use_stage_pred and stage_logits is not None and self.stage_loss_weight > 0:
                ce = F.cross_entropy(
                    stage_logits.reshape(-1, self.stage_num),
                    stage_labels.reshape(-1),
                    reduction='none',
                )
                valid_flat = (~is_pad).reshape(-1).float()
                stage_loss = (ce * valid_flat).sum() / valid_flat.sum().clamp_min(1.0)
                loss_dict['stage_loss'] = stage_loss
            else:
                stage_loss = torch.tensor(0.0, device=actions.device)

            loss = l1 + total_kld[0] * self.kl_weight + stage_loss * self.stage_loss_weight
            loss_dict['kl'] = total_kld[0]

            # TRIR++ is training-only. Validation remains clean for a stable best-ckpt signal.
            if self.training and self.use_trir and self.trir_aug_prob > 0:
                if torch.rand((), device=raw_image.device).item() < self.trir_aug_prob:
                    aug_raw = self._augment_image_trirpp(raw_image)
                    aug_image = self._normalize_image(aug_raw)
                    aug_out = self.model(qpos, aug_image, env_state, actions, is_pad)
                    aug_a_hat, _, aug_mu, aug_logvar, aug_stage_logits = self._unpack_model_output(aug_out)
                    aug_feat = getattr(self.model, 'last_visual_summary', None)

                    aug_l1, _, _, _ = self._supervised_loss(actions, aug_a_hat, is_pad, stage_weights)
                    loss_dict['trir_aug_l1'] = aug_l1
                    loss = loss + self.trir_aug_weight * aug_l1

                    if self.trir_cons_weight > 0:
                        cons_raw = F.smooth_l1_loss(aug_a_hat, a_hat.detach(), reduction='none')
                        cons = (cons_raw * valid_mask.float()).sum() / (valid_mask.float().sum() * actions.shape[-1]).clamp_min(1.0)
                        loss_dict['trir_cons'] = cons
                        loss = loss + self.trir_cons_weight * cons

                    if self.trir_feat_cons_weight > 0 and clean_feat_detached is not None and aug_feat is not None:
                        feat_cons = F.smooth_l1_loss(aug_feat, clean_feat_detached, reduction='mean')
                        loss_dict['trir_feat_cons'] = feat_cons
                        loss = loss + self.trir_feat_cons_weight * feat_cons

                    if self.use_stage_pred and aug_stage_logits is not None and self.stage_loss_weight > 0:
                        ce_aug = F.cross_entropy(
                            aug_stage_logits.reshape(-1, self.stage_num),
                            stage_labels.reshape(-1),
                            reduction='none',
                        )
                        valid_flat = (~is_pad).reshape(-1).float()
                        aug_stage_loss = (ce_aug * valid_flat).sum() / valid_flat.sum().clamp_min(1.0)
                        loss_dict['stage_loss_aug'] = aug_stage_loss
                        loss = loss + 0.5 * self.stage_loss_weight * aug_stage_loss

            loss_dict['loss'] = loss
            if return_outputs:
                loss_dict['a_hat'] = a_hat
                loss_dict['target_actions'] = actions
                loss_dict['valid_mask'] = valid_mask
                loss_dict['stage_labels'] = stage_labels
                loss_dict['stage_weights'] = stage_weights
            return loss_dict
        else: # inference time
            out = self.model(qpos, image, env_state) # no action, sample from prior
            a_hat = out[0]
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


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
