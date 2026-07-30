"""Collect paired Clean-vs-0.60 feature/action drift for Dense-ACT and PC-DVIL.

This diagnostic collector loads both checkpoints at the same time and evaluates
both methods on exactly the same simulation state at every timestep.  The Clean
and Perturbed inputs are paired copies of the same three-camera observation.
TFC/feature caching is forcibly disabled for both policies.

CSV columns (kept intentionally minimal):
    method, intensity, rollout_id, timestep, valid_mask,
    feature_drift_cam1, feature_drift_cam2, feature_drift_cam3,
    feature_drift_mean, action_chunk_drift

Metric definition:
    feature drift: mean absolute difference between spatially pooled, projected
                   per-camera ACT visual features (one vector per camera)
    action drift:  mean absolute difference over the full normalized ACT action
                   chunk [num_queries, action_dim]

The environment is advanced with a shared Clean reference action.  By default,
reference control alternates by rollout (Dense-ACT for even rollout IDs and
PC-DVIL for odd rollout IDs), preventing the state distribution from being
owned by only one method while preserving exact state pairing within a rollout.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np
import torch
from einops import rearrange

from constants import SIM_TASK_CONFIGS
from policy import ACTPolicy
from sim_env import BOX_POSE, _apply_factory_visuals, make_sim_env
from utils import sample_box_pose, sample_insertion_pose, set_seed


METHOD_DENSE = "Dense-ACT"
METHOD_PCDVIL = "PC-DVIL"
CSV_FIELDS = [
    "method",
    "intensity",
    "rollout_id",
    "timestep",
    "valid_mask",
    "feature_drift_cam1",
    "feature_drift_cam2",
    "feature_drift_cam3",
    "feature_drift_mean",
    "action_chunk_drift",
]


@contextmanager
def nested_argparse_defaults():
    """Prevent DETR's internal parser from consuming this collector's CLI.

    The local ACT fork constructs the model by creating a second argparse parser
    inside build_ACT_model_and_optimizer().  That parser normally re-reads
    sys.argv and therefore mistakes collector-only flags such as
    --num_rollouts/--intensity for DETR training arguments.  During model
    construction we bypass only that nested parsing step, retain every parser
    default, and then let args_override provide the actual architecture values.
    """
    original_parse_known_args = argparse.ArgumentParser.parse_known_args

    def _defaults_only(parser, args=None, namespace=None):
        ns = argparse.Namespace() if namespace is None else namespace
        for action in parser._actions:
            if action.dest == "help":
                continue
            if not hasattr(ns, action.dest) and action.default is not argparse.SUPPRESS:
                setattr(ns, action.dest, action.default)
        return ns, []

    argparse.ArgumentParser.parse_known_args = _defaults_only
    try:
        yield
    finally:
        argparse.ArgumentParser.parse_known_args = original_parse_known_args


def configure_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this collector, but torch.cuda.is_available() is False.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def get_amp_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported amp dtype: {name}")


def build_policy_config(args: argparse.Namespace, camera_names: List[str]) -> Dict[str, object]:
    # feature_cache=False is deliberate and mandatory for the drift diagnostic.
    return {
        "lr": 1e-5,
        "lr_backbone": 1e-5,
        "backbone": "dinov2_vits14",
        "camera_names": camera_names,
        "dinov2_train_layers": int(args.dinov2_train_layers),
        "dinov2_pool": int(args.dinov2_pool),
        "feature_cache": False,
        "cache_interval": 1,
        "use_trir": False,
        "trir_aug_prob": 0.0,
        "trir_view_prob": 0.0,
        "trir_aug_weight": 0.0,
        "trir_cons_weight": 0.0,
        "trir_feat_cons_weight": 0.0,
        "trir_brightness": 0.0,
        "trir_contrast": 0.0,
        "trir_gamma": 0.0,
        "trir_saturation": 0.0,
        "trir_blur_prob": 0.0,
        "trir_shadow_prob": 0.0,
        "trir_shadow_strength": 0.0,
        "trir_erasing_prob": 0.0,
        "trir_noise_std": 0.0,
        "use_auto_stage_weight": False,
        "use_stage_pred": False,
        "stage_num": 5,
        "stage_hidden_dim": 128,
        "stage_loss_weight": 0.0,
        "num_queries": int(args.chunk_size),
        "kl_weight": int(args.kl_weight),
        "hidden_dim": int(args.hidden_dim),
        "dim_feedforward": int(args.dim_feedforward),
        "enc_layers": int(args.enc_layers),
        "dec_layers": int(args.dec_layers),
        "nheads": int(args.nheads),
    }


def load_policy_and_stats(
    method: str,
    ckpt_dir: Path,
    policy_config: Mapping[str, object],
) -> Tuple[ACTPolicy, Mapping[str, np.ndarray]]:
    ckpt_path = ckpt_dir / "policy_best.ckpt"
    stats_path = ckpt_dir / "dataset_stats.pkl"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"{method}: missing checkpoint: {ckpt_path}")
    if not stats_path.is_file():
        raise FileNotFoundError(f"{method}: missing dataset stats: {stats_path}")

    # Shield ACT/DETR's nested parser from this collector's command-line flags.
    with nested_argparse_defaults():
        policy = ACTPolicy(dict(policy_config))
    state_dict = torch.load(str(ckpt_path), map_location="cpu")
    status = policy.load_state_dict(state_dict, strict=True)
    policy.cuda().eval()

    # Hard assertion: this experiment must not use TFC.
    if bool(getattr(policy.model, "feature_cache", False)):
        raise RuntimeError(f"{method}: feature_cache unexpectedly enabled")
    if int(getattr(policy.model, "cache_interval", 1)) != 1:
        raise RuntimeError(f"{method}: cache_interval must be 1 when TFC is disabled")

    with stats_path.open("rb") as f:
        stats = pickle.load(f)

    print(f"[{method}] loaded: {ckpt_path}")
    print(f"[{method}] load status: {status}")
    print(f"[{method}] feature_cache=False, cache_interval=1")
    return policy, stats


def images_to_tensor(images: Mapping[str, np.ndarray], camera_names: Iterable[str]) -> torch.Tensor:
    ordered = []
    for cam_name in camera_names:
        if cam_name not in images:
            raise KeyError(f"Missing camera {cam_name!r}; available={list(images.keys())}")
        img = np.asarray(images[cam_name])
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Camera {cam_name} has invalid shape {img.shape}; expected HxWx3")
        ordered.append(rearrange(img, "h w c -> c h w"))
    array = np.stack(ordered, axis=0)
    return torch.from_numpy(array / 255.0).float().cuda(non_blocking=True).unsqueeze(0)


@contextmanager
def temporary_factory_settings(mode: str, strength: float, seed: int):
    keys = ("FACTORY_MODE", "FACTORY_STRENGTH", "FACTORY_SEED")
    old = {key: os.environ.get(key) for key in keys}
    os.environ["FACTORY_MODE"] = str(mode)
    os.environ["FACTORY_STRENGTH"] = str(float(strength))
    os.environ["FACTORY_SEED"] = str(int(seed))
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def make_paired_perturbed_images(
    clean_images: Mapping[str, np.ndarray],
    camera_names: Iterable[str],
    intensity: float,
    perturb_seed: int,
) -> Dict[str, np.ndarray]:
    # The existing 0.60 robustness setting is FACTORY_MODE=hard_noline with
    # FACTORY_STRENGTH=0.60.  Reusing sim_env._apply_factory_visuals prevents a
    # second, inconsistent perturbation implementation.
    perturbed: Dict[str, np.ndarray] = {}
    with temporary_factory_settings("hard_noline", intensity, perturb_seed):
        for cam_name in camera_names:
            perturbed[cam_name] = _apply_factory_visuals(
                np.asarray(clean_images[cam_name]).copy(),
                cam_name,
            )
    return perturbed


def preprocess_qpos(qpos_numpy: np.ndarray, stats: Mapping[str, np.ndarray]) -> torch.Tensor:
    qpos = (qpos_numpy - stats["qpos_mean"]) / stats["qpos_std"]
    return torch.from_numpy(qpos).float().cuda(non_blocking=True).unsqueeze(0)


def postprocess_action(raw_action: torch.Tensor, stats: Mapping[str, np.ndarray]) -> np.ndarray:
    action = raw_action.float().squeeze(0).detach().cpu().numpy()
    return action * stats["action_std"] + stats["action_mean"]


def capture_clean_and_perturbed(
    policy: ACTPolicy,
    qpos: torch.Tensor,
    clean_image: torch.Tensor,
    perturbed_image: torch.Tensor,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, float]:
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        clean_chunk = policy(qpos, clean_image)
    clean_cam_features = getattr(policy.model, "last_cam_visual_summaries", None)
    if clean_cam_features is None:
        raise RuntimeError(
            "Per-camera features were not exposed. Replace detr/models/detr_vae.py "
            "with the patched file supplied with this collector."
        )
    clean_cam_features = clean_cam_features.detach().float().clone()

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
        perturbed_chunk = policy(qpos, perturbed_image)
    perturbed_cam_features = getattr(policy.model, "last_cam_visual_summaries", None)
    if perturbed_cam_features is None:
        raise RuntimeError("Perturbed forward did not expose per-camera features")
    perturbed_cam_features = perturbed_cam_features.detach().float().clone()

    if clean_cam_features.shape != perturbed_cam_features.shape:
        raise RuntimeError(
            f"Feature shape mismatch: clean={tuple(clean_cam_features.shape)}, "
            f"perturbed={tuple(perturbed_cam_features.shape)}"
        )
    if clean_chunk.shape != perturbed_chunk.shape:
        raise RuntimeError(
            f"Action chunk shape mismatch: clean={tuple(clean_chunk.shape)}, "
            f"perturbed={tuple(perturbed_chunk.shape)}"
        )

    # One L1/MAE metric only, as requested.  Feature shape is [1, K, hidden].
    cam_drifts = torch.mean(
        torch.abs(clean_cam_features - perturbed_cam_features),
        dim=-1,
    ).squeeze(0)
    if cam_drifts.numel() != 3:
        raise RuntimeError(f"Expected exactly 3 cameras, got {cam_drifts.numel()}")

    action_chunk_drift = torch.mean(torch.abs(clean_chunk.float() - perturbed_chunk.float()))
    return (
        clean_chunk.detach().float(),
        cam_drifts.detach().cpu().numpy(),
        clean_cam_features.detach().cpu().numpy(),
        float(action_chunk_drift.item()),
    )


def add_chunk_to_temporal_buffer(
    buffer: torch.Tensor,
    filled: torch.Tensor,
    timestep: int,
    chunk: torch.Tensor,
) -> None:
    num_queries = chunk.shape[1]
    end = min(buffer.shape[1], timestep + num_queries)
    length = end - timestep
    buffer[timestep, timestep:end] = chunk[0, :length]
    filled[timestep, timestep:end] = True


def temporal_aggregate_action(buffer: torch.Tensor, filled: torch.Tensor, timestep: int) -> torch.Tensor:
    candidates = buffer[:, timestep][filled[:, timestep]]
    if candidates.shape[0] == 0:
        raise RuntimeError(f"No temporal actions populated at timestep {timestep}")
    # Match the current evaluation code's exponential weighting convention.
    k = 0.01
    weights_np = np.exp(-k * np.arange(candidates.shape[0], dtype=np.float64))
    weights_np /= weights_np.sum()
    weights = torch.from_numpy(weights_np).to(candidates.device, dtype=candidates.dtype).unsqueeze(1)
    return (candidates * weights).sum(dim=0, keepdim=True)


def choose_reference_method(mode: str, rollout_id: int) -> str:
    if mode == "dense":
        return METHOD_DENSE
    if mode == "pcdvil":
        return METHOD_PCDVIL
    if mode == "alternate":
        return METHOD_DENSE if rollout_id % 2 == 0 else METHOD_PCDVIL
    raise ValueError(f"Unknown reference controller: {mode}")


def invalid_row(method: str, intensity_label: str, rollout_id: int, timestep: int) -> Dict[str, object]:
    return {
        "method": method,
        "intensity": intensity_label,
        "rollout_id": rollout_id,
        "timestep": timestep,
        "valid_mask": 0,
        "feature_drift_cam1": np.nan,
        "feature_drift_cam2": np.nan,
        "feature_drift_cam3": np.nan,
        "feature_drift_mean": np.nan,
        "action_chunk_drift": np.nan,
    }


def collect(args: argparse.Namespace) -> None:
    configure_cuda_runtime()
    set_seed(int(args.seed))

    task_config = SIM_TASK_CONFIGS[args.task_name]
    camera_names = list(task_config["camera_names"])
    if len(camera_names) != 3:
        raise ValueError(f"This collector expects exactly three cameras, got {camera_names}")
    max_timesteps = int(task_config["episode_len"])
    state_dim = 14

    # Keep the actual environment Clean. Perturbations are generated from copies
    # of each Clean observation without changing physics or the state trajectory.
    os.environ["FACTORY_MODE"] = "clean"
    os.environ["FACTORY_STRENGTH"] = "1.0"
    os.environ["PEG_COLOR"] = "original"
    os.environ["PEG_SCALE"] = "1.0"
    os.environ["PEG_SHAPE"] = "original"

    policy_config = build_policy_config(args, camera_names)
    policies: Dict[str, ACTPolicy] = {}
    stats: Dict[str, Mapping[str, np.ndarray]] = {}
    policies[METHOD_DENSE], stats[METHOD_DENSE] = load_policy_and_stats(
        METHOD_DENSE, Path(args.dense_ckpt_dir), policy_config
    )
    policies[METHOD_PCDVIL], stats[METHOD_PCDVIL] = load_policy_and_stats(
        METHOD_PCDVIL, Path(args.pcdvil_ckpt_dir), policy_config
    )

    amp_dtype = get_amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.amp) and amp_dtype != torch.float32

    eval_pose_list = None
    if args.eval_pose_path:
        with open(args.eval_pose_path, "rb") as f:
            eval_pose_list = pickle.load(f)
        eval_pose_list = [np.asarray(p, dtype=np.float64).copy() for p in eval_pose_list]
        if not eval_pose_list:
            raise ValueError(f"Empty eval pose list: {args.eval_pose_path}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    intensity_label = f"{float(args.intensity):.2f}"

    print("\n========== Drift collector configuration ==========")
    print(f"task={args.task_name}")
    print(f"camera mapping: cam1={camera_names[0]}, cam2={camera_names[1]}, cam3={camera_names[2]}")
    print(f"intensity={intensity_label} (FACTORY_MODE=hard_noline)")
    print("metric=L1 mean absolute drift")
    print("TFC=OFF for both methods")
    print(f"reference_controller={args.reference_controller}")
    print(f"num_rollouts={args.num_rollouts}, max_timesteps={max_timesteps}")
    print(f"output={output_path}\n")

    env = make_sim_env(args.task_name)
    env_max_reward = env.task.max_reward

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        with torch.inference_mode():
            for rollout_id in range(int(args.num_rollouts)):
                if eval_pose_list is not None:
                    BOX_POSE[0] = eval_pose_list[rollout_id % len(eval_pose_list)].copy()
                elif "sim_transfer_cube" in args.task_name:
                    BOX_POSE[0] = sample_box_pose()
                elif "sim_insertion" in args.task_name:
                    BOX_POSE[0] = np.concatenate(sample_insertion_pose())

                ts = env.reset()
                reference_method = choose_reference_method(args.reference_controller, rollout_id)
                perturb_seed = int(args.perturb_seed) + rollout_id

                temporal_buffers = {
                    method: torch.zeros(
                        [max_timesteps, max_timesteps + int(args.chunk_size), state_dim],
                        device="cuda",
                        dtype=torch.float32,
                    )
                    for method in policies
                }
                temporal_filled = {
                    method: torch.zeros(
                        [max_timesteps, max_timesteps + int(args.chunk_size)],
                        device="cuda",
                        dtype=torch.bool,
                    )
                    for method in policies
                }

                completed = False
                last_valid_t = -1
                for timestep in range(max_timesteps):
                    obs = ts.observation
                    if "images" not in obs:
                        raise KeyError("Simulation observation does not contain a three-camera 'images' dictionary")

                    clean_images_np = obs["images"]
                    perturbed_images_np = make_paired_perturbed_images(
                        clean_images_np,
                        camera_names,
                        intensity=float(args.intensity),
                        perturb_seed=perturb_seed,
                    )
                    clean_image = images_to_tensor(clean_images_np, camera_names)
                    perturbed_image = images_to_tensor(perturbed_images_np, camera_names)
                    qpos_numpy = np.asarray(obs["qpos"])

                    clean_chunks: Dict[str, torch.Tensor] = {}
                    for method in (METHOD_DENSE, METHOD_PCDVIL):
                        qpos = preprocess_qpos(qpos_numpy, stats[method])
                        clean_chunk, cam_drifts, _, action_drift = capture_clean_and_perturbed(
                            policies[method],
                            qpos,
                            clean_image,
                            perturbed_image,
                            amp_enabled,
                            amp_dtype,
                        )
                        clean_chunks[method] = clean_chunk

                        row = {
                            "method": method,
                            "intensity": intensity_label,
                            "rollout_id": rollout_id,
                            "timestep": timestep,
                            "valid_mask": 1,
                            "feature_drift_cam1": float(cam_drifts[0]),
                            "feature_drift_cam2": float(cam_drifts[1]),
                            "feature_drift_cam3": float(cam_drifts[2]),
                            "feature_drift_mean": float(np.mean(cam_drifts)),
                            "action_chunk_drift": float(action_drift),
                        }
                        writer.writerow(row)

                        add_chunk_to_temporal_buffer(
                            temporal_buffers[method],
                            temporal_filled[method],
                            timestep,
                            clean_chunk,
                        )

                    csv_file.flush()
                    last_valid_t = timestep

                    reference_raw_action = temporal_aggregate_action(
                        temporal_buffers[reference_method],
                        temporal_filled[reference_method],
                        timestep,
                    )
                    reference_action = postprocess_action(reference_raw_action, stats[reference_method])
                    ts = env.step(reference_action)

                    if timestep % int(args.print_every) == 0:
                        print(
                            f"rollout={rollout_id:02d} t={timestep:03d} "
                            f"reference={reference_method} reward={ts.reward}"
                        )

                    if args.stop_on_success and ts.reward is not None and ts.reward >= env_max_reward:
                        completed = True
                        break
                    if hasattr(ts, "last") and ts.last():
                        break

                # Keep a rectangular timestep grid for downstream CI code.
                if last_valid_t + 1 < max_timesteps:
                    for timestep in range(last_valid_t + 1, max_timesteps):
                        writer.writerow(invalid_row(METHOD_DENSE, intensity_label, rollout_id, timestep))
                        writer.writerow(invalid_row(METHOD_PCDVIL, intensity_label, rollout_id, timestep))
                    csv_file.flush()

                print(
                    f"Finished rollout {rollout_id}: reference={reference_method}, "
                    f"valid_steps={last_valid_t + 1}, success_reached={completed}"
                )

    print(f"\nSaved paired drift CSV: {output_path}")
    print("Rows with valid_mask=0 are padding after an early terminal/success step.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="sim_insertion_scripted")
    parser.add_argument(
        "--dense_ckpt_dir",
        type=str,
        default="ckpts/dinov2_randpose_last8_pool2_3cam",
    )
    parser.add_argument(
        "--pcdvil_ckpt_dir",
        type=str,
        default="ckpts/dinov2_last8_trir_weak_3cam",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="eval_logs/paired_drift_s060.csv",
    )
    parser.add_argument("--eval_pose_path", type=str, default=None)
    parser.add_argument("--num_rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--perturb_seed", type=int, default=1000)
    parser.add_argument("--intensity", type=float, default=0.60)
    parser.add_argument(
        "--reference_controller",
        type=str,
        choices=["alternate", "dense", "pcdvil"],
        default="alternate",
    )
    parser.add_argument("--stop_on_success", action="store_true")
    parser.add_argument("--print_every", type=int, default=25)

    # Architecture must match both checkpoints.
    parser.add_argument("--dinov2_train_layers", type=int, default=8)
    parser.add_argument("--dinov2_pool", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--kl_weight", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=7)
    parser.add_argument("--nheads", type=int, default=8)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    return parser


if __name__ == "__main__":
    collect(build_parser().parse_args())
