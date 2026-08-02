# PC-DVIL

**Perturbation-Consistent Dense Visual Imitation Learning for Robust and Efficient Bimanual Precision Manipulation**

PC-DVIL is a three-view RGB imitation-learning project for ALOHA-style bimanual precision manipulation.

## Repository structure

```text
PC-DVIL/
├── README.md
├── REPRODUCIBILITY.md
├── DATA_AND_MODEL_AVAILABILITY.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── CITATION.cff
├── act-baseline/      # Original three-view ACT / ResNet baseline
├── act-main/          # Dense-ACT: DINOv2-last8-pool2
├── act-main_trir/     # PC-DVIL: dense visual encoder + perturbation consistency
└── cobot_magic/       # Real-robot battery-assembly code
```

## Method variants

| Folder | Method | Visual encoder | Perturbation consistency | Cache |
|---|---|---|---|---|
| `act-baseline` | ACT-3Cam | ResNet | No | No |
| `act-main` | Dense-ACT | DINOv2 ViT-S/14, last8, pool2 | No | Optional |
| `act-main_trir` | PC-DVIL | DINOv2 ViT-S/14, last8, pool2 | Yes | Cache2 at deployment |
| `cobot_magic` | Real-robot PC-DVIL | Three-view dense visual encoder | Yes | Cache2 |

## Tested environments

Simulation:

```text
Conda environment: act_sim
Python: 3.8
PyTorch: 2.4.1+cu121
GPU: RTX 4090 / RTX 4090D
Task: sim_insertion_scripted
```

Real robot:

```text
Conda environment: aloha
Ubuntu + ROS Noetic
ALOHA-style dual-arm platform
Three RGB cameras
```

Prefer the `conda_env.yaml` files retained inside the code folders. `requirements.txt` is only a compact dependency reference.

## DINOv2 source and weights

Place the local DINOv2 source and ViT-S/14 pretrained weight at:

```text
act-main/dinov2-main/
act-main/dinov2_vits14_pretrain.pth

act-main_trir/dinov2-main/
act-main_trir/dinov2_vits14_pretrain.pth

cobot_magic/dinov2_local/dinov2-main/
cobot_magic/dinov2_local/dinov2_vits14_pretrain.pth
```

Do not commit pretrained weights to normal Git history.

## Main configuration

```text
Task: sim_insertion_scripted
Cameras: left wrist, fixed/top, right wrist
Episode length: 400
Chunk size: 100
Hidden dimension: 512
Feed-forward dimension: 3200
Encoder/decoder layers: 4/7
Attention heads: 8
KL weight: 10
DINOv2 trainable blocks: last 8
Dense-token pooling: 2
Fixed evaluation poses: 50 poses, seed1000
Deployment cache interval: 2
Simulation training: 5,000 epochs, batch size 2
Real-robot training: 6,000 epochs, batch size 4
Optimizer: AdamW, weight decay 1e-4
Checkpoint: lowest validation loss (policy_best.ckpt)
```

Detailed commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Data and checkpoints

Do not upload the following files directly to GitHub:

```text
episode_*.hdf5
*.ckpt
*.pth
*.pt
large videos
ROS bags
training logs
```

Demonstration data and trained checkpoints may be provided through GitHub Releases, an institutional repository, or reasonable request to the corresponding author. See [DATA_AND_MODEL_AVAILABILITY.md](DATA_AND_MODEL_AVAILABILITY.md).

## Legacy stage-aware code

Some historical real-robot checkpoints may contain an auxiliary stage-prediction head for state-dict compatibility. This module is not treated as a core PC-DVIL contribution and is not required for the main drift or Cache2 analysis.

## Citation

Until an article DOI is assigned, cite the software release as follows. The preferred article citation in `CITATION.cff` can be updated after publication.

```bibtex
@software{pcdvil2026,
  title   = {Perturbation-Consistent Dense Visual Imitation Learning for Robust and Efficient Bimanual Precision Manipulation},
  author  = {Jiahui Dai and Zhongqi Zhao and Junhao Yu and Yuzun Cheng},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/zhao1292197115/PC-DVIL}
}
```

## Safety

This code can command physical robots. Verify robot topics, camera topics, joint limits, collision clearance, initial pose, and emergency-stop access before enabling motion.
