# Data and Model Availability

## Recommended paper statement

> The source code is publicly available in the project repository. The demonstration datasets and trained checkpoints used in this study are available from the corresponding author upon reasonable request, subject to institutional and hardware-related constraints.

中文表述：

> 本研究源代码在项目仓库中公开。受数据体量、机器人平台及机构管理要求限制，研究所用示教数据与训练模型可向通讯作者合理申请获取。

## Large files

Do not commit these files to normal Git history:

```text
episode_*.hdf5
*.ckpt
*.pth
*.pt
*.bag
large videos
```

Distribute checkpoints through GitHub Releases, Zenodo, an institutional repository, or a request procedure.

## Expected local assets

```text
act-main_trir/ckpts/dinov2_randpose_last8_pool2_3cam/
act-main_trir/ckpts/dinov2_last8_trir_weak_3cam/
act-main_trir/ckpts/dinov2_trirpp_nostage_sim_insertion_5000/
act-main_trir/eval_poses/sim_insertion_eval_seed1000_50.pkl
cobot_magic/data/
cobot_magic/train/dinov2_trirpp_stageaware_battery_6000/
cobot_magic/dinov2_local/dinov2_vits14_pretrain.pth
```

Requests for the demonstration data or trained checkpoints may be sent to:

```text
Corresponding author: Zhongqi Zhao
Email: 1292197115@qq.com
Institution: Chongqing University of Technology
```
