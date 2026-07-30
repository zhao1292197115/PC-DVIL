# Baseline step logging pack

把这三个文件复制到 `act-baseline` 根目录：

1. `imitate_episodes_metrics_fixedposes_step_log.py`
2. `run_baseline_step_eval_seed1000.sh`
3. `plot_step_curves_from_log.py`

运行评估并保存逐步日志：

```bash
cd ~/act_sim_workspace/act-baseline
bash run_baseline_step_eval_seed1000.sh
```

生成曲线图：

```bash
python3 plot_step_curves_from_log.py \
  --csv eval_logs/resnet_clean_seed1000/step_log_policy_best.csv \
  --out_dir paper_figures/resnet_clean_seed1000_step_curves
```

输出的 step_log CSV 字段包括：

- method, env, seed, rollout_id, eval_pose_id, step
- reward, highest_reward, latency_ms, success_so_far
- action_0...action_N
- qpos_0...qpos_M
- success_final, episode_return_final, episode_highest_reward_final, completion_step_final

这些数据可以画：reward per step 曲线、highest reward / stage progress 曲线、action jerk / 抖动曲线、latency 曲线。
