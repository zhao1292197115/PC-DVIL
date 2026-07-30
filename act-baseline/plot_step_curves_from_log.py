#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def finish(ax):
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def mean_std_by_step(df, value_col):
    g = df.groupby(["method", "step"])[value_col]
    out = g.agg(["mean", "std"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out

def plot_curve(df, value_col, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    stat = mean_std_by_step(df, value_col)
    for method, sub in stat.groupby("method"):
        sub = sub.sort_values("step")
        x = sub["step"].to_numpy()
        y = sub["mean"].to_numpy()
        s = sub["std"].to_numpy()
        ax.plot(x, y, linewidth=1.6, label=method)
        ax.fill_between(x, y - s, y + s, alpha=0.16, linewidth=0)
    ax.set_xlabel("Episode step")
    ax.set_ylabel(ylabel)
    finish(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def add_action_jerk(df):
    action_cols = sorted([c for c in df.columns if c.startswith("action_")], key=lambda x: int(x.split("_")[1]))
    if len(action_cols) == 0:
        raise ValueError("没有 action_0...action_N 列，无法计算 jerk。")
    rows = []
    for (method, env, rollout_id), sub in df.groupby(["method", "env", "rollout_id"]):
        sub = sub.sort_values("step").copy()
        actions = sub[action_cols].to_numpy(dtype=float)
        jerk = np.full(len(sub), np.nan)
        if len(sub) >= 4:
            j = actions[3:] - 3 * actions[2:-1] + 3 * actions[1:-2] - actions[:-3]
            jerk[3:] = np.linalg.norm(j, axis=1)
        sub["action_jerk_norm"] = jerk
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="step_log_policy_best.csv")
    ap.add_argument("--out_dir", default="paper_step_figures")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    plot_curve(df, "reward", "Step reward", out_dir / "baseline_step_reward_curve")
    plot_curve(df, "highest_reward", "Highest reward so far", out_dir / "baseline_stage_progress_curve")
    plot_curve(df, "latency_ms", "Policy latency (ms)", out_dir / "baseline_latency_curve")

    df = add_action_jerk(df)
    df.to_csv(out_dir / "step_log_with_action_jerk.csv", index=False)
    plot_curve(df.dropna(subset=["action_jerk_norm"]), "action_jerk_norm", "Action jerk norm", out_dir / "baseline_action_jerk_curve")

    print(f"Done. Figures saved to: {out_dir}")

if __name__ == "__main__":
    main()
