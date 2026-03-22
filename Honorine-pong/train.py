#!/usr/bin/env python3
"""
Train DQN on ALE/Pong-v5 with Stable-Baselines3.

Rubric — policy comparison (CNN vs MLP):
- `--policy cnn`: Convolutional features over screen input (standard for Atari / Pong).
- `--policy mlp`: Same DQN but `MlpPolicy` on flattened pixels — usually worse because spatial
  structure (ball/paddle location) is not exploited as effectively as with a CNN.
- `--policy both`: Train CNN and MLP in one run (two checkpoints under runs/<exp>/cnn|mlp).

Rubric — logging reward trends and episode length:
- During training, SB3 prints rollout `ep_rew_mean` and `ep_len_mean` in the console / TensorBoard.
- After each run, this script saves `runs/<exp>/<policy>/training_curves.png` from the Monitor CSV
  (per-episode reward + episode length, with a moving average).

Outputs:
- Save trained weights as `dqn_model.zip` (plus per-run paths in `hyperparameter_results.csv`).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecTransposeImage


def register_ale_envs():
    """Required on some Kaggle/Gymnasium setups so ALE/* ids are visible."""
    gym.register_envs(ale_py)


def build_env_fn(env_id: str, monitor_path: Path, flatten_obs: bool, render_mode: str | None = None):
    """Create one Atari env with consistent preprocessing."""

    def _make():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            clip_reward=True,
        )
        if flatten_obs:
            env = gym.wrappers.FlattenObservation(env)
        env = Monitor(env, filename=str(monitor_path))
        return env

    return _make


def make_vec_env(env_id: str, policy_kind: str, monitor_path: Path, render_mode: str | None = None):
    flatten_obs = policy_kind.lower() == "mlp"
    vec_env = DummyVecEnv([build_env_fn(env_id, monitor_path, flatten_obs, render_mode=render_mode)])
    vec_env = VecMonitor(vec_env)
    if policy_kind.lower() == "cnn":
        # Stack 4 frames to provide temporal context for the CNN.
        vec_env = VecFrameStack(vec_env, n_stack=4)
        # SB3 CNN expects channel-first (C, H, W).
        vec_env = VecTransposeImage(vec_env)
    return vec_env


def load_monitor_dataframe(monitor_csv: Path) -> pd.DataFrame:
    if not monitor_csv.exists():
        return pd.DataFrame(columns=["r", "l", "t"])
    # First line in monitor.csv is metadata, second line is header.
    return pd.read_csv(monitor_csv, skiprows=1)


def save_training_plots(monitor_csv: Path, output_png: Path, policy_name: str):
    df = load_monitor_dataframe(monitor_csv)
    if df.empty:
        return

    rewards = df["r"].to_numpy()
    ep_len = df["l"].to_numpy()
    x = np.arange(1, len(df) + 1)

    window = min(25, len(df))
    reward_ma = pd.Series(rewards).rolling(window=window).mean()
    len_ma = pd.Series(ep_len).rolling(window=window).mean()

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"{policy_name} Training Trends")

    axes[0].plot(x, rewards, alpha=0.35, label="episode reward")
    axes[0].plot(x, reward_ma, label=f"moving avg ({window})")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(x, ep_len, alpha=0.35, label="episode length")
    axes[1].plot(x, len_ma, label=f"moving avg ({window})")
    axes[1].set_ylabel("Episode Length")
    axes[1].set_xlabel("Episode")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_png, dpi=140)
    plt.close(fig)


def evaluate_model(
    model: DQN,
    env_id: str,
    policy_kind: str,
    n_episodes: int = 5,
    max_steps_per_episode: int = 20_000,
) -> Dict[str, float]:
    eval_monitor = Path("tmp_eval_monitor.csv")
    eval_env = make_vec_env(env_id, policy_kind=policy_kind, monitor_path=eval_monitor)
    episode_rewards: List[float] = []

    truncated_episodes = 0

    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        while not done and steps < max_steps_per_episode:
            action, _ = model.predict(obs, deterministic=True)  # Greedy (max-Q) evaluation
            obs, rewards, dones, infos = eval_env.step(action)
            total_reward += float(rewards[0])
            done = bool(dones[0])
            steps += 1
        if not done:
            truncated_episodes += 1
        episode_rewards.append(total_reward)

    eval_env.close()
    if eval_monitor.exists():
        eval_monitor.unlink()

    return {
        "eval_mean_reward": float(np.mean(episode_rewards)),
        "eval_std_reward": float(np.std(episode_rewards)),
        "eval_truncated_episodes": int(truncated_episodes),
    }


def append_experiment_row(results_csv: Path, row: Dict[str, str | int | float]):
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_one_policy(args: argparse.Namespace, policy_kind: str, run_dir: Path) -> Dict[str, float]:
    run_dir.mkdir(parents=True, exist_ok=True)
    monitor_csv = run_dir / "monitor.csv"
    eval_monitor_csv = run_dir / "eval_monitor.csv"

    env = make_vec_env(args.env_id, policy_kind=policy_kind, monitor_path=monitor_csv)
    eval_env = make_vec_env(args.env_id, policy_kind=policy_kind, monitor_path=eval_monitor_csv)

    policy_name = "CnnPolicy" if policy_kind.lower() == "cnn" else "MlpPolicy"

    model = DQN(
        policy_name,
        env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_initial_eps=args.epsilon_start,
        exploration_final_eps=args.epsilon_end,
        exploration_fraction=args.epsilon_decay_fraction,
        tensorboard_log=str(args.tensorboard_dir),
        verbose=1,
        seed=args.seed,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    print(f"[{policy_name}] Starting training for {args.total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=eval_callback,
        progress_bar=args.progress_bar,
    )
    print(f"[{policy_name}] Training finished. Saving model and running final evaluation...")

    final_model_path = run_dir / "dqn_model.zip"
    model.save(str(final_model_path))
    save_training_plots(monitor_csv, run_dir / "training_curves.png", policy_name=policy_name)

    metrics = evaluate_model(
        model,
        args.env_id,
        policy_kind=policy_kind,
        n_episodes=args.eval_episodes,
        max_steps_per_episode=args.eval_max_steps,
    )
    metrics["policy"] = policy_kind
    metrics["final_model_path"] = str(final_model_path)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    env.close()
    eval_env.close()
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN for Atari Pong (SB3 + Gymnasium).")
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--policy", type=str, default="both", choices=["cnn", "mlp", "both"])
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp-name", type=str, default="exp01")
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("tb_logs"))
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=20_000,
        help="Hard cap on steps per evaluation episode to prevent very long/stuck eval loops.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Enable tqdm progress bar (can appear stuck in some notebook environments).",
    )

    # Hyperparameters requested by assignment
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument(
        "--epsilon-decay-fraction",
        type=float,
        default=0.1,
        help="Fraction of total timesteps over which epsilon decays.",
    )

    # Useful defaults for Atari DQN
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=20_000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-interval", type=int, default=10_000)
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Short qualitative summary for hyperparameter_results.csv (impact on learning, exploration-exploitation, stability).",
    )

    return parser.parse_args()


def main():
    register_ale_envs()
    args = parse_args()
    base_dir = args.output_dir / args.exp_name
    base_dir.mkdir(parents=True, exist_ok=True)

    policies = ["cnn", "mlp"] if args.policy == "both" else [args.policy]
    summary_rows: List[Dict[str, str | int | float]] = []

    for policy_kind in policies:
        run_dir = base_dir / policy_kind
        metrics = train_one_policy(args, policy_kind=policy_kind, run_dir=run_dir)
        summary_rows.append(
            {
                "member_name": "Honorine",
                "experiment_name": args.exp_name,
                "policy": policy_kind,
                "env_id": args.env_id,
                "timesteps": args.total_timesteps,
                "lr": args.lr,
                "gamma": args.gamma,
                "batch_size": args.batch_size,
                "epsilon_start": args.epsilon_start,
                "epsilon_end": args.epsilon_end,
                "epsilon_decay_fraction": args.epsilon_decay_fraction,
                "eval_mean_reward": round(float(metrics["eval_mean_reward"]), 3),
                "eval_std_reward": round(float(metrics["eval_std_reward"]), 3),
                "noted_behavior": (
                    args.notes.strip()
                    or "Summarize after run: learning speed, stability, exploration vs exploitation, vs baseline (see README Honorine table)."
                ),
                "model_path": str(run_dir / "dqn_model.zip"),
            }
        )

    # Save one top-level model file for assignment convenience.
    # Prefer CNN model by default (usually better on visual Atari input).
    preferred_policy = "cnn" if "cnn" in policies else policies[0]
    preferred_model = base_dir / preferred_policy / "dqn_model.zip"
    assignment_model_path = Path("dqn_model.zip")
    if preferred_model.exists():
        assignment_model_path.write_bytes(preferred_model.read_bytes())

    results_csv = args.output_dir / "hyperparameter_results.csv"
    for row in summary_rows:
        append_experiment_row(results_csv, row)

    print("\nTraining complete.")
    print(f"Assignment model: {assignment_model_path.resolve()}")
    print(f"Results table updated: {results_csv.resolve()}")
    print("Per-policy outputs saved in:", base_dir.resolve())


if __name__ == "__main__":
    main()
