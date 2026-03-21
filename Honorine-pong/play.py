#!/usr/bin/env python3
"""
Load a trained DQN model and run greedy evaluation episodes.

Rubric — GreedyQPolicy (evaluation):
- During training, DQN uses epsilon-greedy exploration.
- For evaluation, the assignment asks for GreedyQPolicy: always pick the action with highest Q-value.
- In Stable-Baselines3 this is done with: model.predict(obs, deterministic=True) each step.

Rubric — visualization:
- `--render-mode human`: calls env.render() for a local GUI (best for live class demo).
- `--render-mode rgb_array` with `--save-video`: records frames from env.render() for Kaggle / headless.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import gymnasium as gym
import ale_py
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage


def register_ale_envs():
    """Required on some Kaggle/Gymnasium setups so ALE/* ids are visible."""
    gym.register_envs(ale_py)


def build_env_fn(env_id: str, policy_kind: str, render_mode: str):
    def _make():
        env = gym.make(env_id, render_mode=render_mode)
        env = AtariWrapper(
            env,
            noop_max=30,
            frame_skip=4,
            screen_size=84,
            terminal_on_life_loss=False,
            clip_reward=False,  # show true game rewards during evaluation
        )
        if policy_kind.lower() == "mlp":
            env = gym.wrappers.FlattenObservation(env)
        return env

    return _make


def make_vec_env(env_id: str, policy_kind: str, render_mode: str):
    vec_env = DummyVecEnv([build_env_fn(env_id, policy_kind, render_mode)])
    if policy_kind.lower() == "cnn":
        vec_env = VecFrameStack(vec_env, n_stack=4)
        vec_env = VecTransposeImage(vec_env)
    return vec_env


def main():
    register_ale_envs()
    parser = argparse.ArgumentParser(description="Play Atari Pong with a trained DQN model.")
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--model-path", type=Path, default=Path("dqn_model.zip"))
    parser.add_argument("--policy", type=str, choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=20_000,
        help="Safety cap to prevent very long episodes during evaluation.",
    )
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "rgb_array"])
    parser.add_argument("--save-video", action="store_true", help="Save video (useful for Kaggle).")
    parser.add_argument("--video-path", type=Path, default=Path("pong_agent.mp4"))
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    env = make_vec_env(args.env_id, args.policy, args.render_mode)
    model = DQN.load(str(args.model_path), env=env)

    all_returns: List[float] = []
    frames: List[np.ndarray] = []

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        while not done and steps < args.max_steps_per_episode:
            # deterministic=True = greedy action (max Q-value)
            action, _state = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_return += float(rewards[0])
            done = bool(dones[0])
            steps += 1

            if args.render_mode == "rgb_array" and args.save_video:
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)

        all_returns.append(ep_return)
        print(f"Episode {episode + 1}/{args.episodes} return: {ep_return:.2f}")
        if not done:
            print(
                f"Episode {episode + 1} reached max step cap ({args.max_steps_per_episode}) and was truncated."
            )

    print("\nEvaluation done.")
    print(f"Mean return over {args.episodes} episodes: {np.mean(all_returns):.2f}")
    print(f"Std return: {np.std(all_returns):.2f}")

    if args.save_video and frames:
        imageio.mimsave(args.video_path, frames, fps=30)
        print(f"Saved video: {args.video_path.resolve()}")

    env.close()


if __name__ == "__main__":
    main()
