import gymnasium as gym
import ale_py
import numpy as np
import os
import gc
import time
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

gym.register_envs(ale_py)

ENV_ID = "BreakoutNoFrameskip-v4"
TIMESTEPS = 500_000


class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    env = make_atari_env(ENV_ID, n_envs=1, seed=0, monitor_dir="logs")
    env = VecFrameStack(env, n_stack=4)

    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        gamma=0.99,
        batch_size=32,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.15,
        buffer_size=100_000,
        optimize_memory_usage=True,
        replay_buffer_kwargs={"handle_timeout_termination": False},
        learning_starts=10_000,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        verbose=1,
        tensorboard_log="logs/tb"
    )

    callback = RewardLoggerCallback()
    print(f"Training DQN on {ENV_ID} for {TIMESTEPS} steps...")
    start = time.time()
    model.learn(total_timesteps=TIMESTEPS, callback=callback)
    print(f"Done in {(time.time() - start)/60:.1f} min")

    model.save("models/dqn_model")
    print("Model saved to models/dqn_model.zip")

    eval_env = make_atari_env(ENV_ID, n_envs=1, seed=42)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"Eval: {mean_reward:.2f} +/- {std_reward:.2f}")
    eval_env.close()
    env.close()
