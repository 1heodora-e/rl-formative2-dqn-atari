import gymnasium as gym
import ale_py
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

ENV_ID = "BreakoutNoFrameskip-v4"
MODEL_PATH = "models/dqn_model.zip"
NUM_EPISODES = 5


if __name__ == "__main__":
    model = DQN.load(MODEL_PATH)

    env = make_atari_env(
        ENV_ID, n_envs=1, seed=0, env_kwargs={"render_mode": "human"}
    )
    env = VecFrameStack(env, n_stack=4)

    for ep in range(NUM_EPISODES):
        obs = env.reset()
        env.render()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            env.render()

        print(f"Episode {ep+1}: reward={total_reward:.1f}, steps={steps}")

    env.close()
