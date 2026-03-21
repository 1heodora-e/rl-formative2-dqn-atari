import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env

gym.register_envs(ale_py)

env = make_atari_env(
    "ALE/SpaceInvaders-v5",
    n_envs=1,
    seed=42,
    env_kwargs={"render_mode": "human"}
)
env = VecFrameStack(env, n_stack=4)

model = DQN.load("dqn_model.zip", env=env, buffer_size=1000)

print("Starting gameplay...")
print("-" * 40)

all_rewards = []

for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, info = env.step(action)
        total_reward += float(reward[0])
        done = bool(dones[0])
        env.render()

    all_rewards.append(total_reward)
    print(f"Episode {episode + 1} - Total Reward: {total_reward:.2f}")

print("-" * 40)
print(f"Average Reward: {sum(all_rewards)/len(all_rewards):.2f}")
print(f"Best Reward:    {max(all_rewards):.2f}")
print(f"Worst Reward:   {min(all_rewards):.2f}")
print("-" * 40)
print("Gameplay finished!")
env.close()