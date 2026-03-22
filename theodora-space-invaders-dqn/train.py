import os
from pathlib import Path

import ale_py
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.callbacks import EvalCallback

gym.register_envs(ale_py)

# Kaggle sets KAGGLE_WORKING_DIR (e.g. /kaggle/working). Locally, use this script's folder.
_WORKSPACE = Path(os.environ.get("KAGGLE_WORKING_DIR", str(Path(__file__).resolve().parent)))
for _sub in ("logs/final_model", "best_model", "tensorboard_logs"):
    (_WORKSPACE / _sub).mkdir(parents=True, exist_ok=True)

env = make_atari_env(
    "ALE/SpaceInvaders-v5",
    n_envs=1,
    seed=42,
    monitor_dir=str(_WORKSPACE / "logs" / "final_model"),
)
env = VecFrameStack(env, n_stack=4)

eval_env = make_atari_env("ALE/SpaceInvaders-v5", n_envs=1, seed=42)
eval_env = VecFrameStack(eval_env, n_stack=4)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(_WORKSPACE / "best_model") + "/",
    log_path=str(_WORKSPACE / "logs" / "final_model") + "/",
    eval_freq=10000,
    verbose=1,
)

final_model = DQN(
    policy="CnnPolicy",
    env=env,
    learning_rate=1e-4,
    gamma=0.999,
    batch_size=64,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    exploration_fraction=0.5,
    verbose=1,
    tensorboard_log=str(_WORKSPACE / "tensorboard_logs"),
)

final_model.learn(
    total_timesteps=500_000,
    callback=eval_callback,
)

final_model.save(str(_WORKSPACE / "dqn_model"))
print(f"Final model saved under {_WORKSPACE / 'dqn_model.zip'}")
env.close()
eval_env.close()
