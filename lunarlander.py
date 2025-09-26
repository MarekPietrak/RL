import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

env_id = "LunarLander-v3"
logdir = "./logs_lander_ppo"
os.makedirs(logdir, exist_ok=True)

def make_env():
    return Monitor(gym.make(env_id))

env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=1.5e-4,
    n_steps=4096,
    batch_size=256,
    gamma=0.999,
    gae_lambda=0.95,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.01,
    target_kl=0.02,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
)

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.0)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=logdir,
    log_path=logdir,
    eval_freq=10_000,
    n_eval_episodes=20,
    deterministic=True,
    render=False
)

model.learn(total_timesteps=1000000, callback=eval_cb)

#add video
from gymnasium.wrappers import RecordVideo

video_dir = os.path.join(logdir, "videos")
os.makedirs(video_dir, exist_ok=True)

# Use a Monitor-wrapped environment for video recording
video_env = gym.make(env_id, render_mode="rgb_array")
video_env = RecordVideo(video_env, video_dir, episode_trigger=lambda x: True)
obs, info = video_env.reset()
total_reward = 0.0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = video_env.step(action)
    total_reward += reward
    if terminated or truncated:
        print("Episode reward:", total_reward)
        break

video_env.close()
print(f"Video saved to {video_dir}")
