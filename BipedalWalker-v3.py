import os
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

#logging data via monitor wrapper
env = Monitor(gym.make("BipedalWalker-v3"), filename=log_dir)

#activate model
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=500000)

#plot results
results_plotter.plot_results([log_dir], 500000, results_plotter.X_TIMESTEPS, "PPO BipedalWalker")

######################################################

# record video
from gymnasium.wrappers import RecordVideo

#create video folder
video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

# create new environment with video recording enabled
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
env = RecordVideo(env, video_dir, episode_trigger=lambda episode_id: True)

obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
env.close()

print(f"Video saved to {video_dir}.")