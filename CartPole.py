import gymnasium as gym
import time
from stable_baselines3 import DQN


env = gym.make("CartPole-v1")
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)

obs, _ = env.reset()  # Unpack obs and info as per Gymnasium API
# reset environment to start a new episode
total_reward = 0
while True:
    action, _state = model.predict(obs, deterministic=True) # choose action (deterministic means no randomness)
    obs, reward, terminated, truncated, info = env.step(action) # Gymnasium API returns 5 values
    total_reward += reward
    
    env.render()
    time.sleep(1/60)

    if terminated or truncated:
        break # episode ended (pole fell or time limit reached)

print("Episode reward:", total_reward)
env.close()

