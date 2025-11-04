import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/MsPacman-v5", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()