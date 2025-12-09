# This has all training loops and playing loops.

import os
import logging
import numpy as np
import csv
import ale_py
import gymnasium as gym
from agents.base import BaseAgent
from config.base import BaseConfig
from gymnasium.wrappers import AtariPreprocessing, FrameStack
import imageio
import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

logger.info("Registering environment.")
gym.register_envs(ale_py)

# Train the agent based on provided config
def train(agent_class: type[BaseAgent], config: BaseConfig, render_human: bool = False):
    
    logger.info("Entered training function.")
    
    # Create directory (this is where we save model checkpoints, if it alr exists, dw about it)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    # Directory for training logs
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger.info("Created directories for training logs and save files.")

    # Paths for CSV logging
    reward_csv = os.path.join(config.LOG_DIR, "reward_log.csv")
    eps_csv = os.path.join(config.LOG_DIR, "epsilon_log.csv")

    # Reward log header
    with open(reward_csv, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "reward"])

    # Only log epsilon for DQLAgent
    log_epsilon = (agent_class.__name__ == "DQLAgent")
    if log_epsilon:
        with open(eps_csv, "w", newline="") as f:
            csv.writer(f).writerow(["episode", "epsilon"])

    # Create gymnasium environment for training
    # https://gymnasium.farama.org/api/env/
    game_environment = "ALE/MsPacman-v5"

    if render_human:
        env = gym.make(game_environment, render_mode="human", frameskip=1)
    else:
        env = gym.make(game_environment, render_mode="rgb_array", frameskip=1)
    
    # Preprocessed environment for the agent
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        terminal_on_life_loss=True
    )
    env = FrameStack(env, 4)

    logger.info("Created training environment.")

    training_agent = agent_class(env.action_space, config)
    logger.info("Created training agent.")

    # Initialize training loop
    for episode in range(config.NUM_EPISODES):

        # For each training episode, reset the environment completely
        observation, info = env.reset()
        reward = 0
        
        # Observation: shud be the initial game state iirc
        # Info: data like # of hearts, score, and more
        # Total Reward: reward counter to keep track of reinforcements

        # Go through a bunch of steps, see what agent does
        for step in range(config.MAX_STEPS):
            # Get an action from the agent based on what it observes from the environment
            action = training_agent.get_action(observation) 

            """
                From: https://gymnasium.farama.org/api/env/#gymnasium.Env.step
                observation (ObsType) An element of the environment's observation_space as the next observation due to the agent actions. An example is a numpy array containing the positions and velocities of the pole in CartPole.
                reward (SupportsFloat) The reward as a result of taking the action.
                terminated (bool) Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or negative. An example is reaching the goal state or moving into the lava from the Sutton and Barto Gridworld. If true, the user needs to call reset().
                truncated (bool) Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the user needs to call reset().
                info (dict) Contains auxiliary diagnostic information (helpful for debugging, learning, and logging). This might, for instance, contain: metrics that describe the agent’s performance state, variables that are hidden from observations, or individual reward terms that are combined to produce the total reward. In OpenAI Gym <v26, it contains “TimeLimit.truncated” to distinguish truncation and termination, however this is deprecated in favour of returning terminated and truncated variables.
            """
            # Get data from the environment frame after agent performs action
            next_observation, step_reward, step_terminated, step_truncated, step_info = env.step(action)
            needs_to_reset = step_terminated or step_truncated

            # Update the agent based on what the results are
            training_agent.update(observation, action, step_reward, next_observation, needs_to_reset)

            # Update the current state, and increment reward accordingly
            observation = next_observation
            info = step_info
            reward += step_reward

            if needs_to_reset:
                # observation, info = env.reset()
                break

        # Rewards logging
        with open(reward_csv, "a", newline="") as f:
            csv.writer(f).writerow([episode + 1, reward])

        # Epsilon logging (only used for DQL so far)
        if log_epsilon:
            eps_val = training_agent._epsilon()
            with open(eps_csv, "a", newline="") as f:
                csv.writer(f).writerow([episode + 1, eps_val])

        # Mod for periodic logging based on config so you don't get 50B logs !
        if (episode + 1) % config.LOG_INTERVAL == 0:
            logger.info(f"Episode {episode + 1}/{config.NUM_EPISODES}")
            logger.info(f"Reward: {reward}")

        # Mod to check periodically for saving the agent
        if (episode + 1) % config.SAVE_INTERVAL == 0:
            path_name = os.path.join(config.MODEL_DIR, f"agent_ep{episode + 1}.pkl")
            training_agent.save(path_name)
            logger.info(f"Saved trained agent after episode {episode} as {path_name}")

    # Plot results (learning and epsilon curves)
    results_dir = config.LOG_DIR.replace("logs", "results")
    os.makedirs(results_dir, exist_ok=True)

    make_learning_curve(reward_csv, os.path.join(results_dir, "learning_curve.png"))
    
    if log_epsilon:
        plot_epsilon(eps_csv, os.path.join(results_dir, "epsilon_curve.png"))
    else:
        logger.info("Skipping epsilon plot (agent has no epsilon).")

    logger.info("Finished training.")

    env.close()

    model_path = os.path.join(config.MODEL_DIR, "agent_final.pkl")
    training_agent.save(model_path)
    logger.info(f"Saved final model to {model_path}")
    logger.info("Done with training.")

    return model_path

# Make agent play the game for num_episodes given that you already trained it.
def play(agent_class: type[BaseAgent], config: BaseConfig, model_path: str, num_episodes: int = 5):

    env = gym.make("ALE/MsPacman-v5", render_mode="human", frameskip=1)
    
    # Preprocessed environment for the agent
    env = AtariPreprocessing(
        env,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        terminal_on_life_loss=True
    )
    env = FrameStack(env, 4)

    agent = agent_class(env.action_space, config)
    
    if model_path != '':
        agent.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.info('No save file (.pkl) provided, just using untrained model.')

    for episode in range(num_episodes):
        observation, info = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.get_action(observation)
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            if terminated or truncated:
                break
        
        logger.info(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")

    env.close()
    logger.info("Done playing!")

# Learning curve plot func
# Uses reward_log.csv file and creates a reward curve
# Shows raw rewards + smoothed moving average
def make_learning_curve(csv_file, out_path):

    episodes, rewards = [], []

    with open(csv_file, "r") as f:
        next(f)
        for ep, rew in csv.reader(f):
            episodes.append(int(ep))
            rewards.append(float(rew))

    # Window size for smoothing
    W = 20
    smoothed = [np.mean(rewards[max(0, i-W):i+1]) for i in range(len(rewards))]

    plt.figure(figsize=(12, 5))
    plt.plot(episodes, rewards, alpha=0.4, label="Raw reward")
    plt.plot(episodes, smoothed, linewidth=2, label=f"Smoothed ({W}-episode MA)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve")
    plt.legend()
    plt.savefig(out_path)
    plt.close()

    logger.info(f"Saved the learning curve to {out_path}")


# Epsilon curve plot func (Only used in DQL so far)
# Epsilon decay over episodes
def plot_epsilon(csv_file, out_path):

    episodes, epsilons = [], []

    with open(csv_file, "r") as f:
        next(f)
        for ep, eps in csv.reader(f):
            episodes.append(int(ep))
            epsilons.append(float(eps))

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, epsilons, color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay")
    plt.grid(alpha=0.3)
    plt.savefig(out_path)
    plt.close()

    logger.info(f"Saved epsilon curve to {out_path}")


# GIF func (might not need it due to a built in func)
def record_gif(agent_class, config, model_path, gif_path="agent_play.gif", upscale=3):
    
    # Creates result/agent folder if none exist
    results_dir = config.LOG_DIR.replace("logs", "results")
    os.makedirs(results_dir, exist_ok=True)

    # Force saves gif in results folder
    if not os.path.isabs(gif_path):
        gif_path = os.path.join(results_dir, os.path.basename(gif_path))

    env_raw = gym.make("ALE/MsPacman-v5", render_mode="rgb_array", frameskip=1)
    
    env_agent = AtariPreprocessing(
        env_raw,
        screen_size=84,
        grayscale_obs=True,
        frame_skip=4,
        terminal_on_life_loss=True
    )
    env_agent = FrameStack(env_agent, 4)

    agent = agent_class(env_agent.action_space, config)
    agent.load(model_path)

    frames = []
    obs, info = env_agent.reset()
    done = False
    reward = 0

    # While episode ends
    while not done:
        frame = env_raw.render()

        if upscale > 1: # Scaled so its not pixelated
            H, W = frame.shape[:2]
            frame = cv2.resize(frame, (W * upscale, H * upscale),
                               interpolation=cv2.INTER_NEAREST)

        frames.append(frame)

        # Agent makes an action
        action = agent.get_action(obs)\
        
        # Step forward in environment
        obs, r, term, trunc, info = env_agent.step(action)
        reward += r
        done = term or trunc

    # Cleanup
    env_raw.close()
    env_agent.close()

    # Save gif
    imageio.mimsave(gif_path, frames, fps=30)
    logger.info(f"Saved GIF to {gif_path} (Reward = {reward})")