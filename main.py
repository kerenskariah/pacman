# This has all training loops and playing loops.

import os
import logging
import csv
import ale_py
import gymnasium as gym
from agents.base import BaseAgent
from config.base import BaseConfig

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

logger.info("Registering environment.")
gym.register_envs(ale_py)

# Train the agent based on provided config
def train(agent: type[BaseAgent], config: BaseConfig, render_human: bool = False):
    
    logger.info("Entered training function.")
    
    # Create directory (this is where we save model checkpoints, if it alr exists, dw about it)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    # Directory for training logs
    os.makedirs(config.LOG_DIR, exist_ok=True)
    logger.info("Created directories for training logs and save files.")

    # Create gymnasium environment for training
    # https://gymnasium.farama.org/api/env/
    game_environment = "ALE/MsPacman-v5"
    
    # Check if agent needs RAM observations (for Q-learning)
    use_ram = getattr(config, 'USE_RAM_OBS', False)

    if render_human:
        if use_ram:
            env = gym.make(game_environment, render_mode="human", obs_type="ram")
        else:
            env = gym.make(game_environment, render_mode="human")
    else:
        if use_ram:
            env = gym.make(game_environment, obs_type="ram")
        else:
            env = gym.make(game_environment)
    
    logger.info("Created training environment.")

    training_agent = agent(env.action_space, config)
    logger.info("Created training agent.")

    # Initialize training loop
    episode_rewards = []
    episode_steps = []
    episode_eps = []
    best_avg = None
    best_state = None
    best_at_episode = None  # 1-based episode index when best rolling avg was achieved
    window = getattr(config, 'BEST_AVG_WINDOW', 100)
    for episode in range(config.NUM_EPISODES):

        # For each training episode, reset the environment completely
        observation, info = env.reset()
        reward = 0
        # Observation: shud be the initial game state iirc
        # Info: data like # of hearts, score, and more
        # Reward: reward counter to keep track of reinforcements

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

        # Per-episode bookkeeping
        steps_taken = step + 1 if 'step' in locals() else 0
        episode_rewards.append(reward)
        episode_steps.append(steps_taken)
        if hasattr(training_agent, 'epsilon'):
            episode_eps.append(getattr(training_agent, 'epsilon'))
        else:
            episode_eps.append('')

        # Update best by rolling average
        if len(episode_rewards) >= window:
            avg = sum(episode_rewards[-window:]) / float(window)
            if (best_avg is None) or (avg > best_avg):
                best_avg = avg
                if hasattr(training_agent, 'state_dict'):
                    best_state = training_agent.state_dict()
                best_at_episode = episode + 1

        # Mod for periodic logging based on config so you don't get 50B logs !
        if (episode + 1) % config.LOG_INTERVAL == 0:
            logger.info(f"Episode {episode + 1}/{config.NUM_EPISODES}, Reward: {reward}")
            
            # Log Q-learning specific stats if available
            if hasattr(training_agent, 'get_stats'):
                stats = training_agent.get_stats()
                logger.info(f"  Q-table: {stats['q_table_states']} states, {stats['q_table_entries']} entries, epsilon={stats['epsilon']:.4f}")

        # Mod to check periodically for saving the agent
        if (episode + 1) % config.SAVE_INTERVAL == 0:
            path_name = f"{config.MODEL_DIR}/agent_ep{episode + 1}.pkl"
            training_agent.save(path_name)
            logger.info(f"Saved trained agent after episode {episode} as {path_name}")

    env.close()

    # Minimal CSV + optional plot
    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        csv_path = os.path.join(config.LOG_DIR, 'episode_log.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'reward', 'steps', 'epsilon'])
            for i, (r, s, e) in enumerate(zip(episode_rewards, episode_steps, episode_eps), start=1):
                w.writerow([i, r, s, e])

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            xs = np.arange(1, len(episode_rewards) + 1)
            win = max(1, int(getattr(config, 'BEST_AVG_WINDOW', 100)))

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            # Rewards with moving average
            ax1.plot(xs, episode_rewards, label='reward', alpha=0.7)
            if len(episode_rewards) >= win:
                ma = np.convolve(episode_rewards, np.ones(win)/win, mode='valid')
                ax1.plot(np.arange(win, len(episode_rewards) + 1), ma, label=f'moving avg ({win})', linewidth=2)
            ax1.set_ylabel('reward')
            ax1.legend(loc='best')

            # Steps per episode
            ax2.plot(xs, episode_steps, label='steps', color='tab:orange', alpha=0.8)
            ax2.set_xlabel('episode')
            ax2.set_ylabel('steps')
            ax2.legend(loc='best')

            out_png = os.path.join(config.LOG_DIR, 'rewards.png')
            plt.tight_layout()
            plt.savefig(out_png, dpi=120)
            plt.close(fig)
            logger.info(f"Saved reward/steps plot to {out_png}")
        except Exception:
            pass
    except Exception:
        pass

    # If we tracked a best model by rolling average, restore it before saving
    used_best = False
    if best_state is not None and hasattr(training_agent, 'load_state'):
        training_agent.load_state(best_state)
        used_best = True

    # Save a single checkpoint (align with PPO naming)
    model_path = f"{config.MODEL_DIR}/agent_latest.pkl"
    training_agent.save(model_path)
    if used_best:
        logger.info(
            f"Saved best model (rolling avg over {window}) from episode {best_at_episode} "
            f"with avg reward {best_avg:.2f} to {model_path}"
        )
    else:
        logger.info(
            f"Saved final model (no full {window}-episode window reached) to {model_path}"
        )
    logger.info("Done with training.")

    return model_path

# Make agent play the game for num_episodes given that you already trained it.
def play(agent_class: type[BaseAgent], config: BaseConfig, model_path: str, num_episodes: int = 5):

    # Check if agent needs RAM observations (for Q-learning)
    use_ram = getattr(config, 'USE_RAM_OBS', False)
    
    if use_ram:
        env = gym.make("ALE/MsPacman-v5", render_mode="human", obs_type="ram")
    else:
        env = gym.make("ALE/MsPacman-v5", render_mode="human")    

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