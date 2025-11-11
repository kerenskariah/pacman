# Use this to test your models.
import os
import argparse
from main import train, play
from logging import getLogger

from agents.random_agent import RandomAgent
from config.random_config import RandomConfig
from agents.deep_ql_agent import DQNAgent, DQNConfig

# Disable the SDL audio driver warnings so there's no console warning output fo rit
os.environ['SDL_AUDIODRIVER'] = 'dummy'
logger = getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train or play a Pac-Man agent.")
    parser.add_argument("--agent", type=str, required=True, help="Specify the agent to use (e.g., 'random', 'dqn')")
    args = parser.parse_args()

    agent_to_run = None
    config_to_use = None

    if args.agent.lower() == 'random':
        agent_to_run = RandomAgent
        config_to_use = RandomConfig()
    elif args.agent.lower() == 'dqn':
        agent_to_run = DQNAgent
        config_to_use = DQNConfig()
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    logger.info(f"Running training for agent: {args.agent}")
    model_path = train(agent_to_run, config_to_use, render_human=False)
    
    logger.info("Training complete. Starting a play session...")
    play(agent_to_run, config_to_use, model_path, num_episodes=5)

if __name__ == '__main__':
    main()