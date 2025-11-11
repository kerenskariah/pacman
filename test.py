# Use this to test your models.

import os
from main import train, play
from agents.deep_ql_agent import DQNAgent, DQNConfig
from logging import getLogger

# Disable the SDL audio driver warnings so there's no console warning output fo rit
os.environ['SDL_AUDIODRIVER'] = 'dummy'
logger = getLogger(__name__)

def main():
    logger.info("In main.")
    _ = train(DQNAgent, DQNConfig, False)
    #play(DQNAgent, DQNConfig, '', num_episodes=1)

if __name__ == '__main__':
    main()