from config.base import BaseConfig

class RandomConfig(BaseConfig):    
    NUM_EPISODES = 1
    MAX_STEPS = 10
    
    # Logging
    LOG_INTERVAL = 1
    SAVE_INTERVAL = 10
    
    # Paths
    MODEL_DIR = "models/random"
    LOG_DIR = "results/random"