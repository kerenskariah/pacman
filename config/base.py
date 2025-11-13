from abc import ABC
from dataclasses import dataclass

@dataclass
class BaseConfig(ABC):    
    # Training parameters
    NUM_EPISODES: int = 1000
    MAX_STEPS: int = 1000
    
    # Logging
    LOG_INTERVAL: int = 10
    SAVE_INTERVAL: int = 100
    
    # Paths
    MODEL_DIR: str = "models"
    LOG_DIR: str = "logs"