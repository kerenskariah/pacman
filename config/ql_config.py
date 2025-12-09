from config.base import BaseConfig
from dataclasses import dataclass

@dataclass
class QLearningConfig(BaseConfig):
    """Configuration for Q-Learning Agent."""
    
    # Environment
    USE_RAM_OBS: bool = True  # Use RAM observations instead of pixels
    
    # Training
    NUM_EPISODES: int = 8000
    MAX_STEPS: int = 4000
    
    # Q-Learning hyperparameters
    ALPHA: float = 0.1          # Learning rate
    GAMMA: float = 0.99         # Discount factor
    EPSILON_START: float = 1.0  # Initial exploration rate
    EPSILON_END: float = 0.01   # Final exploration rate
    EPSILON_DECAY: float = 0.995  # Epsilon decay per episode
    
    # State discretization
    RAM_STRIDE: int = 16  # Select every 16th RAM byte (128/16 = 8 bytes)
    RAM_BINS: int = 8     # Bin each byte into 8 discrete values
    
    # Logging
    LOG_INTERVAL: int = 100
    BEST_AVG_WINDOW: int = 100  # choose best model by 100-episode moving average
    SAVE_INTERVAL: int = 1000000000
    
    # Paths
    MODEL_DIR: str = "models/qlearning"
    LOG_DIR: str = "results/qlearning"
    
    # Misc
    SEED: int = 42
