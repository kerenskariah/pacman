from config.base import BaseConfig
from dataclasses import dataclass
import torch

@dataclass
class PPOConfig(BaseConfig):
    
    # Environment
    ENV_ID: str = "ALE/MsPacman-v5"
    # Increased from 8 to 16 for better gradient estimation and faster wall-clock collection
    NUM_ENVS: int = 16  
    FRAME_STACK: int = 4
    FRAME_SKIP: int = 4
    STICKY_PROB: float = 0.0 # Reduced from 0.25 to 0.0 for deterministic control

    # Training
    # Increased from 4000 (~4M frames) to 10000 (~10M frames) for convergence
    TOTAL_UPDATES: int = 10000          
    ROLLOUT_STEPS: int = 128           
    PPO_EPOCHS: int = 4                
    MINIBATCH_SIZE: int = 128 # Reduced from 256 to 128 for more frequent updates
    MAX_GRAD_NORM: float = 0.5

    # Loss / algo
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.1              
    VALUE_CLIP: float = 0.5 # Increased from 0.2 to 0.5
    # Increased entropy to prevent premature convergence and encourage exploration
    ENTROPY_COEF: float = 0.05
    VALUE_COEF: float = 0.5
    TARGET_KL: float = 0.02            

    # Optimizer
    LR: float = 2.5e-4

    # Logging / saving
    LOG_INTERVAL: int = 10 # Log every 10 updates
    SAVE_INTERVAL: int = 50 # Save every 50 updates
    MODEL_DIR: str = "models/ppo"
    LOG_DIR: str = "logs/ppo"

    # Misc
    DEVICE: str = ("mps" if torch.backends.mps.is_available() 
                   else "cuda" if torch.cuda.is_available() 
                   else "cpu")
    SEED: int = 42
    USE_AMP: bool = True