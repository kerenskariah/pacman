from config.base import BaseConfig
from dataclasses import dataclass

@dataclass
class PPOConfig(BaseConfig):
    
    # Environment
    ENV_ID: str = "ALE/MsPacman-v5"
    NUM_ENVS: int = 8
    FRAME_STACK: int = 4
    FRAME_SKIP: int = 4
    STICKY_PROB: float = 0.25

    # Training
    TOTAL_UPDATES: int = 4000          # number of PPO updates
    ROLLOUT_STEPS: int = 128           # steps per env before one PPO update
    PPO_EPOCHS: int = 4                # SGD passes over the collected batch
    MINIBATCH_SIZE: int = 256
    MAX_GRAD_NORM: float = 0.5

    # Loss / algo
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_EPS: float = 0.1              # policy clip
    VALUE_CLIP: float = 0.2            # value clip range
    ENTROPY_COEF: float = 0.01
    VALUE_COEF: float = 0.5
    TARGET_KL: float = 0.02            # early-stop PPO epochs when exceeded

    # Optimizer
    LR: float = 2.5e-4

    # Logging / saving
    LOG_INTERVAL: int = 1             # in updates
    SAVE_INTERVAL: int = 10           # in updates
    MODEL_DIR: str = "models/ppo"
    LOG_DIR: str = "results/ppo"

    # Misc
    DEVICE: str = "cuda"  # Will be set properly in code
    SEED: int = 42
    USE_AMP: bool = True               # mixed precision on GPU


