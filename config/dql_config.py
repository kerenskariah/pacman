# config/dql_config.py
from config.base import BaseConfig
import torch


class DQLConfig(BaseConfig):

    MODEL_DIR = "models/dql"
    LOG_DIR = "logs/dql"

    # Training length
    NUM_EPISODES = 5000
    MAX_STEPS = 6000

    # DQN hyperparameters
    gamma = 0.99
    lr = 1e-4

    # Replay buffer
    buffer_size = 100_000
    learning_starts = 5_000
    batch_size = 128
    train_freq = 4
    target_update_freq = 4_000

    # Exploration schedule
    eps_start = 1.0
    eps_final = 0.02
    eps_decay_steps = 500_000

    # Stability options
    max_grad_norm = 10.0
    double_dqn = True

    # Image hints (auto-detected)
    is_image = None
    image_size = None

    # Logging / Checkpoint intervals
    SAVE_INTERVAL = 200
    LOG_INTERVAL = 100
    MAX_CHECKPOINTS = 5

    # Evaluation / GIF
    EVAL_INTERVAL = 200
    GIF_INTERVAL = 500

    # FORCE CPU — always
    device = "cpu"