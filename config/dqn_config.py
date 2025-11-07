# config/dqn_config.py
from config.base import BaseConfig

class DQNConfig(BaseConfig):
    # --- Train loop ---
    NUM_EPISODES = 50          # keep small to smoke-test
    MAX_STEPS = 5000
    LOG_INTERVAL = 5
    SAVE_INTERVAL = 25

    # --- Paths ---
    MODEL_DIR = "models/dqn_simple"
    LOG_DIR = "results/dqn_simple"

    # --- DQN hyperparams (kept simple) ---
    GAMMA = 0.99
    LR = 1e-4

    # Replay buffer
    REPLAY_CAPACITY = 20_000
    BATCH_SIZE = 64
    TRAIN_START_SIZE = 2_000    # start learning after buffer is warm

    # Epsilon-greedy
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 100_000  # linear decay over these steps

    # Device preference: "cuda", "mps" (Apple), "cpu", or None for auto-pick
    DEVICE = None
