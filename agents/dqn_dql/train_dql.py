# train_dql.py
from agents.dqn_dql.deep_ql_agent import DQLAgent
from config.dql_config import DQLConfig
from main import train, play, record_gif
import os

if __name__ == "__main__":
    print("\n=== DQL Training Script ===\n")

    # Load config
    config = DQLConfig()
    config.device = "cpu"  # Force CPU

    config.MODEL_DIR = "models/dql"
    config.LOG_DIR   = "logs/dql"

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Quick Test Settings
    config.NUM_EPISODES = 20
    config.MAX_STEPS = 500
    config.SAVE_INTERVAL = 10
    config.EVAL_INTERVAL = 10
    config.GIF_INTERVAL = 10000  # basically disables GIFs for test
    
    # Long Run Settings
    config.NUM_EPISODES = 5000
    config.MAX_STEPS = 6000
    config.SAVE_INTERVAL = 200
    config.EVAL_INTERVAL = 200
    config.GIF_INTERVAL = 1000

    print(f"NUM_EPISODES = {config.NUM_EPISODES}")
    print(f"MAX_STEPS    = {config.MAX_STEPS}\n")
    print("Starting training...\n")

    # Train
    model_path = train(DQLAgent, config)

    print("\nTraining completed.")
    print(f"Your DQL model saved at: {model_path}\n")

    # Evaluate test run
    print("Running evaluation episodes...\n")
    play(DQLAgent, config, model_path, num_episodes=3)

    # Record GIF
    print("\nRecording GIF (high-res)...")
    try:
        gif_path = "agent_play.gif"
        record_gif(DQLAgent, config, model_path, gif_path=gif_path)
        print(f"GIF created: {gif_path}\n")
    except Exception as e:
        print("GIF failed:", e)
