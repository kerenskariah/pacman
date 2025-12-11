# train_dqn_agent.py
import logging
from main import train
from agents.dqn_agent import DQNAgent
from config.dqn_config import DQNConfig

if __name__ == "__main__":
    # 1) Turn on INFO logs
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # 2) Small, fast test so you see logs ASAP
    cfg = DQNConfig()
    cfg.NUM_EPISODES = 5       # first log prints at episode 5 with LOG_INTERVAL=5
    cfg.MAX_STEPS = 500        # keep episodes short so you don't wait long

    print("Starting DQN training...")
    model_path = train(DQNAgent, cfg, render_human=False)
    print(f"✅ Training complete. Model saved to: {model_path}")
