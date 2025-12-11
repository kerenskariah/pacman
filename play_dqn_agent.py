# play_dqn_agent.py
from main import play
from agents.dqn_agent import DQNAgent
from config.dqn_config import DQNConfig

if __name__ == "__main__":
    cfg = DQNConfig()
    model_path = f"{cfg.MODEL_DIR}/agent_final.pkl"

    print("Starting Ms. Pac-Man with trained DQN agent...")
    play(DQNAgent, cfg, model_path, num_episodes=2)
    print("✅ Done playing!")
