import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from config.dqn_config import DQNConfig

DEFAULT_WINDOW = 100

def main():
    cfg = DQNConfig()
    csv_path = os.path.join(cfg.LOG_DIR, "train_log.csv")

    episodes = []
    rewards = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))

    episodes = np.array(episodes)
    rewards = np.array(rewards)

    if len(rewards) == 0:
        print("No data in train_log.csv")
        return

    window = min(DEFAULT_WINDOW, max(1, len(rewards) // 5))
    print(f"Using moving average window = {window}")

    # moving average
    kernel = np.ones(window) / window
    rewards_ma = np.convolve(rewards, kernel, mode="valid")
    episodes_ma = episodes[window - 1 :]

    # rolling std for shaded band
    if len(rewards) >= window:
        rolling_std = np.array([
            rewards[i - window + 1 : i + 1].std()
            for i in range(window - 1, len(rewards))
        ])
        upper = rewards_ma + rolling_std
        lower = rewards_ma - rolling_std
    else:
        rolling_std = None
        upper = rewards_ma
        lower = rewards_ma

    plt.figure(figsize=(10, 5))

    # noisy episode rewards
    plt.plot(episodes, rewards, alpha=0.25, linewidth=1, label="Episode reward")

    # shaded band
    if rolling_std is not None:
        plt.fill_between(episodes_ma, lower, upper, alpha=0.2)

    # moving average line
    plt.plot(episodes_ma, rewards_ma, linewidth=2, label=f"Moving average ({window})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Rewards Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # save PNG as well as show
    out_path = os.path.join(cfg.LOG_DIR, "train_rewards.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

    plt.show()

if __name__ == "__main__":
    main()
