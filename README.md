# Agent Plays Ms. Pac-Man

ECS 170 – Group 9

## Why Ms. Pac-Man?

We chose **Ms. Pac-Man** because it provides a more challenging and dynamic environment for reinforcement learning compared to the original Pac-Man. Unlike Pac-Man’s single maze and predictable ghost paths, Ms. Pac-Man features:

- Four unique mazes
- Randomized fruit spawns
- Less predictable ghost movement patterns

This unpredictability forces agents to generalize, adapt, and balance survival with reward acquisition — making it a significantly more interesting environment for RL research.

## Environment

We use the **Atari Learning Environment (ALE)** together with **Gymnasium** as our RL environment. These frameworks provide:

- Standard RL interfaces (`reset`, `step`)
- Support for Atari preprocessing
- Integration with GPU-accelerated training setups  
- Simplified switching between agents without modifying the environment logic

To improve training stability and reduce variance, we incorporate:

- Frame skipping (4)
- Frame stacking (4)
- Sticky actions (25%)
- Reward clipping (-1, 0, +1)

These match the preprocessing pipeline used in many deep RL Atari benchmarks.

## Prerequisites

- Python **3.8 or later**
- pip

## Create and activate a virtual environment (Linux / macOS)

1. Create the virtual environment:

```
python3 -m venv .venv
```

2. Activate it:

```
source .venv/bin/activate
```

On Windows (PowerShell):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Install dependencies with pip

```
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```


## Deactivate the virtual environment

```
deactivate
```

## If requirements.txt is missing

Create it from an active virtual environment:

```
pip freeze > requirements.txt
```

## Project Structure


```
.
├── main.py                     # Shared training & evaluation utilities
├── requirements.txt
├── train_ppo.py                # PPO training script
├── train_dql.py                # DQN / Deep Q training script
├── test.py                     # Simple environment or baseline test script
├── play_5_ppo.sh               # Run PPO agent for 5 evaluation episodes
│
├── agents/
│   ├── base.py                 # Abstract base class defining agent API
│   ├── random_agent.py         # Random policy agent
│   ├── ql_agent.py             # Tabular Q-Learning agent
│   ├── deep_ql_agent.py        # Deep Q-Network (DQN) agent
│   ├── ppo_agent.py            # PPO actor-critic agent
│   └── duel-qn-agent.py        # Prototype Dueling DQN agent (experimental)
│
├── config/
│   ├── base.py
│   ├── random_config.py
│   ├── ql_config.py
│   ├── dql_config.py
│   └── ppo_config.py
│
├── models/
│   ├── qlearning/
│   │   └── agent_latest.pkl
│   └── ppo/
│       └── agent_latest.pt
│
├── results/
│   ├── qlearning/
│   │   ├── rewards.png
│   │   └── episode_log.csv
│   └── ppo/
│       └── rewards_latest.png
│
├── reports/
│   └── project_check_in/
│       ├── project_check_in.md
│       ├── ppo_rewards_latest.png
│       └── group_members.yaml
│
└── README.md
```

## Functions in `main.py`

### `train(agent_class, config, render_human=False)`

* **agent_class**: The agent class to instantiate (not an instance)
* **config**: Configuration object specifying hyperparameters and environment settings
* **render_human**: If `True`, visualizes training (much slower)
* **Returns**: Path to the saved model checkpoint created during training

### `play(agent_class, config, model_path, num_episodes=5)`

* **agent_class**: Agent class to load and run
* **config**: Configuration object
* **model_path**: Path to a saved `.pkl` or `.pt` model (empty string loads an untrained model)
* **num_episodes**: Number of episodes to run for evaluation or human-visible gameplay


## Implemented Agents

All agents inherit from a unified interface (`agents/base.py`).

### Random Agent

Ensures preprocessing, rendering, and episode logic work.

### Q-Learning Agent

- Tabular Q-learning using RAM features — learns simple behaviors but quickly plateaus.

### Deep Q-Learning (DQN)

- Neural network approximator, experience replay, target network. Learns meaningful behaviors but is noisy and unstable.

### PPO Agent

- Actor–critic architecture with clipped updates. Demonstrated the strongest and most stable performance in our experiments.

### Dueling DQN (Prototype)

- Splits Q-values into Value + Advantage streams. Not fully implemented but included for completeness.


## Training & Evaluation

### Train PPO

```
python train_ppo.py
```

### Train DQN / Deep Q-Learning

```
python train_dql.py
```

### Test baseline or environment

```
python test.py
```

### Watch PPO agent play 5 episodes

```
bash play_5_ppo.sh
```


## Results (High-Level Summary)

Across thousands of training episodes:

* **Random Agent**: Near-zero scores; used for debugging.
* **Q-Learning**: Learns pellet collection but plateaus quickly.
* **DQN**: Learns strategic behavior but with unstable reward curves.
* **PPO**: Most reliable learning; smooth upward trend in moving-average reward; best overall performance.

Plots are available under:

```
results/qlearning/
results/ppo/
reports/project_check_in/
```

## Future Improvements

* Improve PPO CNN architecture
* Finish Dueling DQN implementation
* Add reward scaling (log rewards)
* Add full UI (Streamlit or Next.js) for interactive model demos
* Add more robust experiment management and logging

## Contributors

* Joanne Lai
* Ann Le
* Grace Zhang
* Haylie Tan
* Sandeep Reehal
* Keren Skariah
* Shreyans Porwal
* Ian Wong
* Sathvik Parasa