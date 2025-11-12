#!/bin/bash
# Script to play 5 episodes with trained PPO agent
python3 test.py --agent ppo --mode play --model-path models/ppo/agent_latest.pt --episodes 5