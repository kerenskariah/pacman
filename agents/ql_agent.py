import numpy as np
import pickle
from collections import defaultdict
from typing import Tuple, Dict, Any
from agents.base import BaseAgent
from config.base import BaseConfig


class RamFeaturizer:
    
    def __init__(self, stride: int = 16, bins: int = 8):
        self.stride = stride
        self.bins = bins
        self.bin_edges = np.linspace(0, 256, num=bins + 1, endpoint=True)
    
    def __call__(self, obs: np.ndarray) -> Tuple[int, ...]:
        if obs.ndim == 3:
            # If we get an RGB image instead of RAM, throw error
            raise ValueError("Expected RAM observation (1D), got image observation (3D). "
                           "Use ALE/MsPacman-ram-v5 or similar RAM variant.")
        
        # Select bytes at stride intervals
        selected = obs[::self.stride].astype(np.int32)
        
        # Discretize into bins
        binned = np.digitize(selected, self.bin_edges, right=False) - 1
        binned = np.clip(binned, 0, self.bins - 1)
        
        return tuple(binned.tolist())


class QLearningAgent(BaseAgent):
    
    def __init__(self, action_space, config: BaseConfig):
        super().__init__(action_space, config)
        
        # Q-table: dict mapping (state, action) -> Q-value
        self.q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        
        # Hyperparams
        self.alpha = getattr(config, 'ALPHA', 0.1)  # learning rate
        self.gamma = getattr(config, 'GAMMA', 0.99)  # discount factor
        self.epsilon_start = getattr(config, 'EPSILON_START', 1.0)
        self.epsilon_end = getattr(config, 'EPSILON_END', 0.01)
        self.epsilon_decay = getattr(config, 'EPSILON_DECAY', 0.995)
        
        self.epsilon = self.epsilon_start
        self.n_actions = action_space.n
        
        # State featurizer
        stride = getattr(config, 'RAM_STRIDE', 16)
        bins = getattr(config, 'RAM_BINS', 8)
        self.featurizer = RamFeaturizer(stride=stride, bins=bins)
        
        # Training statistics
        self.episode_count = 0
        self.step_count = 0
        
        # RNG
        self.rng = np.random.default_rng(getattr(config, 'SEED', 42)) 
    
    def get_action(self, observation) -> int:
        state = self.featurizer(observation)
        
        # Epsilon-greedy
        if self.rng.random() < self.epsilon:
            # Explore: random action
            return int(self.rng.integers(0, self.n_actions))
        else:
            # Best action from Q-table
            q_values = self.q_table[state]
            
            if not q_values:
                # State never seen, return random action
                return int(self.rng.integers(0, self.n_actions))
            
            # Get action with max Q-value (break ties randomly)
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return int(self.rng.choice(best_actions))
    
    def update(self, state, action: int, reward: float, next_state, needs_to_reset: bool):
        self.step_count += 1
        
        # Convert observations to state keys
        s = self.featurizer(state)
        
        # Q-learning update
        current_q = self.q_table[s][action]
        
        if needs_to_reset:
            # Terminal state: target is just the reward
            target = reward
        else:
            # Non-terminal: target is r + gamma * max Q(s', a')
            s_next = self.featurizer(next_state)
            next_q_values = self.q_table[s_next]
            
            if next_q_values:
                max_next_q = max(next_q_values.values())
            else:
                max_next_q = 0.0
            
            target = reward + self.gamma * max_next_q
        
        # Update Q-value
        self.q_table[s][action] = current_q + self.alpha * (target - current_q)
        
        # Decay epsilon at episode end
        if needs_to_reset:
            self.episode_count += 1
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        # Convert defaultdict to regular dict for pickling
        q_table_serializable = {
            state: dict(actions) for state, actions in self.q_table.items()
        }
        
        save_data = {
            'q_table': q_table_serializable,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'config': {
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon_start': self.epsilon_start,
                'epsilon_end': self.epsilon_end,
                'epsilon_decay': self.epsilon_decay,
                'n_actions': self.n_actions,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)

    # In-memory export/import for best-model selection
    def state_dict(self) -> Dict[str, Any]:
        q_table_serializable = {state: dict(actions) for state, actions in self.q_table.items()}
        return {
            'q_table': q_table_serializable,
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
        }

    def load_state(self, state: Dict[str, Any]):
        q_table_loaded = state.get('q_table', {})
        self.q_table = defaultdict(lambda: defaultdict(float))
        for s, actions in q_table_loaded.items():
            for a, v in actions.items():
                self.q_table[s][a] = v
        self.epsilon = state.get('epsilon', self.epsilon)
        self.episode_count = state.get('episode_count', self.episode_count)
        self.step_count = state.get('step_count', self.step_count)
    
    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore Q-table
        q_table_loaded = save_data['q_table']
        self.q_table = defaultdict(lambda: defaultdict(float))
        for state, actions in q_table_loaded.items():
            for action, value in actions.items():
                self.q_table[state][action] = value
        
        # Restore training state
        self.epsilon = save_data.get('epsilon', self.epsilon_end)
        self.episode_count = save_data.get('episode_count', 0)
        self.step_count = save_data.get('step_count', 0)
        
        # Set epsilon to 0 for evaluation (greedy policy)
        self.epsilon = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'q_table_states': len(self.q_table),
            'q_table_entries': sum(len(actions) for actions in self.q_table.values()),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'step_count': self.step_count,
        }
