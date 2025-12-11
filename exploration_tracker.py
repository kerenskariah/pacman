"""
Exploration Tracking for PPO Training
Tracks multiple exploration metrics to understand agent behavior and learning progress.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class ExplorationTracker:
    """
    Comprehensive exploration tracking for RL agents.
    
    Most Important Metrics:
    1. Policy Entropy - Direct measure of action randomness
    2. Action Distribution - Shows if agent has biases/gets stuck
    3. State Visitation Diversity - Approximate measure of exploration breadth
    4. Advantage Statistics - Shows value function quality
    """
    
    def __init__(self, num_actions: int, num_envs: int, tracking_window: int = 1000):
        self.num_actions = num_actions
        self.num_envs = num_envs
        self.tracking_window = tracking_window
        
        # Action distribution tracking
        self.action_counts = np.zeros(num_actions, dtype=np.int64)
        self.action_counts_window = np.zeros(num_actions, dtype=np.int64)
        self.action_history = deque(maxlen=tracking_window)
        
        # Entropy tracking (per update)
        self.entropy_history = []
        
        # State diversity tracking (approximate)
        self.state_hashes = set()
        self.state_hash_history = deque(maxlen=tracking_window * num_envs)
        
        # Advantage statistics
        self.advantage_stats = {
            'mean': [],
            'std': [],
            'min': [],
            'max': [],
            'positive_frac': []
        }
        
        # Value prediction quality
        self.value_stats = {
            'explained_variance': [],
            'value_mean': [],
            'return_mean': []
        }
        
        # Per-action entropy (shows which actions have uncertain policy)
        self.per_action_entropy = {i: [] for i in range(num_actions)}
        
        # Unique states per episode
        self.states_per_episode = []
        self.current_episode_states = set()
        
    def update_action_distribution(self, actions: np.ndarray):
        """Track which actions are being taken."""
        unique, counts = np.unique(actions, return_counts=True)
        for action, count in zip(unique, counts):
            self.action_counts[action] += count
            self.action_counts_window[action] += count
            
        for action in actions:
            self.action_history.append(action)
            
        # Reset window if it's full
        if len(self.action_history) >= self.tracking_window:
            self.action_counts_window = np.zeros(self.num_actions, dtype=np.int64)
            for a in self.action_history:
                self.action_counts_window[a] += 1
    
    def update_entropy(self, entropy: float):
        """Track policy entropy over time."""
        self.entropy_history.append(entropy)
    
    def update_per_action_entropy(self, action_probs: torch.Tensor):
        """
        Track entropy for each action's probability distribution.
        Shows which actions the policy is most uncertain about.
        
        Args:
            action_probs: [batch_size, num_actions] probability distribution
        """
        # Calculate entropy contribution per action
        # H = -sum(p * log(p))
        log_probs = torch.log(action_probs + 1e-8)
        entropy_per_action = -(action_probs * log_probs).mean(dim=0)  # [num_actions]
        
        for i, ent in enumerate(entropy_per_action):
            self.per_action_entropy[i].append(ent.item())
    
    def update_state_diversity(self, observations: np.ndarray):
        """
        Track approximate state diversity using hashing.
        More unique states = better exploration.
        
        Args:
            observations: [batch_size, ...] state observations
        """
        # Hash observations to track unique states
        # Use mean pooling to create a signature
        for obs in observations:
            # Create a simple hash from the observation
            obs_flat = obs.flatten()
            # Sample key positions to create hash (for efficiency)
            sample_idx = np.linspace(0, len(obs_flat)-1, 16, dtype=int)
            obs_signature = tuple(np.round(obs_flat[sample_idx], decimals=2))
            
            # Track global diversity
            self.state_hashes.add(obs_signature)
            self.state_hash_history.append(obs_signature)
            
            # Track per-episode diversity
            self.current_episode_states.add(obs_signature)
    
    def on_episode_end(self):
        """Called when an episode ends to track per-episode exploration."""
        self.states_per_episode.append(len(self.current_episode_states))
        self.current_episode_states = set()
    
    def update_advantage_stats(self, advantages: torch.Tensor):
        """
        Track advantage statistics.
        High variance = value function struggling to predict
        Mostly positive/negative = biased estimates
        
        Args:
            advantages: [batch_size] advantage estimates
        """
        adv_np = advantages.detach().cpu().numpy()
        
        self.advantage_stats['mean'].append(float(np.mean(adv_np)))
        self.advantage_stats['std'].append(float(np.std(adv_np)))
        self.advantage_stats['min'].append(float(np.min(adv_np)))
        self.advantage_stats['max'].append(float(np.max(adv_np)))
        self.advantage_stats['positive_frac'].append(float(np.mean(adv_np > 0)))
    
    def update_value_stats(self, values: torch.Tensor, returns: torch.Tensor):
        """
        Track value function quality.
        Explained variance close to 1.0 = good value predictions
        
        Args:
            values: [batch_size] predicted values
            returns: [batch_size] actual returns
        """
        values_np = values.detach().cpu().numpy()
        returns_np = returns.detach().cpu().numpy()
        
        # Explained variance: 1 - Var(returns - values) / Var(returns)
        var_returns = np.var(returns_np)
        if var_returns > 1e-8:
            explained_var = 1.0 - np.var(returns_np - values_np) / var_returns
            explained_var = np.clip(explained_var, -1.0, 1.0)  # Can be negative if bad
        else:
            explained_var = 0.0
        
        self.value_stats['explained_variance'].append(float(explained_var))
        self.value_stats['value_mean'].append(float(np.mean(values_np)))
        self.value_stats['return_mean'].append(float(np.mean(returns_np)))
    
    def get_action_distribution(self, window: bool = False) -> Dict[int, float]:
        """
        Get current action distribution as percentages.
        
        Args:
            window: If True, use recent window; if False, use all history
        """
        counts = self.action_counts_window if window else self.action_counts
        total = counts.sum()
        if total == 0:
            return {i: 0.0 for i in range(self.num_actions)}
        return {i: float(counts[i] / total) for i in range(self.num_actions)}
    
    def get_action_diversity_score(self, window: bool = True) -> float:
        """
        Calculate action diversity score (entropy of action distribution).
        
        Returns:
            Float from 0 (deterministic, always same action) to log(num_actions) (uniform)
        """
        dist = self.get_action_distribution(window=window)
        probs = np.array([dist[i] for i in range(self.num_actions)])
        probs = probs[probs > 0]  # Remove zeros for log
        if len(probs) == 0:
            return 0.0
        return float(-np.sum(probs * np.log(probs)))
    
    def get_state_diversity_score(self, window: bool = True) -> int:
        """
        Get number of unique states visited.
        
        Args:
            window: If True, use recent window; if False, use all history
        """
        if window:
            return len(set(self.state_hash_history))
        return len(self.state_hashes)
    
    def get_entropy_trend(self, window: int = 100) -> Optional[float]:
        """
        Calculate trend in entropy (positive = increasing exploration).
        
        Returns:
            Slope of linear fit to recent entropy values
        """
        if len(self.entropy_history) < window:
            return None
        
        recent = self.entropy_history[-window:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return float(slope)
    
    def get_summary_stats(self) -> Dict:
        """Get comprehensive summary of exploration metrics."""
        stats = {
            # Action distribution
            'action_distribution': self.get_action_distribution(window=True),
            'action_diversity_score': self.get_action_diversity_score(window=True),
            'most_common_action': int(np.argmax(self.action_counts_window)),
            'least_common_action': int(np.argmin(self.action_counts_window)),
            
            # Entropy
            'current_entropy': self.entropy_history[-1] if self.entropy_history else 0.0,
            'avg_entropy_last_100': float(np.mean(self.entropy_history[-100:])) if len(self.entropy_history) >= 100 else 0.0,
            'entropy_trend': self.get_entropy_trend(),
            
            # State diversity
            'unique_states_window': self.get_state_diversity_score(window=True),
            'unique_states_total': self.get_state_diversity_score(window=False),
            'avg_states_per_episode': float(np.mean(self.states_per_episode[-100:])) if len(self.states_per_episode) >= 100 else 0.0,
            
            # Advantages
            'avg_advantage': float(np.mean(self.advantage_stats['mean'][-100:])) if len(self.advantage_stats['mean']) >= 100 else 0.0,
            'advantage_std': float(np.mean(self.advantage_stats['std'][-100:])) if len(self.advantage_stats['std']) >= 100 else 0.0,
            'positive_advantage_frac': float(np.mean(self.advantage_stats['positive_frac'][-100:])) if len(self.advantage_stats['positive_frac']) >= 100 else 0.0,
            
            # Value function
            'explained_variance': float(np.mean(self.value_stats['explained_variance'][-100:])) if len(self.value_stats['explained_variance']) >= 100 else 0.0,
        }
        
        return stats
    
    def log_summary(self, update: int, logger_obj: Optional[logging.Logger] = None):
        """Log exploration summary."""
        if logger_obj is None:
            logger_obj = logger
        
        stats = self.get_summary_stats()
        
        logger_obj.info(f"\n{'='*60}")
        logger_obj.info(f"EXPLORATION SUMMARY - Update {update}")
        logger_obj.info(f"{'='*60}")
        
        # Action distribution
        logger_obj.info("\nAction Distribution (recent):")
        action_dist = stats['action_distribution']
        for action, prob in action_dist.items():
            logger_obj.info(f"  Action {action}: {prob*100:.1f}%")
        logger_obj.info(f"Action Diversity Score: {stats['action_diversity_score']:.3f} (max={np.log(self.num_actions):.3f})")
        
        # Entropy
        logger_obj.info(f"\nPolicy Entropy:")
        logger_obj.info(f"  Current: {stats['current_entropy']:.3f}")
        logger_obj.info(f"  Avg (last 100): {stats['avg_entropy_last_100']:.3f}")
        if stats['entropy_trend'] is not None:
            trend_str = "increasing" if stats['entropy_trend'] > 0 else "decreasing"
            logger_obj.info(f"  Trend: {trend_str} ({stats['entropy_trend']:.6f} per update)")
        
        # State diversity
        logger_obj.info(f"\nState Diversity:")
        logger_obj.info(f"  Unique states (window): {stats['unique_states_window']}")
        logger_obj.info(f"  Unique states (total): {stats['unique_states_total']}")
        logger_obj.info(f"  Avg per episode: {stats['avg_states_per_episode']:.1f}")
        
        # Value function quality
        logger_obj.info(f"\nValue Function Quality:")
        logger_obj.info(f"  Explained Variance: {stats['explained_variance']:.3f}")
        logger_obj.info(f"  Advantage Std: {stats['advantage_std']:.3f}")
        logger_obj.info(f"  Positive Advantage %: {stats['positive_advantage_frac']*100:.1f}%")
        
        logger_obj.info(f"{'='*60}\n")
    
    def save_to_dict(self) -> Dict:
        """Save all tracking data to dictionary for checkpointing."""
        return {
            'action_counts': self.action_counts.tolist(),
            'action_counts_window': self.action_counts_window.tolist(),
            'action_history': list(self.action_history),
            'entropy_history': self.entropy_history,
            'advantage_stats': self.advantage_stats,
            'value_stats': self.value_stats,
            'states_per_episode': self.states_per_episode,
            'unique_states_total': len(self.state_hashes),
            'per_action_entropy': {k: v for k, v in self.per_action_entropy.items()},
        }
    
    def load_from_dict(self, data: Dict):
        """Load tracking data from dictionary."""
        self.action_counts = np.array(data['action_counts'], dtype=np.int64)
        self.action_counts_window = np.array(data['action_counts_window'], dtype=np.int64)
        self.action_history = deque(data['action_history'], maxlen=self.tracking_window)
        self.entropy_history = data['entropy_history']
        self.advantage_stats = data['advantage_stats']
        self.value_stats = data['value_stats']
        self.states_per_episode = data['states_per_episode']
        if 'per_action_entropy' in data:
            self.per_action_entropy = {int(k): v for k, v in data['per_action_entropy'].items()}


def calculate_action_entropy_per_action(logits: torch.Tensor) -> torch.Tensor:
    """
    Helper function to calculate per-action entropy from policy logits.
    
    Args:
        logits: [batch_size, num_actions] raw policy outputs
        
    Returns:
        [num_actions] entropy contribution per action
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy_per_action = -(probs * log_probs).mean(dim=0)
    return entropy_per_action
