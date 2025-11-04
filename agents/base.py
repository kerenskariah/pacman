from abc import ABC, abstractmethod

class BaseAgent(ABC):    
    def __init__(self, action_space):
        self.action_space = action_space
    
    @abstractmethod
    def get_action(self, observation):
        """Select an action given an observation."""
        pass
    
    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Update agent based on experience."""
        pass
    
    @abstractmethod
    def save(self, filepath):
        """Save agent to disk."""
        pass
    
    @abstractmethod
    def load(self, filepath):
        """Load agent from disk."""
        pass