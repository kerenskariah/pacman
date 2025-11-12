from agents.base import BaseAgent
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from typing import Tuple


class ACNetCNN(nn.Module):
    def __init__(self, action_dim: int, use_cnn: bool = False):
        super().__init__()
        self.use_cnn = use_cnn
        
        if use_cnn:
            # CNN for image inputs (4, 84, 84)
            self.conv = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            )
            self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512), nn.ReLU())
            self.pi = nn.Linear(512, action_dim)
            self.v = nn.Linear(512, 1)
            
            # Orthogonal initialization
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                    nn.init.constant_(m.bias, 0)
            nn.init.orthogonal_(self.pi.weight, gain=0.01)
            nn.init.constant_(m.bias, 0)
        else:
            # Simple MLP for flattened inputs
            self.shared = nn.Sequential(
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU()
            )
            self.pi = nn.Linear(128, action_dim)
            self.v = nn.Linear(128, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_cnn:
            if x.dtype != torch.float32:
                x = x.float()
            x = x / 255.0
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = self.shared(x)
        
        logits = self.pi(x)
        value = self.v(x).squeeze(-1)
        return logits, value


class PPOAgent(BaseAgent):
    def __init__(self, action_space, config):
        self.action_space = action_space
        self.action_size = action_space.n
        self.config = config
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Network
        self.network = ACNetCNN(self.action_size, use_cnn=config.USE_CNN).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.LR, eps=1e-5)
        
        # Memory for collecting rollout data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # Metrics
        self.episode_count = 0
        self.update_count = 0
    
    def _preprocess_state(self, state):
        if self.config.USE_CNN:
            # Expecting (4, 84, 84) from FrameStack
            if len(state.shape) == 3:
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            else:
                state = torch.FloatTensor(state).to(self.device)
        else:
            # MLP path: flatten and normalize
            if isinstance(state, np.ndarray):
                state = state.flatten()
            else:
                state = np.array(state).flatten()
            
            # Normalize to [0, 1]
            state = state.astype(np.float32) / 255.0
            
            # Pad or truncate to fixed size
            if len(state) < 128:
                state = np.pad(state, (0, 128 - len(state)))
            elif len(state) > 128:
                state = state[:128]
            
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        return state
    
    def get_action(self, observation):
        state = self._preprocess_state(observation)
        
        with torch.no_grad():
            logits, value = self.network(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        if self.config.USE_CNN:
            self.states.append(observation)  # Store raw observation
        else:
            self.states.append(state.cpu().numpy()[0])
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        
        return action.item()
    
    def update(self, state, action, reward, next_state, done):
        self.rewards.append(reward)
        self.dones.append(done)
        
        if done or len(self.states) >= self.config.BATCH_SIZE:
            if done:
                self.episode_count += 1
            
            # Only train if we have enough data
            if len(self.states) >= self.config.MIN_BATCH_SIZE:
                loss_info = self._train()
                
                # Log periodically
                if self.episode_count % self.config.LOG_INTERVAL == 0:
                    print(f"Episode {self.episode_count}, Update {self.update_count}, "
                          f"Loss: {loss_info['total_loss']:.4f}, "
                          f"Policy Loss: {loss_info['policy_loss']:.4f}, "
                          f"Value Loss: {loss_info['value_loss']:.4f}")
            
            # Clear memory
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.log_probs.clear()
            self.values.clear()
            self.dones.clear()
    
    def _compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        
        # Append next_value for bootstrapping
        values = values + [next_value]
        
        # Compute advantages in reverse
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.config.GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.config.GAMMA * self.config.GAE_LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        # Returns = advantages + values
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def _train(self):
        self.update_count += 1
        
        # Get next state value for bootstrapping
        if len(self.states) > 0:
            last_state = self._preprocess_state(self.states[-1])
            with torch.no_grad():
                _, next_value = self.network(last_state)
                next_value = next_value.item()
        else:
            next_value = 0
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(
            self.rewards, self.values, self.dones, next_value
        )
        
        # Convert to tensors
        if self.config.USE_CNN:
            # Stack images: (B, 4, 84, 84)
            states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        else:
            states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device)
        
        actions_tensor = torch.LongTensor(self.actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(self.log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # PPO update for multiple epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.config.PPO_EPOCHS):
            # Forward pass
            logits, values = self.network(states_tensor)
            dist = Categorical(logits=logits)
            
            # Calculate losses
            log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy().mean()
            
            # Policy loss with clipping
            ratio = torch.exp(log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.config.CLIP_EPS, 1 + self.config.CLIP_EPS) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss with clipping
            value_pred_clipped = self.values[0] + torch.clamp(
                values - self.values[0], -self.config.VALUE_CLIP, self.config.VALUE_CLIP
            )
            value_loss_unclipped = (values - returns_tensor) ** 2
            value_loss_clipped = (value_pred_clipped - returns_tensor) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            
            # Total loss
            loss = policy_loss + self.config.VALUE_COEF * value_loss - self.config.ENTROPY_COEF * entropy
            
            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        return {
            'total_loss': loss.item(),
            'policy_loss': total_policy_loss / self.config.PPO_EPOCHS,
            'value_loss': total_value_loss / self.config.PPO_EPOCHS,
            'entropy': total_entropy / self.config.PPO_EPOCHS
        }
    
    def save(self, filepath):
        """Save model weights."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'update_count': self.update_count
        }, filepath)
    
    def load(self, filepath):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.update_count = checkpoint.get('update_count', 0)

