import typing as t
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base import BaseAgent, AgentConfig

class RLConfig(AgentConfig):
    """Configuration for RL agents"""
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        learning_rate: float = 3e-4,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        **kwargs
    ):
        super().__init__(
            model_type="rl",
            input_size=state_size,
            output_size=action_size,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            **kwargs
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

class ActorCritic(nn.Module):
    """Combined actor-critic network"""
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(hidden_size, action_size)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(x)
        action_probs = torch.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value

class RLAgent(BaseAgent):
    """Reinforcement Learning agent using PPO"""
    
    def __init__(self, config: RLConfig):
        super().__init__(config)
        self.config = config
        
    def _build_model(self) -> nn.Module:
        return ActorCritic(
            self.config.input_size,
            self.config.output_size,
            self.config.hidden_size
        ).to(self.config.device)
        
    def train_step(self, batch: dict) -> dict:
        """Perform one PPO update"""
        states = torch.FloatTensor(batch['states']).to(self.config.device)
        actions = torch.LongTensor(batch['actions']).to(self.config.device)
        old_probs = torch.FloatTensor(batch['old_probs']).to(self.config.device)
        advantages = torch.FloatTensor(batch['advantages']).to(self.config.device)
        returns = torch.FloatTensor(batch['returns']).to(self.config.device)
        
        # Forward pass
        action_probs, values = self.model(states)
        dist = Categorical(action_probs)
        
        # Calculate losses
        ratio = torch.exp(dist.log_prob(actions) - torch.log(old_probs))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.config.clip_epsilon, 1+self.config.clip_epsilon) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = 0.5 * (values - returns).pow(2).mean()
        entropy_loss = -dist.entropy().mean()
        
        total_loss = (
            actor_loss +
            self.config.value_loss_coef * critic_loss +
            self.config.entropy_coef * entropy_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'loss': total_loss.item(),
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy_loss.item()
        }
        
    def predict(self, state: np.ndarray) -> int:
        """Select an action for the given state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.config.device)
        
        with torch.no_grad():
            action_probs, _ = self.model(state)
            dist = Categorical(action_probs)
            action = dist.sample()
            
        return action.item()

