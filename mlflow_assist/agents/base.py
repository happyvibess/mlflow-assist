from abc import ABC, abstractmethod
import typing as t
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch

@dataclass
class AgentConfig:
    """Base configuration for agents"""
    model_type: str
    input_size: int
    output_size: int
    hidden_size: int = 256
    learning_rate: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: Path = Path("checkpoints")
    
class BaseAgent(ABC):
    """Base class for all trainable agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        
    @abstractmethod
    def _build_model(self) -> torch.nn.Module:
        """Build the core model architecture"""
        pass
        
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer"""
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
    @abstractmethod
    def train_step(self, batch: t.Any) -> dict:
        """Perform one training step"""
        pass
        
    @abstractmethod
    def predict(self, state: t.Any) -> t.Any:
        """Make a prediction given a state"""
        pass
        
    def save(self, path: Path) -> None:
        """Save agent state"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config
        }
        torch.save(checkpoint, path)
        
    def load(self, path: Path) -> None:
        """Load agent state"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.config = checkpoint['config']
        
    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics to MLflow"""
        mlflow.log_metrics(metrics, step=step)

