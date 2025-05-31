import typing as t
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from .base import BaseAgent, AgentConfig

class AgentTrainer:
    """Handles the training loop and metrics tracking for agents"""
    
    def __init__(
        self,
        agent: BaseAgent,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        experiment_name: str = "agent_training",
        checkpoint_dir: Path = Path("checkpoints")
    ):
        self.agent = agent
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        mlflow.set_experiment(experiment_name)
        
    def train(
        self,
        num_epochs: int,
        eval_every: int = 1,
        save_every: int = 5,
        early_stopping_patience: int = 5
    ) -> None:
        """Train the agent"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        with mlflow.start_run():
            mlflow.log_params(self.agent.config.__dict__)
            
            for epoch in range(num_epochs):
                # Training
                train_metrics = self._train_epoch()
                self.agent.log_metrics(
                    {f"train_{k}": v for k, v in train_metrics.items()},
                    step=epoch
                )
                
                # Validation
                if self.val_loader and epoch % eval_every == 0:
                    val_metrics = self._validate()
                    self.agent.log_metrics(
                        {f"val_{k}": v for k, v in val_metrics.items()},
                        step=epoch
                    )
                    
                    # Early stopping
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        patience_counter = 0
                        self._save_checkpoint("best.pt")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                            break
                            
                # Regular checkpointing
                if epoch % save_every == 0:
                    self._save_checkpoint(f"epoch_{epoch}.pt")
                    
    def _train_epoch(self) -> dict:
        """Run one epoch of training"""
        self.agent.model.train()
        total_metrics = {}
        num_batches = 0
        
        for batch in self.train_loader:
            metrics = self.agent.train_step(batch)
            
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            num_batches += 1
            
        return {k: v / num_batches for k, v in total_metrics.items()}
        
    def _validate(self) -> dict:
        """Run validation"""
        self.agent.model.eval()
        total_metrics = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = self.agent.train_step(batch)
                
                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0) + v
                num_batches += 1
                
        return {k: v / num_batches for k, v in total_metrics.items()}
        
    def _save_checkpoint(self, filename: str) -> None:
        """Save a checkpoint"""
        path = self.checkpoint_dir / filename
        self.agent.save(path)
        mlflow.log_artifact(str(path))

