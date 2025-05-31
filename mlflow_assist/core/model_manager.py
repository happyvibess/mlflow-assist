"""
Core functionality for ML model management.
"""

from typing import Any, Dict, Optional, Union
import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.base import BaseEstimator
import torch

class ModelConfig(BaseModel):
    """Model configuration settings."""
    name: str
    version: Optional[str] = "1.0.0"
    framework: str = "sklearn"
    hyperparameters: Dict[str, Any] = {}

class ModelManager:
    """Handles ML model operations including training, evaluation, and deployment."""
    
    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize ModelManager with optional MLflow tracking URI."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.active_experiment = None

    def train(
        self,
        model: Union[BaseEstimator, torch.nn.Module],
        config: ModelConfig,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        **kwargs
    ) -> Any:
        """
        Train a model with automatic tracking and logging.
        
        Args:
            model: The model to train
            config: Model configuration
            X_train: Training features
            y_train: Training labels
            **kwargs: Additional training parameters
        
        Returns:
            Trained model
        """
        with mlflow.start_run() as run:
            # Log model configuration
            mlflow.log_params(config.hyperparameters)
            
            # Train based on framework
            if config.framework == "pytorch":
                self._train_pytorch(model, X_train, y_train, **kwargs)
            else:
                model.fit(X_train, y_train)
            
            # Log the model
            mlflow.sklearn.log_model(model, "model")
            
            return model

    def deploy(
        self,
        model_name: str,
        stage: str = "production",
        version: Optional[str] = None
    ) -> None:
        """
        Deploy a model to the specified stage.
        
        Args:
            model_name: Name of the model to deploy
            stage: Target stage (production, staging, etc.)
            version: Specific model version to deploy
        """
        client = mlflow.tracking.MlflowClient()
        
        if version:
            model_version = client.get_model_version(model_name, version)
        else:
            # Get latest version
            versions = client.get_latest_versions(model_name)
            model_version = versions[0]
        
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )

    def _train_pytorch(
        self,
        model: torch.nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        **kwargs
    ) -> torch.nn.Module:
        """
        Internal method for PyTorch model training.
        
        Args:
            model: PyTorch model
            X_train: Training features
            y_train: Training labels
            **kwargs: Training parameters
        
        Returns:
            Trained PyTorch model
        """
        epochs = kwargs.get("epochs", 10)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            mlflow.log_metric("loss", loss.item(), step=epoch)
        
        return model

