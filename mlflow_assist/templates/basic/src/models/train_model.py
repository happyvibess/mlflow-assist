"""
Model training functions.
"""

import mlflow
from sklearn.base import BaseEstimator
import pandas as pd
from typing import Optional

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Optional[BaseEstimator] = None,
    **params
) -> BaseEstimator:
    """
    Train a machine learning model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model: Sklearn-compatible model instance
        **params: Model parameters
    
    Returns:
        Trained model
    """
    if model is None:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
    
    # Log parameters and metrics with MLflow
    with mlflow.start_run():
        mlflow.log_params(params)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Log metrics
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_score", train_score)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
    
    return model

