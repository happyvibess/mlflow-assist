"""
Tests for the ModelManager class.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from mlflow_assist.core.model_manager import ModelManager, ModelConfig

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y

@pytest.fixture
def model_manager():
    """Create a ModelManager instance for testing."""
    return ModelManager()

def test_model_training(model_manager, sample_data):
    """Test basic model training functionality."""
    X, y = sample_data
    model = LogisticRegression()
    config = ModelConfig(
        name="test_model",
        framework="sklearn",
        hyperparameters={"max_iter": 100}
    )
    
    trained_model = model_manager.train(model, config, X, y)
    assert trained_model is not None
    assert hasattr(trained_model, "predict")
    
    # Test prediction
    predictions = trained_model.predict(X)
    assert len(predictions) == len(y)

