"""
Tests for utility functions.
"""

import os
import pytest
from pathlib import Path
import tempfile
import yaml

from mlflow_assist.utils.helpers import (
    setup_logging,
    load_config,
    save_config,
    ensure_dir,
    get_project_root,
    validate_model_path,
    create_experiment_name
)

def test_setup_logging():
    """Test logging setup."""
    logger = setup_logging()
    assert logger.name == "mlflow_assist"
    assert logger.level == 20  # INFO level

def test_config_operations():
    """Test configuration loading and saving."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
        config = {
            "test": True,
            "nested": {"value": 42}
        }
        
        # Test saving
        save_config(config, tmp.name)
        
        # Test loading
        loaded = load_config(tmp.name)
        assert loaded == config
        
        os.unlink(tmp.name)

def test_ensure_dir():
    """Test directory creation."""
    with tempfile.TemporaryDirectory() as tmp:
        test_dir = Path(tmp) / "test_dir"
        result = ensure_dir(test_dir)
        assert result.exists()
        assert result.is_dir()

def test_create_experiment_name():
    """Test experiment name creation."""
    base_name = "test_experiment"
    
    # Without version
    name = create_experiment_name(base_name)
    assert name == base_name
    
    # With version
    name = create_experiment_name(base_name, "v1")
    assert name == "test_experiment_v1"

def test_validate_model_path():
    """Test model path validation."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        
        # Create required files
        (tmp_path / "model.pkl").touch()
        (tmp_path / "config.yaml").touch()
        
        assert validate_model_path(tmp_path) is True
        assert validate_model_path(tmp_path / "nonexistent") is False

