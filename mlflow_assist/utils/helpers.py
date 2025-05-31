"""
Utility functions for MLFlow-Assist.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Path to log file
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger("mlflow_assist")
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root() -> Path:
    """
    Get the root directory of the current project.
    
    Returns:
        Path to project root
    """
    current = Path.cwd()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()

def validate_model_path(model_path: Union[str, Path]) -> bool:
    """
    Validate if a model path exists and contains required files.
    
    Args:
        model_path: Path to model directory
    
    Returns:
        True if valid, False otherwise
    """
    path = Path(model_path)
    required_files = ["model.pkl", "config.yaml"]
    return path.exists() and all((path / file).exists() for file in required_files)

def create_experiment_name(
    base_name: str,
    version: Optional[str] = None
) -> str:
    """
    Create a standardized experiment name.
    
    Args:
        base_name: Base name for the experiment
        version: Optional version string
    
    Returns:
        Formatted experiment name
    """
    if version:
        return f"{base_name}_v{version}"
    return base_name

