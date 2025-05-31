"""
Configuration management utilities.
"""

import os
from pathlib import Path
import yaml
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root
    """
    return Path(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

