"""
MLFlow-Assist: A comprehensive toolkit for ML and LLM development.
"""

from mlflow_assist.core.model_manager import ModelManager
from mlflow_assist.llm.llm_handler import LLMHandler
from mlflow_assist.utils.helpers import setup_logging

__version__ = "0.1.0"

# Configure logging
logger = setup_logging()

__all__ = ["ModelManager", "LLMHandler", "logger"]

