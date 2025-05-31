"""
Command-line interface for MLFlow-Assist.
"""

import os
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from mlflow_assist.core.model_manager import ModelManager
from mlflow_assist.llm.llm_handler import LLMHandler
from mlflow_assist.utils.helpers import setup_logging

app = typer.Typer(help="MLFlow-Assist CLI for ML and LLM operations")
console = Console()
logger = setup_logging()

@app.command()
def init(
    project_name: str,
    template: str = "basic",
    path: Optional[str] = None
):
    """
    Initialize a new ML/LLM project with recommended structure.
    
    Args:
        project_name: Name of the project
        template: Project template to use
        path: Project directory path
    """
    project_path = Path(path or ".") / project_name
    
    # Create project structure
    folders = [
        "data/raw",
        "data/processed",
        "models",
        "notebooks",
        "src",
        "config",
    ]
    
    try:
        for folder in folders:
            (project_path / folder).mkdir(parents=True, exist_ok=True)
        
        # Create initial files
        (project_path / "README.md").write_text(
            f"# {project_name}\n\nML/LLM project created with MLFlow-Assist"
        )
        
        (project_path / "requirements.txt").write_text(
            "mlflow-assist\nmlflow\npandas\nscikit-learn\ntorch\ntransformers"
        )
        
        console.print(f"✨ Created new project: {project_name}", style="bold green")
    except Exception as e:
        logger.error(f"Failed to initialize project: {e}")
        raise typer.Exit(1)

@app.command()
def train(
    model_name: str,
    data_path: str,
    config_path: Optional[str] = None,
    output_path: Optional[str] = None
):
    """
    Train a model using the specified configuration.
    
    Args:
        model_name: Name of the model
        data_path: Path to training data
        config_path: Path to model configuration
        output_path: Path to save the trained model
    """
    try:
        manager = ModelManager()
        # Implementation details would depend on data format and model type
        console.print(f"🚀 Training model: {model_name}", style="bold blue")
        # Add training logic here
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(1)

@app.command()
def deploy(
    model_name: str,
    stage: str = "production",
    version: Optional[str] = None
):
    """
    Deploy a trained model to the specified stage.
    
    Args:
        model_name: Name of the model to deploy
        stage: Deployment stage
        version: Model version
    """
    try:
        manager = ModelManager()
        manager.deploy(model_name, stage, version)
        console.print(
            f"✅ Deployed {model_name} to {stage}",
            style="bold green"
        )
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise typer.Exit(1)

@app.command()
def generate(
    prompt: str,
    model: str = "gpt-3.5-turbo",
    max_length: int = 512
):
    """
    Generate text using an LLM.
    
    Args:
        prompt: Input prompt for generation
        model: Name of the LLM to use
        max_length: Maximum length of generated text
    """
    try:
        handler = LLMHandler()
        response = handler.generate(prompt, max_length=max_length)
        console.print("Generated text:", style="bold blue")
        console.print(response)
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise typer.Exit(1)

@app.command()
def list_models():
    """List all available models and their status."""
    try:
        # Create a table
        table = Table(title="Available Models")
        table.add_column("Name", style="cyan")
        table.add_column("Version", style="magenta")
        table.add_column("Stage", style="green")
        table.add_column("Status", style="yellow")
        
        # Add implementation to fetch and display models
        console.print(table)
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()

