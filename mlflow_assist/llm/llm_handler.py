"""
Handler for LLM operations and interactions.
"""

from typing import Any, Dict, List, Optional, Union
import os
from pydantic import BaseModel
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline

class LLMConfig(BaseModel):
    """Configuration for LLM operations."""
    model_name: str
    max_length: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    api_key: Optional[str] = None

class LLMHandler:
    """Handles LLM operations including text generation and model management."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM handler with optional configuration.
        
        Args:
            config: LLM configuration settings
        """
        self.config = config or LLMConfig(model_name="gpt-3.5-turbo")
        self.model = None
        self.tokenizer = None
        
        # Load API key from environment if not provided
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY")

    def load_model(self, model_name: Optional[str] = None) -> None:
        """
        Load a local transformer model.
        
        Args:
            model_name: Name of the model to load
        """
        model_name = model_name or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the configured LLM.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Temperature for text generation
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        # Use API for specific models
        if self.config.model_name.startswith("gpt"):
            return self._generate_api(prompt, max_length, temperature, **kwargs)
        
        # Use local model
        return self._generate_local(prompt, max_length, temperature, **kwargs)

    def _generate_api(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using API-based models.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Temperature for text generation
            **kwargs: Additional API parameters
        
        Returns:
            Generated text
        """
        api_url = "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_length or self.config.max_length,
            "temperature": temperature or self.config.temperature,
            **kwargs
        }
        
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]

    def _generate_local(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using local transformer models.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Temperature for text generation
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        if not self.model or not self.tokenizer:
            self.load_model()
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length or self.config.max_length,
            temperature=temperature or self.config.temperature,
            **kwargs
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

