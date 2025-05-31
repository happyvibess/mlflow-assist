"""
Advanced LLM chain management and prompt engineering tools.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel
import torch
from transformers import Pipeline, AutoModelForCausalLM, AutoTokenizer
import json
import re

class PromptTemplate(BaseModel):
    """Template for structured prompts."""
    template: str
    input_variables: List[str]
    
    def format(self, **kwargs: Any) -> str:
        """Format the template with provided variables."""
        return self.template.format(**kwargs)

class LLMChain:
    """Chain of LLM operations with prompt management."""
    
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        max_length: int = 512,
        device: str = "auto"
    ):
        self.device = "cuda" if torch.cuda.is_available() and device == "auto" else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.temperature = temperature
        self.max_length = max_length
        self.conversation_history: List[Dict[str, str]] = []

    def add_prompt_template(self, template: Union[str, PromptTemplate]) -> None:
        """Add a prompt template to the chain."""
        if isinstance(template, str):
            variables = re.findall(r"{(\w+)}", template)
            template = PromptTemplate(
                template=template,
                input_variables=variables
            )
        self.prompt_template = template

    def generate(
        self,
        prompt: Union[str, Dict[str, Any]],
        **kwargs: Any
    ) -> str:
        """
        Generate response using the LLM.
        
        Args:
            prompt: Input prompt or variables for template
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text
        """
        if isinstance(prompt, dict) and hasattr(self, "prompt_template"):
            final_prompt = self.prompt_template.format(**prompt)
        else:
            final_prompt = prompt if isinstance(prompt, str) else json.dumps(prompt)
        
        # Add conversation history context
        if self.conversation_history:
            context = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in self.conversation_history[-5:]  # Last 5 messages
            )
            final_prompt = f"{context}\n\nHuman: {final_prompt}\nAssistant:"
        
        inputs = self.tokenizer(
            final_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=self.max_length,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        # Update conversation history
        self.conversation_history.append(
            {"role": "human", "content": final_prompt}
        )
        self.conversation_history.append(
            {"role": "assistant", "content": response}
        )
        
        return response

    def create_chain(
        self,
        steps: List[Dict[str, Any]]
    ) -> "LLMChainPipeline":
        """Create a pipeline of LLM operations."""
        return LLMChainPipeline(self, steps)

class LLMChainPipeline:
    """Pipeline for chaining multiple LLM operations."""
    
    def __init__(
        self,
        llm_chain: LLMChain,
        steps: List[Dict[str, Any]]
    ):
        self.llm_chain = llm_chain
        self.steps = steps

    def execute(
        self,
        initial_input: Union[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Execute the pipeline of LLM operations.
        
        Args:
            initial_input: Starting input for the chain
        
        Returns:
            List of responses from each step
        """
        results = []
        current_input = initial_input
        
        for step in self.steps:
            template = step.get("template")
            if template:
                self.llm_chain.add_prompt_template(template)
            
            response = self.llm_chain.generate(
                current_input,
                **step.get("parameters", {})
            )
            results.append(response)
            
            # Update input for next step if specified
            if step.get("use_response_as_input", False):
                current_input = response
        
        return results

