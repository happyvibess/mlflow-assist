"""Advanced LLM chain management and prompt engineering"""

import logging
import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

class PromptTemplate(BaseModel):
    """Template for structured prompts"""
    template: str
    input_variables: List[str]
    
    def format(self, **kwargs: Any) -> str:
        """Format template with provided variables"""
        return self.template.format(**kwargs)

@dataclass
class LLMConfig:
    """Configuration for LLM chains"""
    model_name: str = "facebook/opt-125m"  # Default to a small local model
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1
    device: str = "auto"

class LLMChain:
    """Advanced LLM chain with conversation history and structured prompting"""
    
    def __init__(
        self,
        config: Optional[Union[LLMConfig, str]] = None,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """Initialize LLM chain with model and configuration"""
        if isinstance(config, str):
            config = LLMConfig(model_name=config)
        self.config = config or LLMConfig()
        
        # Set device
        self.device = (
            "cuda" if torch.cuda.is_available() and self.config.device == "auto"
            else "cpu"
        )
        
        try:
            if model is None or tokenizer is None:
                logger.info(f"Loading model: {self.config.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name
                ).to(self.device)
            else:
                self.model = model.to(self.device)
                self.tokenizer = tokenizer
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
            
        self.prompt_template = None
        self.conversation_history: List[Dict[str, str]] = []
        
    def add_prompt_template(self, template: Union[str, PromptTemplate]) -> None:
        """Add a prompt template for generation"""
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
        **kwargs
    ) -> str:
        """Generate text from prompt with conversation history"""
        try:
            # Format prompt
            if isinstance(prompt, dict) and self.prompt_template:
                formatted_prompt = self.prompt_template.format(**prompt)
            elif isinstance(prompt, str) and self.prompt_template:
                formatted_prompt = self.prompt_template.format(text=prompt)
            else:
                formatted_prompt = prompt if isinstance(prompt, str) else json.dumps(prompt)
                
            # Add conversation history context
            if self.conversation_history:
                context = "\n".join(
                    f"{msg['role']}: {msg['content']}"
                    for msg in self.conversation_history[-5:]  # Last 5 messages
                )
                formatted_prompt = f"{context}\n\nHuman: {formatted_prompt}\nAssistant:"
                
            # Prepare inputs with attention mask
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_attention_mask=True
            ).to(self.device)
            
            # Generate
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                num_return_sequences=self.config.num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                **kwargs
            )
            
            # Decode and clean response
            response = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            ).replace(formatted_prompt, "").strip()
            
            # Update conversation history
            self.conversation_history.append(
                {"role": "human", "content": formatted_prompt}
            )
            self.conversation_history.append(
                {"role": "assistant", "content": response}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
            
    def create_chain(
        self,
        steps: List[Dict[str, Any]]
    ) -> "ChainExecutor":
        """Create a chain of LLM operations"""
        return ChainExecutor(self, steps)

class ChainExecutor:
    """Executes a chain of LLM operations"""
    
    def __init__(self, llm: LLMChain, steps: List[Dict[str, Any]]):
        self.llm = llm
        self.steps = steps
        
    def execute(self, initial_input: Union[str, Dict[str, Any]]) -> List[str]:
        """Execute the chain of operations"""
        results = []
        current_input = initial_input
        
        for step in self.steps:
            template = step.get("template")
            use_response = step.get("use_response_as_input", False)
            
            if template:
                self.llm.add_prompt_template(template)
                
            output = self.llm.generate(current_input)
            results.append(output)
            
            if use_response:
                current_input = output
                
        return results


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

