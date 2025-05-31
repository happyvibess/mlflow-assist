"""
Model optimization and compression utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic, prepare, convert
import numpy as np
from pydantic import BaseModel

class OptimizationConfig(BaseModel):
    """Configuration for model optimization."""
    compression_method: str = "pruning"  # or "quantization", "distillation"
    target_sparsity: float = 0.5
    pruning_method: str = "l1"  # or "random", "structured"
    quantization_dtype: str = "qint8"
    distillation_temperature: float = 2.0

class ModelOptimizer:
    """Model optimization and compression manager."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.compression_history: List[Dict[str, Any]] = []

    def optimize(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor] = None
    ) -> nn.Module:
        """
        Apply optimization techniques to the model.
        
        Args:
            model: PyTorch model to optimize
            example_inputs: Example inputs for quantization calibration
        
        Returns:
            Optimized model
        """
        if self.config.compression_method == "pruning":
            return self.apply_pruning(model)
        elif self.config.compression_method == "quantization":
            return self.apply_quantization(model, example_inputs)
        elif self.config.compression_method == "distillation":
            return self.prepare_distillation(model)
        else:
            raise ValueError(f"Unknown compression method: {self.config.compression_method}")

    def apply_pruning(self, model: nn.Module) -> nn.Module:
        """Apply model pruning."""
        parameters_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                parameters_to_prune.append((module, "weight"))
        
        if self.config.pruning_method == "l1":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=self.config.target_sparsity
            )
        elif self.config.pruning_method == "random":
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=self.config.target_sparsity
            )
        elif self.config.pruning_method == "structured":
            for module, _ in parameters_to_prune:
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=self.config.target_sparsity,
                    n=2,
                    dim=0
                )
        
        # Make pruning permanent
        for module, name in parameters_to_prune:
            prune.remove(module, name)
        
        self._log_compression_stats("pruning", model)
        return model

    def apply_quantization(
        self,
        model: nn.Module,
        example_inputs: Optional[torch.Tensor]
    ) -> nn.Module:
        """Apply model quantization."""
        model.eval()
        
        if self.config.quantization_dtype == "qint8":
            # Dynamic quantization
            quantized_model = quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
        else:
            # Static quantization
            if example_inputs is None:
                raise ValueError("example_inputs required for static quantization")
            
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            prepared_model = prepare(model)
            
            # Calibration
            prepared_model(example_inputs)
            quantized_model = convert(prepared_model)
        
        self._log_compression_stats("quantization", quantized_model)
        return quantized_model

    def prepare_distillation(self, model: nn.Module) -> nn.Module:
        """Prepare model for knowledge distillation."""
        class DistillationWrapper(nn.Module):
            def __init__(
                self,
                student_model: nn.Module,
                temperature: float
            ):
                super().__init__()
                self.student = student_model
                self.temperature = temperature
            
            def forward(
                self,
                x: torch.Tensor,
                teacher_logits: Optional[torch.Tensor] = None
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                student_logits = self.student(x)
                if teacher_logits is not None:
                    soft_targets = nn.functional.softmax(
                        teacher_logits / self.temperature, dim=1
                    )
                    return student_logits, soft_targets
                return student_logits
        
        wrapped_model = DistillationWrapper(
            model,
            self.config.distillation_temperature
        )
        self._log_compression_stats("distillation", wrapped_model)
        return wrapped_model

    def _log_compression_stats(
        self,
        method: str,
        model: nn.Module
    ) -> None:
        """Log compression statistics."""
        stats = {
            "method": method,
            "timestamp": torch.cuda.Event(enable_timing=True),
            "param_count": sum(p.numel() for p in model.parameters()),
            "param_size_mb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        }
        
        if method == "pruning":
            # Calculate sparsity
            zero_params = sum(
                (p == 0).sum().item() for p in model.parameters()
            )
            total_params = sum(p.numel() for p in model.parameters())
            stats["sparsity"] = zero_params / total_params
        
        self.compression_history.append(stats)

    def get_compression_stats(self) -> List[Dict[str, Any]]:
        """Get compression history and statistics."""
        return self.compression_history

