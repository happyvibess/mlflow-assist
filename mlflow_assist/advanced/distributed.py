"""
Distributed training support with various backends.
"""

from typing import Any, Dict, Optional, Union
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import horovod.torch as hvd
from pydantic import BaseModel

class DistributedConfig(BaseModel):
    """Configuration for distributed training."""
    backend: str = "nccl"  # or "gloo", "mpi", "horovod"
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    init_method: Optional[str] = None

class DistributedTrainer:
    """Distributed training manager."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: DistributedConfig
    ):
        self.model = model
        self.config = config
        self.initialized = False
        
        if config.backend == "horovod":
            self._init_horovod()
        else:
            self._init_pytorch_distributed()

    def _init_pytorch_distributed(self) -> None:
        """Initialize PyTorch distributed backend."""
        if not dist.is_initialized():
            env = {
                "MASTER_ADDR": self.config.master_addr,
                "MASTER_PORT": self.config.master_port,
                "WORLD_SIZE": str(self.config.world_size),
                "RANK": str(self.config.rank)
            }
            
            for k, v in env.items():
                import os
                os.environ[k] = v
            
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
        self.initialized = True

    def _init_horovod(self) -> None:
        """Initialize Horovod backend."""
        hvd.init()
        self.initialized = True

    def prepare_model(
        self,
        model: torch.nn.Module,
        device: Optional[torch.device] = None
    ) -> torch.nn.Module:
        """Prepare model for distributed training."""
        if not device:
            if self.config.backend == "horovod":
                device = torch.device(f"cuda:{hvd.local_rank()}")
            else:
                device = torch.device(f"cuda:{self.config.rank}")
        
        model = model.to(device)
        
        if self.config.backend == "horovod":
            return model
        else:
            return DistributedDataParallel(
                model,
                device_ids=[self.config.rank]
            )

    def prepare_optimizer(
        self,
        optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        """Prepare optimizer for distributed training."""
        if self.config.backend == "horovod":
            return hvd.DistributedOptimizer(
                optimizer,
                named_parameters=self.model.named_parameters()
            )
        return optimizer

    def prepare_dataloader(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = True,
        **kwargs: Any
    ) -> DataLoader:
        """Prepare dataloader for distributed training."""
        if self.config.backend == "horovod":
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=hvd.size(),
                rank=hvd.rank(),
                shuffle=shuffle
            )
        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank,
                shuffle=shuffle
            )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            **kwargs
        )

    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.config.backend == "horovod":
            hvd.allreduce(torch.tensor(0), name="barrier")
        else:
            dist.barrier()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "mean"
    ) -> torch.Tensor:
        """Perform all-reduce operation."""
        if self.config.backend == "horovod":
            if op == "mean":
                return hvd.allreduce(tensor, op=hvd.Average)
            elif op == "sum":
                return hvd.allreduce(tensor, op=hvd.Sum)
        else:
            dist.all_reduce(tensor)
            if op == "mean":
                tensor /= self.config.world_size
        return tensor

    def cleanup(self) -> None:
        """Clean up distributed training resources."""
        if self.initialized:
            if self.config.backend != "horovod":
                dist.destroy_process_group()
            self.initialized = False

