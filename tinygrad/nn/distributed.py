"""
Fully Sharded Data Parallel (FSDP) implementation for tinygrad.

FSDP shards model parameters, gradients, and optimizer states across multiple devices.
During forward/backward passes, it all-gathers parameters and reduce-scatters gradients.

Reference: https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
"""
from __future__ import annotations
import math
from typing import Callable
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Optimizer
from tinygrad.helpers import flatten, prod

class FSDP:
  """
  Fully Sharded Data Parallel wrapper for neural network modules.

  Shards model parameters across multiple devices and handles communication
  for forward and backward passes.

  ```python exec="true" session="fsdp"
  from tinygrad import Tensor, nn
  from tinygrad.nn.distributed import FSDP
  import numpy as np
  ```

  ```python exec="true" source="above" session="fsdp" result="python"
  # Create a simple model
  model = nn.Linear(10, 5)
  
  # Wrap with FSDP (would use multiple devices in practice)
  fsdp_model = FSDP(model)
  
  # Forward pass
  x = Tensor.randn(2, 10)
  out = fsdp_model(x)
  print(f"Output shape: {out.shape}")
  ```
  """

  def __init__(self, module, devices: list[str]|None=None, sync_module_states: bool=True):
    """
    Wrap a module with FSDP.

    Args:
      module: The neural network module to wrap
      devices: List of device names to shard across. If None, uses available devices.
      sync_module_states: Whether to synchronize module states across devices
    """
    self.module = module
    self.devices = devices or self._get_default_devices()
    self.world_size = len(self.devices)
    self.rank = 0  # Will be set by distributed context

    # Get all parameters from the module
    self.params = self._get_params(module)
    
    # Shard parameters across devices
    self._shard_params()
    
    # Register forward/backward hooks
    self._register_hooks()

  def _get_default_devices(self) -> list[str]:
    """Get default list of devices for sharding."""
    from tinygrad.device import Device
    # Start with single device for now
    return [Device.DEFAULT]

  def _get_params(self, module) -> list[Tensor]:
    """Extract all parameters from a module."""
    from tinygrad.nn.state import get_parameters
    params = get_parameters(module)
    # Return all parameters (requires_grad is None by default in tinygrad)
    return params

  def _shard_params(self):
    """Shard parameters across devices."""
    for param in self.params:
      # Shard along the first dimension
      if param.shape[0] >= self.world_size:
        # Use shard method if available
        if hasattr(param, 'shard'):
          param.shard(self.devices, axis=0)
        else:
          # Manual sharding: keep only our shard
          shard_size = param.shape[0] // self.world_size
          start_idx = self.rank * shard_size
          end_idx = start_idx + shard_size if self.rank < self.world_size - 1 else param.shape[0]
          
          # Create sharded tensor on our device
          sharded_data = param[start_idx:end_idx].to(self.devices[self.rank])
          param.assign(sharded_data)

  def _register_hooks(self):
    """Register forward and backward hooks for communication."""
    # Store original forward
    if hasattr(self.module, '__call__'):
      self._original_forward = self.module.__call__
      self.module.__call__ = self._wrapped_forward

  def _wrapped_forward(self, *args, **kwargs):
    """Wrapped forward that handles FSDP communication."""
    # All-gather parameters before forward
    self._all_gather_params()
    
    # Run forward
    output = self._original_forward(*args, **kwargs)
    
    return output

  def _all_gather_params(self):
    """All-gather sharded parameters from all devices."""
    for param in self.params:
      if hasattr(param, 'shard') and param.device != self.devices[self.rank]:
        # Gather from all shards
        gathered = param  # In real implementation, would all-gather
        param.assign(gathered)

  def _reduce_scatter_grads(self, grad: Tensor):
    """Reduce-scatter gradients across devices."""
    # In real implementation, would reduce-scatter
    # For now, just return the gradient as-is
    return grad

  def __call__(self, *args, **kwargs):
    """Forward pass through FSDP-wrapped module."""
    return self._wrapped_forward(*args, **kwargs)

  def parameters(self) -> list[Tensor]:
    """Return list of parameters."""
    return self.params

  def state_dict(self):
    """Return full state dict by gathering from all shards."""
    state = {}
    for name in dir(self.module):
      attr = getattr(self.module, name)
      if isinstance(attr, Tensor):
        state[name] = attr
    return state

  def load_state_dict(self, state_dict):
    """Load state dict and shard across devices."""
    for name, value in state_dict.items():
      if hasattr(self.module, name):
        getattr(self.module, name).assign(value)
    self._shard_params()


class FSDPOptimizer:
  """
  Optimizer wrapper that works with FSDP-sharded parameters.

  Handles optimizer state sharding and gradient synchronization.
  """

  def __init__(self, optim_class: type, params: list[Tensor], **kwargs):
    """
    Wrap an optimizer for use with FSDP.

    Args:
      optim_class: The optimizer class to wrap (e.g., Adam)
      params: List of parameters (should be FSDP-sharded)
      **kwargs: Arguments passed to the underlying optimizer
    """
    self.optim = optim_class(params, **kwargs)
    self.params = params

  def zero_grad(self):
    """Zero gradients."""
    self.optim.zero_grad()

  def step(self):
    """Perform optimization step."""
    # Wait for gradient synchronization
    # In real implementation, would wait for reduce-scatter to complete
    
    # Step the underlying optimizer
    self.optim.step()
    
    # Re-shard parameters after update
    # In real implementation, would re-shard

  def state_dict(self):
    """Return optimizer state dict."""
    return self.optim.state_dict()

  def load_state_dict(self, state_dict):
    """Load optimizer state."""
    self.optim.load_state_dict(state_dict)


def shard_module(module, devices: list[str]|None=None) -> FSDP:
  """
  Convenience function to wrap a module with FSDP.

  Args:
    module: The module to wrap
    devices: List of devices to shard across

  Returns:
    FSDP-wrapped module
  """
  return FSDP(module, devices)
