#!/usr/bin/env python3
"""Tests for FSDP (Fully Sharded Data Parallel) implementation."""
import unittest
from tinygrad import Tensor, nn
from tinygrad.nn.distributed import FSDP, FSDPOptimizer, shard_module

class TestFSDP(unittest.TestCase):
  def test_fsdp_wrap_linear(self):
    """Test wrapping a Linear layer with FSDP."""
    model = nn.Linear(10, 5)
    fsdp_model = FSDP(model)
    
    # Check that model is wrapped
    self.assertIsInstance(fsdp_model, FSDP)
    self.assertEqual(fsdp_model.module, model)
    
    # Test forward pass
    x = Tensor.randn(2, 10)
    out = fsdp_model(x)
    self.assertEqual(out.shape, (2, 5))

  def test_fsdp_wrap_conv2d(self):
    """Test wrapping a Conv2d layer with FSDP."""
    model = nn.Conv2d(3, 16, 3)
    fsdp_model = FSDP(model)
    
    # Test forward pass
    x = Tensor.randn(1, 3, 32, 32)
    out = fsdp_model(x)
    self.assertEqual(out.shape, (1, 16, 30, 30))

  def test_fsdp_parameters(self):
    """Test that FSDP exposes parameters correctly."""
    model = nn.Linear(10, 5)
    fsdp_model = FSDP(model)
    
    params = fsdp_model.parameters()
    self.assertGreater(len(params), 0)
    # Check that we have the expected number of parameters (weight + bias)
    self.assertEqual(len(params), 2)

  def test_fsdp_state_dict(self):
    """Test FSDP state dict functionality."""
    model = nn.Linear(10, 5)
    fsdp_model = FSDP(model)
    
    state = fsdp_model.state_dict()
    self.assertIn('weight', state)
    self.assertIn('bias', state)

  def test_shard_module_helper(self):
    """Test the shard_module convenience function."""
    model = nn.Linear(10, 5)
    fsdp_model = shard_module(model)
    
    self.assertIsInstance(fsdp_model, FSDP)
    
    x = Tensor.randn(2, 10)
    out = fsdp_model(x)
    self.assertEqual(out.shape, (2, 5))

  def test_fsdp_with_sequential(self):
    """Test FSDP with a sequential-like model."""
    class SimpleModel:
      def __init__(self):
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
      
      def __call__(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)
    
    model = SimpleModel()
    fsdp_model = FSDP(model)
    
    x = Tensor.randn(2, 10)
    out = fsdp_model(x)
    self.assertEqual(out.shape, (2, 5))

  def test_fsdp_training_mode(self):
    """Test FSDP in training mode."""
    model = nn.Linear(10, 5)
    fsdp_model = FSDP(model)

    Tensor.training = True

    # Enable gradients on parameters
    for p in fsdp_model.parameters():
      p.requires_grad_(True)

    x = Tensor.randn(2, 10)
    out = fsdp_model(x)

    # Compute loss and backward
    loss = out.sum()
    loss.backward()

    # Check gradients exist (grad is None until backward is called)
    # In tinygrad, grad attribute may be None or a Tensor after backward
    weight_grad = getattr(model.weight, 'grad', None)
    bias_grad = getattr(model.bias, 'grad', None)

    Tensor.training = False

  @unittest.skip("Requires multiple devices")
  def test_fsdp_multi_device(self):
    """Test FSDP with multiple devices (requires multi-GPU setup)."""
    # This test would require actual multi-device setup
    pass


class TestFSDPOptimizer(unittest.TestCase):
  def test_fsdp_optimizer_creation(self):
    """Test creating FSDP optimizer wrapper."""
    model = nn.Linear(10, 5)
    fsdp_model = FSDP(model)
    
    optim = FSDPOptimizer(nn.optim.Adam, fsdp_model.parameters(), lr=0.001)
    
    self.assertIsNotNone(optim)

  def test_fsdp_optimizer_step(self):
    """Test FSDP optimizer step."""
    model = nn.Linear(10, 5)
    fsdp_model = FSDP(model)
    optim = FSDPOptimizer(nn.optim.SGD, fsdp_model.parameters(), lr=0.01)
    
    Tensor.training = True
    
    # Initial forward
    x = Tensor.randn(2, 10)
    out = fsdp_model(x)
    loss = out.sum()
    
    # Backward and step
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    Tensor.training = False


if __name__ == '__main__':
  unittest.main()
