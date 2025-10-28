"""
Window Manager
Manages the sliding window of SSM outputs and input tokens
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Optional, Tuple, List


class WindowManager:
    """
    Manages sliding window: [n SSM outputs | m input tokens]
    
    Window structure at step t:
        window[t] = [ssm_out[t-n+1], ..., ssm_out[t],  # n SSM outputs
                     x[t-m+1], ..., x[t]]              # m input tokens
    
    Operations:
    1. append_input: Add new input token
    2. append_ssm: Add new SSM output (only on write steps)
    3. get_window: Return current window for Transformer
    """
    
    def __init__(self, n_ssm_outputs, m_input_tokens, d_model, device='cuda'):
        """
        Args:
            n_ssm_outputs: Number of SSM outputs to keep
            m_input_tokens: Number of input tokens to keep
            d_model: Model dimension
            device: Device to store tensors
        """
        self.n = n_ssm_outputs
        self.m = m_input_tokens
        self.d_model = d_model
        self.device = device
        
        # Use deque for efficient rotation
        self.ssm_outputs = deque(maxlen=n_ssm_outputs)
        self.input_tokens = deque(maxlen=m_input_tokens)
        
        # Initialize with zeros
        self._initialize()
        
    def _initialize(self):
        """Initialize with zero vectors"""
        zero_vec = torch.zeros(self.d_model, device=self.device)
        
        # Fill SSM outputs with zeros
        for _ in range(self.n):
            self.ssm_outputs.append(zero_vec.clone())
        
        # Input tokens will be filled as we process
        # Start empty and fill up to m
        
    def append_input(self, x):
        """
        Add new input token embedding
        
        Args:
            x: (d_model,) - Single token embedding
        """
        if x.dim() == 1:
            self.input_tokens.append(x)
        else:
            # Batch dimension exists, take first item
            self.input_tokens.append(x[0])
    
    def append_ssm(self, ssm_output):
        """
        Add new SSM output (only called on write steps)
        
        Args:
            ssm_output: (d_model,) - SSM output vector
        """
        if ssm_output.dim() == 1:
            self.ssm_outputs.append(ssm_output)
        else:
            self.ssm_outputs.append(ssm_output[0])
    
    def get_window(self, batch_size=1):
        """
        Get current window for Transformer
        
        Returns:
            window: (batch_size, window_size, d_model)
        """
        # Stack SSM outputs
        if len(self.ssm_outputs) > 0:
            ssm_part = torch.stack(list(self.ssm_outputs), dim=0)  # (n, d_model)
        else:
            ssm_part = torch.zeros(self.n, self.d_model, device=self.device)
        
        # Stack input tokens (may be less than m initially)
        if len(self.input_tokens) > 0:
            input_part = torch.stack(list(self.input_tokens), dim=0)  # (len, d_model)
            # Pad if needed
            if len(self.input_tokens) < self.m:
                padding = torch.zeros(
                    self.m - len(self.input_tokens), 
                    self.d_model, 
                    device=self.device
                )
                input_part = torch.cat([padding, input_part], dim=0)
        else:
            input_part = torch.zeros(self.m, self.d_model, device=self.device)
        
        # Concatenate
        window = torch.cat([ssm_part, input_part], dim=0)  # (n+m, d_model)
        
        # Add batch dimension
        window = window.unsqueeze(0).expand(batch_size, -1, -1)
        
        return window
    
    def reset(self):
        """Reset to initial state"""
        self._initialize()
    
    @property
    def window_size(self):
        return self.n + self.m
    
    def get_state(self):
        """Get current state for checkpointing"""
        return {
            'ssm_outputs': list(self.ssm_outputs),
            'input_tokens': list(self.input_tokens),
        }
    
    def load_state(self, state):
        """Load state from checkpoint"""
        self.ssm_outputs = deque(state['ssm_outputs'], maxlen=self.n)
        self.input_tokens = deque(state['input_tokens'], maxlen=self.m)
    
    def __repr__(self):
        return (f"WindowManager(n={self.n}, m={self.m}, "
                f"window_size={self.window_size}, "
                f"ssm_filled={len(self.ssm_outputs)}/{self.n}, "
                f"input_filled={len(self.input_tokens)}/{self.m})")


class BatchedWindowManager:
    """
    Batched version of WindowManager for parallel processing
    Manages windows for entire batch at once
    """
    
    def __init__(self, batch_size, n_ssm_outputs, m_input_tokens, d_model, device='cuda'):
        """
        Args:
            batch_size: Batch size
            n_ssm_outputs: Number of SSM outputs to keep
            m_input_tokens: Number of input tokens to keep  
            d_model: Model dimension
            device: Device to store tensors
        """
        self.batch_size = batch_size
        self.n = n_ssm_outputs
        self.m = m_input_tokens
        self.d_model = d_model
        self.device = device
        
        # Store as tensors for efficiency
        # Shape: (batch, capacity, d_model)
        self.ssm_outputs = torch.zeros(
            batch_size, n_ssm_outputs, d_model, device=device
        )
        self.input_tokens = torch.zeros(
            batch_size, m_input_tokens, d_model, device=device
        )
        
        # Track current positions
        self.ssm_pos = 0
        self.input_pos = 0
        
    def append_input(self, x):
        """
        Add new input token embeddings for batch
        
        Args:
            x: (batch, d_model) - Input embeddings
        """
        # Rotate left and add new
        self.input_tokens = torch.cat([
            self.input_tokens[:, 1:, :],  # Remove oldest
            x.unsqueeze(1)  # Add newest
        ], dim=1)
        
        self.input_pos = min(self.input_pos + 1, self.m)
    
    def append_ssm(self, ssm_output):
        """
        Add new SSM outputs for batch
        
        Args:
            ssm_output: (batch, d_model) - SSM outputs
        """
        # Rotate left and add new
        self.ssm_outputs = torch.cat([
            self.ssm_outputs[:, 1:, :],  # Remove oldest
            ssm_output.unsqueeze(1)  # Add newest
        ], dim=1)
        
        self.ssm_pos = min(self.ssm_pos + 1, self.n)
    
    def get_window(self):
        """
        Get current windows for entire batch
        
        Returns:
            window: (batch, n+m, d_model)
        """
        # Concatenate SSM outputs and input tokens
        window = torch.cat([self.ssm_outputs, self.input_tokens], dim=1)
        return window
    
    def reset(self):
        """Reset all windows"""
        self.ssm_outputs.zero_()
        self.input_tokens.zero_()
        self.ssm_pos = 0
        self.input_pos = 0
    
    @property
    def window_size(self):
        return self.n + self.m
    
    def __repr__(self):
        return (f"BatchedWindowManager(batch={self.batch_size}, "
                f"n={self.n}, m={self.m}, window_size={self.window_size})")


if __name__ == "__main__":
    # Test WindowManager
    print("Testing WindowManager...")
    
    n, m = 15, 50
    d_model = 768
    device = 'cpu'
    
    manager = WindowManager(n, m, d_model, device)
    print(f"✅ Created: {manager}")
    
    # Simulate processing
    for step in range(100):
        # Add input
        x = torch.randn(d_model, device=device)
        manager.append_input(x)
        
        # Write every 16 steps
        if step % 16 == 0:
            ssm_out = torch.randn(d_model, device=device)
            manager.append_ssm(ssm_out)
        
        # Get window
        window = manager.get_window(batch_size=2)
        
        if step % 20 == 0:
            print(f"Step {step}: window shape = {window.shape}")
    
    print(f"\n✅ Final state: {manager}")
    
    # Test batched version
    print("\nTesting BatchedWindowManager...")
    batch_size = 4
    batched_manager = BatchedWindowManager(batch_size, n, m, d_model, device)
    
    for step in range(10):
        x = torch.randn(batch_size, d_model, device=device)
        batched_manager.append_input(x)
        
        if step % 3 == 0:
            ssm_out = torch.randn(batch_size, d_model, device=device)
            batched_manager.append_ssm(ssm_out)
    
    window = batched_manager.get_window()
    print(f"✅ Batched window shape: {window.shape}")
