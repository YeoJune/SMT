"""
SMT (Stride Memory Transformer) with TBPTT
Efficient long-context processing with truncated backpropagation
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict, Tuple

from .components.ssm import SSMMemory
from .window_manager import WindowManager
from .components.attention_pooling import AttentionPooling
from .components.transformer import WindowedTransformer


class StrideMemoryTransformer(nn.Module):
    """
    Stride Memory Transformer with TBPTT
    
    Architecture:
        Input → Window [n SSM outputs | m input tokens]
        → Transformer → Logits
        
        Every stride steps:
        → Attention Pool window → SSM compress → Add to window
    
    Memory efficiency:
        - Transformer only sees small window (e.g., 65 tokens)
        - Chunks process with truncated gradients
        - SSM state carries long-range information
    
    Usage:
        # Training with long context
        model = StrideMemoryTransformer(config)
        logits = model(input_ids, chunk_size=512)  # TBPTT enabled
        
        # Inference
        generated = model.generate(prompt, max_new_tokens=1000)
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        
        self.config = config
        
        # Extract config
        model_cfg = config.model_config
        window_cfg = config.window_config
        transformer_cfg = config.transformer_config
        ssm_cfg = config.ssm_config
        pooling_cfg = config.pooling_config
        
        self.n = window_cfg['n_memory_tokens']
        self.m = window_cfg['n_input_tokens']
        self.stride = window_cfg['stride']
        self.d_model = model_cfg['d_model']
        self.vocab_size = model_cfg['vocab_size']
        self.pad_token_id = model_cfg['pad_token_id']
        
        # Use gradient checkpointing
        self.use_checkpointing = model_cfg.get('use_gradient_checkpointing', False)
        
        # Embedding (shared)
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Components
        self.transformer = WindowedTransformer(
            d_model=self.d_model,
            n_layers=transformer_cfg['num_layers'],
            n_heads=transformer_cfg['num_attention_heads'],
            d_ff=transformer_cfg['intermediate_size'],
            dropout=transformer_cfg['hidden_dropout_prob'],
            pretrained_name=transformer_cfg.get('pretrained_model'),
        )
        
        self.attention_pooling = AttentionPooling(
            d_model=self.d_model,
            d_k=pooling_cfg['query_dim'],
            dropout=model_cfg['dropout'],
        )
        
        self.ssm = SSMMemory(
            d_model=self.d_model,
            n_layers=ssm_cfg['num_layers'],
            d_state=ssm_cfg['state_size'],
            d_conv=ssm_cfg['conv_kernel'],
            expand_factor=ssm_cfg['expand_factor'],
        )
        
        # LM head (tied with embedding)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        print(f"✅ SMT: window={self.n}+{self.m}, stride={self.stride}")
        if self.use_checkpointing:
            print(f"✅ Gradient checkpointing enabled")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        chunk_size: Optional[int] = None,
        return_aux: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional TBPTT chunking.
        
        Args:
            input_ids: (batch, seq_len) - Input token IDs
            chunk_size: If provided, use TBPTT with this chunk size
                       If None, process entire sequence (may OOM on long sequences)
            return_aux: If True, return auxiliary information
        
        Returns:
            logits: (batch, seq_len, vocab_size)
            aux (optional): Dict with auxiliary outputs
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # Decide whether to use chunking
        if chunk_size is None or S <= chunk_size:
            # Process entire sequence (no TBPTT)
            return self._forward_full(input_ids, return_aux)
        else:
            # Process with TBPTT
            return self._forward_chunked(input_ids, chunk_size, return_aux)
    
    def _forward_full(
        self,
        input_ids: torch.Tensor,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Process entire sequence without chunking.
        
        Use for short sequences or when full gradients are needed.
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # Initialize
        self.ssm.init_cache(batch_size=B, device=device)
        window_mgr = WindowManager(
            batch_size=B,
            n_ssm_outputs=self.n,
            m_input_tokens=self.m,
            d_model=self.d_model,
            device=device,
        )
        
        # Embed
        embeddings = self.embedding(input_ids)  # (B, S, D)
        
        # Process each step
        all_logits = []
        write_steps = []
        
        for t in range(S):
            # Add input token to window
            x_t = embeddings[:, t, :]  # (B, D)
            window_mgr.append_input(x_t)
            
            # Get window and process
            window = window_mgr.get_window()  # (B, n+m, D)
            
            # Transformer (with optional checkpointing)
            if self.use_checkpointing and self.training:
                transformer_out = checkpoint(
                    self.transformer, window, use_reentrant=False
                )
            else:
                transformer_out = self.transformer(window)
            
            # Get logits from last position
            logits_t = self.lm_head(transformer_out[:, -1, :])  # (B, vocab)
            all_logits.append(logits_t)
            
            # Stride-based SSM update
            if t > 0 and t % self.stride == 0:
                pooled, _ = self.attention_pooling(window)
                ssm_output = self.ssm(pooled)
                window_mgr.append_ssm(ssm_output)
                write_steps.append(t)
        
        # Stack logits
        logits = torch.stack(all_logits, dim=1)  # (B, S, vocab)
        
        # Cleanup
        self.ssm.clear_cache()
        
        if return_aux:
            aux = {
                'write_steps': write_steps,
                'n_writes': len(write_steps),
            }
            return logits, aux
        
        return logits
    
    def _forward_chunked(
        self,
        input_ids: torch.Tensor,
        chunk_size: int,
        return_aux: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Process sequence with TBPTT chunking.
        
        IMPORTANT: This returns DETACHED logits!
        For training with TBPTT, you should NOT call backward on the output.
        Instead, use the training loop's TBPTT-aware training step.
        
        For inference, this is fine since no gradients are needed.
        """
        B, S = input_ids.shape
        device = input_ids.device
        
        # Initialize
        self.ssm.init_cache(batch_size=B, device=device)
        window_mgr = WindowManager(
            batch_size=B,
            n_ssm_outputs=self.n,
            m_input_tokens=self.m,
            d_model=self.d_model,
            device=device,
        )
        
        # Process in chunks
        all_logits = []
        all_write_steps = []
        
        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            
            # Embed only this chunk
            chunk_input_ids = input_ids[:, chunk_start:chunk_end]
            chunk_embeddings = self.embedding(chunk_input_ids)
            
            # Process this chunk
            chunk_logits, write_steps = self._process_chunk(
                chunk_embeddings,
                window_mgr,
                global_offset=chunk_start,
            )
            
            # Store DETACHED logits (for output only, no gradients)
            all_logits.append(chunk_logits.detach())
            all_write_steps.extend(write_steps)
            
            # Truncate gradients at chunk boundary
            if chunk_end < S:
                self.ssm.detach_cache()
                window_mgr.detach()
        
        # Concatenate all chunks (all detached)
        logits = torch.cat(all_logits, dim=1)  # (B, S, vocab)
        
        # Cleanup
        self.ssm.clear_cache()
        
        if return_aux:
            aux = {
                'write_steps': all_write_steps,
                'n_writes': len(all_write_steps),
                'n_chunks': (S + chunk_size - 1) // chunk_size,
            }
            return logits, aux
        
        return logits
    
    def _process_chunk(
        self,
        chunk_embeddings: torch.Tensor,
        window_mgr: WindowManager,
        global_offset: int,
    ) -> Tuple[torch.Tensor, list]:
        """
        Process a single chunk with parallelized transformer inference.
        
        Phase 1: SSM updates (sequential, but infrequent)
        Phase 2: Construct all windows in parallel
        Phase 3: Batch transformer processing
        
        Args:
            chunk_embeddings: (B, chunk_len, D) - Embedded tokens for this chunk
            window_mgr: Window manager (state carries from previous chunk)
            global_offset: Global step offset (for stride calculation)
        
        Returns:
            logits: (B, chunk_len, vocab)
            write_steps: List of global steps where SSM was updated
        """
        B, chunk_len, D = chunk_embeddings.shape
        
        # Phase 1: SSM updates and track SSM state timeline
        ssm_outputs_timeline = []  # SSM state at each step
        write_steps = []
        
        current_ssm_outputs = window_mgr.ssm_outputs.clone()
        
        for t in range(chunk_len):
            x_t = chunk_embeddings[:, t, :]
            window_mgr.append_input(x_t)
            
            # Save current SSM state for this step
            ssm_outputs_timeline.append(current_ssm_outputs.clone())
            
            # SSM update at stride steps
            global_t = global_offset + t
            if global_t > 0 and global_t % self.stride == 0:
                window = window_mgr.get_window()
                pooled, _ = self.attention_pooling(window)
                ssm_output = self.ssm(pooled)
                window_mgr.append_ssm(ssm_output)
                current_ssm_outputs = window_mgr.ssm_outputs.clone()
                write_steps.append(global_t)
        
        # Phase 2: Construct all windows in parallel
        # Get initial input tokens from window manager
        initial_inputs = window_mgr.input_tokens[:, -(self.m-1):, :].contiguous()
        
        # Pad chunk embeddings with initial inputs
        padded_inputs = torch.cat([initial_inputs, chunk_embeddings], dim=1)
        # (B, m-1+chunk_len, D)
        
        # Create sliding windows using unfold
        input_windows = padded_inputs.unfold(dimension=1, size=self.m, step=1)
        # (B, chunk_len, D, m)
        input_windows = input_windows.permute(0, 1, 3, 2).contiguous()
        # (B, chunk_len, m, D)
        
        # Stack SSM outputs for all steps
        ssm_outputs_batch = torch.stack(ssm_outputs_timeline, dim=1)
        # (B, chunk_len, n, D)
        
        # Combine SSM outputs and input windows
        all_windows = torch.cat([ssm_outputs_batch, input_windows], dim=2)
        # (B, chunk_len, n+m, D)
        
        # Phase 3: Batch transformer processing
        # Reshape to (B*chunk_len, n+m, D) for batch processing
        batch_windows = all_windows.view(B * chunk_len, self.n + self.m, D)
        
        if self.use_checkpointing and self.training:
            transformer_out = checkpoint(
                self.transformer, batch_windows, use_reentrant=False
            )
        else:
            transformer_out = self.transformer(batch_windows)
        
        # Get last position output: (B*chunk_len, n+m, D) → (B*chunk_len, D)
        last_hidden = transformer_out[:, -1, :]
        
        # Compute logits
        logits = self.lm_head(last_hidden)  # (B*chunk_len, vocab)
        
        # Reshape back: (B*chunk_len, vocab) → (B, chunk_len, vocab)
        logits = logits.view(B, chunk_len, -1)
        
        return logits, write_steps
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Auto-regressive generation.
        
        Args:
            input_ids: (batch, prefix_len) - Prefix tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        
        Returns:
            generated: (batch, prefix_len + max_new_tokens)
        """
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize
        self.ssm.init_cache(batch_size=B, device=device)
        window_mgr = WindowManager(
            batch_size=B,
            n_ssm_outputs=self.n,
            m_input_tokens=self.m,
            d_model=self.d_model,
            device=device,
        )
        
        generated = input_ids.clone()
        
        # Process prefix
        embeddings = self.embedding(input_ids)
        for t in range(input_ids.shape[1]):
            x_t = embeddings[:, t, :]
            window_mgr.append_input(x_t)
            
            if t > 0 and t % self.stride == 0:
                window = window_mgr.get_window()
                pooled, _ = self.attention_pooling(window)
                ssm_output = self.ssm(pooled)
                window_mgr.append_ssm(ssm_output)
        
        # Generate new tokens
        for step in range(max_new_tokens):
            # Get window and process
            window = window_mgr.get_window()
            transformer_out = self.transformer(window)
            logits = self.lm_head(transformer_out[:, -1, :])
            
            # Sample
            next_token = self._sample(logits, temperature, top_k, top_p)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update window
            next_embedding = self.embedding(next_token).squeeze(1)
            window_mgr.append_input(next_embedding)
            
            # SSM update
            current_step = generated.shape[1] - 1
            if current_step % self.stride == 0:
                window = window_mgr.get_window()
                pooled, _ = self.attention_pooling(window)
                ssm_output = self.ssm(pooled)
                window_mgr.append_ssm(ssm_output)
        
        # Cleanup
        self.ssm.clear_cache()
        
        return generated
    
    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Sample next token from logits."""
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        
        logits = logits / temperature
        
        # Top-k
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Top-p (nucleus)
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits, dim=-1), dim=-1
            )
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = 0
            
            for i in range(logits.shape[0]):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i, indices_to_remove] = -float('Inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        breakdown = {
            'embedding': sum(p.numel() for p in self.embedding.parameters()),
            'transformer': sum(p.numel() for p in self.transformer.parameters()),
            'attention_pooling': sum(p.numel() for p in self.attention_pooling.parameters()),
            'ssm': sum(p.numel() for p in self.ssm.parameters()),
            'lm_head': 0,  # Tied
        }
        
        return {
            'total': total,
            'trainable': trainable,
            'breakdown': breakdown,
        }