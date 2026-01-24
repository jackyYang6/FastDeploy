"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

"""
PaddleFormers Fleet MOE Model Fallback for FastDeploy.

This module provides a fallback mechanism for MOE models using PaddleFleet backend.
It wraps Fleet models with FastDeploy's inference capabilities:
- Attention backend replacement (KV Cache management)
- Weight loading with fusion (QKV, Gate+Up)
- Continuous batching and chunked prefill support

Design Principles:
1. No layer replacement - TP/EP/Fused layers are handled by Fleet
2. Only replace Attention backend for KV Cache management
3. Weight loading bypasses FlexCheckpoint to avoid OOM
4. Forward/compute_logits interface compatible with FastDeploy

Supported Models:
- Qwen3-MoE (is_fleet=True)
- GLM4-MoE (is_fleet=True)
- Other MOE models using GPTModelProvider
"""

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Optional

import paddle
from paddle import nn

from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.model_executor.graph_optimization.decorator import (
    support_graph_optimization,
)
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig


def _register_fastdeploy_attention_backend():
    """
    Register FastDeploy attention backend to PaddleFormers.
    
    This replaces Fleet's DotProductAttention with FD's implementation
    that includes KV Cache management.
    """
    try:
        from paddleformers.nn.attention.interface import ALL_ATTENTION_FUNCTIONS
    except ImportError:
        raise ImportError(
            "paddleformers is required for Fleet fallback. "
            "Please install paddleformers."
        )
    
    if "fastdeploy_fleet" in ALL_ATTENTION_FUNCTIONS._global_mapping:
        return  # Already registered
    
    def fastdeploy_fleet_attention_forward(
        module: nn.Layer,
        query: paddle.Tensor,
        key: paddle.Tensor,
        value: paddle.Tensor,
        attention_mask: paddle.Tensor,
        scaling: Optional[float] = None,
        **kwargs,
    ):
        """
        FastDeploy attention forward for Fleet models.
        
        Replaces Fleet's DotProductAttention with FD's implementation.
        
        Args:
            module: The attention module (DotProductAttention)
            query: [B, num_heads, S, head_dim] or [B, S, num_heads, head_dim]
            key: [B, num_kv_heads, S, head_dim] or [B, S, num_kv_heads, head_dim]
            value: [B, num_kv_heads, S, head_dim] or [B, S, num_kv_heads, head_dim]
            attention_mask: Attention mask (unused, FD uses forward_meta)
            scaling: Optional attention scaling factor
            
        Returns:
            Tuple of (attn_output, None)
            attn_output: [S, hidden_size]
        """
        # Get config from module's config attribute
        config = getattr(module, "config", None)
        if config is None:
            raise ValueError(
                f"Module {type(module).__name__} does not have 'config' attribute. "
                "Ensure the Fleet model config is properly set."
            )
        
        # Get attention instances and forward_meta from config
        attention_instances = getattr(config, "attention_instances", None)
        forward_meta = getattr(config, "forward_meta", None)
        
        if attention_instances is None:
            raise ValueError(
                "attention_instances not found in config. "
                "Ensure PaddleFormersFleetForCausalLM._setup_attention_backend() was called."
            )
        if forward_meta is None:
            raise ValueError(
                "forward_meta not found in config. "
                "Ensure forward_meta is set before model forward."
            )
        
        # Get layer index (Fleet uses 1-based layer_number)
        layer_number = getattr(module, "layer_number", None)
        if layer_number is None:
            # Try alternative attribute names
            layer_number = getattr(module, "layer_idx", None)
            if layer_number is None:
                raise ValueError(
                    "Cannot determine layer index from module. "
                    f"Module type: {type(module).__name__}"
                )
        
        # Convert to 0-based index
        layer_idx = int(layer_number) - 1 if layer_number >= 1 else int(layer_number)
        
        if layer_idx not in attention_instances:
            raise ValueError(
                f"Layer index {layer_idx} not found in attention_instances. "
                f"Available: {list(attention_instances.keys())}"
            )
        
        attn = attention_instances[layer_idx]
        
        # Set scaling if provided
        if scaling is not None:
            attn.scale = float(scaling)
        
        # Convert QKV format from Fleet to FD
        # Fleet format: [B, num_heads, S, head_dim] or [B, S, num_heads, head_dim]
        # FD format: [S, (num_heads + 2*num_kv_heads) * head_dim]
        qkv = _convert_fleet_qkv_to_fd_format(query, key, value)
        
        # Call FD attention with KV Cache
        output = attn.forward(qkv=qkv, forward_meta=forward_meta)
        
        return output, None
    
    # Register to PaddleFormers
    ALL_ATTENTION_FUNCTIONS._global_mapping["fastdeploy_fleet"] = fastdeploy_fleet_attention_forward


def _convert_fleet_qkv_to_fd_format(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
) -> paddle.Tensor:
    """
    Convert Fleet QKV tensors to FD packed format.
    
    Fleet DotProductAttention QKV format:
        query: [B, num_heads, S, head_dim]
        key:   [B, num_kv_heads, S, head_dim]  
        value: [B, num_kv_heads, S, head_dim]
    
    FD Attention expected format:
        qkv: [S, (num_heads + 2*num_kv_heads) * head_dim]
    
    Args:
        query: Query tensor from Fleet
        key: Key tensor from Fleet
        value: Value tensor from Fleet
        
    Returns:
        Packed QKV tensor for FD attention
    """
    # Handle different input formats
    if query.ndim == 4:
        # [B, H, S, D] format
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Remove batch dimension (FD uses packed format, batch=1 expected)
        if batch_size != 1:
            raise ValueError(
                f"Fleet attention batch size {batch_size} != 1. "
                "FD expects packed sequences with batch=1."
            )
        
        # [B, H, S, D] -> [S, H, D] -> [S, H*D]
        q_flat = query.squeeze(0).transpose([1, 0, 2]).reshape([seq_len, -1])
        k_flat = key.squeeze(0).transpose([1, 0, 2]).reshape([seq_len, -1])
        v_flat = value.squeeze(0).transpose([1, 0, 2]).reshape([seq_len, -1])
        
    elif query.ndim == 3:
        # [S, H, D] format (already no batch)
        seq_len = query.shape[0]
        q_flat = query.reshape([seq_len, -1])
        k_flat = key.reshape([seq_len, -1])
        v_flat = value.reshape([seq_len, -1])
        
    else:
        raise ValueError(
            f"Unexpected query shape {query.shape}. "
            "Expected 4D [B, H, S, D] or 3D [S, H, D]."
        )
    
    # Concatenate Q, K, V
    qkv = paddle.concat([q_flat, k_flat, v_flat], axis=-1)
    
    return qkv


@support_graph_optimization
class PaddleFormersFleetModelBase(nn.Layer):
    """
    Base class for Fleet MOE model fallback.
    
    This class provides:
    - Fleet model creation with proper config
    - Attention backend registration and setup
    - Weight loading with fusion (QKV, Gate+Up)
    - Forward interface compatible with FastDeploy
    """
    
    def __init__(self, fd_config: "FDConfig", **kwargs):
        super().__init__(fd_config)
        
        from paddleformers.transformers import AutoConfig
        from paddleformers.utils.log import logger
        
        logger.info("Initializing PaddleFormers Fleet backend for MOE model.")
        
        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        
        # Load PaddleFormers config
        self.paddleformers_config = AutoConfig.from_pretrained(
            self.model_config.model,
            trust_remote_code=True,
        )
        
        # Setup Fleet-specific config
        self._setup_fleet_config()
        
        # Create Fleet model (empty weights)
        self.model = self._create_fleet_model()
        
        # Get text config for model parameters
        self.text_config = self.paddleformers_config
        
        # Create FD attention instances
        self.attention_instances = self._create_attention_instances()
        
        # Setup attention backend
        self._setup_attention_backend()
        
        # Store important config values
        self.ori_vocab_size = self.model_config.ori_vocab_size
        self.tie_word_embeddings = getattr(
            self.paddleformers_config, "tie_word_embeddings", False
        )
        
        # Parallel config
        self.parallel_config = fd_config.parallel_config
        self.tp_group = self.parallel_config.tp_group
        self.tp_rank = self.parallel_config.tensor_parallel_rank
        
        logger.info(
            f"Fleet model initialized: {type(self.model).__name__}, "
            f"is_fleet={getattr(self.model, 'is_fleet', False)}"
        )
    
    def _setup_fleet_config(self):
        """
        Setup Fleet-specific configuration.
        
        This ensures the config has proper values for:
        - Tensor parallel size
        - Fused layers (QKV, Gate+Up)
        - Inference mode settings
        """
        config = self.paddleformers_config
        
        # Parallel config
        tp_size = self.fd_config.parallel_config.tensor_parallel_size
        config.tensor_model_parallel_size = max(tp_size, 1)
        config.pipeline_model_parallel_size = 1  # FD doesn't use PP
        config.context_parallel_size = 1
        config.virtual_pipeline_model_parallel_size = 1
        config.expert_model_parallel_size = 1  # TODO: Support EP
        
        # Enable fused layers for correct weight loading
        config.fuse_attention_qkv = True
        config.fuse_attention_ffn = True
        
        # Disable recompute (inference mode)
        config.recompute_granularity = None
        config.recompute_modules = None
        
        # Ensure these are set for MOE
        if not hasattr(config, 'moe_grouped_gemm'):
            config.moe_grouped_gemm = False  # Use StandardMLPExpert
    
    def _create_fleet_model(self):
        """
        Create Fleet model using PaddleFormers AutoModel.
        
        Note: This creates an empty model without loading weights.
        Weights are loaded separately via load_weights() to avoid OOM.
        
        Returns:
            Fleet model (GPTModel)
        """
        import paddle
        from paddleformers.transformers import AutoModelForCausalLM
        from paddleformers.utils.log import logger
        
        logger.info("Creating Fleet model from config...")
        
        # Create model from config (no weights)
        model = AutoModelForCausalLM.from_config(
            self.paddleformers_config,
            dtype=self.model_config.dtype,
        )
        
        # Verify it's a Fleet model
        is_fleet = getattr(model, "is_fleet", False)
        if not is_fleet:
            logger.warning(
                f"Model {type(model).__name__} may not be a Fleet model "
                f"(is_fleet={is_fleet}). Proceeding anyway."
            )
        
        # Set to eval mode
        model.eval()
        
        logger.info(f"Fleet model created: {type(model).__name__}")
        
        return model
    
    def _create_attention_instances(self) -> dict[int, Attention]:
        """
        Create FastDeploy attention instances for all layers.
        
        Returns:
            Dict mapping layer_idx to Attention instance
        """
        num_layers = self.paddleformers_config.num_hidden_layers
        
        attention_instances = {}
        for i in range(num_layers):
            attention_instances[i] = Attention(
                fd_config=self.fd_config,
                layer_id=i,
            )
        
        return attention_instances
    
    def _setup_attention_backend(self):
        """
        Setup FastDeploy attention backend.
        
        This:
        1. Registers the FD attention function to PaddleFormers
        2. Sets the config to use FD attention
        3. Attaches attention_instances to config for access in forward
        """
        # Register FD attention backend
        _register_fastdeploy_attention_backend()
        
        # Set attention implementation
        self.paddleformers_config._attn_implementation = "fastdeploy_fleet"
        
        # Attach attention instances to config for access in attention forward
        self.paddleformers_config.attention_instances = self.attention_instances
    
    @paddle.no_grad()
    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
        **kwargs,
    ) -> paddle.Tensor:
        """
        Forward pass through the Fleet model.
        
        Args:
            ids_remove_padding: Input token IDs [num_tokens]
            forward_meta: FastDeploy forward metadata
            **kwargs: Additional arguments
            
        Returns:
            hidden_states: [num_tokens, hidden_size]
        """
        num_tokens = ids_remove_padding.shape[0]
        
        # Compute position IDs
        position_ids = self._compute_position_ids(forward_meta, num_tokens)
        
        # Attach forward_meta to config for attention backend access
        self.paddleformers_config.forward_meta = forward_meta
        
        # Fleet model forward
        # Fleet expects [B, S] input, add batch dimension
        input_ids = ids_remove_padding.unsqueeze(0)
        position_ids = position_ids.unsqueeze(0)
        
        # Fleet PipelineLayer uses dict_args format for input
        dict_args = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        
        # Call Fleet model with dict input
        outputs = self.model(dict_args)
        
        # Extract hidden states from Fleet output
        # Fleet PipelineLayer output is dict with "hidden_states" or direct tensor
        if isinstance(outputs, dict):
            hidden_states = outputs.get("hidden_states", outputs.get("logits"))
        elif isinstance(outputs, tuple):
            hidden_states = outputs[0]
        else:
            hidden_states = outputs
        
        # Remove batch dimension: [B, S, H] -> [S, H]
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.squeeze(0)
        
        return hidden_states
    
    def _compute_position_ids(
        self,
        forward_meta: ForwardMeta,
        num_tokens: int,
    ) -> paddle.Tensor:
        """
        Compute position IDs from forward metadata.
        
        Args:
            forward_meta: FastDeploy forward metadata
            num_tokens: Number of tokens
            
        Returns:
            position_ids: [num_tokens]
        """
        batch_id_per_token = forward_meta.batch_id_per_token
        seq_lens_decoder = forward_meta.seq_lens_decoder
        
        if batch_id_per_token is not None and seq_lens_decoder is not None:
            # Compute absolute positions for each token
            decoder_offsets = seq_lens_decoder.squeeze(-1)
            token_decoder_offsets = paddle.index_select(
                decoder_offsets, batch_id_per_token, axis=0
            )
            
            cu_seqlens = forward_meta.cu_seqlens_q
            if cu_seqlens is not None:
                token_global_idx = paddle.arange(num_tokens, dtype="int64")
                request_start_idx = paddle.index_select(
                    cu_seqlens[:-1], batch_id_per_token, axis=0
                )
                relative_positions = token_global_idx - request_start_idx.astype("int64")
            else:
                relative_positions = paddle.zeros([num_tokens], dtype="int64")
            
            position_ids = token_decoder_offsets.astype("int64") + relative_positions
        else:
            # Simple sequential positions
            position_ids = paddle.arange(num_tokens, dtype="int64")
            if seq_lens_decoder is not None:
                position_ids = position_ids + seq_lens_decoder[0, 0].astype("int64")
        
        return position_ids


class FleetMOEMixin():
    """
    Fleet MOE model for causal language modeling.
    
    Provides:
    - load_weights() with Fleet QKV/Gate+Up fusion
    """
    
    @paddle.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, paddle.Tensor]]):
        """
        Load weights from checkpoint with fusion.
        
        This method:
        1. Buffers Q/K/V weights and fuses them into qkv_proj
        2. Buffers gate/up weights and fuses them into up_gate_proj
        3. Maps weight names from HF to Fleet format
        4. Handles weight transpose as needed
        
        Args:
            weights: Iterable of (weight_name, weight_tensor) tuples
        """
        from paddleformers.utils.log import logger
        
        logger.info("Loading weights into Fleet model with fusion...")
        
        # Get model parameters
        params_dict = dict(self.model.named_parameters())
        
        # Also include lm_head
        for name, param in self.lm_head.named_parameters():
            params_dict[f"lm_head.{name}"] = param
        
        # Buffers for weight fusion
        qkv_buffer = {}  # layer_idx -> {"q": weight, "k": weight, "v": weight}
        gate_up_buffer = {}  # (layer_idx, expert_idx) -> {"gate": weight, "up": weight}
        
        # Config for fusion
        num_heads = self.text_config.num_attention_heads
        num_kv_heads = getattr(self.text_config, "num_key_value_heads", num_heads)
        
        loaded_count = 0
        skipped_count = 0
        
        for loaded_name, loaded_weight in weights:
            try:
                # Check if this is a QKV weight
                qkv_info = self._parse_qkv_weight_name(loaded_name)
                if qkv_info is not None:
                    layer_idx, proj_type = qkv_info
                    
                    if layer_idx not in qkv_buffer:
                        qkv_buffer[layer_idx] = {}
                    
                    # Transpose weight: HF [out, in] -> Paddle [in, out]
                    qkv_buffer[layer_idx][proj_type] = loaded_weight.T
                    
                    # Check if all Q/K/V are collected
                    if len(qkv_buffer[layer_idx]) == 3:
                        fused_weight = self._fuse_qkv_weights(
                            qkv_buffer[layer_idx]["q"],
                            qkv_buffer[layer_idx]["k"],
                            qkv_buffer[layer_idx]["v"],
                            num_heads,
                            num_kv_heads,
                        )
                        
                        # Find target parameter
                        target_name = self._get_qkv_target_name(layer_idx)
                        if target_name in params_dict:
                            params_dict[target_name].set_value(fused_weight)
                            loaded_count += 3
                        else:
                            logger.warning(f"QKV target not found: {target_name}")
                            skipped_count += 3
                        
                        del qkv_buffer[layer_idx]
                    continue
                
                # Check if this is a gate/up weight
                gate_up_info = self._parse_gate_up_weight_name(loaded_name)
                if gate_up_info is not None:
                    layer_idx, expert_idx, proj_type = gate_up_info
                    key = (layer_idx, expert_idx)
                    
                    if key not in gate_up_buffer:
                        gate_up_buffer[key] = {}
                    
                    # Transpose weight: HF [out, in] -> Paddle [in, out]
                    gate_up_buffer[key][proj_type] = loaded_weight.T
                    
                    # Check if both gate and up are collected
                    if len(gate_up_buffer[key]) == 2:
                        fused_weight = self._fuse_gate_up_weights(
                            gate_up_buffer[key]["gate"],
                            gate_up_buffer[key]["up"],
                        )
                        
                        # Find target parameter
                        target_name = self._get_gate_up_target_name(layer_idx, expert_idx)
                        if target_name in params_dict:
                            params_dict[target_name].set_value(fused_weight)
                            loaded_count += 2
                        else:
                            logger.warning(f"Gate+Up target not found: {target_name}")
                            skipped_count += 2
                        
                        del gate_up_buffer[key]
                    continue
                
                # Regular weight - map name and load
                target_name = self._map_weight_name(loaded_name)
                if target_name is None:
                    skipped_count += 1
                    continue
                
                if target_name not in params_dict:
                    # Try with different prefixes
                    for prefix in ["", "model.", "0.embedding.", "49.", "50."]:
                        alt_name = prefix + target_name
                        if alt_name in params_dict:
                            target_name = alt_name
                            break
                    else:
                        skipped_count += 1
                        continue
                
                param = params_dict[target_name]
                
                # Handle weight transpose if needed
                weight_to_load = self._maybe_transpose_weight(
                    loaded_name, loaded_weight, param.shape
                )
                
                # Handle dtype conversion for MOE gate weights (requires float32)
                if "mlp.gate.weight" in target_name or "mlp.gate.weight" in loaded_name:
                    if weight_to_load.dtype != param.dtype:
                        weight_to_load = weight_to_load.astype(param.dtype)
                
                # Load weight
                if param.shape == weight_to_load.shape:
                    param.set_value(weight_to_load)
                    loaded_count += 1
                else:
                    logger.warning(
                        f"Shape mismatch for {target_name}: "
                        f"param={param.shape}, weight={weight_to_load.shape}"
                    )
                    skipped_count += 1
                    
            except Exception as e:
                logger.warning(f"Error loading {loaded_name}: {e}")
                skipped_count += 1
        
        # Load lm_head if tie_word_embeddings
        if self.tie_word_embeddings:
            self._handle_tied_embeddings(params_dict)
        
        logger.info(
            f"Weight loading completed: {loaded_count} loaded, {skipped_count} skipped"
        )
    
    def _parse_qkv_weight_name(self, name: str) -> Optional[tuple[int, str]]:
        """
        Parse QKV weight name to extract layer index and projection type.
        
        Args:
            name: Weight name like "model.layers.0.self_attn.q_proj.weight"
            
        Returns:
            (layer_idx, proj_type) or None if not a QKV weight
        """
        patterns = [
            r"model\.layers\.(\d+)\.self_attn\.(q|k|v)_proj\.weight",
            r"layers\.(\d+)\.self_attn\.(q|k|v)_proj\.weight",
        ]
        
        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                layer_idx = int(match.group(1))
                proj_type = match.group(2)
                return layer_idx, proj_type
        
        return None
    
    def _parse_gate_up_weight_name(self, name: str) -> Optional[tuple[int, int, str]]:
        """
        Parse gate/up weight name to extract layer index, expert index, and type.
        
        Args:
            name: Weight name like "model.layers.0.mlp.experts.0.gate_proj.weight"
            
        Returns:
            (layer_idx, expert_idx, proj_type) or None if not a gate/up weight
        """
        patterns = [
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up)_proj\.weight",
            r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up)_proj\.weight",
        ]
        
        for pattern in patterns:
            match = re.match(pattern, name)
            if match:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(2))
                proj_type = match.group(3)
                return layer_idx, expert_idx, proj_type
        
        return None
    
    def _fuse_qkv_weights(
        self,
        q_weight: paddle.Tensor,
        k_weight: paddle.Tensor,
        v_weight: paddle.Tensor,
        num_heads: int,
        num_kv_heads: int,
    ) -> paddle.Tensor:
        """
        Fuse Q/K/V weights into qkv_proj format (fused_qkv).
        
        The fused format interleaves Q, K, V per KV group:
        [Q_group0, K0, V0, Q_group1, K1, V1, ...]
        
        Args:
            q_weight: [hidden_size, num_heads * head_dim] (transposed)
            k_weight: [hidden_size, num_kv_heads * head_dim] (transposed)
            v_weight: [hidden_size, num_kv_heads * head_dim] (transposed)
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads
            
        Returns:
            fused_weight: [hidden_size, (num_heads + 2*num_kv_heads) * head_dim]
        """
        hidden_size = q_weight.shape[0]
        head_dim = q_weight.shape[1] // num_heads
        num_kv_groups = num_heads // num_kv_heads
        
        # Reshape for interleaving
        # Q: [H, num_kv_heads, num_kv_groups, head_dim]
        q_reshaped = q_weight.reshape([hidden_size, num_kv_heads, num_kv_groups, head_dim])
        # K: [H, num_kv_heads, 1, head_dim]
        k_reshaped = k_weight.reshape([hidden_size, num_kv_heads, 1, head_dim])
        # V: [H, num_kv_heads, 1, head_dim]
        v_reshaped = v_weight.reshape([hidden_size, num_kv_heads, 1, head_dim])
        
        # Interleave: [Q_group, K, V] per KV head
        # Result: [H, num_kv_heads, num_kv_groups + 2, head_dim]
        fused = paddle.concat([q_reshaped, k_reshaped, v_reshaped], axis=2)
        
        # Flatten: [H, total_heads * head_dim]
        fused = fused.reshape([hidden_size, -1])
        
        return fused
    
    def _fuse_gate_up_weights(
        self,
        gate_weight: paddle.Tensor,
        up_weight: paddle.Tensor,
    ) -> paddle.Tensor:
        """
        Fuse gate/up weights into up_gate_proj format.
        
        Args:
            gate_weight: [hidden_size, intermediate_size] (transposed)
            up_weight: [hidden_size, intermediate_size] (transposed)
            
        Returns:
            fused_weight: [hidden_size, 2 * intermediate_size]
        """
        # Concatenate on axis=1: [gate | up]
        return paddle.concat([gate_weight, up_weight], axis=1)
    
    def _get_qkv_target_name(self, layer_idx: int) -> str:
        """Get Fleet parameter name for fused QKV."""
        # Fleet uses layer numbers starting from 1 for transformer layers
        # Layer 0 is embedding
        fleet_layer_idx = layer_idx + 1
        return f"{fleet_layer_idx}.self_attn.qkv_proj.weight"
    
    def _get_gate_up_target_name(self, layer_idx: int, expert_idx: int) -> str:
        """Get Fleet parameter name for fused gate+up."""
        fleet_layer_idx = layer_idx + 1
        return f"{fleet_layer_idx}.mlp.experts.{expert_idx}.up_gate_proj.weight"
    
    def _map_weight_name(self, hf_name: str) -> Optional[str]:
        """
        Map HuggingFace weight name to Fleet parameter name.
        
        Args:
            hf_name: HuggingFace weight name
            
        Returns:
            Fleet parameter name or None if not mappable
        """
        # Skip QKV and gate/up (handled separately)
        if any(x in hf_name for x in ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]):
            return None
        
        # Common mappings
        mappings = [
            # Embedding
            (r"model\.embed_tokens\.weight", "0.embedding.embed_tokens.weight"),
            
            # Attention o_proj
            (r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight", 
             lambda m: f"{int(m.group(1))+1}.self_attn.o_proj.weight"),
            
            # Layer norms
            (r"model\.layers\.(\d+)\.input_layernorm\.weight",
             lambda m: f"{int(m.group(1))+1}.input_layernorm.weight"),
            (r"model\.layers\.(\d+)\.post_attention_layernorm\.weight",
             lambda m: f"{int(m.group(1))+1}.post_attention_layernorm.weight"),
            
            # QK norms
            (r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight",
             lambda m: f"{int(m.group(1))+1}.self_attn.q_layernorm.weight"),
            (r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight",
             lambda m: f"{int(m.group(1))+1}.self_attn.k_layernorm.weight"),
            
            # MOE gate
            (r"model\.layers\.(\d+)\.mlp\.gate\.weight",
             lambda m: f"{int(m.group(1))+1}.mlp.gate.weight"),
            
            # Expert down_proj
            (r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.weight",
             lambda m: f"{int(m.group(1))+1}.mlp.experts.{m.group(2)}.down_proj.weight"),
            
            # Final norm (last layer + 1)
            (r"model\.norm\.weight", 
             lambda m: f"{self.text_config.num_hidden_layers+1}.weight"),
            
            # LM head
            (r"lm_head\.weight", "lm_head.linear.weight"),
        ]
        
        for pattern, replacement in mappings:
            match = re.match(pattern, hf_name)
            if match:
                if callable(replacement):
                    return replacement(match)
                return replacement
        
        return None
    
    def _maybe_transpose_weight(
        self,
        name: str,
        weight: paddle.Tensor,
        target_shape: list[int],
    ) -> paddle.Tensor:
        """
        Transpose weight if needed to match target shape.
        
        HuggingFace uses [out_features, in_features] for Linear weights.
        Paddle uses [in_features, out_features].
        
        Args:
            name: Weight name
            weight: Weight tensor
            target_shape: Expected shape
            
        Returns:
            Possibly transposed weight
        """
        if weight.shape == target_shape:
            return weight
        
        # Try transpose
        if len(weight.shape) == 2 and len(target_shape) == 2:
            transposed = weight.T
            if transposed.shape == target_shape:
                return transposed
        
        return weight
    
    def _handle_tied_embeddings(self, params_dict: dict):
        """Handle tied word embeddings (embed_tokens -> lm_head)."""
        from paddleformers.utils.log import logger
        
        # Find embedding weight
        embed_weight = None
        for name in ["0.embedding.embed_tokens.weight", "model.embed_tokens.weight"]:
            if name in params_dict:
                embed_weight = params_dict[name]
                break
        
        if embed_weight is None:
            logger.warning("Cannot find embedding weight for tie_word_embeddings")
            return
        
        # Set lm_head weight
        lm_head_name = "lm_head.linear.weight"
        if lm_head_name in params_dict:
            # lm_head expects transposed weight for output projection
            params_dict[lm_head_name].set_value(embed_weight.T)
            logger.info("Tied lm_head weight to embed_tokens")
    
    def set_state_dict(self, state_dict):
        """Compatibility method for PaddleFormers."""
        self.load_weights(state_dict.items())


def name() -> str:
    """Return backend name for registration."""
    return "paddleformers_fleet"
