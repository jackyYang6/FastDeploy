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

"""Generic PaddleFormers modeling backend base class."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Dict, Any
from paddleformers.utils.log import logger

import regex as re
import paddle
from paddle.distributed import fleet
from paddle import nn
from paddleformers.transformers import AutoModel
from paddleformers.nn.attention.interface import ALL_ATTENTION_FUNCTIONS

from fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear, ReplicatedLinear, QKVParallelLinear
from fastdeploy.model_executor.layers.normalization import RMSNorm
from fastdeploy.model_executor.layers.embeddings import VocabParallelEmbedding
from fastdeploy.model_executor.utils import WeightsMapper


from paddleformers.transformers import PretrainedModel
from fastdeploy.config import FDConfig, ModelConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta

class PPMissingLayer(nn.Layer):
    """A placeholder for a layer that is not present on this pipeline parallel rank."""
    def forward(self, *args, **kwargs):
        raise RuntimeError("This layer should not be called.")

def getattr_iter(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default

def get_pp_indices(num_layers, pp_rank, pp_size):
    layers_per_rank = num_layers // pp_size
    start_layer = pp_rank * layers_per_rank
    end_layer = (pp_rank + 1) * layers_per_rank
    if pp_rank == pp_size - 1:
        end_layer = num_layers
    return start_layer, end_layer

def maybe_prefix(prefix, name):
    if prefix:
        return f"{prefix}.{name}"
    return name

def fastdeploy_append_attention_forward(
    module: paddle.nn.Layer,
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    attention_mask: paddle.Tensor,
    scaling: float | None = None,
    **kwargs,
):
    """将 PaddleFormers 的 q/k/v 直接转换为 FD 需要的 qkv: [S, hidden_dim] 格式。
    
    优化版本：避免双重 transpose 开销。
    PaddleFormers 输入: [B, H, S, D] (已经 transpose 过)
    FD 需要: qkv [S, (H_q + 2*H_kv) * D]
    """
    config = getattr(module, "config", None)
    if config is None:
        raise ValueError(f"Module {module} does not have 'config' attribute.")

    attention_instances = getattr(config, 'attention_instances', None)
    forward_meta = getattr(config, 'forward_meta', None)

    if attention_instances is None:
        raise ValueError("attention_instances not found in module.config")
    if forward_meta is None:
        raise ValueError("forward_meta not found in module.config")

    layer_idx = getattr(module, "layer_idx", getattr(module, "layer_id", None))
    if layer_idx is None:
        raise ValueError("layer_idx not found on attention module")

    self_attn = attention_instances[int(layer_idx)]
    if scaling is not None:
        self_attn.scale = float(scaling)

    # query shape is either [1, H, S, D] or [S, H, D]
    seq_len = query.shape[-2] if query.ndim == 4 else query.shape[0]

    if not hasattr(config, "_logged_attn_shapes"):
        logger.info(
            f"[FD-PF] layer={layer_idx} seq_len={seq_len} "
            f"q.shape={tuple(query.shape)} k.shape={tuple(key.shape)} v.shape={tuple(value.shape)}"
        )
        config._logged_attn_shapes = True

    def flatten_to_sd(t: paddle.Tensor, name: str) -> paddle.Tensor:
        """将 [B, H, S, D] 展平为 [S, H*D] 供 FD attention 使用。
        """
        if t.ndim == 3:
            return t.reshape([t.shape[0], -1])
        if t.ndim != 4:
            raise ValueError(f"{name} has unexpected dims {t.ndim}, expect 3 or 4")
    
        batch, dim1, dim2, dim3 = t.shape
        if batch != 1:
            raise ValueError(f"{name} batch size {batch} not supported")
        
        squeezed = t.squeeze(0)  # [dim1, dim2, dim3]
        
        if dim2 == seq_len:
            # [H, S, D] -> transpose to [S, H, D] -> reshape [S, H*D]
            return squeezed.transpose([1, 0, 2]).reshape([seq_len, -1])
        elif dim1 == seq_len:
            # [S, H, D] -> reshape [S, H*D]
            return squeezed.reshape([seq_len, -1])
        else:
            # Fallback: assume [H, S, D] format
            return squeezed.transpose([1, 0, 2]).reshape([seq_len, -1])

    q_flat = flatten_to_sd(query, "query")
    k_flat = flatten_to_sd(key, "key")
    v_flat = flatten_to_sd(value, "value")
    qkv = paddle.concat([q_flat, k_flat, v_flat], axis=-1)
    
    output = self_attn.forward(qkv=qkv, forward_meta=forward_meta)

    return output, None

ALL_ATTENTION_FUNCTIONS._global_mapping['fastdeploy'] = fastdeploy_append_attention_forward

@support_graph_optimization
class PaddleFormersModelBase(nn.Layer):
    """
    A mixin class to provide PaddleFormers backend logic.
    This class is not a nn.Layer itself but provides methods to
    initialize and manage a PaddleFormers model.
    """
    pf_to_fd_mapper = WeightsMapper(
        orig_to_new_prefix={
            "": "model.",
            "model.model.": "model.",
            "model.embed_tokens.weight": "model.embed_tokens.embeddings.weight",
            "embed_tokens.weight": "model.embed_tokens.embeddings.weight",
            "model.lm_head.weight": "lm_head.linear.weight",
            "model.score.": "classifier.",
            "model.classifier.": "classifier.",
        }
    )

    def __init_subclass__(cls, *args, **kwargs):
        """Merge pf_to_fd_mapper in MRO from most specific to least specific."""
        super().__init_subclass__(*args, **kwargs)

        # Collect all mappings from base classes
        merged_mappings = {}
        for base in reversed(cls.__mro__):  # Reverse to go from least to most specific
            if base_pf_to_fd_mapper := getattr(base, "pf_to_fd_mapper", None):
                if hasattr(base_pf_to_fd_mapper, 'orig_to_new_prefix'):
                    merged_mappings.update(base_pf_to_fd_mapper.orig_to_new_prefix)

        # Create new mapper with merged mappings
        cls.pf_to_fd_mapper = WeightsMapper(orig_to_new_prefix=merged_mappings)

    def __init__(self, fd_config: "FDConfig", **kwargs):
        super().__init__(fd_config)
        logger.info("Initializing PaddleFormers backend logic.")

        # 1. Storing all three levels of config
        self.fd_config = fd_config # FastDeploy's top-level FDConfig
        self.model_config = fd_config.model_config # FastDeploy's ModelConfig

        # Reload config using AutoConfig to ensure we get the specific config class (e.g. Qwen2Config)
        # instead of a generic PretrainedConfig. This is crucial for attributes like 'layer_types'.
        from paddleformers.transformers import AutoConfig
        self.paddleformers_config = AutoConfig.from_pretrained(self.model_config.model)
        self.paddleformers_config.fuse_rms_norm = True
        
        # fuse_attention_qkv: Only enable for models with FD-compatible QKV layout
        # Currently only Qwen3 (after modification) uses [Q_all, K_all, V_all] layout
        # that's compatible with FD's QKVParallelLinear.
        # Other models use interleaved layout or don't support fusion at all.
        model_type = getattr(self.paddleformers_config, 'model_type', '').lower()
        self._use_fused_qkv = False
        # self._use_fused_qkv = model_type in ['qwen3']  # Models with FD-compatible QKV layout
        # if self._use_fused_qkv:
        #     self.paddleformers_config.fuse_attention_qkv = True
        #     logger.info(f"Enabled fuse_attention_qkv for model_type={model_type}")
        
        # NOTE: TP is handled entirely by FastDeploy via layer replacement:
        # 1. FD replaces Linear layers with ColumnParallelLinear/RowParallelLinear
        # 2. FD's weight loader splits weights across TP ranks
        # 3. FD's attention backend handles multi-rank computation
        # 
        # PaddleFormers model is created as if TP=1 (single GPU).
        # We do NOT set tensor_model_parallel_size because that causes PaddleFormers
        # to create TP-aware layers internally, which conflicts with FD's approach.
        #
        # KNOWN ISSUE: LLaMA models have hardcoded reshape with num_heads in attention,
        # which doesn't account for TP. After FD replaces layers, output dims are halved
        # but num_heads remains full, causing shape mismatch. LLaMA TP>1 is not supported.
        if fd_config.parallel_config.tensor_parallel_size > 1:
            logger.info(f"TP size={fd_config.parallel_config.tensor_parallel_size} - Handled by FD layer replacement (not PaddleFormers TP)")

        self.text_config = self.paddleformers_config # The specific text model config
        
        # Sync important config values from text_config to model_config
        # This ensures fallback models use their actual config values instead of FD defaults
        self._sync_config_from_text_config()

        # For convenience, keep direct access to some FD configs
        self.quant_config = self.fd_config.quant_config
        self.parallel_config = self.fd_config.parallel_config

        # Parallel group setup - use parallel_config from FDConfig
        # For single GPU (centralized) deployment, these will be 1
        # For distributed deployment, these should come from fleet
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size if hasattr(self.parallel_config, 'pipeline_parallel_size') else 1
        
        # Use real tp_group from fd_config (already set by ParallelConfig.set_communicate_group())
        # This is critical for TP communication (all_reduce, alltoall, etc.)
        if hasattr(self.parallel_config, 'tp_group') and self.parallel_config.tp_group is not None:
            self.tp_group = self.parallel_config.tp_group
            self.tp_rank = self.parallel_config.tensor_parallel_rank
            logger.info(f"Using real tp_group from parallel_config: rank={self.tp_rank}, size={tp_size}")
        else:
            # Fallback to SimpleGroup for TP=1 case
            class SimpleGroup:
                def __init__(self, rank: int, world_size: int):
                    self.rank = rank
                    self.world_size = world_size
            self.tp_group = SimpleGroup(rank=0, world_size=tp_size)
            self.tp_rank = 0
            logger.warning(f"parallel_config.tp_group is None, using SimpleGroup (TP={tp_size})")
        
        # PP group (not critical for current debugging)
        class SimpleGroup:
            def __init__(self, rank: int, world_size: int):
                self.rank = rank
                self.world_size = world_size
        self.pp_group = SimpleGroup(rank=0, world_size=pp_size)
        
        logger.info(f"Parallel config: TP={tp_size} (rank={self.tp_rank}), PP={pp_size}")

        # 2. Initialize the actual PaddleFormers model
        # Check if head_dim is supported by FD Attention
        # FD Attention kernel only supports head_dim in [64, 128]
        head_dim = getattr(self.paddleformers_config, 'head_dim', None)
        if head_dim is None:
            # Calculate from hidden_size and num_attention_heads
            hidden_size = getattr(self.paddleformers_config, 'hidden_size', 0)
            num_heads = getattr(self.paddleformers_config, 'num_attention_heads', 1)
            head_dim = hidden_size // num_heads if num_heads > 0 else 0
        
        FD_SUPPORTED_HEAD_DIMS = {64, 128}
        if head_dim in FD_SUPPORTED_HEAD_DIMS:
            self.paddleformers_config._attn_implementation = "fastdeploy"
            logger.info(f"Using FD Attention (head_dim={head_dim})")
        else:
            self.paddleformers_config._attn_implementation = "eager"
            logger.warning(f"head_dim={head_dim} not supported by FD Attention, falling back to SDPA")

        # NOTE: Do NOT set paddle.set_device() here - we need paddle.LazyGuard() 
        # from the external loader to take effect for empty weight initialization
        
        logger.info("Creating model from config (weights will be loaded in load_weights())")
        
        self.model: PretrainedModel = AutoModel.from_config(
            self.paddleformers_config,
            dtype=self.model_config.dtype,
        )
        self.model.eval()

        # 3. Adapt the model to the FastDeploy environment
        # self.pipeline_parallel()  # Disabled for centralized deployment
        self.recursive_replace()  # Replace Linear/RMSNorm with FD optimized versions
        
        # Only create FD Attention instances when using FD Attention backend
        if self.paddleformers_config._attn_implementation == "fastdeploy":
            self.attention_instances = self.create_attention_instances()
            self.paddleformers_config.attention_instances = self.attention_instances
        else:
            self.attention_instances = {}
            logger.info(f"Using {self.paddleformers_config._attn_implementation} attention, skipping FD Attention creation")

        input_embeddings = self.model.get_input_embeddings()
        if not isinstance(input_embeddings, PPMissingLayer):
            self.embed_scale = getattr(input_embeddings, "embed_scale", None)
            embedding_dim = getattr_iter(self.text_config, ("embedding_size", "hidden_size"))
            assert embedding_dim is not None
            self.model.set_input_embeddings(
                VocabParallelEmbedding(
                    fd_config=self.fd_config,
                    num_embeddings=self.text_config.vocab_size,
                    embedding_dim=embedding_dim,
                )
            )

    def _sync_config_from_text_config(self) -> None:
        """
        Sync important config values from text_config (PaddleFormers/HF config)
        to model_config. This ensures fallback models use their actual config
        values instead of FD's defaults.
        
        This is crucial for models with unique configs like:
        - Gemma3: tie_word_embeddings=True, layer_types, sliding_window
        - Mistral: sliding_window
        - etc.
        """
        mc = self.model_config
        tc = self.text_config
        
        # List of important config fields to sync
        # Format: (field_name, compute_function_or_None)
        # If compute_function is None, just copy the value directly
        sync_fields = [
            "tie_word_embeddings",
            "sliding_window",
            "sliding_window_pattern",
            "layer_types",  # May be computed as property
            "rope_theta",
            "rope_scaling",
            "head_dim",
            "rms_norm_eps",

            "rope_local_base_freq",  # Gemma3 specific
            "query_pre_attn_scalar",  # Gemma3 specific
        ]
        
        synced = []
        for field in sync_fields:
            text_value = getattr(tc, field, None)
            if text_value is not None:
                # Only sync if not already set or if FD default differs
                current_value = getattr(mc, field, None) if hasattr(mc, field) else None
                if current_value is None or current_value != text_value:
                    setattr(mc, field, text_value)
                    synced.append(f"{field}={text_value}")
        
        if synced:
            logger.info(f"[PF Fallback] Synced config from text_config: {', '.join(synced[:5])}")
            if len(synced) > 5:
                logger.info(f"  ... and {len(synced) - 5} more fields")

    def pipeline_parallel(self):
        """Pipeline parallel setup - currently disabled for centralized deployment."""
        # TODO: Re-enable for distributed deployment
        logger.info("Skipping pipeline parallel setup (centralized deployment)")
        return

        # Original implementation (commented out for centralized deployment):
        # if self.pp_group.world_size <= 1:
        #     return
        # if not hasattr(self.model, '_pp_plan'):
        #     raise ValueError(f"{type(self.model)} does not support pipeline parallel.")
        #
        # module_lists = []
        # module_list_idx = None
        # pp_plan = list(self.model._pp_plan.keys())
        # for i, name in enumerate(pp_plan):
        #     if isinstance(getattr(self.model, name), nn.LayerList):
        #         module_lists.append(name)
        #         module_list_idx = i
        #
        # if len(module_lists) > 1:
        #     raise ValueError("Pipeline parallel of models with multiple `ModuleList`s in the base model are not supported yet!")
        # if module_list_idx is None:
        #     raise ValueError(f"Could not find `ModuleList` in {type(self.model)}")
        #
        # for name in pp_plan[:module_list_idx]:
        #     if self.pp_group.rank == 0 or (self.text_config.tie_word_embeddings and self.pp_group.rank == self.pp_group.world_size - 1):
        #         continue
        #     setattr(self.model, name, PPMissingLayer())
        #
        # start_layer, end_layer = get_pp_indices(
        #     self.text_config.num_hidden_layers,
        #     self.pp_group.rank,
        #     self.pp_group.world_size,
        # )
        # layers_name = pp_plan[module_list_idx]
        # layers = getattr(self.model, layers_name)
        # for i in range(len(layers)):
        #     if not (start_layer <= i < end_layer):
        #         layers[i] = PPMissingLayer()
        #
        # for name in pp_plan[module_list_idx + 1 :]:
        #     if self.pp_group.rank != self.pp_group.world_size - 1:
        #         setattr(self.model, name, PPMissingLayer())

    def recursive_replace(self):
        """Recursively replace modules in the model as needed.
        
        Replaces:
        - nn.Linear with FD's tensor parallel linear classes (based on naming rules)
        - *RMSNorm with FD's RMSNorm
        
        Aligned with vLLM transformers/base.py implementation.
        Uses naming-based strategy instead of model.tp_plan for compatibility.
        """
        tp_plan = self._get_tp_plan()
        
        def _get_linear_style(qual_name: str) -> str:
            """Determine linear style based on layer name patterns."""
            for pattern, style in tp_plan.items():
                if re.search(pattern, qual_name):
                    return style
            return "replicate"

        def _recursive_replace(module: nn.Layer, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                new_module = child_module
                
                if isinstance(child_module, nn.Linear):
                    style = _get_linear_style(qual_name)
                    
                    # PaddlePaddle nn.Linear: weight shape is [in_features, out_features]
                    # PyTorch nn.Linear: has in_features/out_features attributes
                    if hasattr(child_module, 'weight') and child_module.weight is not None:
                        weight_shape = child_module.weight.shape
                        in_features = weight_shape[0]
                        out_features = weight_shape[1]
                    else:
                        in_features = getattr(child_module, 'in_features', None)
                        out_features = getattr(child_module, 'out_features', None)
                    
                    with_bias = hasattr(child_module, 'bias') and child_module.bias is not None

                    if style == "colwise":
                        # Special handling for qkv_proj: use QKVParallelLinear (only when fused QKV is enabled)
                        if "qkv_proj" in qual_name and self._use_fused_qkv:
                            from fastdeploy.model_executor.layers.linear import QKVParallelLinear
                            new_module = QKVParallelLinear(
                                self.fd_config, prefix=qual_name,
                                with_bias=with_bias
                                # num_heads, kv_num_heads, hidden_size, head_dim are read from fd_config
                            )
                            logger.info(f"Replacing {qual_name} with QKVParallelLinear for fused QKV")
                        else:
                            new_module = ColumnParallelLinear(
                                self.fd_config, prefix=qual_name, 
                                input_size=in_features, output_size=out_features, 
                                with_bias=with_bias
                            )
                    elif style == "rowwise":
                        new_module = RowParallelLinear(
                            self.fd_config, prefix=qual_name,
                            input_size=in_features, output_size=out_features,
                            with_bias=with_bias
                        )
                    else:  # replicate
                        new_module = ReplicatedLinear(
                            self.fd_config, prefix=qual_name,
                            input_size=in_features, output_size=out_features,
                            with_bias=with_bias
                        )

                # NOTE: Skip RMSNorm replacement for now
                # FD RMSNorm returns (out, residual_out) tuple, but PaddleFormers expects single tensor
                # elif child_module.__class__.__name__.endswith("RMSNorm"):
                #     if hasattr(child_module, 'weight') and child_module.weight is not None:
                #         hidden_size = child_module.weight.shape[0]
                #     else:
                #         hidden_size = getattr(self.text_config, 'hidden_size', None)
                #     eps = getattr(child_module, 'epsilon', getattr(child_module, 'variance_epsilon', 1e-6))
                #     new_module = RMSNorm(
                #         self.fd_config, hidden_size=hidden_size, 
                #         eps=eps, prefix=qual_name
                #     )
                else:
                    _recursive_replace(child_module, prefix=qual_name)
                
                if new_module is not child_module:
                    setattr(module, child_name, new_module)
                    logger.info(f"Replacing {qual_name} with {new_module.__class__.__name__}")

        _recursive_replace(self.model, prefix="model")

    def _get_tp_plan(self) -> dict[str, str]:
        """Get TP plan for linear layer replacement.
        
        Priority:
        1. Try to get from PaddleFormers model's _get_tensor_parallel_mappings classmethod
        2. Fall back to default naming-based rules
        
        Returns:
            Dict mapping regex patterns to style ("colwise", "rowwise", "replicate")
        """
        # Try to get TP mappings from PaddleFormers model class
        model_cls = type(self.model)
        if hasattr(model_cls, '_get_tensor_parallel_mappings'):
            try:
                # Call the classmethod with config
                mappings = model_cls._get_tensor_parallel_mappings(self.text_config, is_split=True)
                if mappings:
                    # Convert PaddleFormers mappings to our format
                    # mappings is like: {"model.layers.0.self_attn.q_proj.weight": partial(fn, is_column=True)}
                    # Extract layer name patterns and determine colwise/rowwise
                    colwise_layers = set()
                    rowwise_layers = set()
                    
                    for key, func in mappings.items():
                        # Extract the layer suffix (e.g., "self_attn.q_proj.weight" -> "q_proj")
                        parts = key.split('.')
                        if len(parts) >= 2:
                            # Find the layer name (second to last before .weight/.bias)
                            for i, part in enumerate(parts):
                                if part.endswith('_proj') or part in ('up_proj', 'gate_proj', 'down_proj', 'o_proj', 'q_proj', 'k_proj', 'v_proj', 'qkv_proj'):
                                    # Check is_column from partial func
                                    if hasattr(func, 'keywords') and func.keywords.get('is_column', False):
                                        colwise_layers.add(part)
                                    else:
                                        rowwise_layers.add(part)
                    
                    if colwise_layers or rowwise_layers:
                        # Handle QKV fusion: if not using fused QKV, ensure separate projections
                        if not self._use_fused_qkv:
                            colwise_layers.discard('qkv_proj')
                            colwise_layers.update(['q_proj', 'k_proj', 'v_proj'])
                        
                        converted_plan = {}
                        for layer in colwise_layers:
                            converted_plan[rf"\.{layer}$"] = "colwise"
                        for layer in rowwise_layers:
                            converted_plan[rf"\.{layer}$"] = "rowwise"
                        logger.info(f"Using PaddleFormers TP mappings: colwise={list(colwise_layers)}, rowwise={list(rowwise_layers)}")
                        return converted_plan
            except Exception as e:
                logger.warning(f"Failed to get PaddleFormers TP mappings: {e}, using default")
        
        # Default naming-based TP plan (aligned with PaddleFormers LAYER_COLWISE/ROWWISE)
        logger.info("Using default naming-based TP plan")
        return {
            # Column Parallel (output dimension split)
            r"\.q_proj$": "colwise",
            r"\.k_proj$": "colwise",
            r"\.v_proj$": "colwise",
            r"\.gate_proj$": "colwise",
            r"\.up_proj$": "colwise",
            # Row Parallel (input dimension split)
            r"\.o_proj$": "rowwise",
            r"\.down_proj$": "rowwise",
        }

    def create_attention_instances(self) -> dict[int, Attention]:
        """Create FastDeploy attention instances for all layers.
        
        These instances replace PaddleFormers' attention and are passed to model.forward().
        For centralized deployment, create instances for all layers.
        """
        num_layers = self.text_config.num_hidden_layers
        
        # Debug: Log GQA config from fd_config.model_config
        mc = self.fd_config.model_config
        tp = self.fd_config.parallel_config.tensor_parallel_size
        logger.info(f"=== Create Attention Instances ===")
        logger.info(f"  num_attention_heads (total): {mc.num_attention_heads}")
        logger.info(f"  num_key_value_heads (total): {mc.num_key_value_heads}")
        logger.info(f"  head_dim: {mc.head_dim}")
        logger.info(f"  TP size: {tp}")
        logger.info(f"  Q heads per GPU: {mc.num_attention_heads // tp}")
        logger.info(f"  KV heads per GPU: {max(1, mc.num_key_value_heads // tp)}")
        logger.info(f"  Creating {num_layers} attention instances")
        
        # Handle interleaved sliding window attention (like Gemma3)
        # First check text_config for layer_types (explicit or computed by transformers/paddleformers)
        layer_types = getattr(self.text_config, 'layer_types', None)
        sliding_window = getattr(self.text_config, 'sliding_window', None)
        
        # If layer_types not present, try to compute from sliding_window_pattern (Gemma3 style)
        if layer_types is None:
            sliding_window_pattern = getattr(self.text_config, 'sliding_window_pattern', None)
            if sliding_window_pattern is not None and sliding_window is not None:
                # Compute layer_types like Gemma3: 
                # "sliding_attention" if (i+1) % pattern != 0 else "full_attention"
                layer_types = [
                    "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
                    for i in range(num_layers)
                ]
                logger.info(f"  Computed layer_types from sliding_window_pattern={sliding_window_pattern}")
                logger.info(f"  sliding_window: {sliding_window}")
                sliding_count = sum(1 for lt in layer_types if lt == "sliding_attention")
                logger.info(f"  Pattern: {sliding_count} sliding, {num_layers - sliding_count} full attention layers")
        elif layer_types is not None:
            # layer_types could be explicit in config or computed as @property by transformers/paddleformers
            sliding_count = sum(1 for lt in layer_types if lt == "sliding_attention")
            logger.info(f"  Detected layer_types (from config or computed property): sliding window enabled")
            logger.info(f"  sliding_window: {sliding_window}")
            logger.info(f"  Pattern: {sliding_count} sliding, {len(layer_types) - sliding_count} full attention layers")
            # Debug: print first 10 layer_types
            logger.info(f"  layer_types[0:10]: {list(layer_types)[:10]}")
        
        # Copy to model_config for Attention to access
        if layer_types is not None:
            if not hasattr(self.fd_config.model_config, 'layer_types'):
                self.fd_config.model_config.layer_types = layer_types
            if not hasattr(self.fd_config.model_config, 'sliding_window') and sliding_window is not None:
                self.fd_config.model_config.sliding_window = sliding_window

        attention_instances = {}
        for i in range(num_layers):
            attention_instances[i] = Attention(
                fd_config=self.fd_config,
                layer_id=i,
            )

        # TODO: For distributed deployment, use PP indices:
        # pp_rank = self.pp_group.rank
        # pp_size = self.pp_group.world_size
        # start, end = get_pp_indices(num_layers, pp_rank, pp_size)
        # for i in range(start, end):
        #     attention_instances[i] = Attention(...)

        return attention_instances

    def embed_input_ids(self, input_ids: paddle.Tensor) -> paddle.Tensor:
        """Embed input_ids using the model's embedding layer."""
        embedding_layer = self.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)
        
        if hasattr(self, 'embed_scale') and self.embed_scale is not None:
            inputs_embeds *= self.embed_scale
        return inputs_embeds

    @paddle.no_grad()
    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
        **kwargs,
    ):
        """Full transformer forward: input_ids -> hidden_states.
        
        This method is the primary forward pass for the model, computing:
        1. Position IDs based on seq_lens_decoder (absolute positions for RoPE)
        2. Token embeddings via embed_input_ids
        3. Transformer layers via self.model()
        
        Returns:
            hidden_states: [TotalTokens, HiddenDim]
        """
        num_tokens = ids_remove_padding.shape[0]
        
        batch_id_per_token = forward_meta.batch_id_per_token  # [num_tokens]
        seq_lens_decoder = forward_meta.seq_lens_decoder      # [batch_size, 1]
        seq_lens_this_time = forward_meta.seq_lens_this_time  # [batch_size, 1]
        
        if batch_id_per_token is not None and seq_lens_decoder is not None:
            decoder_offsets = seq_lens_decoder.squeeze(-1)  # [batch_size]
            token_decoder_offsets = paddle.index_select(decoder_offsets, batch_id_per_token, axis=0)  # [num_tokens]

            cu_seqlens = forward_meta.cu_seqlens_q  # [batch_size + 1]
            if cu_seqlens is not None:
                token_global_idx = paddle.arange(num_tokens, dtype="int64")
                request_start_idx = paddle.index_select(cu_seqlens[:-1], batch_id_per_token, axis=0)
                relative_positions = token_global_idx - request_start_idx.astype("int64")
            else:
                relative_positions = paddle.zeros([num_tokens], dtype="int64")
            position_ids = token_decoder_offsets.astype("int64") + relative_positions
        else:
            position_ids = paddle.arange(num_tokens, dtype="int64")
            if seq_lens_decoder is not None:
                position_ids = position_ids + seq_lens_decoder[0, 0].astype("int64")
        
        inputs_embeds = self.embed_input_ids(ids_remove_padding).unsqueeze(0)
        
        if hasattr(self, 'embed_scale') and self.embed_scale is not None:
            inputs_embeds *= self.embed_scale
        if getattr(self.text_config, 'uses_mrope', False):
            position_ids = position_ids.unsqueeze(1)
        else:
            position_ids = position_ids.unsqueeze(0)
        
        forward_meta.rope_already_applied = True
        self.paddleformers_config.forward_meta = forward_meta
        
        model_output = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=position_ids,
            return_dict=False,
            **kwargs,
        )
        
        hidden_states = model_output[0][0, ...]  # Remove batch dim
        
        return hidden_states
    
    @paddle.no_grad()
    def load_weights(self, weights: Iterable[tuple[str, paddle.Tensor]]):
        """Load weights from checkpoint into model parameters.
        
        Using FD native pattern: iterate weights and use param.weight_loader()
        for each FD layer (handles shape conversion automatically).
        """
        import re
        from fastdeploy.model_executor.utils import (
            default_weight_loader,
            process_weights_after_loading,
        )

        sublayers_dict = dict(self.named_sublayers())
        process_fn = process_weights_after_loading(sublayers_dict, self.fd_config)
        params_dict = dict(self.named_parameters())
        
        logger.info(f"Starting weight loading (FD native pattern)...")
        logger.info(f"Total parameters: {len(params_dict)}")
        
        # Debug: print first few param names to understand naming
        logger.info("=== First 10 parameter names ===")
        for i, name in enumerate(list(params_dict.keys())[:10]):
            logger.info(f"  param[{i}]: {name}")

        # Weight name mapping: HF name -> FD param name + shard_id
        # Like native Qwen3, includes lm_head in stacked_params_mapping
        stacked_params_mapping = [
            # Embeddings and lm_head (same as native)
            ("embed_tokens.embeddings", "embed_tokens", None),
            ("lm_head.linear", "lm_head", None),
        ]

        loaded_count = 0
        skipped_count = 0
        
        # QKV weight loading: map q/k/v_proj weights to qkv_proj with shard_id
        # QKVParallelLinear.weight_loader handles the proper placement of each shard
        def get_qkv_shard_info(weight_name):
            """Extract qkv_proj param name and shard_id from q/k/v_proj weight name."""
            for proj, shard_id in [("q_proj", "q"), ("k_proj", "k"), ("v_proj", "v")]:
                if proj in weight_name:
                    qkv_param_name = weight_name.replace(proj, "qkv_proj")
                    return qkv_param_name, shard_id
            return None, None

        for loaded_weight_name, loaded_weight in weights:
            # Debug: log first few and lm_head weight names
            if loaded_count + skipped_count < 5 or 'lm_head' in loaded_weight_name:
                logger.info(f"  Checkpoint weight: {loaded_weight_name}, shape={loaded_weight.shape}")
            
            # Handle QKV weight loading: map q/k/v_proj to qkv_proj with shard_id
            # Only when fused QKV is enabled for this model
            if self._use_fused_qkv:
                qkv_param_name, shard_id = get_qkv_shard_info(loaded_weight_name)
                if qkv_param_name is not None and ".weight" in loaded_weight_name:
                    if qkv_param_name not in params_dict:
                        # Try with/without model. prefix
                        if "model." + qkv_param_name in params_dict:
                            qkv_param_name = "model." + qkv_param_name
                        elif qkv_param_name.startswith("model.") and qkv_param_name[6:] in params_dict:
                            qkv_param_name = qkv_param_name[6:]
                    
                    if qkv_param_name in params_dict:
                        param = params_dict[qkv_param_name]
                        weight_loader = getattr(param, "weight_loader", None)
                        if weight_loader is not None:
                            # Use QKVParallelLinear.weight_loader with shard_id
                            weight_loader(param, loaded_weight, shard_id)
                            loaded_count += 1
                            if loaded_count <= 5:
                                logger.info(f"  QKV shard loaded: {loaded_weight_name} -> {qkv_param_name} (shard={shard_id})")
                            continue
                        else:
                            logger.warning(f"  QKV param {qkv_param_name} has no weight_loader, trying manual merge")
                    else:
                        logger.warning(f"  QKV param {qkv_param_name} not found in params_dict")
                        skipped_count += 1
                    continue
            
            # Try stacked params mapping first
            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in loaded_weight_name:
                    continue
                model_param_name = loaded_weight_name.replace(weight_name, param_name)
                if model_param_name not in params_dict:
                    logger.warning(f"  Stacked mapping: {loaded_weight_name} -> {model_param_name} NOT FOUND in params_dict!")
                    continue
                    
                param = params_dict[model_param_name]
                logger.info(f"  Loaded via stacked mapping: {loaded_weight_name} -> {model_param_name}")
                weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
                weight_loader(param, loaded_weight, shard_id)
                loaded_count += 1
                matched = True
                break

            if matched:
                continue

            # Direct mapping: try to find parameter with same name
            # Handle "model." prefix variations
            model_param_name = loaded_weight_name
            if model_param_name not in params_dict:
                # Try with "model." prefix
                model_param_name = "model." + loaded_weight_name
            if model_param_name not in params_dict:
                # Try without "model." prefix
                if loaded_weight_name.startswith("model."):
                    model_param_name = loaded_weight_name[6:]
            
            if model_param_name not in params_dict:
                skipped_count += 1
                continue

            param = params_dict[model_param_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader(self.fd_config))
            
            try:
                weight_loader(param, loaded_weight)
                loaded_count += 1
            except Exception as e:
                logger.warning(f"Failed to load {model_param_name}: {e}")
                skipped_count += 1
            
            # Post-process (for quantization etc)
            model_sublayer_name = re.sub(r"\.(weight|bias)$", "", model_param_name)
            process_fn(model_sublayer_name, param)

        logger.info(f"✅ Weight loading completed: {loaded_count} loaded, {skipped_count} skipped")
        
        # NOTE: FD's default_loader_v1.py L60 calls process_final_after_loading()
        # which already transposes weights for all FD Linear layers when model_format="torch".
        # DO NOT add another transpose here - it would cause double transpose!
        
        # Handle lm_head weight for tied case only
        # For non-tied case, lm_head is already loaded via stacked_params_mapping
        if hasattr(self, 'lm_head'):
            if hasattr(self, 'tie_word_embeddings') and self.tie_word_embeddings:
                # For tied embeddings, embed_tokens.embeddings.weight is already in correct format
                # [vocab_sharded, hidden] - just copy directly
                embed_weight = self.model.get_input_embeddings()
                if hasattr(embed_weight, 'embeddings') and hasattr(embed_weight.embeddings, 'weight'):
                    embed_tensor = embed_weight.embeddings.weight
                    self.lm_head.linear.weight.set_value(lm_head_weight)
                    logger.info(f"✅ Tied lm_head weight from embed_tokens (transposed): shape={lm_head_weight.shape}")
                else:
                    logger.warning("⚠️ tie_word_embeddings=True but embed_tokens.embeddings.weight not found!")
            else:
                # Non-tied: lm_head was already loaded via stacked_params_mapping
                logger.info(f"✅ lm_head loaded via stacked_params_mapping (non-tied)")
        
        # Verify weights loaded correctly
        self._verify_weights_loaded()
        
        # Debug: check weight stats for first few layers to diagnose collapsed hidden_state
        logger.info("=== Checking weight statistics for first 3 layers ===")
        for i in range(min(3, len(self.model.layers))):
            layer = self.model.layers[i]
            # Check attention weights
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    if hasattr(attn, name):
                        proj = getattr(attn, name)
                        if hasattr(proj, 'weight'):
                            w = proj.weight
                            logger.info(f"  layers.{i}.self_attn.{name}.weight: shape={w.shape}, "
                                       f"mean={float(w.mean()):.6f}, std={float(w.std()):.6f}")
            # Check MLP weights
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                for name in ['gate_proj', 'up_proj', 'down_proj']:
                    if hasattr(mlp, name):
                        proj = getattr(mlp, name)
                        if hasattr(proj, 'weight'):
                            w = proj.weight
                            logger.info(f"  layers.{i}.mlp.{name}.weight: shape={w.shape}, "
                                       f"mean={float(w.mean()):.6f}, std={float(w.std()):.6f}")
            # Check LayerNorm
            for norm_name in ['input_layernorm', 'post_attention_layernorm']:
                if hasattr(layer, norm_name):
                    norm = getattr(layer, norm_name)
                    if hasattr(norm, 'weight'):
                        w = norm.weight
                        logger.info(f"  layers.{i}.{norm_name}.weight: shape={w.shape}, "
                                   f"mean={float(w.mean()):.6f}, std={float(w.std()):.6f}")
            
            # Check QK-Norm weights
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                for qk_norm_name in ['q_norm', 'k_norm']:
                    if hasattr(attn, qk_norm_name):
                        norm = getattr(attn, qk_norm_name)
                        if hasattr(norm, 'weight'):
                            w = norm.weight
                            logger.info(f"  layers.{i}.self_attn.{qk_norm_name}.weight: shape={w.shape}, "
                                       f"mean={float(w.mean()):.6f}, std={float(w.std()):.6f}")
        logger.info("=" * 60)

    def _get_sublayer_by_path(self, path: str):
        """Get sublayer by dot-separated path like 'model.layers.0.self_attn.q_proj'"""
        parts = path.split(".")
        obj = self
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif part.isdigit() and hasattr(obj, '__getitem__'):
                obj = obj[int(part)]
            else:
                return None
        return obj

    def _verify_weights_loaded(self):
        """Verify key weights were actually loaded (not random)."""
        logger.info("Verifying weights were loaded correctly...")
        
        # Check FD Linear layers first
        fd_linear_count = 0
        cpu_weight_count = 0
        for name, sublayer in self.model.named_sublayers():
            if sublayer.__class__.__name__ in ['ColumnParallelLinear', 'RowParallelLinear', 'ReplicatedLinear']:
                fd_linear_count += 1
                if hasattr(sublayer, 'weight') and sublayer.weight is not None:
                    if sublayer.weight.place.is_cpu_place():
                        cpu_weight_count += 1
                        if fd_linear_count <= 3:  # Log first few
                            logger.warning(f"  ⚠️ {name}.weight on CPU!")
                    else:
                        if fd_linear_count <= 3:
                            logger.info(f"  ✅ {name}.weight on GPU, shape={sublayer.weight.shape}")
        
        logger.info(f"FD Linear layers: {fd_linear_count}, weights on CPU: {cpu_weight_count}")
        
        def check_weight(name: str, weight):
            """Check a single weight tensor using paddle operations (no numpy for bfloat16)."""
            try:
                # Check device first
                device = "GPU" if not weight.place.is_cpu_place() else "CPU"
                # Use paddle operations directly to avoid bfloat16 precision loss
                mean_val = float(weight.mean())
                std_val = float(weight.std())
                
                # Check for random initialization (std > 1.0 is suspicious for normalized weights)
                if std_val > 1.0:
                    logger.error(f"  {name}: RANDOM! mean={mean_val:.4f}, std={std_val:.4f} [{device}]")
                else:
                    logger.info(f"  ✅ {name}: mean={mean_val:.6f}, std={std_val:.6f} [{device}]")
            except Exception as e:
                logger.warning(f"  {name}: Failed to check - {e}")
        
        # 1. Check VocabParallelEmbedding (the actual layer used in forward)
        embed_layer = self.model.get_input_embeddings()
        if hasattr(embed_layer, 'embeddings') and hasattr(embed_layer.embeddings, 'weight'):
            check_weight("embed_tokens (VocabParallelEmbedding)", embed_layer.embeddings.weight)
        else:
            logger.warning("  embed_tokens: Could not access VocabParallelEmbedding weight")
        
        # 2. Check ParallelLMHead (the actual layer used in forward)
        if hasattr(self, 'lm_head') and hasattr(self.lm_head, 'linear'):
            check_weight("lm_head (ParallelLMHead)", self.lm_head.linear.weight)
        else:
            logger.warning("  lm_head: Could not access ParallelLMHead weight")
        
        # 3. Check a transformer layer weight via sublayer path
        layer0_qproj = self._get_sublayer_by_path("model.layers.0.self_attn.q_proj")
        if layer0_qproj is not None and hasattr(layer0_qproj, 'weight'):
            check_weight("model.layers.0.self_attn.q_proj", layer0_qproj.weight)
        else:
            logger.warning("  model.layers.0.self_attn.q_proj: Not found")
