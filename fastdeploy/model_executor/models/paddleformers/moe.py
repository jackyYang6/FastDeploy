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

"""MOE Mixin for PaddleFormers fallback.

This module follows vLLM's MoEMixin pattern: MOE-specific replacement logic
is encapsulated in a Mixin class that can be inherited by MOE model classes.

Reference: vllm/model_executor/models/transformers/moe.py
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import paddle
from paddle import nn

from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from loguru import logger

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig


def maybe_prefix(prefix: str, name: str) -> str:
    """Concatenate prefix and name with dot separator."""
    if prefix:
        return f"{prefix}.{name}"
    return name


class PaddleFormersFusedMoE(FusedMoE):
    """Custom FusedMoE for PaddleFormers fallback.
    
    Replaces PaddleFormers' Experts class (e.g., GptOssExperts) with FD's FusedMoE.
    
    NOTE: FD's FusedMoE.forward(x, gate) computes routing internally by calling gate(x).
    So we ignore the pre-computed router_indices/routing_weights from PaddleFormers
    and let FD's optimized kernel handle everything.
    """
    
    def __init__(self, *args, gate: nn.Layer = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._gate = gate  # Keep reference to original gate
    
    def forward(
        self,
        hidden_states: paddle.Tensor,
        router_indices: paddle.Tensor = None,  # Ignored - FD FusedMoE re-computes
        routing_weights: paddle.Tensor = None,  # Ignored - FD FusedMoE re-computes
        **kwargs: Any,
    ) -> paddle.Tensor:
        """Forward matching PaddleFormers Experts interface.
        
        PaddleFormers calls:
            output = self.experts(hidden_states, router_indices=..., routing_weights=...)
            
        We ignore the pre-computed routing and use FD's FusedMoE internal routing via gate.
        """
        # FD's FusedMoE handles routing internally
        return super().forward(hidden_states, self._gate)


class MoEMixin:
    """Mixin class for MOE models in PaddleFormers fallback.
    
    Usage:
        class PaddleFormersForCausalLMMoE(MoEMixin, PaddleFormersForCausalLM):
            pass
    
    This mixin provides:
    - recursive_replace() for MOE experts replacement
    - get_expert_mapping() for weight loading
    """
    
    def recursive_replace(self):
        """Replace MOE experts with FusedMoE layers."""
        text_config = self.text_config
        
        # Get MOE config
        num_experts = getattr(text_config, 'num_experts',
                     getattr(text_config, 'num_local_experts',
                     getattr(text_config, 'n_routed_experts', None)))
        
        if num_experts is None:
            # Not a MOE model, skip
            super().recursive_replace()
            return
        
        top_k = getattr(text_config, 'num_experts_per_tok',
                getattr(text_config, 'top_k', 2))
        hidden_size = getattr(text_config, 'hidden_size', None)
        intermediate_size = getattr(text_config, 'moe_intermediate_size',
                            getattr(text_config, 'intermediate_size', None))
        
        # Check for shared experts
        num_shared_experts = getattr(text_config, 'n_shared_experts',
                             getattr(text_config, 'moe_num_shared_experts', 0))
        reduce_results = num_shared_experts == 0
        
        # Track MOE layers for later use
        self.mlp_moe_layers = []
        self.moe_layers = []
        self.num_moe_layers = 0
        
        def _recursive_replace_moe(module: nn.Layer, prefix: str):
            for child_name, child_module in module.named_children():
                qual_name = maybe_prefix(prefix, child_name)
                
                # Detect MOE experts: child_name == "experts"
                # GPT-OSS: GptOssExperts(nn.Layer) with forward(hidden, router_indices, routing_weights)
                # Qwen3: nn.LayerList of MLP modules
                if child_name == "experts":
                    parent_mlp = module
                    
                    # Check if parent has gate/router (this is a MOE block)
                    gate_layer = getattr(parent_mlp, 'gate', None) or getattr(parent_mlp, 'router', None)
                    if gate_layer is not None:
                        # Extract layer_idx from prefix
                        layer_match = re.search(r'layers\.(\d+)', qual_name)
                        layer_idx = int(layer_match.group(1)) if layer_match else 0
                        
                        # Check for bias in experts
                        has_bias = False
                        for param_name, _ in child_module.named_parameters():
                            if "bias" in param_name:
                                has_bias = True
                                break
                        
                        # Create FusedMoE to replace experts
                        fused_experts = PaddleFormersFusedMoE(
                            self.fd_config,
                            gate=gate_layer,
                            moe_intermediate_size=intermediate_size,
                            num_experts=num_experts,
                            top_k=top_k,
                            layer_idx=layer_idx,
                            reduce_results=reduce_results,
                        )
                        
                        # Replace experts with fused version
                        parent_mlp.experts = fused_experts
                        
                        # Track for weight loading
                        self.mlp_moe_layers.append(parent_mlp)
                        self.moe_layers.append(fused_experts)
                        self.num_moe_layers += 1
                        
                        logger.info(f"Replaced MOE experts at {qual_name} with PaddleFormersFusedMoE "
                                   f"(num_experts={num_experts}, top_k={top_k})")
                    else:
                        # Has "experts" but no gate, recurse into it
                        _recursive_replace_moe(child_module, prefix=qual_name)
                else:
                    _recursive_replace_moe(child_module, prefix=qual_name)
        
        _recursive_replace_moe(self.model, prefix="model")
        
        # Call parent's recursive_replace for non-MOE replacements
        super().recursive_replace()
    
    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Generate expert weight mapping for load_weights.
        
        Returns list of (param_name, weight_name, expert_id, shard_id) tuples.
        """
        num_experts = getattr(self.text_config, 'num_experts',
                     getattr(self.text_config, 'num_local_experts',
                     getattr(self.text_config, 'n_routed_experts', 0)))
        
        if num_experts == 0:
            return []
        
        expert_mapping = []
        
        # Check if using fused up_gate_proj format (Qwen3-MoE style)
        use_fused_ffn = getattr(self, '_use_fused_ffn', False)
        
        if use_fused_ffn:
            # Fused format: up_gate_proj (already fused in checkpoint)
            expert_mapping.extend(
                FusedMoE.make_expert_params_mapping(
                    ckpt_gate_up_proj_name="up_gate_proj",
                    ckpt_down_proj_name="down_proj",
                    num_experts=num_experts,
                )
            )
        else:
            # Separate format: gate_proj + up_proj
            ckpt_patterns = [
                ("gate_proj", "down_proj", "up_proj"),
                ("w1", "w2", "w3"),
            ]
            
            for gate_proj, down_proj, up_proj in ckpt_patterns:
                expert_mapping.extend(
                    FusedMoE.make_expert_params_mapping(
                        ckpt_gate_proj_name=gate_proj,
                        ckpt_down_proj_name=down_proj,
                        ckpt_up_proj_name=up_proj,
                        num_experts=num_experts,
                    )
                )
        
        return expert_mapping
