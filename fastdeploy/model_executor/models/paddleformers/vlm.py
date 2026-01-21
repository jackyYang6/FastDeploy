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

"""VLM (Vision-Language Model) Mixin for PaddleFormers fallback.

VLM models have both Vision Encoder and Language Model components.
The fallback strategy:
- Keep Vision Encoder unchanged (PaddleFormers handles it)
- Replace LLM Attention/MLP/Norm with FD optimized layers

This allows FD to accelerate VL models that don't have native implementations.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

import paddle
from paddle import nn

from loguru import logger

if TYPE_CHECKING:
    from fastdeploy.config import FDConfig
    from fastdeploy.model_executor.forward_meta import ForwardMeta


# Common vision-related module name patterns to skip during replacement
VISION_MODULE_PATTERNS = [
    r"visual",
    r"vision",
    r"vit",
    r"image_encoder",
    r"vision_tower",
    r"vision_model",
    r"patch_embed",
    r"merger",  # Qwen2.5-VL patch merger
]


def is_vision_module(name: str) -> bool:
    """Check if a module name belongs to vision encoder."""
    name_lower = name.lower()
    for pattern in VISION_MODULE_PATTERNS:
        if re.search(pattern, name_lower):
            return True
    return False


class VLMMixin:
    """Mixin class for VLM (Vision-Language) models in PaddleFormers fallback.
    
    Usage:
        class PaddleFormersVLMForConditionalGeneration(VLMMixin, CausalLMMixin, PaddleFormersModelBase):
            pass
    
    This mixin:
    - Skips Vision Encoder during layer replacement
    - Preserves vision-related modules (patch_embed, merger, etc.)
    - Only replaces LLM components with FD optimized layers
    """
    
    # Override to specify which parts are the language model
    # These patterns will be processed by recursive_replace
    LLM_MODULE_PATTERNS = [
        r"language_model",
        r"text_model", 
        r"llm",
        r"model\.layers",  # decoder layers
    ]
    
    def __init__(self, fd_config: "FDConfig", **kwargs):
        """Initialize VLM model with vision encoder extraction.
        
        This must be called after PaddleFormersModelBase.__init__ which loads the model.
        The visual encoder is extracted and stored as self.visual for FD's
        gpu_model_runner.extract_vision_features() to use.
        """
        super().__init__(fd_config, **kwargs)
        
        # Extract vision encoder from the loaded PaddleFormers model
        # This is done AFTER base.__init__ which runs recursive_replace
        # Vision modules are preserved (not replaced) by recursive_replace
        self.visual = self.get_vision_encoder()
        
        if self.visual is not None:
            logger.info(f"[VLM Fallback] Vision encoder extracted: {type(self.visual).__name__}")
        else:
            logger.warning("[VLM Fallback] No vision encoder found in model")
    
    def recursive_replace(self):
        """Replace layers, skipping Vision Encoder components."""
        
        def should_replace(qual_name: str) -> bool:
            """Determine if a module should be replaced based on its name."""
            # Skip vision-related modules
            if is_vision_module(qual_name):
                return False
            return True
        
        def _selective_replace(module: nn.Layer, prefix: str):
            """Recursively replace modules, skipping vision components."""
            for child_name, child_module in list(module.named_children()):
                qual_name = f"{prefix}.{child_name}" if prefix else child_name
                
                if is_vision_module(child_name):
                    logger.info(f"[VLM Fallback] Keeping vision module: {qual_name}")
                    continue
                
                # Continue recursion for non-vision modules
                _selective_replace(child_module, qual_name)
        
        # Log VLM-specific info
        logger.info("[VLM Fallback] Starting VLM layer replacement (vision encoder will be preserved)")
        
        # Call parent's recursive_replace which handles the actual replacement
        # The replacement logic in base.py will process all modules
        # Vision modules are kept because they don't match replacement patterns
        super().recursive_replace()
        
        # Log what was preserved
        for name, module in self.model.named_modules():
            if is_vision_module(name):
                logger.debug(f"[VLM Fallback] Preserved: {name} ({type(module).__name__})")
    
    def get_vision_encoder(self) -> Optional[nn.Layer]:
        """Get the vision encoder module if it exists."""
        # Common attribute names for vision encoder
        for attr in ["visual", "vision_tower", "vision_model", "image_encoder"]:
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
            # Also check nested structure
            if hasattr(self, "model") and hasattr(self.model, "model"):
                inner_model = self.model.model
                if hasattr(inner_model, attr):
                    return getattr(inner_model, attr)
        return None
    
    def get_language_model(self) -> Optional[nn.Layer]:
        """Get the language model module if it exists.
        
        For VLM, this returns the LLM component (e.g., Qwen3VLTextModel).
        Used by SupportsMultiModal protocol.
        """
        for attr in ["language_model", "text_model", "llm"]:
            if hasattr(self.model, attr):
                return getattr(self.model, attr)
            if hasattr(self, "model") and hasattr(self.model, "model"):
                inner_model = self.model.model
                if hasattr(inner_model, attr):
                    return getattr(inner_model, attr)
        # Fallback: return self.model if it has layers
        if hasattr(self.model, "layers"):
            return self.model
        return None

    @paddle.no_grad()
    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor],
        forward_meta: "ForwardMeta",
    ) -> paddle.Tensor:
        """VLM forward pass matching FD's multimodal calling convention.
        
        FD's gpu_model_runner calls VLM models with signature:
            model(ids_remove_padding, image_features, forward_meta)
        
        For VLM fallback:
        1. Get text embeddings from token IDs
        2. Merge with image_features if present (vision tokens are placeholder IDs)
        3. Pass merged embeddings to language model
        4. Return hidden states (lm_head handled by CausalLMMixin.compute_logits)
        """
        # === FALLBACK DEBUG LOGGING START ===
        from fastdeploy.model_executor.models.paddleformers.fallback_logger import get_fallback_logger
        fb_logger = get_fallback_logger()
        fb_logger.debug(f"[VLMMixin.forward] ENTERED")
        fb_logger.debug(f"[VLMMixin.forward] ids_remove_padding shape: {ids_remove_padding.shape}")
        fb_logger.debug(f"[VLMMixin.forward] image_features: {image_features is not None}")
        if image_features is not None:
            fb_logger.debug(f"[VLMMixin.forward] image_features shape: {image_features.shape}")
        fb_logger.debug(f"[VLMMixin.forward] forward_meta type: {type(forward_meta)}")
        # === FALLBACK DEBUG LOGGING END ===
        
        num_tokens = ids_remove_padding.shape[0]
        fb_logger.debug(f"[VLMMixin.forward] num_tokens: {num_tokens}")
        
        # Get text embeddings
        fb_logger.debug(f"[VLMMixin.forward] Getting text embeddings...")
        inputs_embeds = self.embed_input_ids(ids_remove_padding)
        fb_logger.debug(f"[VLMMixin.forward] text embeddings shape: {inputs_embeds.shape}")
        
        # Merge image features into embeddings if present
        if image_features is not None:
            fb_logger.debug(f"[VLMMixin.forward] Merging image features...")
            inputs_embeds = self.get_input_embeddings(ids_remove_padding, image_features)
            fb_logger.debug(f"[VLMMixin.forward] merged embeddings shape: {inputs_embeds.shape}")
        
        inputs_embeds = inputs_embeds.unsqueeze(0)  # Add batch dim
        fb_logger.debug(f"[VLMMixin.forward] inputs_embeds (with batch dim): {inputs_embeds.shape}")
        
        # Compute position IDs
        fb_logger.debug(f"[VLMMixin.forward] Computing position IDs...")
        batch_id_per_token = forward_meta.batch_id_per_token
        seq_lens_decoder = forward_meta.seq_lens_decoder

        if batch_id_per_token is not None and seq_lens_decoder is not None:
            decoder_offsets = seq_lens_decoder.squeeze(-1)
            token_decoder_offsets = paddle.index_select(decoder_offsets, batch_id_per_token, axis=0)

            cu_seqlens = forward_meta.cu_seqlens_q
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

        # VLM with M-RoPE (Qwen3VL) requires 3D position_ids [3, batch, seq_len]
        # The 3 dimensions are for temporal, height, width positions
        # For text-only or when vision positions are already handled, all 3 dims use same positions
        # Construct 3D directly to bypass PaddleFormers' buggy expand logic at line 1429
        fb_logger.debug(f"[VLMMixin.forward] 1D position_ids before expand: {position_ids.tolist()}")
        position_ids = position_ids.unsqueeze(0).unsqueeze(0)  # [seq_len] -> [1, 1, seq_len]
        position_ids = position_ids.expand([3, 1, -1])  # [3, 1, seq_len] for M-RoPE
        fb_logger.debug(f"[VLMMixin.forward] position_ids shape (3D for M-RoPE): {position_ids.shape}")

        forward_meta.rope_already_applied = True
        self.paddleformers_config.forward_meta = forward_meta
        fb_logger.debug(f"[VLMMixin.forward] Set rope_already_applied=True")

        # Call language model directly (bypass vision encoder since we have image_features)
        language_model = self.get_language_model()
        fb_logger.debug(f"[VLMMixin.forward] language_model type: {type(language_model).__name__}")
        # Ensure language_model.config (which is text_config) has the correct settings
        # to trigger FastDeploy attention path
        if hasattr(language_model, "config"):
            lm_config = language_model.config
            # Critical: Set _attn_implementation on the text config so Qwen3VLAttention 
            # selects the 'fastdeploy' backend instead of 'eager'
            if getattr(lm_config, "_attn_implementation", "") != "fastdeploy":
                fb_logger.debug(f"[VLMMixin.forward] Enabling fastdeploy attention on language_model.config")
                lm_config._attn_implementation = "fastdeploy"
            
            # Pass metadata required by the FD attention backend
            lm_config.attention_instances = self.attention_instances
            lm_config.forward_meta = forward_meta
        
        fb_logger.debug(f"[VLMMixin.forward] Calling language_model.forward()...")
        
        model_output = language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=position_ids,
            return_dict=False,
        )
        fb_logger.debug(f"[VLMMixin.forward] language_model returned, output type: {type(model_output)}")

        hidden_states = model_output[0][0, ...]  # Remove batch dim
        fb_logger.debug(f"[VLMMixin.forward] hidden_states shape: {hidden_states.shape}")
        fb_logger.debug(f"[VLMMixin.forward] RETURNING")

        return hidden_states
    
    def get_input_embeddings(
        self,
        ids_remove_padding: paddle.Tensor,
        image_features: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Get merged text + vision embeddings.
        
        Similar to FD's native VLM implementation: scatters image_features
        into text embeddings at vision token positions.
        """
        # Get text embeddings
        inputs_embeds = self.embed_input_ids(ids_remove_padding)
        
        if image_features is None:
            return inputs_embeds
            
        # Get vision token IDs from config
        image_token_id = getattr(self.model_config, "image_token_id", None)
        video_token_id = getattr(self.model_config, "video_token_id", None)
        
        if image_token_id is None and video_token_id is None:
            logger.warning("No image/video token ID found in config, returning text embeddings only")
            return inputs_embeds
        
        # Find positions of vision tokens and scatter image_features
        vision_mask = paddle.zeros_like(ids_remove_padding, dtype="bool")
        if image_token_id is not None:
            vision_mask = vision_mask | (ids_remove_padding == image_token_id)
        if video_token_id is not None:
            vision_mask = vision_mask | (ids_remove_padding == video_token_id)
        
        # Scatter image features into vision token positions
        if vision_mask.any():
            vision_positions = paddle.nonzero(vision_mask).squeeze(-1)
            num_vision_tokens = vision_positions.shape[0]
            if num_vision_tokens > 0 and image_features.shape[0] >= num_vision_tokens:
                inputs_embeds[vision_positions] = image_features[:num_vision_tokens]
        
        return inputs_embeds
