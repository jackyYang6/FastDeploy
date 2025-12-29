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

"""Pure PaddleFormers Causal LM Mixin - no layer replacements.

This module provides a Mixin for pure PaddleFormers CausalLM backend that uses all
PaddleFormers operators for training-inference consistency.
"""

from paddleformers.utils.log import logger
from collections.abc import Iterable
from typing import Dict, Any

import paddle

from fastdeploy.model_executor.forward_meta import ForwardMeta
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.models.model_base import ModelForCasualLM

from .base_pure import PaddleFormersPureModelBase


class CausalLMPureMixin:
    """Mixin class that provides causal LM functionality for pure PaddleFormers models.
    
    This is a private mixin class and should NOT be instantiated directly.
    Use PaddleFormersPureForCausalLM instead.
    """
    def __init__(self, fd_config: FDConfig, **kwargs):
        # Skip ModelForCasualLM.__init__ and call PaddleFormersPureModelBase directly
        super(ModelForCasualLM, self).__init__(fd_config, **kwargs)
        
        logger.info("[Pure Mode] Initializing CausalLMPureMixin")
        
        # Load the complete PaddleFormers model (with LM Head)
        self.model = self._load_paddleformers_model()
        
        # Store original vocab size for logits masking
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        
        # Single request KV cache state
        self._current_past_kv = None
        self._current_position = 0
        
        logger.info(f"[Pure Mode] Model initialized. Vocab size: {self.ori_vocab_size}")

    @paddle.no_grad()
    def forward(
        self,
        ids_remove_padding: paddle.Tensor,
        forward_meta: ForwardMeta,
        **kwargs,
    ):
        """
        Forward pass using pure PaddleFormers (single request mode).
        
        Calls self.model.model (base transformer) to get hidden_states,
        NOT self.model (which would compute logits too).
        
        Args:
            ids_remove_padding: Packed 1D token IDs [total_tokens]
            forward_meta: FastDeploy forward metadata
            
        Returns:
            hidden_states: [total_tokens, hidden_size]
        """
        try:
            seq_lens = forward_meta.seq_lens_this_time
            total_tokens = int(paddle.sum(seq_lens).item())
            
            # Handle empty input (preempted requests)
            if total_tokens == 0:
                logger.warning("[Pure Mode] Empty input, returning empty hidden states")
                hidden_size = self.model.config.hidden_size
                return paddle.zeros([0, hidden_size], dtype=self.model_config.dtype)
            
            # Convert packed tokens to batched format [B=1, S]
            input_ids = ids_remove_padding.unsqueeze(0)
            
            # Determine if this is prefill or decode based on whether we have cached KV
            is_prefill = self._current_past_kv is None or total_tokens > 1
            
            if is_prefill:
                # Prefill: reset position and KV cache
                past_key_values = None
                self._current_position = 0
                position_ids = paddle.arange(total_tokens, dtype="int64").unsqueeze(0)
            else:
                # Decode: use cached KV and update position
                past_key_values = self._current_past_kv
                position_ids = paddle.full([1, 1], self._current_position, dtype="int64")
            
            # Call model.model (base transformer) NOT model (which includes lm_head)
            base_model = self.model.model
            outputs = base_model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Store KV cache for next decode step
            self._current_past_kv = outputs.past_key_values
            self._current_position = position_ids[0, -1].item() + 1
            
            # Get hidden states: [B, S, H] -> [S, H]
            hidden_states = outputs.last_hidden_state[0]  # Remove batch dim
            
            return hidden_states
            
        except Exception as e:
            logger.error(f"[Pure Mode] Error in forward: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise e

    def compute_logits(self, hidden_states: paddle.Tensor, **kwargs) -> paddle.Tensor:
        """
        Compute logits from hidden states.
        
        In pure mode, we use PaddleFormers' LM head directly.
        """
        lm_head = self.model.lm_head
        logits = lm_head(hidden_states)
        logits = logits.astype(paddle.float32)
        
        if self.ori_vocab_size < logits.shape[-1]:
            logits[:, self.ori_vocab_size:] = -float("inf")
        
        return logits

    def reset_kv_cache(self):
        """Reset KV cache for new generation."""
        self._current_past_kv = None
        self._current_position = 0

    @classmethod
    def name(cls):
        return "PaddleFormersPureForCausalLM"

    @paddle.no_grad()
    def load_weights(self, weights_iterator: Iterable) -> None:
        """
        Load weights - for pure mode, weights are loaded via AutoModelForCausalLM.
        This method consumes the iterator to satisfy FD's interface.
        """
        logger.info("[Pure Mode] Weights already loaded via AutoModelForCausalLM.from_pretrained")
        for _ in weights_iterator:
            pass  # Consume iterator

    def set_state_dict(self, state_dict: Dict[str, Any]):
        """Set state dict - not used in pure mode."""
        logger.warning("[Pure Mode] set_state_dict called but not used in pure mode")