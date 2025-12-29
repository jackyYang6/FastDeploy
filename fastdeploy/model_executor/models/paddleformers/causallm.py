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

"""Causal LM Mixin for PaddleFormers models.

This mixin provides lm_head and compute_logits functionality.
The forward() method is implemented in PaddleFormersModelBase.
"""

from typing import Dict, Any
from paddleformers.utils.log import logger

import paddle
from fastdeploy.model_executor.models.model_base import ModelForCasualLM
from fastdeploy.model_executor.layers.lm_head import ParallelLMHead


class CausalLMMixin:
    """Mixin class that provides causal LM functionality for PaddleFormers models.
    
    This mixin only handles:
    - lm_head initialization
    - compute_logits (hidden_states -> logits)
    
    The forward() method is inherited from PaddleFormersModelBase which computes
    input_ids -> hidden_states.
    
    This is a private mixin class and should NOT be instantiated directly.
    Use PaddleFormersForCausalLM instead.
    """
    def __init__(self, fd_config, **kwargs):

        super().__init__(fd_config, **kwargs)
        
        logger.info("Initializing CausalLMMixin.")
        
        # Store original vocab size for masking invalid tokens in compute_logits
        # This is CRITICAL - without this, model generates invalid tokens beyond vocab
        self.ori_vocab_size = fd_config.model_config.ori_vocab_size
        
        # Set up lm_head using ParallelLMHead for better compatibility (e.g. weight tying)
        # tie_word_embeddings is now synced from text_config in base._sync_config_from_text_config()
        self.tie_word_embeddings = fd_config.model_config.tie_word_embeddings
        logger.info(f"  tie_word_embeddings: {self.tie_word_embeddings}")
        
        # Try to detect if bias is needed (default to False for most modern LLMs)
        with_bias = getattr(self.text_config, "use_bias", False) or getattr(self.text_config, "bias", False)
        
        self.lm_head = ParallelLMHead(
            fd_config=fd_config,
            embedding_dim=self.text_config.hidden_size,
            num_embeddings=self.text_config.vocab_size,
            prefix="lm_head",
            with_bias=with_bias,
        )

    def compute_logits(self, hidden_state, **kwargs):
        """Compute logits from hidden states using lm_head."""
        # Debug: log hidden state stats
        if not hasattr(self, '_logits_call_count'):
            self._logits_call_count = 0
        self._logits_call_count += 1
        if self._logits_call_count <= 3:
            logger.info(f"[FD-PF] compute_logits #{self._logits_call_count}: hidden_state.shape={hidden_state.shape}, mean={float(hidden_state.mean()):.6f}, std={float(hidden_state.std()):.6f}")
        
        logits = self.lm_head(hidden_state)
        
        if self._logits_call_count <= 3:
            # Log top-5 predicted tokens
            top5_vals, top5_ids = paddle.topk(logits[-1], k=5)
            logger.info(f"[FD-PF] logits.shape={logits.shape}, top5_token_ids={top5_ids.tolist()}, top5_vals={top5_vals.tolist()}")
        
        logits = logits.astype(paddle.float32)
        
        # Mask logits beyond original vocab size
        logits[:, self.ori_vocab_size:] = -float("inf")

        return logits
    
    def set_state_dict(self, state_dict: Dict[str, Any]):
        self.load_weights(state_dict.items())

    @classmethod
    def name(cls):
        # This will be inherited by PaddleFormersForCausalLM
        return "PaddleFormersForCausalLM"    
