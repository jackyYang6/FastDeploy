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

"""Pure PaddleFormers modeling backend - no layer replacements.

This module provides a pure PaddleFormers backend that doesn't replace any layers,
ensuring training-inference consistency by using all PaddleFormers operators.
"""
from paddleformers.utils.log import logger

import paddle
from paddle import nn
from paddleformers.transformers import AutoModelForCausalLM, AutoConfig

from fastdeploy.config import FDConfig

class PaddleFormersPureModelBase(nn.Layer):
    """
    Pure PaddleFormers backend base class.
    
    This class uses PaddleFormers model directly without any layer replacements:
    - No Embedding replacement
    - No Attention replacement  
    - No LM Head replacement
    - PaddleFormers manages its own KV Cache (single request mode)
    
    This ensures complete training-inference consistency.
    
    Note: Current implementation supports single request only.
    FD does NOT manage KV Cache in this mode.
    """
    
    def __init__(self, fd_config: FDConfig, **kwargs):
        # Call nn.Layer.__init__()
        super().__init__()
        logger.info("[Pure Mode] Initializing PaddleFormers backend logic.")
        
        # Store config references (same pattern as base.py)
        self.fd_config = fd_config
        self.model_config = fd_config.model_config
        
        # Load PaddleFormers config (use model_config.model, not model_name)
        self.paddleformers_config = AutoConfig.from_pretrained(
            self.model_config.model,
            dtype="bfloat16", 
            # fuse_attention_qkv=True,
            use_cache=True,
            # fuse_attention_ffn=True,
            _attn_implementation="sdpa",)
        self.paddleformers_config.fuse_rms_norm = True  
        self.text_config = self.paddleformers_config
        
        logger.info(f"[Pure Mode] Model: {self.model_config.model}")
        logger.info(f"[Pure Mode] Attention: {self.paddleformers_config._attn_implementation}")

    def _load_paddleformers_model(self) -> nn.Layer:
        """Load the complete PaddleFormers CausalLM model."""
        logger.info("[Pure Mode] Loading PaddleFormers model...")
        logger.info(f"[Pure Mode] Model path: {self.model_config.model}")
        logger.info(f"[Pure Mode] dtype: {self.model_config.dtype}")
        
        # Set device before model creation
        if paddle.device.is_compiled_with_cuda():
            paddle.set_device("gpu:0")
            logger.info("[Pure Mode] Set device to gpu:0")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model,
            config=self.paddleformers_config,
            dtype=self.model_config.dtype,
            convert_from_hf=True,
        )
        model.eval()
        
        # Explicitly move model to GPU
        model = model.to("gpu:0")
        logger.info("[Pure Mode] Moved model to GPU")
        
        # Verify model is on GPU
        for name, param in model.named_parameters():
            logger.info(f"[Pure Mode] First param '{name}' on device: {param.place}")
            break
        
        logger.info("[Pure Mode] Model loaded successfully")
        return model
