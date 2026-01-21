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

"""Generic PaddleFormers VL Processor for fallback support.

This processor wraps PaddleFormers' AutoProcessor to handle VLM models
that don't have a native FastDeploy processor implementation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from fastdeploy.engine.request import Request
from fastdeploy.input.text_processor import DataProcessor as TextProcessor


class PaddleFormersVLProcessor(TextProcessor):
    """Generic VL Processor using PaddleFormers AutoProcessor.
    
    This processor enables fallback support for any PaddleFormers VLM model
    by wrapping its AutoProcessor and converting outputs to FD format.
    
    Data Flow:
        FD Request -> _parse_request() -> _call_pf_processor() -> 
        _convert_to_fd_format() -> FD multimodal_inputs
    """
    
    def __init__(
        self,
        config,
        model_name_or_path: str,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        reasoning_parser_obj=None,
        tool_parser_obj=None,
        **kwargs,
    ):
        """Initialize the processor.
        
        Args:
            config: FD model config
            model_name_or_path: Path to PaddleFormers model
            limit_mm_per_prompt: Limits for multimodal items per prompt
            mm_processor_kwargs: Additional kwargs for PF processor
        """
        super().__init__(model_name_or_path, reasoning_parser_obj, tool_parser_obj)
        
        # Load PaddleFormers AutoProcessor
        from paddleformers.transformers import AutoProcessor
        self.pf_processor = AutoProcessor.from_pretrained(model_name_or_path)
        
        # Get token IDs
        self.image_token_id = getattr(self.pf_processor, 'image_token_id', None)
        if self.image_token_id is None:
            self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        
        self.video_token_id = getattr(self.pf_processor, 'video_token_id', None)
        if self.video_token_id is None:
            self.video_token_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
        
        # Get merge_size for grid calculation
        self.merge_size = getattr(
            getattr(self.pf_processor, 'image_processor', None),
            'merge_size', 2
        )
        
        self.config = config
        self.limit_mm_per_prompt = self._parse_limits(limit_mm_per_prompt)
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        
        logger.info(f"[PF VL Processor] Loaded for {model_name_or_path}")
        logger.debug(f"  image_token_id={self.image_token_id}, video_token_id={self.video_token_id}")
    
    def process_request(self, request, max_model_len=None, **kwargs):
        """FD entry point: process request and return with multimodal_inputs."""
        logger.info("[PF VL Processor] process_request STARTED")
        try:
            task = request.to_dict()
            logger.info(f"[PF VL Processor] task keys: {task.keys()}")
            self.process_request_dict(task, max_model_len)
            logger.info("[PF VL Processor] process_request_dict completed")
            request = Request.from_dict(task)
            request = self._apply_default_parameters(request)
            logger.info("[PF VL Processor] process_request COMPLETED")
            return request
        except Exception as e:
            logger.error(f"[PF VL Processor] process_request FAILED: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def process_request_dict(self, request: Dict, max_model_len=None) -> Dict:
        """Process request dictionary."""
        logger.info("[PF VL Processor] process_request_dict STARTED")
        request = self._apply_default_parameters(request)
        if not request.get("eos_token_ids"):
            request["eos_token_ids"] = self.eos_token_ids
        
        # 1. Parse request to get text, images, videos
        logger.info("[PF VL Processor] Parsing request...")
        text, images, videos = self._parse_request(request)
        logger.info(f"[PF VL Processor] Parsed: text_len={len(text)}, images={len(images)}, videos={len(videos)}")
        
        # 2. Call PaddleFormers Processor
        logger.info("[PF VL Processor] Calling PF processor...")
        pf_outputs = self._call_pf_processor(text, images, videos)
        logger.info(f"[PF VL Processor] PF processor returned, keys: {pf_outputs.keys()}")
        
        # 3. Convert to FD format
        logger.info("[PF VL Processor] Converting to FD format...")
        outputs = self._convert_to_fd_format(pf_outputs)
        logger.info(f"[PF VL Processor] Converted, output keys: {outputs.keys()}")
        
        # 4. Set outputs to request
        request["prompt_token_ids"] = outputs["input_ids"].tolist()
        request["prompt_token_ids_len"] = len(request["prompt_token_ids"])
        request["multimodal_inputs"] = outputs
        logger.info(f"[PF VL Processor] Set prompt_token_ids_len={request['prompt_token_ids_len']}")
        
        # Handle length limits
        if max_model_len and len(request["prompt_token_ids"]) > max_model_len:
            request["prompt_token_ids"] = request["prompt_token_ids"][:max_model_len - 1]
        
        if request.get("max_tokens") is None:
            request["max_tokens"] = max(1, max_model_len - len(request["prompt_token_ids"]))
        
        return request
    
    def _parse_request(self, request: Dict) -> tuple:
        """Parse request into text, images, videos."""
        images, videos = [], []
        
        if request.get("messages"):
            # OpenAI-style messages format
            text = self.tokenizer.apply_chat_template(
                request["messages"], tokenize=False, add_generation_prompt=True
            )
            # Extract multimodal data from messages
            for msg in request["messages"]:
                content = msg.get("content", [])
                if isinstance(content, str):
                    continue
                for item in content:
                    item_type = item.get("type")
                    if item_type == "image":
                        img_data = item.get("data") or item.get("image") or item.get("image_url", {}).get("url")
                        if img_data:
                            images.append(self._load_image(img_data))
                    elif item_type == "video":
                        vid_data = item.get("data") or item.get("video")
                        if vid_data:
                            videos.append(vid_data)
        elif request.get("prompt"):
            text = request["prompt"]
            mm_data = request.get("multimodal_data", {})
            images = [self._load_image(img) for img in mm_data.get("image", [])]
            videos = mm_data.get("video", [])
        else:
            raise ValueError("Request must contain 'prompt' or 'messages'")
        
        return text, images, videos
    
    def _load_image(self, image_data):
        """Load image from various sources."""
        from PIL import Image
        import io
        import base64
        
        if isinstance(image_data, Image.Image):
            return image_data
        elif isinstance(image_data, str):
            if image_data.startswith("data:"):
                # Base64 encoded
                _, data = image_data.split(",", 1)
                return Image.open(io.BytesIO(base64.b64decode(data)))
            elif image_data.startswith(("http://", "https://")):
                # URL
                import requests
                response = requests.get(image_data)
                return Image.open(io.BytesIO(response.content))
            else:
                # File path
                return Image.open(image_data)
        elif isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data))
        return image_data
    
    def _call_pf_processor(self, text: str, images: List, videos: List) -> Dict:
        """Call PaddleFormers Processor."""
        kwargs = {
            "text": text,
            "return_tensors": "np",  # Return numpy format
            **self.mm_processor_kwargs,
        }
        
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        
        return self.pf_processor(**kwargs)
    
    def _convert_to_fd_format(self, pf_outputs: Dict) -> Dict:
        """Convert PaddleFormers BatchFeature to FD multimodal_inputs format."""
        input_ids = np.array(pf_outputs["input_ids"]).flatten()
        seq_len = len(input_ids)
        
        # Compute token_type_ids
        token_type_ids = np.zeros_like(input_ids)
        if self.image_token_id:
            token_type_ids[input_ids == self.image_token_id] = 1  # image
        if self.video_token_id:
            token_type_ids[input_ids == self.video_token_id] = 2  # video
        
        # Process pixel_values
        images = None
        if "pixel_values" in pf_outputs:
            images = np.array(pf_outputs["pixel_values"])
        
        # Process grid_thw
        grid_thw = []
        if "image_grid_thw" in pf_outputs:
            grid_thw.extend(np.array(pf_outputs["image_grid_thw"]).tolist())
        if "video_grid_thw" in pf_outputs:
            grid_thw.extend(np.array(pf_outputs["video_grid_thw"]).tolist())
        
        # Generate position_ids for 3D rope (Qwen3VL format)
        # Shape: [seq_len, 3] - for (temporal, height, width) positions
        # For text-only, all 3 dimensions use same position sequence
        # For multimodal with images, FD expects this from the processor
        if "position_ids" in pf_outputs:
            # Use position_ids from PaddleFormers processor if available
            position_ids = np.array(pf_outputs["position_ids"])
            if position_ids.ndim == 3:
                position_ids = position_ids.squeeze(0)  # Remove batch dimension
            # Ensure shape is [seq_len, 3]
            if position_ids.shape[0] == 3 and position_ids.shape[1] == seq_len:
                position_ids = position_ids.T  # Transpose to [seq_len, 3]
        else:
            # Generate default position_ids for text-only
            # Shape: [seq_len, 3] with same positions for all 3 dimensions
            positions = np.arange(seq_len, dtype=np.int64)
            position_ids = np.stack([positions, positions, positions], axis=1)
        
        return {
            "input_ids": input_ids.astype(np.int64),
            "token_type_ids": token_type_ids.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
            "images": images,
            "grid_thw": np.array(grid_thw) if grid_thw else None,
            "image_type_ids": [0] * len(grid_thw) if grid_thw else None,
            "image_patch_id": self.image_token_id,
            "video_patch_id": self.video_token_id,
            "merge_size": self.merge_size,
        }
    
    def _mm_num_tokens(self, grid_thw) -> Union[int, List[int]]:
        """Calculate number of multimodal tokens from grid_thw."""
        if grid_thw is None or len(grid_thw) == 0:
            return 0
        
        def calc_one(thw):
            t, h, w = map(int, thw)
            return t * h * w // (self.merge_size ** 2)
        
        if isinstance(grid_thw[0], (list, tuple, np.ndarray)):
            return [calc_one(x) for x in grid_thw]
        return calc_one(grid_thw)
    
    def _parse_limits(self, limits):
        """Parse multimodal limits."""
        DEFAULT_LIMITS = {"image": 4, "video": 1, "audio": 1}
        if not limits:
            return DEFAULT_LIMITS
        return {**DEFAULT_LIMITS, **limits}
