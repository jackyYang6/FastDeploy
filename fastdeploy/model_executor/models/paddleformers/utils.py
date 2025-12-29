
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

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional

from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.utils import get_logger

if TYPE_CHECKING:
    from paddleformers.transformers.configuration_utils import PretrainedConfig

logger = get_logger("paddleformers_backend", "paddleformers_backend.log")


def infer_architecture_name(fd_config: FDConfig, default: Optional[str] = None) -> str:
    """
    Determine the architecture name that PaddleFormers should instantiate.

    Args:
        fd_config: FastDeploy configuration container.
        default: Optional fallback architecture name.

    Returns:
        The resolved architecture name.

    Raises:
        ValueError: If no architecture information can be inferred.
    """

    model_config = getattr(fd_config, "model_config", None)
    if model_config is None:
        raise ValueError("FDConfig.model_config is required for PaddleFormers fallback.")

    architecture_hint = getattr(model_config, "_architecture", None)
    if architecture_hint:
        return architecture_hint

    pretrained_config = getattr(model_config, "pretrained_config", None)
    if pretrained_config is None:
        if default is not None:
            return default
        raise ValueError("FDConfig.model_config.pretrained_config is required for PaddleFormers fallback.")

    architectures = getattr(pretrained_config, "architectures", None) or []
    if architectures:
        return architectures[0]

    if default is not None:
        return default

    raise ValueError(
        "Cannot infer target architecture from pretrained_config.architectures; "
        "please ensure the original configuration keeps this field."
    )


def ensure_prefix_name(config: "PretrainedConfig", default: str = "model") -> None:
    """
    Ensure the PaddleFormers config has a prefix_name set so that parameter prefixes
    match FastDeploy expectations.
    """

    prefix = getattr(config, "prefix_name", None)
    if not prefix:
        setattr(config, "prefix_name", default)


def clone_pretrained_config(fd_config: FDConfig, architecture: str) -> "PretrainedConfig":
    """
    Create a deep copy of the original pretrained configuration and update the
    architecture entry for the PaddleFormers backend.
    """

    pretrained_config = getattr(fd_config.model_config, "pretrained_config", None)
    if pretrained_config is None:
        raise ValueError("FDConfig.model_config.pretrained_config is missing; cannot bootstrap PaddleFormers model.")

    config_copy = copy.deepcopy(pretrained_config)
    if architecture:
        setattr(config_copy, "architectures", [architecture])
    ensure_prefix_name(config_copy)
    return config_copy


def get_lm_head(module: nn.Layer) -> Optional[nn.Layer]:
    """
    Try to locate the LM head on a PaddleFormers model instance.
    """

    if hasattr(module, "lm_head"):
        lm_head = getattr(module, "lm_head")
        if isinstance(lm_head, nn.Layer):
            return lm_head

    get_output_embeddings = getattr(module, "get_output_embeddings", None)
    if callable(get_output_embeddings):
        embeddings = get_output_embeddings()
        if isinstance(embeddings, nn.Layer):
            return embeddings

    return None


def inject_fastdeploy_attention(model: nn.Layer, fd_config: FDConfig) -> None:
    """
    Tag attention-like modules with FastDeploy metadata so that customized kernels can hook in.
    """

    tagged = []
    for name, module in model.named_sublayers():
        module_name = module.__class__.__name__.lower()
        if "attn" in name or "attention" in module_name:
            setattr(module, "_fastdeploy_config", fd_config)
            tagged.append(name)

    if tagged:
        logger.debug(
            "PaddleFormers backend: marked %d attention modules for FastDeploy integration (sample: %s).",
            len(tagged),
            ", ".join(tagged[:5]),
        )
    else:
        logger.debug(
            "PaddleFormers backend: no attention modules detected when scanning model %s.",
            getattr(fd_config.model_config, "model", "<unknown>"),
        )


def apply_tensor_parallel(model: nn.Layer, fd_config: FDConfig) -> None:
    """
    Populate tensor-parallel metadata on the PaddleFormers model/config so that downstream
    PaddleFormers components are aware of FastDeploy's parallel topology.
    """

    parallel_cfg = getattr(fd_config, "parallel_config", None)
    if parallel_cfg is None:
        return

    tp_size = getattr(parallel_cfg, "tensor_parallel_size", 1)
    if tp_size > 1:
        tp_rank = getattr(parallel_cfg, "tensor_parallel_rank", 0)

        for target in (model, getattr(model, "config", None)):
            if target is None:
                continue
            setattr(target, "tensor_parallel_degree", tp_size)
            setattr(target, "tensor_parallel_rank", tp_rank)

        logger.debug(
            "PaddleFormers backend: tensor-parallel stub invoked (size=%s); "
            "modules have been annotated with tensor-parallel metadata.",
            tp_size,
        )


def apply_pipeline_parallel(model: nn.Layer, fd_config: FDConfig) -> None:
    """
    Populate pipeline-parallel metadata on the PaddleFormers model/config.
    """

    parallel_cfg = getattr(fd_config, "parallel_config", None)
    if parallel_cfg is None:
        return

    pp_size = getattr(parallel_cfg, "pipeline_parallel_size", 1)
    if pp_size > 1:
        pp_rank = getattr(parallel_cfg, "pipeline_parallel_rank", 0)

        for target in (model, getattr(model, "config", None)):
            if target is None:
                continue
            setattr(target, "pipeline_parallel_degree", pp_size)
            setattr(target, "pipeline_parallel_rank", pp_rank)

        logger.debug(
            "PaddleFormers backend: pipeline-parallel stub invoked (size=%s); "
            "modules have been annotated with pipeline-parallel metadata.",
            pp_size,
        )


__all__ = [
    "apply_pipeline_parallel",
    "apply_tensor_parallel",
    "clone_pretrained_config",
    "ensure_prefix_name",
    "get_lm_head",
    "infer_architecture_name",
    "inject_fastdeploy_attention",
    "logger",
]
