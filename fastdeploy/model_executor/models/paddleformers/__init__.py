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

from fastdeploy.model_executor.models.model_base import ModelRegistry, ModelCategory, ModelForCasualLM
from .base import PaddleFormersModelBase
from .base_pure import PaddleFormersPureModelBase
from .causallm import CausalLMMixin
from .causallm_pure import CausalLMPureMixin

from fastdeploy.model_executor.graph_optimization.decorator import support_graph_optimization

__all__ = ["PaddleFormersForCausalLM", "PaddleFormersPureForCausalLM"]

@ModelRegistry.register_model_class(
    architecture="PaddleFormersForCausalLM",
    module_name="paddleformers",
    category=ModelCategory.TEXT_GENERATION,
)
class PaddleFormersForCausalLM(CausalLMMixin, PaddleFormersModelBase, ModelForCasualLM): ...

# Pure PaddleFormers mode (no layer replacements, for training-inference consistency)

@ModelRegistry.register_model_class(
    architecture="PaddleFormersPureForCausalLM",
    module_name="paddleformers",
    category=ModelCategory.TEXT_GENERATION,
)
@support_graph_optimization
class PaddleFormersPureForCausalLM(CausalLMPureMixin, ModelForCasualLM, PaddleFormersPureModelBase): ...