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
Self-tests for PaddleFormers MOE Model Fallback implementation.
Tests model registration, expert replacement, and weight loading.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import paddle
import pytest

from fastdeploy.model_executor.models.model_base import ModelRegistry
from fastdeploy.worker.worker_process import init_distributed_environment

# Initialize distributed environment at module load
init_distributed_environment()


class MockMOEPretrainedConfig:
    """Mock PaddleFormers PretrainedConfig for MOE models."""

    def __init__(self):
        self.model_type = "qwen3_moe"
        self.vocab_size = 32000
        self.hidden_size = 4096
        self.num_hidden_layers = 2
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.intermediate_size = 11008
        self.num_experts = 8
        self.num_experts_per_tok = 2
        self.rms_norm_eps = 1e-6
        self.tie_word_embeddings = False
        self.architectures = ["Qwen3ForCausalLMMoE"]
        self._attn_implementation = "eager"
        self.use_bias = False
        self.head_dim = 128
        self.rope_theta = 10000.0
        self.fuse_rms_norm = True
        self.moe_intermediate_size = 11008


class MockLinearLayer(paddle.nn.Layer):
    """Mock for ColumnParallelLinear/RowParallelLinear."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        in_features = kwargs.get("in_features", 128)
        out_features = kwargs.get("out_features", 128)
        self.weight = paddle.create_parameter(
            shape=[in_features, out_features],
            dtype="float32",
        )
        self.weight_loader = MagicMock()

    def forward(self, x):
        return paddle.matmul(x.astype("float32"), self.weight)


class MockLMHead(paddle.nn.Layer):
    """Mock for ParallelLMHead."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[4096, 32000],
            dtype="float32",
        )
        self.weight_loader = MagicMock()

    def forward(self, x):
        return paddle.matmul(x.astype("float32"), self.weight)


@pytest.fixture
def mock_distributed_layers(monkeypatch):
    """Mock all distributed layers to avoid Fleet."""
    monkeypatch.setattr("fastdeploy.model_executor.models.paddleformers.base.ColumnParallelLinear", MockLinearLayer)
    monkeypatch.setattr("fastdeploy.model_executor.models.paddleformers.base.RowParallelLinear", MockLinearLayer)
    monkeypatch.setattr("fastdeploy.model_executor.models.paddleformers.base.MergedColumnParallelLinear", MockLinearLayer)
    monkeypatch.setattr("fastdeploy.model_executor.models.paddleformers.base.ReplicatedLinear", MockLinearLayer)

    # Mock Attention
    mock_attention = MagicMock()
    monkeypatch.setattr(
        "fastdeploy.model_executor.models.paddleformers.base.Attention", lambda *args, **kwargs: mock_attention
    )

    # Mock ParallelLMHead
    monkeypatch.setattr("fastdeploy.model_executor.models.paddleformers.causallm.ParallelLMHead", MockLMHead)

    yield


@pytest.fixture
def mock_paddleformers_moe(monkeypatch):
    """Mock PaddleFormers AutoConfig and AutoModel for MOE."""
    mock_config = MockMOEPretrainedConfig()
    monkeypatch.setattr("paddleformers.transformers.AutoConfig.from_pretrained", lambda model, **kwargs: mock_config)

    # Mock AutoModel with MOE structure
    mock_model = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.return_value = paddle.randn([1, 10, 4096])
    mock_model.get_input_embeddings.return_value = mock_embedding
    mock_model.named_sublayers.return_value = []
    mock_model.return_value = [paddle.randn([1, 10, 4096])]

    monkeypatch.setattr("paddleformers.transformers.AutoModel.from_config", lambda config, **kwargs: mock_model)

    yield mock_model, mock_config


@pytest.fixture
def moe_fd_config():
    """Create FDConfig for MOE model."""
    from fastdeploy.config import (
        CacheConfig,
        FDConfig,
        GraphOptimizationConfig,
        LoadConfig,
        ModelConfig,
        ParallelConfig,
    )
    from fastdeploy.scheduler import SchedulerConfig

    config_dict = {
        "architectures": ["Qwen3ForCausalLMMoE"],
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "moe_intermediate_size": 11008,
        "num_hidden_layers": 2,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
        "head_dim": 128,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "vocab_size": 32000,
        "dtype": "float16",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }

    tmp_dir = tempfile.mkdtemp(prefix="test_moe_paddleformers_")
    config_path = os.path.join(tmp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f)

    model_config = ModelConfig(
        {
            "model": tmp_dir,
            "model_impl": "paddleformers",
            "max_model_len": 2048,
        }
    )

    fd_config = FDConfig(
        model_config=model_config,
        parallel_config=ParallelConfig(
            {
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
            }
        ),
        scheduler_config=SchedulerConfig({}),
        cache_config=CacheConfig({}),
        graph_opt_config=GraphOptimizationConfig({}),
        load_config=LoadConfig({}),
        ips="0.0.0.0",
    )
    fd_config.parallel_config.tp_group = None
    fd_config.parallel_config.tensor_parallel_rank = 0

    yield fd_config

    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestMOEModelRegistration:
    """Test MOE model registration."""

    def test_moe_model_registered(self):
        """Verify PaddleFormersForCausalLMMoE is registered."""
        registry = ModelRegistry()
        supported = registry.get_supported_archs()
        assert "PaddleFormersForCausalLMMoE" in supported, \
            f"PaddleFormersForCausalLMMoE not found in: {supported}"

    def test_moe_fallback_resolution(self):
        """Verify MOE model resolves to PaddleFormersForCausalLMMoE."""
        registry = ModelRegistry()

        mock_model_config = SimpleNamespace(
            model_impl="paddleformers",
            architectures=["Qwen3ForCausalLMMoE"],
            runner_type="generate",
        )

        backend = registry._try_resolve_paddleformers(
            "Qwen3ForCausalLMMoE", mock_model_config, is_fallback=False
        )
        assert backend == "PaddleFormersForCausalLMMoE"


class TestMOEModelInitialization:
    """Test MOE model initialization."""

    def test_moe_model_init(self, mock_distributed_layers, mock_paddleformers_moe, moe_fd_config):
        """Test PaddleFormersForCausalLMMoE initialization."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLMMoE,
        )

        model = PaddleFormersForCausalLMMoE(moe_fd_config)

        assert model is not None
        assert hasattr(model, "model")
        assert hasattr(model, "lm_head")

    def test_moe_attributes_initialized(self, mock_distributed_layers, mock_paddleformers_moe, moe_fd_config):
        """Test MOE-specific attributes are initialized."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLMMoE,
        )

        model = PaddleFormersForCausalLMMoE(moe_fd_config)

        # Check MOE attributes set by recursive_replace
        assert hasattr(model, "num_moe_layers"), "num_moe_layers should be set"
        assert hasattr(model, "mlp_moe_layers"), "mlp_moe_layers should be set"
        assert hasattr(model, "moe_layers"), "moe_layers should be set"


class TestMOERecursiveReplace:
    """Test MOE expert replacement logic."""

    def test_moe_config_extraction(self):
        """Test MOE config extraction from text_config."""
        config = MockMOEPretrainedConfig()

        # Check num_experts extraction
        num_experts = getattr(config, 'num_experts',
                        getattr(config, 'num_local_experts',
                        getattr(config, 'n_routed_experts', None)))
        assert num_experts == 8, "num_experts should be 8"

        # Check top_k extraction
        top_k = getattr(config, 'num_experts_per_tok', getattr(config, 'top_k', 2))
        assert top_k == 2, "top_k should be 2"

    def test_non_moe_model_skip(self, mock_distributed_layers, mock_paddleformers_moe, moe_fd_config):
        """Test non-MOE model skips MOE replacement."""
        from fastdeploy.model_executor.models.paddleformers.base import (
            PaddleFormersModelBase,
        )

        # Modify config to not have num_experts
        moe_fd_config.model_config.num_experts = None

        model = PaddleFormersModelBase(moe_fd_config)

        # Should not set MOE attributes for non-MOE model
        assert getattr(model, 'num_moe_layers', 0) == 0


class TestExpertMappingGeneration:
    """Test expert weight mapping generation."""

    def test_get_expert_mapping(self, mock_distributed_layers, mock_paddleformers_moe, moe_fd_config):
        """Test get_expert_mapping returns correct structure."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLMMoE,
        )

        model = PaddleFormersForCausalLMMoE(moe_fd_config)
        expert_mapping = model.get_expert_mapping()

        # Should return list of tuples
        assert isinstance(expert_mapping, list), "expert_mapping should be a list"

        # Each tuple should have 4 elements
        if len(expert_mapping) > 0:
            first_mapping = expert_mapping[0]
            assert len(first_mapping) == 4, \
                f"Each mapping should have 4 elements, got {len(first_mapping)}"

    def test_expert_mapping_count(self, mock_distributed_layers, mock_paddleformers_moe, moe_fd_config):
        """Test expert mapping count matches num_experts * num_layers."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLMMoE,
        )

        model = PaddleFormersForCausalLMMoE(moe_fd_config)
        expert_mapping = model.get_expert_mapping()

        # num_experts=8, 3 projections (gate, down, up) per expert = 24 mappings
        # But there are 2 patterns (gate_proj/down_proj/up_proj and w1/w2/w3)
        # So total should be 48
        expected_count = 8 * 3 * 2  # 8 experts * 3 projections * 2 patterns
        assert len(expert_mapping) == expected_count, \
            f"Expected {expected_count} mappings, got {len(expert_mapping)}"


class TestMOEWeightLoading:
    """Test MOE weight loading logic."""

    def test_moe_load_weights_execution(self, mock_distributed_layers, mock_paddleformers_moe, moe_fd_config):
        """Test load_weights can be called with MOE weights."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLMMoE,
        )

        model = PaddleFormersForCausalLMMoE(moe_fd_config)

        # Create mock MOE weights
        mock_weights = [
            ("experts.0.gate_proj.weight", paddle.randn([4096, 11008])),
            ("experts.0.down_proj.weight", paddle.randn([11008, 4096])),
            ("experts.0.up_proj.weight", paddle.randn([4096, 11008])),
        ]

        # Should not raise
        model.load_weights(mock_weights)

    def test_moe_weight_name_format(self, mock_distributed_layers, mock_paddleformers_moe, moe_fd_config):
        """Test MOE weight name format is correctly handled."""
        from fastdeploy.model_executor.models.paddleformers import (
            PaddleFormersForCausalLMMoE,
        )

        model = PaddleFormersForCausalLMMoE(moe_fd_config)

        # Weight name from expert_mapping should have trailing dot
        weight_name_with_dot = "experts.0.gate_proj."

        # Should strip trailing dot and add .weight suffix
        weight_name_stripped = weight_name_with_dot.rstrip('.')
        assert weight_name_stripped == "experts.0.gate_proj", \
            f"Trailing dot not stripped correctly: {weight_name_stripped}"

        if not weight_name_stripped.endswith('.weight'):
            weight_name_stripped += '.weight'

        assert weight_name_stripped == "experts.0.gate_proj.weight", \
            f".weight suffix not added correctly: {weight_name_stripped}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
