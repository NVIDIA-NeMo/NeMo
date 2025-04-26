# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import pytest

from nemo.deploy.deploy_base import DeployBase


class MockDeployable(DeployBase):
    def deploy(self):
        pass

    def serve(self):
        pass

    def run(self):
        pass

    def stop(self):
        pass


class MockTritonDeployable:
    pass


@pytest.fixture
def mock_model():
    return MagicMock()


@pytest.fixture
def deploy_base(mock_model):
    return MockDeployable(
        triton_model_name="test_model",
        model=mock_model,
        max_batch_size=128,
        http_port=8000,
        grpc_port=8001,
    )


def test_initialization_with_model(deploy_base, mock_model):
    assert deploy_base.triton_model_name == "test_model"
    assert deploy_base.model == mock_model
    assert deploy_base.max_batch_size == 128
    assert deploy_base.http_port == 8000
    assert deploy_base.grpc_port == 8001
    assert deploy_base.address == "0.0.0.0"
    assert deploy_base.allow_grpc is True
    assert deploy_base.allow_http is True
    assert deploy_base.streaming is False


def test_initialization_with_checkpoint():
    with patch('nemo.deploy.deploy_base.ModelPT') as mock_model_pt:
        mock_model_pt.restore_from.return_value = MagicMock()
        deploy_base = MockDeployable(
            triton_model_name="test_model",
            checkpoint_path="test.ckpt",
        )
        assert deploy_base.checkpoint_path == "test.ckpt"


def test_initialization_without_model_or_checkpoint():
    with pytest.raises(Exception) as exc_info:
        MockDeployable(triton_model_name="test_model")
    assert "Either checkpoint_path or model should be provided" in str(exc_info.value)


def test_get_module_and_class():
    module, class_name = DeployBase.get_module_and_class("nemo.models.test_model.TestModel")
    assert module == "nemo.models.test_model"
    assert class_name == "TestModel"


def test_is_model_deployable_valid(deploy_base):
    deploy_base.model = MockTritonDeployable()
    with patch('nemo.deploy.deploy_base.ITritonDeployable', MockTritonDeployable):
        assert deploy_base._is_model_deployable() is True


def test_is_model_deployable_invalid(deploy_base):
    deploy_base.model = MagicMock()
    with patch('nemo.deploy.deploy_base.ITritonDeployable', MockTritonDeployable):
        with pytest.raises(Exception) as exc_info:
            deploy_base._is_model_deployable()
        assert "This model is not deployable to Triton" in str(exc_info.value)
