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

from nemo.deploy import ITritonDeployable
from nemo.deploy.deploy_pytriton import DeployPyTriton


class MockModel(ITritonDeployable):
    def triton_infer_fn(self, *args, **kwargs):
        return {"output": "test output"}

    def triton_infer_fn_streaming(self, *args, **kwargs):
        yield {"output": "test output"}

    def get_triton_input(self):
        return [{"name": "input", "dtype": "string", "shape": (-1,)}]

    def get_triton_output(self):
        return [{"name": "output", "dtype": "string", "shape": (-1,)}]


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def deploy_pytriton(mock_model):
    return DeployPyTriton(triton_model_name="test_model", model=mock_model, http_port=8000, grpc_port=8001)


@patch('nemo.deploy.deploy_pytriton.Triton')
def test_deploy_success(mock_triton, deploy_pytriton):
    deploy_pytriton.deploy()
    assert deploy_pytriton.triton is not None
    mock_triton.return_value.bind.assert_called_once()


@patch('nemo.deploy.deploy_pytriton.Triton')
def test_deploy_streaming_success(mock_triton):
    deploy = DeployPyTriton(triton_model_name="test_model", model=MockModel(), streaming=True)
    deploy.deploy()
    assert deploy.triton is not None
    mock_triton.return_value.bind.assert_called_once()


@patch('nemo.deploy.deploy_pytriton.Triton')
def test_deploy_failure(mock_triton, deploy_pytriton):
    mock_triton.side_effect = Exception("Deployment failed")
    deploy_pytriton.deploy()
    assert deploy_pytriton.triton is None


def test_serve_success(deploy_pytriton):
    deploy_pytriton.triton = MagicMock()
    deploy_pytriton.serve()
    deploy_pytriton.triton.serve.assert_called_once()


def test_serve_failure(deploy_pytriton):
    deploy_pytriton.triton = None
    with pytest.raises(Exception, match="deploy should be called first."):
        deploy_pytriton.serve()


def test_run_success(deploy_pytriton):
    deploy_pytriton.triton = MagicMock()
    deploy_pytriton.run()
    deploy_pytriton.triton.run.assert_called_once()


def test_run_failure(deploy_pytriton):
    deploy_pytriton.triton = None
    with pytest.raises(Exception, match="deploy should be called first."):
        deploy_pytriton.run()


def test_stop_success(deploy_pytriton):
    deploy_pytriton.triton = MagicMock()
    deploy_pytriton.stop()
    deploy_pytriton.triton.stop.assert_called_once()


def test_stop_failure(deploy_pytriton):
    deploy_pytriton.triton = None
    with pytest.raises(Exception, match="deploy should be called first."):
        deploy_pytriton.stop()
