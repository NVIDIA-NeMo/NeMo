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

import numpy as np
import pytest

from nemo.deploy.nlp.query_llm import NemoQueryLLM, NemoQueryLLMBase, NemoQueryLLMHF, NemoQueryLLMPyTorch


class TestNemoQueryLLMBase:
    def test_base_initialization(self):
        url = "localhost:8000"
        model_name = "test-model"
        query = NemoQueryLLMBase(url=url, model_name=model_name)
        assert query.url == url
        assert query.model_name == model_name


class TestNemoQueryLLMPyTorch:
    @pytest.fixture
    def query(self):
        return NemoQueryLLMPyTorch(url="localhost:8000", model_name="test-model")

    def test_initialization(self, query):
        assert isinstance(query, NemoQueryLLMBase)
        assert query.url == "localhost:8000"
        assert query.model_name == "test-model"

    @patch('nemo.deploy.nlp.query_llm.ModelClient')
    def test_query_llm_basic(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"sentences": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test basic query
        response = query.query_llm(prompts=["test prompt"], max_length=100, temperature=0.7, top_k=1, top_p=0.9)

        assert isinstance(response, dict)
        assert "choices" in response
        assert response["choices"][0]["text"] == "test response"

    @patch('nemo.deploy.nlp.query_llm.ModelClient')
    def test_query_llm_with_logprobs(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {
            "sentences": np.array([b"test response"]),
            "log_probs": np.array([0.1, 0.2, 0.3]),
        }
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with logprobs
        response = query.query_llm(prompts=["test prompt"], max_length=100, compute_logprob=True)

        assert "logprobs" in response["choices"][0]
        assert "token_logprobs" in response["choices"][0]["logprobs"]


class TestNemoQueryLLMHF:
    @pytest.fixture
    def query(self):
        return NemoQueryLLMHF(url="localhost:8000", model_name="test-model")

    def test_initialization(self, query):
        assert isinstance(query, NemoQueryLLMBase)
        assert query.url == "localhost:8000"
        assert query.model_name == "test-model"

    @patch('nemo.deploy.nlp.query_llm.ModelClient')
    def test_query_llm_basic(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"sentences": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test basic query
        response = query.query_llm(prompts=["test prompt"], max_length=100, temperature=0.7, top_k=1, top_p=0.9)

        assert isinstance(response, dict)
        assert "choices" in response
        assert response["choices"][0]["text"] == "test response"

    @patch('nemo.deploy.nlp.query_llm.ModelClient')
    def test_query_llm_with_logits(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {
            "sentences": np.array([b"test response"]),
            "logits": np.array([[0.1, 0.2, 0.3]]),
        }
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with logits
        response = query.query_llm(prompts=["test prompt"], max_length=100, output_logits=True)

        assert "logits" in response


class TestNemoQueryLLM:
    @pytest.fixture
    def query(self):
        return NemoQueryLLM(url="localhost:8000", model_name="test-model")

    def test_initialization(self, query):
        assert isinstance(query, NemoQueryLLMBase)
        assert query.url == "localhost:8000"
        assert query.model_name == "test-model"

    @patch('nemo.deploy.nlp.query_llm.ModelClient')
    def test_query_llm_basic(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test basic query
        response = query.query_llm(prompts=["test prompt"], max_output_len=100, temperature=0.7, top_k=1, top_p=0.9)

        assert isinstance(response[0], str)
        assert response[0] == "test response"

    @patch('nemo.deploy.nlp.query_llm.ModelClient')
    def test_query_llm_openai_format(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with OpenAI format
        response = query.query_llm(prompts=["test prompt"], max_output_len=100, openai_format_response=True)

        assert isinstance(response, dict)
        assert "choices" in response
        assert response["choices"][0]["text"] == "test response"

    @patch('nemo.deploy.nlp.query_llm.DecoupledModelClient')
    def test_query_llm_streaming(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = [
            {"outputs": np.array([b"test"])},
            {"outputs": np.array([b" response"])},
        ]
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test streaming query
        responses = list(query.query_llm_streaming(prompts=["test prompt"], max_output_len=100))

        assert len(responses) == 2
        assert responses[0] == "test"
        assert responses[1] == " response"

    @patch('nemo.deploy.nlp.query_llm.ModelClient')
    def test_query_llm_with_stop_words(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with stop words
        response = query.query_llm(prompts=["test prompt"], max_output_len=100, stop_words_list=["stop"])

        assert isinstance(response[0], str)
        assert response[0] == "test response"

    @patch('nemo.deploy.nlp.query_llm.ModelClient')
    def test_query_llm_with_bad_words(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with bad words
        response = query.query_llm(prompts=["test prompt"], max_output_len=100, bad_words_list=["bad"])

        assert isinstance(response[0], str)
        assert response[0] == "test response"
