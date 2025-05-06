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


import os
import tempfile
import unittest
from unittest.mock import MagicMock

import numpy as np
import sentencepiece
import torch

from nemo.export.sentencepiece_tokenizer import SentencePieceTokenizer


class TestSentencePieceTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary directory for test files
        cls.test_dir = tempfile.mkdtemp()

        # Create a simple sentencepiece model for testing
        with open(os.path.join(cls.test_dir, "test.txt"), "w") as f:
            f.write("Hello world\nThis is a test\n")

        # Train a simple sentencepiece model
        sentencepiece.SentencePieceTrainer.Train(
            f'--input={os.path.join(cls.test_dir, "test.txt")} '
            f'--model_prefix={os.path.join(cls.test_dir, "test_model")} '
            '--vocab_size=55 --model_type=bpe'
        )

        cls.model_path = os.path.join(cls.test_dir, "test_model.model")

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files
        import shutil

        shutil.rmtree(cls.test_dir)

    def setUp(self):
        self.tokenizer = SentencePieceTokenizer(model_path=self.model_path)

    def test_initialization(self):
        # Test initialization with model path
        tokenizer = SentencePieceTokenizer(model_path=self.model_path)
        self.assertIsNotNone(tokenizer.tokenizer)
        self.assertEqual(tokenizer.original_vocab_size, tokenizer.vocab_size)

        # Test initialization with invalid model path
        with self.assertRaises(ValueError):
            SentencePieceTokenizer(model_path="nonexistent.model")

        # Test initialization with both model_path and tokenizer
        mock_tokenizer = MagicMock()
        with self.assertRaises(ValueError):
            SentencePieceTokenizer(model_path=self.model_path, tokenizer=mock_tokenizer)

        # Test initialization with neither model_path nor tokenizer
        with self.assertRaises(ValueError):
            SentencePieceTokenizer()

    def test_text_to_tokens(self):
        text = "Hello world"
        tokens = self.tokenizer.text_to_tokens(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_encode(self):
        text = "Hello world"
        ids = self.tokenizer.encode(text)
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_tokens_to_text(self):
        text = "Hello world"
        tokens = self.tokenizer.text_to_tokens(text)
        reconstructed_text = self.tokenizer.tokens_to_text(tokens)
        self.assertIsInstance(reconstructed_text, str)
        self.assertNotEqual(reconstructed_text, "")  # Should not be empty

    def test_batch_decode(self):
        text = "Hello world"
        ids = self.tokenizer.encode(text)

        # Test with list
        decoded_text = self.tokenizer.batch_decode(ids)
        self.assertIsInstance(decoded_text, str)

        # Test with numpy array
        ids_np = np.array(ids)
        decoded_text_np = self.tokenizer.batch_decode(ids_np)
        self.assertIsInstance(decoded_text_np, str)

        # Test with torch tensor
        ids_torch = torch.tensor(ids)
        decoded_text_torch = self.tokenizer.batch_decode(ids_torch)
        self.assertIsInstance(decoded_text_torch, str)

    def test_token_to_id(self):
        text = "Hello"
        tokens = self.tokenizer.text_to_tokens(text)
        token_id = self.tokenizer.token_to_id(tokens[0])
        self.assertIsInstance(token_id, int)

    def test_ids_to_tokens(self):
        text = "Hello world"
        ids = self.tokenizer.encode(text)
        tokens = self.tokenizer.ids_to_tokens(ids)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_tokens_to_ids(self):
        text = "Hello"
        tokens = self.tokenizer.text_to_tokens(text)
        ids = self.tokenizer.tokens_to_ids(tokens)
        self.assertIsInstance(ids, list)
        self.assertTrue(all(isinstance(i, int) for i in ids))

    def test_legacy_mode(self):
        special_tokens = ["[PAD]", "[BOS]", "[EOS]"]
        tokenizer = SentencePieceTokenizer(model_path=self.model_path, special_tokens=special_tokens, legacy=True)

        # Test adding special tokens
        self.assertGreater(tokenizer.vocab_size, tokenizer.original_vocab_size)

        # Test special token encoding
        text = "Hello [PAD] world"
        tokens = tokenizer.text_to_tokens(text)
        self.assertIn("[PAD]", tokens)

        # Test special token decoding
        ids = tokenizer.encode(text)
        decoded_text = tokenizer.batch_decode(ids)
        self.assertIn("[PAD]", decoded_text)

    def test_properties(self):
        # Test pad_id property
        self.assertIsInstance(self.tokenizer.pad_id, int)

        # Test bos_token_id property
        self.assertIsInstance(self.tokenizer.bos_token_id, int)

        # Test eos_token_id property
        self.assertIsInstance(self.tokenizer.eos_token_id, int)

        # Test unk_id property
        self.assertIsInstance(self.tokenizer.unk_id, int)

    def test_vocab_property(self):
        vocab = self.tokenizer.vocab
        self.assertIsInstance(vocab, list)
        self.assertTrue(all(isinstance(t, str) for t in vocab))

    def test_convert_ids_to_tokens(self):
        text = "Hello world"
        ids = self.tokenizer.encode(text)
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_convert_tokens_to_string(self):
        text = "Hello world"
        tokens = self.tokenizer.text_to_tokens(text)
        string = self.tokenizer.convert_tokens_to_string(tokens)
        self.assertIsInstance(string, str)

    def test_len(self):
        self.assertEqual(len(self.tokenizer), self.tokenizer.vocab_size)

    def test_is_fast(self):
        self.assertTrue(self.tokenizer.is_fast)

    def test_get_added_vocab(self):
        self.assertIsNone(self.tokenizer.get_added_vocab())


if __name__ == '__main__':
    unittest.main()
