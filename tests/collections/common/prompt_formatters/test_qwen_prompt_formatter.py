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
import torch

from nemo.collections.common.prompts.qwen import Qwen3_5PromptFormatter, Qwen3PromptFormatter, QwenPromptFormatter


def test_qwen_prompt_formatter_training(bpe_tokenizer):
    formatter = QwenPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    # The test tokenizer inserts an extra space, but it was verified that AutoTokenizer("Qwen/Qwen3-1.7B") doesn't.
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\nTEST<|im_end|>\n'
    assert bpe_tokenizer.ids_to_text(ans["context_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()) == '<|im_start|>assistant\nTEST<|im_end|>\n'
    assert torch.is_tensor(ans["mask"])
    # fmt: on


def test_qwen_prompt_formatter_inference(bpe_tokenizer):
    formatter = QwenPromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids"}
    # fmt: off
    # The test tokenizer inserts an extra space, but it was verified that AutoTokenizer("Qwen/Qwen3-1.7B") doesn't.
    assert ans["input_ids"].tolist() == ans["context_ids"].tolist()
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n'
    # fmt: on


def test_qwen3_5_prompt_formatter_training(bpe_tokenizer):
    formatter = Qwen3_5PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog(
        [
            {"role": "user", "slots": {"message": "TEST"}},
            {"role": "assistant", "slots": {"message": "TEST"}},
        ]
    )
    assert set(ans) == {"input_ids", "context_ids", "answer_ids", "mask"}
    # fmt: off
    # Qwen3.5 always emits a thinking block; an empty one means "no reasoning for this turn".
    assert bpe_tokenizer.ids_to_text(ans["input_ids"].tolist()) == '<|im_start|>user\nTEST<|im_end|>\n <|im_start|>assistant\n<think>\n\n</think>\n\nTEST<|im_end|>\n'
    assert bpe_tokenizer.ids_to_text(ans["answer_ids"].tolist()) == '<|im_start|>assistant\n<think>\n\n</think>\n\nTEST<|im_end|>\n'
    # fmt: on
    assert torch.is_tensor(ans["mask"])


def test_qwen3_5_prompt_formatter_inference_opens_thinking_block(bpe_tokenizer):
    formatter = Qwen3_5PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog([{"role": "user", "slots": {"message": "TEST"}}], enable_thinking=True)
    assert set(ans) == {"input_ids", "context_ids"}
    text = bpe_tokenizer.ids_to_text(ans["input_ids"].tolist())
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    assert "</think>" not in text


def test_qwen3_5_prompt_formatter_inference_closes_thinking_block(bpe_tokenizer):
    formatter = Qwen3_5PromptFormatter(bpe_tokenizer)
    ans = formatter.encode_dialog([{"role": "user", "slots": {"message": "TEST"}}], enable_thinking=False)
    text = bpe_tokenizer.ids_to_text(ans["input_ids"].tolist())
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_qwen3_5_prompt_formatter_defaults_to_no_thinking(bpe_tokenizer):
    """Qwen3.5 opens a reasoning block only when explicitly asked, unlike Qwen3.

    Training targets always carry a closed block, so a thinking default would prime reasoning
    at inference for a prompt the model was fine-tuned to answer directly.
    """
    formatter = Qwen3_5PromptFormatter(bpe_tokenizer)
    turns = [{"role": "user", "slots": {"message": "TEST"}}]
    default = formatter.encode_dialog([dict(t) for t in turns])
    explicit = formatter.encode_dialog([dict(t) for t in turns], enable_thinking=False)
    assert default["input_ids"].tolist() == explicit["input_ids"].tolist()
    assert bpe_tokenizer.ids_to_text(default["input_ids"].tolist()).endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def test_qwen3_5_prompt_formatter_does_not_inject_think_tags(bpe_tokenizer):
    """Qwen3.5's chat template has no ``/think`` markers, unlike Qwen3's."""

    def user_messages_after_encoding(formatter_cls):
        seen = set()
        for _ in range(50):
            turns = [
                {"role": "user", "slots": {"message": "TEST"}},
                {"role": "assistant", "slots": {"message": "TEST"}},
            ]
            formatter_cls(bpe_tokenizer).encode_dialog(turns)
            seen.add(turns[0]["slots"]["message"])
        return seen

    assert user_messages_after_encoding(Qwen3_5PromptFormatter) == {"TEST"}
    # Sanity check that the shared code path still injects tags for Qwen3 itself.
    assert user_messages_after_encoding(Qwen3PromptFormatter) != {"TEST"}
