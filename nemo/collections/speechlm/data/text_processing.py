# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Union

from lhotse.cut import Cut

from nemo.collections.multimodal.speech_llm.parts.utils.data_utils import PromptFormatterTextProcessing


__all__ = ['TextProcesserWithPromptFormatter']


@dataclass
class MockCutSupervision:
    """
    A dummy class for MockCut to support behavior of Cut.supervisions[0].text
    """

    text: str = ""


class MockCut(Cut):
    """
    A warpper class to wrap the input data for the prompt formatter.
    """

    def __init__(self, sample: Union[str, dict], text_key="output"):
        """
        Args:
            sample: The input data to be formatted.
        """
        super().__init__()
        if isinstance(sample, str):
            self.context = sample
        elif isinstance(sample, dict):
            for k, v in sample.items():
                setattr(self, k, v)
                if k != text_key:
                    self.context = v
                    self.question = v
                else:
                    self.answer = v

        self.supervisions = [MockCutSupervision(text=sample[text_key])]

    def has_custom(self, key):
        """Utility function to mimic Cut.has_custom()."""
        return hasattr(self, key)


class TextProcesserWithPromptFormatter(PromptFormatterTextProcessing):
    """
    Wrapper class that processes text and uses the prompt formatter,
    such that the same text processor can be used by both lhotse and non-lhotse datasets.
    """

    def __init__(
        self,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        prompt_format: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        audio_locator: Optional[str] = None,
    ):
        """
        Args:
            tokenizer: The tokenizer to use.
            prompt_format: The prompt format string.
            max_seq_length: The maximum sequence length.
            audio_locator: The audio locator for inserting audio.
        """
        super().__init__(
            tokenizer=tokenizer,
            prompt_format=prompt_format,
            max_seq_length=max_seq_length,
            audio_locator=audio_locator,
        )

    def __call__(self, *args, **kwargs):
        return self.process_sample(*args, **kwargs)

    def wrap_inputs(self, **kwargs):
        """wrap the input data for the prompt formatter."""
        inputs = {k: v for k, v in kwargs.items()}
        return MockCut(inputs)

    def process_sample(self, *args, **kwargs):
        """process the input sample, wether it's a Cut or a dict."""
        if len(args) == 1 and isinstance(args[0], Cut):
            # lhotse dataset and dataloader
            return self._process_example(args[0])
        elif len(args) > 1 or (len(args) == 1 and not isinstance(args[0], Cut)):
            raise ValueError("The input to the text processor must be a single Cut object.")
        elif len(kwargs) > 0 and len(args) > 0:
            raise ValueError("The text processor does not accept both positional and keyword arguments.")
        else:
            # nemo dataset
            return self._process_example(self.wrap_inputs(**kwargs))
