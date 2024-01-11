# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from transformers import T5Config, T5EncoderModel, T5Tokenizer


class T5Encoder(torch.nn.Module):
    def __init__(self, max_seq_len=512, encoder_path=None):
        """
        Initialize the T5 Encoder.

        :param max_seq_len: Maximum token length, defaults to 512
        :param encoder_path: Optional if loaded T5 on the disk, defaults to None
        """
        super().__init__()
        self.max_seq_len = max_seq_len

        self.model_seq_len = 512
        # Initializing T5 model
        self.tokenizer = T5Tokenizer.from_pretrained("t5-11b", model_max_length=self.model_seq_len)

        if encoder_path is None:
            self.model = T5EncoderModel.from_pretrained("t5-11b", low_cpu_mem_usage=True)
        else:
            print(f'Load T5 encoder from {encoder_path}')
            hard_coded_encoder_weight_location = os.path.join(encoder_path, "t5xxl-encoder.bin")
            hard_coded_encoder_config_location = os.path.join(os.path.dirname(__file__), "t5encoder.json")
            self.model = T5EncoderModel.from_pretrained(
                hard_coded_encoder_weight_location,
                config=T5Config.from_json_file(hard_coded_encoder_config_location),
                low_cpu_mem_usage=True,
            )

    def encode(self, text_batch, device='cuda'):
        '''
        Encode a batch of text to T5 embeddings.
        '''
        encoded = self.tokenizer.batch_encode_plus(
            text_batch, return_tensors="pt", padding="max_length", max_length=self.model_seq_len, truncation=True
        )
        # We expect all the processing is done in GPU.
        input_ids = encoded.input_ids.to(device=device)
        attn_mask = encoded.attention_mask.to(device=device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attn_mask)
            encoded_text = output.last_hidden_state.detach()

        encoded_text = encoded_text[:, 0 : self.max_seq_len]
        attn_mask = attn_mask[:, 0 : self.max_seq_len]
        for bnum in range(encoded_text.shape[0]):
            nvalid_elem = attn_mask[bnum].sum().item()
            encoded_text[bnum][nvalid_elem:] = 0

        return encoded_text, attn_mask
