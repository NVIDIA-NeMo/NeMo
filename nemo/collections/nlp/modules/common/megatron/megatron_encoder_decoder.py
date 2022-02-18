# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer based language model."""

from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.utils import ApexGuardDefaults

try:
    from apex.transformer.enums import AttnMaskType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False
    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()


__all__ = ["MegatronTransformerEncoderDecoderModule"]


class MegatronTransformerEncoderDecoderModule(MegatronModule):
    """Transformer encoder-decoder model.
    """

    def __init__(
        self,
        encoder,
        decoder,
        # AttnMaskType enum mask type (e.g., padding, casual)
        encoder_attn_mask_type: AttnMaskType = None,
        decoder_attn_mask_type: AttnMaskType = None,
    ):
        super(MegatronTransformerEncoderDecoderModule, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        # try to infer mask_type if not given
        if encoder_attn_mask_type is None:
            try:
                encoder_attn_mask_type = encoder.model.self_attn_mask_type
            except Exception as e:
                raise ValueError("Failed inferring encoder_attn_mask_type, please provide AttnMaskType value")
        if decoder_attn_mask_type is None:
            try:
                decoder_attn_mask_type = decoder.model.self_attn_mask_type
            except Exception as e:
                raise ValueError("Failed inferring decoder_attn_mask_type, please provide AttnMaskType value")

        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.decoder_attn_mask_type = decoder_attn_mask_type

        self._encoder_key = "encoder"
        self._decoder_key = "decoder"

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        self.encoder.set_input_tensor(input_tensor)

    def encode(
        self, enc_input, enc_attn_mask, enc_layer_past=None, enc_get_key_value=False,
    ):
        """Encodes embedder input using encoder"""
        enc_output, enc_output_mask = self.encoder(
            enc_input=enc_input,
            enc_attn_mask=enc_attn_mask,
            layer_past=enc_layer_past,
            get_key_value=enc_get_key_value,
        )

        return enc_output, enc_output_mask

    def decode(
        self, dec_input, dec_attn_mask, enc_output, enc_output_mask, dec_layer_past=None, dec_get_key_value=False,
    ):
        """Decodes embedder input using decoder and encoder input"""
        dec_output = self.decoder(
            dec_input=dec_input,
            dec_attn_mask=dec_attn_mask,
            layer_past=dec_layer_past,
            get_key_value=dec_get_key_value,
            enc_output=enc_output,
            enc_output_mask=enc_output_mask,
        )

        return dec_output

    def forward(
        self,
        enc_input,
        enc_attn_mask,
        dec_input,
        dec_attn_mask,
        enc_layer_past=None,
        enc_get_key_value=False,
        enc_output=None,
        enc_output_mask=None,
        dec_layer_past=None,
        dec_get_key_value=False,
    ):
        # encoder
        if enc_output is None:
            enc_output, enc_output_mask = self.encode(
                enc_input=enc_input,
                enc_attn_mask=enc_attn_mask,
                enc_layer_past=enc_layer_past,
                enc_get_key_value=enc_get_key_value,
            )
        elif enc_output_mask is None:
            raise ValueError("enc_output_mask cannot be None when enc_output is not None")

        # decoder
        dec_output = self.decode(
            dec_input=dec_input,
            dec_attn_mask=dec_attn_mask,
            enc_output=enc_output,
            enc_output_mask=enc_output_mask,
            dec_layer_past=dec_layer_past,
            dec_get_key_value=dec_get_key_value,
        )

        ret_dict = {
            "enc_output": enc_output,
            "enc_output_mask": enc_output_mask,
            "dec_output": dec_output,
        }

        return ret_dict

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}

        state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._decoder_key] = self.decoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        self.encoder.load_state_dict(state_dict[self._encoder_key], strict=strict)
        self.decoder.load_state_dict(state_dict[self._decoder_key], strict=strict)
