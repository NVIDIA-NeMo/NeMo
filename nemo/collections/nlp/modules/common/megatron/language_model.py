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
import math

import torch
import torch.nn as nn
import torch.nn.init as init

from nemo.collections.nlp.modules.common.megatron.module import MegatronModule
from nemo.collections.nlp.modules.common.megatron.transformer import ParallelTransformer
from nemo.collections.nlp.modules.common.megatron.utils import (
    ApexGuardDefaults,
    get_linear_layer,
    init_method_normal,
    scaled_init_method_normal,
)

try:
    from apex.transformer import parallel_state, tensor_parallel
    from apex.transformer.enums import AttnMaskType, LayerType

    HAVE_APEX = True
except (ImportError, ModuleNotFoundError):
    HAVE_APEX = False

    # fake missing classes with None attributes
    AttnMaskType = ApexGuardDefaults()
    LayerType = ApexGuardDefaults()


def get_language_model(
    hidden_size,
    ffn_hidden_size,
    num_layers,
    max_position_embeddings,
    num_tokentypes,
    add_pooler,
    vocab_size,
    num_attention_heads,
    encoder_attn_mask_type,
    apply_query_key_layer_scaling=True,
    kv_channels=None,
    init_method=None,
    scaled_init_method=None,
    add_decoder=False,
    decoder_attn_mask_type=AttnMaskType.causal,
    pre_process=True,
    post_process=True,
    init_method_std=0.02,
    use_cpu_initialization=False,
    hidden_dropout=0.1,
    precision=16,
    fp32_residual_connection=False,
    activations_checkpoint_method=None,
    activations_checkpoint_num_layers=1,
    layernorm_epsilon=1e-5,
    bias_gelu_fusion=True,
    masked_softmax_fusion=True,
    persist_layer_norm=False,
    openai_gelu=False,
    onnx_safe=False,
    use_soft_prompts=False,
    num_prompt_tokens=10,
    existing_prompt_tags=None,
):
    """Build language model and return along with the key to save."""

    if kv_channels is None:
        assert (
            hidden_size % num_attention_heads == 0
        ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
        kv_channels = hidden_size // num_attention_heads

    if init_method is None:
        init_method = init_method_normal(init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(init_method_std, num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        encoder_attn_mask_type=encoder_attn_mask_type,
        num_tokentypes=num_tokentypes,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        apply_query_key_layer_scaling=apply_query_key_layer_scaling,
        kv_channels=kv_channels,
        ffn_hidden_size=ffn_hidden_size,
        add_decoder=add_decoder,
        decoder_attn_mask_type=decoder_attn_mask_type,
        add_pooler=add_pooler,
        pre_process=pre_process,
        post_process=post_process,
        use_cpu_initialization=use_cpu_initialization,
        hidden_dropout=hidden_dropout,
        precision=precision,
        fp32_residual_connection=fp32_residual_connection,
        activations_checkpoint_method=activations_checkpoint_method,
        activations_checkpoint_num_layers=activations_checkpoint_num_layers,
        layernorm_epsilon=layernorm_epsilon,
        bias_gelu_fusion=bias_gelu_fusion,
        masked_softmax_fusion=masked_softmax_fusion,
        persist_layer_norm=persist_layer_norm,
        openai_gelu=openai_gelu,
        onnx_safe=onnx_safe,
        use_soft_prompts=use_soft_prompts,
        num_prompt_tokens=num_prompt_tokens,
        existing_prompt_tags=existing_prompt_tags,
    )
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [b, s, h]prompt_embeddings
        # sequence_index: index of the token to pool.
        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        use_cpu_initialization=False,
    ):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            vocab_size, self.hidden_size, init_method=self.init_method, use_cpu_initialization=use_cpu_initialization,
        )
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)
        self._position_embeddings_key = 'position_embeddings'
        # Initialize the position embeddings.
        self.init_method(self.position_embeddings.weight)

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes), flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes, self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None, separate_embeddings=False):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Want word embeddings and position embeddings before addition for soft prompt initalization
        if separate_embeddings:
            return words_embeddings, position_embeddings

        embeddings = words_embeddings + position_embeddings
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] = self.position_embeddings.state_dict(
            destination, prefix, keep_vars
        )
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = self.tokentype_embeddings.state_dict(
                destination, prefix, keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'position_embeddings' in key:
                    state_dict_[key.split('position_embeddings.')[1]] = state_dict[key]
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_, strict=strict)
            else:
                print(
                    '***WARNING*** expected tokentype embeddings in the ' 'checkpoint but could not find it',
                    flush=True,
                )


class PromptEmbedding(MegatronModule):
    """Prompt embeddings

    Arugments:
        init_from_prompt_text: Whether to intialize prompt embeddings
                               from from certain lm embeddings
                               corresponding to a prompt string
        hidden_size: hidden size should match lm embedding size
        num_prompt_tokens: length of prompt initalized from torch init method
        position_embedding_weights: embedding vectors for positions 0 through 
                                    num_prompt_tokens
        word_embedding_weights: token embedding vectors for text init option
        init_method: pytorch init method
        embedding_weights: token embeddings from prompt text
        prompt_embedding_dropout_prob: dropout probablity
    """

    def __init__(
        self,
        init_from_prompt_text,
        hidden_size,
        num_prompt_tokens,
        position_embedding_weights=None,
        word_embedding_weights=None,
        init_method=init.xavier_normal_,
        prompt_embedding_dropout_prob=0.1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_prompt_tokens = num_prompt_tokens

        # Randomly init token and position embeddings
        self.prompt_embeddings = torch.nn.Embedding(self.num_prompt_tokens, self.hidden_size)
        self.position_embeddings = torch.nn.Embedding(self.num_prompt_tokens, self.hidden_size)
        init_method(self.prompt_embeddings.weight)
        init_method(self.position_embeddings.weight)

        # Set embedding weights to be embeddings from prompt tokens
        if init_from_prompt_text:
            self.prompt_embeddings.weight = nn.Parameter(word_embedding_weights)
        if position_embedding_weights != None:
            self.position_embeddings.weight = nn.Parameter(position_embedding_weights)

        # Set keys for loading and saving weights
        self._prompt_embeddings_key = 'prompt_embeddings'
        self._position_embeddings_key = 'position_embeddings'

        # Set ids needed for forward pass and broadcast them
        # ids = {'ids': torch.arange(self.num_prompt_tokens, dtype=torch.int64)}
        # ids_b = tensor_parallel.broadcast_data(['ids'], ids, torch.int64)
        # self.ids = ids_b['ids'].long()
        self.ids = torch.arange(self.num_prompt_tokens, dtype=torch.int64)

        self.embedding_dropout = torch.nn.Dropout(prompt_embedding_dropout_prob)

    def forward(self, tokentype_ids=None):
        # Embeddings.
        device = next(self.prompt_embeddings.parameters()).device
        prompt_embeddings = self.prompt_embeddings(self.ids.to(device))
        position_embeddings = self.position_embeddings(self.ids.to(device))
        embeddings = prompt_embeddings + position_embeddings

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    # These save and load methods don't actually seem to be called during training or when restoring the model
    # But I've added them because the other transformer submodules have them
    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""
        state_dict_ = {}
        state_dict_[self._prompt_embeddings_key] = self.prompt_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] = self.position_embeddings.state_dict(
            destination, prefix, keep_vars
        )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._prompt_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'prompt_embeddings' in key:
                    state_dict_[key.split('prompt_embeddings.')[1]] = state_dict[key]
        self.prompt_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'position_embeddings' in key:
                    state_dict_[key.split('position_embeddings.')[1]] = state_dict[key]
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)


class PromptTable(torch.nn.Module):
    def __init__(
        self, existing_prompt_tags, num_prompt_tokens, hidden_size,
    ):
        super().__init__()

        self.num_prompt_tokens = num_prompt_tokens
        self.hidden_size = hidden_size
        self.prompt_table = torch.nn.ModuleDict()
        self.prompt_id_to_tag = {}

        if existing_prompt_tags:
            for tag, prompt_id in existing_prompt_tags:
                self.prompt_id_to_tag[prompt_id] = tag
                self.prompt_table[tag] = PromptEmbedding(
                    init_from_prompt_text=False,
                    hidden_size=self.hidden_size,
                    num_prompt_tokens=self.num_prompt_tokens,
                )

    def forward(self, prompt_id):
        prompt_id = prompt_id.item()
        prompt_tag = self.prompt_id_to_tag[prompt_id]
        return self.prompt_table[prompt_tag]()

    def remove_prompt(self, prompt_tag):
        if prompt_tag not in prompt_table:
            return

        # find the prompt_id assocaited with the tag to delete
        prompt_id = None
        for key, value in prompt_id_to_tag.items():
            if value == prompt_tag:
                prompt_id = key
                break

        del self.prompt_id_to_tag[prompt_id]
        del self.prompt_table[prompt_tag]

    def init_prompt_from_random(self, prompt_tag, prompt_id, embeddings):
        """Add new soft prompt to be tuned.
           Intialize prompt weights using pytorch init method

        """
        # Initalize prompt embeddings from a pytorch random init method
        prompt_embeddings = PromptEmbedding(
            init_from_prompt_text=False, hidden_size=self.hidden_size, num_prompt_tokens=self.num_prompt_tokens,
        )

        self.prompt_table[prompt_tag] = prompt_embeddings
        self.prompt_id_to_tag[prompt_id] = prompt_tag

    def init_prompt_from_text(self, prompt_tag, prompt_id, init_token_ids, embeddings):
        """Add new soft prompt to be tuned.
           Intialize prompt weights from existing embeddings from specific vocab tokens.

        """
        # Trim or iterate until num_text_tokens matches num_prompt_tokens
        num_text_tokens = len(init_token_ids)
        num_prompt_tokens = self.num_prompt_tokens

        if num_text_tokens > num_prompt_tokens:
            init_token_ids = init_token_ids[:num_prompt_tokens]
        elif num_text_tokens < num_prompt_tokens:
            num_reps = math.ceil(num_prompt_tokens / num_text_tokens)
            init_token_ids = init_token_ids * num_reps

        # Set dictionary item keys and datatypes for broadcasting
        keys = ['text']
        datatype = torch.int64

        # Broadcast int ids across gpus for tensor parallel
        init_token_ids = init_token_ids[:num_prompt_tokens]
        init_token_ids = {'text': torch.tensor(init_token_ids, dtype=torch.int64)}
        init_token_ids_b = tensor_parallel.broadcast_data(keys, init_token_ids, datatype)
        init_token_ids = init_token_ids_b['text'].long()
        init_position_ids = torch.arange(self.num_prompt_tokens, dtype=torch.long, device=init_token_ids.device)

        # Use a copy of token embedding weights to initalize the prompt embeddings
        word_embeddings, position_embeddings = embeddings(init_token_ids, init_position_ids, separate_embeddings=True)

        word_embeddings = word_embeddings.detach().clone()
        position_embeddings = position_embeddings.detach().clone()

        prompt_embeddings = PromptEmbedding(
            init_from_prompt_text=True,
            hidden_size=self.hidden_size,
            num_prompt_tokens=self.num_prompt_tokens,
            word_embedding_weights=word_embeddings,
            position_embedding_weights=position_embeddings,
        )

        self.prompt_table[prompt_tag] = prompt_embeddings
        self.prompt_id_to_tag[prompt_id] = prompt_tag

    def load_state_dict(self, state_dict_, strict):
        for prompt_tag in self.prompt_table:
            self.prompt_table[prompt_tag].load_state_dict(state_dict_[prompt_tag], strict=strict)

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        prompt_state_dict_ = {}
        for prompt_tag in self.prompt_table:
            prompt_state_dict_[prompt_tag] = self.prompt_table[prompt_tag].state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )

        return prompt_state_dict_


class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        encoder_attn_mask_type,
        vocab_size,
        max_position_embeddings,
        hidden_size,
        ffn_hidden_size,
        num_layers,
        num_tokentypes,
        num_attention_heads,
        apply_query_key_layer_scaling=True,
        kv_channels=None,
        add_decoder=False,
        decoder_attn_mask_type=AttnMaskType.causal,
        add_pooler=False,
        pre_process=True,
        post_process=True,
        use_cpu_initialization=False,
        hidden_dropout=0.1,
        precision=16,
        fp32_residual_connection=False,
        activations_checkpoint_method=None,
        activations_checkpoint_num_layers=1,
        layernorm_epsilon=1e-5,
        bias_gelu_fusion=True,
        masked_softmax_fusion=True,
        persist_layer_norm=False,
        openai_gelu=False,
        onnx_safe=False,
        use_soft_prompts=False,
        num_prompt_tokens=100,
        existing_prompt_tags=None,
    ):
        super(TransformerLanguageModel, self).__init__()

        self.pre_process = pre_process
        self.post_process = post_process
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.encoder_attn_mask_type = encoder_attn_mask_type
        self.add_decoder = add_decoder
        self.decoder_attn_mask_type = decoder_attn_mask_type
        self.add_pooler = add_pooler
        self.hidden_dropout = hidden_dropout
        self.output_layer_init_method = output_layer_init_method
        self.use_soft_prompts = use_soft_prompts
        self.existing_prompt_tags = existing_prompt_tags
        self.num_prompt_tokens = num_prompt_tokens

        if kv_channels is None:

            assert (
                hidden_size % num_attention_heads == 0
            ), 'hidden_size must be divisible by num_attention_heads if kv_channels is None'
            kv_channels = hidden_size // num_attention_heads

        # Embeddings.
        if self.pre_process:
            self.embedding = Embedding(
                hidden_size=self.hidden_size,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_position_embeddings,
                init_method=self.init_method,
                num_tokentypes=self.num_tokentypes,
                use_cpu_initialization=use_cpu_initialization,
                embedding_dropout_prob=self.hidden_dropout,
            )
            self._embedding_key = 'embedding'

        # Soft Prompts
        if self.use_soft_prompts:
            self.prompt_table = PromptTable(
                existing_prompt_tags=self.existing_prompt_tags,
                num_prompt_tokens=self.num_prompt_tokens,
                hidden_size=self.hidden_size,
            )
            self._prompt_table_key = 'prompt_table'

        # Transformer.
        self.encoder = ParallelTransformer(
            init_method=self.init_method,
            output_layer_init_method=self.output_layer_init_method,
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=num_attention_heads,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            kv_channels=kv_channels,
            ffn_hidden_size=ffn_hidden_size,
            self_attn_mask_type=self.encoder_attn_mask_type,
            pre_process=self.pre_process,
            post_process=self.post_process,
            precision=precision,
            fp32_residual_connection=fp32_residual_connection,
            activations_checkpoint_method=activations_checkpoint_method,
            activations_checkpoint_num_layers=activations_checkpoint_num_layers,
            layernorm_epsilon=layernorm_epsilon,
            hidden_dropout=hidden_dropout,
            use_cpu_initialization=use_cpu_initialization,
            bias_gelu_fusion=bias_gelu_fusion,
            persist_layer_norm=persist_layer_norm,
            openai_gelu=openai_gelu,
            onnx_safe=onnx_safe,
            masked_softmax_fusion=masked_softmax_fusion,
        )
        self._encoder_key = 'encoder'

        # Decoder
        if self.add_decoder:
            assert (
                parallel_state.get_pipeline_model_parallel_world_size() == 1
            ), 'pipeline parallelism is not supported in the presence of decoder'
            self.decoder = ParallelTransformer(
                layer_type=LayerType.decoder,
                self_attn_mask_type=self.decoder_attn_mask_type,
                init_method=self.init_method,
                output_layer_init_method=self.output_layer_init_method,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                num_attention_heads=num_attention_heads,
                apply_query_key_layer_scaling=apply_query_key_layer_scaling,
                kv_channels=kv_channels,
                ffn_hidden_size=ffn_hidden_size,
                pre_process=self.pre_process,
                post_process=self.post_process,
                precision=precision,
                fp32_residual_connection=fp32_residual_connection,
                activations_checkpoint_method=activations_checkpoint_method,
                activations_checkpoint_num_layers=activations_checkpoint_num_layers,
                layernorm_epsilon=layernorm_epsilon,
                hidden_dropout=hidden_dropout,
                use_cpu_initialization=use_cpu_initialization,
                bias_gelu_fusion=bias_gelu_fusion,
                persist_layer_norm=persist_layer_norm,
                openai_gelu=openai_gelu,
                onnx_safe=onnx_safe,
                masked_softmax_fusion=masked_softmax_fusion,
            )
            self._decoder_key = 'decoder'

        if self.post_process:
            # Pooler.
            if self.add_pooler:
                self.pooler = Pooler(self.hidden_size, self.init_method)
                self._pooler_key = 'pooler'

    def set_input_tensor(self, input_tensor):
        """ See megatron.model.transformer.set_input_tensor()"""
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        self.encoder.set_input_tensor(input_tensor[0])

    def forward(
        self,
        enc_input_ids,
        enc_position_ids,
        enc_attn_mask,
        prompt_ids=None,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attn_mask=None,
        enc_dec_attn_mask=None,
        tokentype_ids=None,
        layer_past=None,
        get_key_value=False,
        pooling_sequence_index=0,
        enc_hidden_states=None,
        output_enc_hidden_only=False,
        encoder_input=None,
    ):
        # Embeddings.
        if self.pre_process and encoder_input is None:
            embedding_output = self.embedding(enc_input_ids, enc_position_ids, tokentype_ids=tokentype_ids)

            # Soft prompts
            if self.use_soft_prompts and prompt_ids != None:
                prompt_embeddings = [self.prompt_table(prompt_id) for prompt_id in prompt_ids]
                prompt_embeddings = torch.stack(prompt_embeddings)
                encoder_input = torch.cat((prompt_embeddings, embedding_output), dim=1)
            else:
                encoder_input = embedding_output
        else:
            pass

        # encoder.
        if enc_hidden_states is None:
            encoder_output = self.encoder(
                encoder_input, enc_attn_mask, layer_past=layer_past, get_key_value=get_key_value
            )
        else:
            encoder_output = enc_hidden_states.to(encoder_input.dtype)

        if self.post_process:
            if self.add_pooler:
                pooled_output = self.pooler(encoder_output, pooling_sequence_index)

        # output_enc_hidden_only refers to when we just need the encoder's
        # output. For example, it is helpful to compute
        # similarity between two sequences by average pooling
        if not self.add_decoder or output_enc_hidden_only:
            if self.add_pooler and self.post_process:
                return encoder_output, pooled_output
            else:
                return encoder_output

        # Decoder Embedding
        dec_embedding_output = self.embedding(dec_input_ids, dec_position_ids)
        # decoder
        decoder_output = self.decoder(
            dec_embedding_output,
            dec_attn_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            encoder_output=encoder_output,
            enc_dec_attn_mask=enc_dec_attn_mask,
        )

        if self.add_pooler and self.post_process:
            return decoder_output, encoder_output, pooled_output
        else:
            return decoder_output, encoder_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.pre_process:
            state_dict_[self._embedding_key] = self.embedding.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )

        if self.use_soft_prompts:
            state_dict_[self._prompt_table_key] = self.prompt_table.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )

        state_dict_[self._encoder_key] = self.encoder.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        if self.post_process:
            if self.add_pooler:
                state_dict_[self._pooler_key] = self.pooler.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars
                )
        if self.add_decoder:
            state_dict_[self._decoder_key] = self.decoder.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars
            )

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self.pre_process:
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Prompt Table
        if self.use_soft_prompts:
            if self._prompt_table_key in state_dict:
                state_dict_ = state_dict[self._prompt_table_key]

                self.prompt_table.load_state_dict(state_dict_, strict=strict)

        # Encoder.
        if self._encoder_key in state_dict:
            state_dict_ = state_dict[self._encoder_key]

        # for backward compatibility.
        elif 'transformer' in state_dict:
            state_dict_ = state_dict['transformer']
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]

        # for backward compatibility.
        state_dict_self_attention = {}
        for key in state_dict_.keys():
            if '.attention.' in key:
                state_dict_self_attention[key.replace(".attention.", ".self_attention.")] = state_dict_[key]
            else:
                state_dict_self_attention[key] = state_dict_[key]
        state_dict_ = state_dict_self_attention

        self.encoder.load_state_dict(state_dict_, strict=strict)

        if self.post_process:
            # pooler
            if self.add_pooler:
                assert 'pooler' in state_dict, 'could not find data for pooler in the checkpoint'
                self.pooler.load_state_dict(state_dict[self._pooler_key], strict=strict)
        # decoder
        if self.add_decoder:
            assert 'decoder' in state_dict, 'could not find data for pooler in the checkpoint'
            self.decoder.load_state_dict(state_dict[self._decoder_key], strict=strict)

    def _init_prompt_from_random(self, prompt_tag, prompt_id):
        """Add new soft prompt to be tuned.
           Intialize prompt weights using pytorch init method

        """
        if self.pre_process:
            if not hasattr(self, 'prompt_table'):
                raise AttributeError('Please set "use_soft_prompts" in the config to True')

            self.prompt_table.init_prompt_from_random(prompt_tag, prompt_id, embeddings=self.embedding)

    def _init_prompt_from_text(self, prompt_tag, prompt_id, init_token_ids):
        """Add new soft prompt to be tuned.
           Intialize prompt weights from existing embeddings from specific vocab tokens.

        """
        if self.pre_process:
            if not hasattr(self, 'prompt_table'):
                raise AttributeError('Please set "use_soft_prompts" in the config to True')

            self.prompt_table.init_prompt_from_text(
                prompt_tag, prompt_id, init_token_ids, embeddings=self.embedding,
            )
