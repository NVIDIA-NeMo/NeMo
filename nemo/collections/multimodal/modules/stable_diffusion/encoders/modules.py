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
import tempfile
from functools import partial

import open_clip
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.checkpoint import checkpoint
from transformers import CLIPTextModel, CLIPTokenizer

from nemo.collections.multimodal.data.clip.clip_dataset import get_preprocess_fns
from nemo.collections.multimodal.models.vision_language_foundation.clip.megatron_clip_models import CLIPModel
from nemo.collections.multimodal.modules.stable_diffusion.encoders.x_transformer import (
    TransformerWrapper,  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
)
from nemo.collections.multimodal.modules.stable_diffusion.encoders.x_transformer import Encoder
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.collections.nlp.parts.nlp_overrides import NLPSaveRestoreConnector
from nemo.utils import logging

try:
    from megatron.core import ModelParallelConfig, parallel_state

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    ModelParallelConfig = ApexGuardDefaults

    HAVE_MEGATRON_CORE = False


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size, max_seq_len=max_seq_len, attn_layers=Encoder(dim=n_embed, depth=n_layer)
        )

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(
        self,
        n_embed,
        n_layer,
        vocab_size=30522,
        max_seq_len=77,
        device="cuda",
        use_tokenizer=True,
        embedding_dropout=0.0,
    ):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=n_embed, depth=n_layer),
            emb_dropout=embedding_dropout,
        )

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self, n_stages=1, method='bilinear', multiplier=0.5, in_channels=3, out_channels=None, bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77, capture_cudagraph_iters: int = -1
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

        # CUDA graph captured sub-modules
        self.capture_cudagraph_iters = capture_cudagraph_iters
        self.iterations = 0
        self.stream = torch.cuda.Stream()
        self.transformer_graph = torch.cuda.CUDAGraph()
        self.static_tokens = None
        self.static_outputs = None

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        if self.capture_cudagraph_iters < 0:
            tokens = batch_encoding["input_ids"].to(self.device, non_blocking=True)
            outputs = self.transformer(input_ids=tokens)
            z = outputs.last_hidden_state

        else:
            if self.static_tokens is None:
                self.static_tokens = batch_encoding["input_ids"].to(device=self.device, non_blocking=True)
            self.static_tokens.copy_(batch_encoding["input_ids"], non_blocking=True)

            if self.iterations == self.capture_cudagraph_iters:
                # cuda graph capture
                logging.info("Capturing CUDA graph for module: %s", self.transformer.__class__.__name__)
                with torch.cuda.graph(self.transformer_graph):
                    self.static_outputs = self.transformer(input_ids=self.static_tokens)

            if 0 <= self.capture_cudagraph_iters <= self.iterations:
                # cuda graph replay
                self.transformer_graph.replay()
            else:
                # warmup
                self.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.stream):
                    self.static_outputs = self.transformer(input_ids=self.static_tokens)
                torch.cuda.current_stream().wait_stream(self.stream)
            self.iterations += 1
            z = self.static_outputs.last_hidden_state

        # # Pad the seq length to multiple of 8
        seq_len = (z.shape[1] + 8 - 1) // 8 * 8
        z = torch.nn.functional.pad(z, (0, 0, 0, seq_len - z.shape[1]), value=0.0)
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        use_fp16=False,
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)


class FrozenMegatronCLIPEmbedder(AbstractEncoder):
    def __init__(self, restore_from_path, device="cuda", layer="last", freeze=True, cfg=None, use_fp16=False):
        super().__init__()
        if restore_from_path is not None:
            cfg, state_dict = self.load_config_and_state_from_nemo(restore_from_path)
        elif cfg is not None:
            state_dict = None
        else:
            raise ValueError("Either restore_from_path or cfg should not be None")

        self.cfg = cfg
        self.build_tokenizer(cfg)
        self.load_model(cfg, state_dict)

        self.device = device
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def load_config_and_state_from_nemo(self, nemo_path):
        if torch.cuda.is_available():
            map_location = torch.device('cuda')
        else:
            map_location = torch.device('cpu')
        save_restore_connector = NLPSaveRestoreConnector()
        cwd = os.getcwd()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                save_restore_connector._unpack_nemo_file(path2file=nemo_path, out_folder=tmpdir)

                # Change current working directory to
                os.chdir(tmpdir)
                config_yaml = os.path.join(tmpdir, save_restore_connector.model_config_yaml)
                cfg = OmegaConf.load(config_yaml)

                model_weights = os.path.join(tmpdir, save_restore_connector.model_weights_ckpt)
                state_dict = save_restore_connector._load_state_dict_from_disk(
                    model_weights, map_location=map_location
                )
            finally:
                os.chdir(cwd)

        return cfg, state_dict

    def build_tokenizer(self, cfg):
        legacy = cfg.tokenizer.sentencepiece_legacy
        self.tokenizer = get_nmt_tokenizer(
            library=cfg.tokenizer.library,
            model_name=cfg.tokenizer.type,
            tokenizer_model=cfg.tokenizer.model,
            vocab_file=cfg.tokenizer.vocab_file,
            merges_file=cfg.tokenizer.merge_file,
            delimiter=cfg.tokenizer.get('delimiter', None),
            legacy=legacy,
        )

        _, self.text_transform = get_preprocess_fns(cfg, self.tokenizer, is_train=False,)
        self.max_length = cfg.text.get("max_position_embeddings")

    def load_model(self, cfg, state_dict):
        padded_vocab_size = self._vocab_size_with_padding(
            orig_vocab_size=self.tokenizer.vocab_size,
            make_vocab_size_divisible_by=cfg.get('make_vocab_size_divisible_by', 128),
            tensor_model_parallel_size=cfg.get('tensor_model_parallel_size', 1),
        )
        model = CLIPModel(
            model_cfg=cfg,
            model_parallel_config=ModelParallelConfig(),
            padded_vocab_size=padded_vocab_size,
            pre_process=cfg.text.pre_process,
            post_process=cfg.text.post_process,
        )

        if state_dict is not None:
            clip_state_dict = {}
            for key, value in state_dict.items():
                key = key[6:]
                clip_state_dict[key] = value
            model.load_state_dict(clip_state_dict)

        del model.vision_encoder
        self.model = model.text_encoder

    def _vocab_size_with_padding(self, orig_vocab_size, make_vocab_size_divisible_by, tensor_model_parallel_size):
        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tensor_model_parallel_size
        while (after % multiple) != 0:
            after += 1
        return after

    def forward(self, text):
        '''
        Get embeddings from input text
        '''
        texts = self.text_transform(text)
        z = self.encode_with_transformer(texts.to(self.device))
        # # Pad the seq length to multiple of 8
        seq_len = (z.shape[1] + 8 - 1) // 8 * 8
        z = torch.nn.functional.pad(z, (0, 0, 0, seq_len - z.shape[1]), value=0.0)
        return z

    def encode_with_transformer(self, text):
        x = self.model.language_model.embedding.word_embeddings(text)
        x += self.model.language_model.embedding.position_embeddings
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = self.model.language_model.encoder.final_layernorm(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.language_model.encoder.layers):
            if i == len(self.model.language_model.encoder.layers) - self.layer_idx:
                break
            x = r(x, attn_mask)
        return x

    def encode(self, text):
        return self(text)


if __name__ == "__main__":
    from ldm.util import count_params

    model = FrozenCLIPEmbedder()
    count_params(model, verbose=True)
