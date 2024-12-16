# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
#
# Code for building NVIDIA proprietary Edify APIs.
# It cannot be released to the public in any form.
# If you want to use the code for other NVIDIA proprietary products,
# please contact the Deep Imagination Research Team (dir@exchange.nvidia.com).
# -----------------------------------------------------------------------------

"""A library for Causal Video Tokenizer inference."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import get_token as get_hf_token
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from tqdm import tqdm

from nemo.collections.common.video_tokenizers.utils import (
    get_tokenizer_config,
    load_jit_model,
    load_pytorch_model,
    numpy2tensor,
    pad_video_batch,
    tensor2numpy,
    unpad_video_batch,
)
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.modelPT import ModelPT


class CausalVideoTokenizer(ModelPT):
    """Causal Video tokenization with the NVIDIA Cosmos Tokenizer"""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        checkpoint = Path(cfg.checkpoint_dir)
        self._full_model_path = str(checkpoint / "autoencoder.jit")
        self._enc_model_path = str(checkpoint / "encoder.jit")
        self._dec_model_path = str(checkpoint / "decoder.jit")
        self._dtype = getattr(torch, cfg.dtype)

        self._device = "cuda"

        if cfg.use_pytorch:
            tokenizer_config = get_tokenizer_config(cfg.tokenizer_type)
            tokenizer_config["dtype"] = self._dtype
            self._full_model = (
                load_pytorch_model(self._full_model_path, tokenizer_config, "full", self._device)
                if cfg.load_full_model
                else None
            )
            self._enc_model = (
                load_pytorch_model(self._enc_model_path, tokenizer_config, "enc", self._device)
                if cfg.load_enc_model
                else None
            )
            self._dec_model = (
                load_pytorch_model(self._dec_model_path, tokenizer_config, "dec", self._device)
                if cfg.load_dec_model
                else None
            )
        else:
            self._full_model = load_jit_model(self._full_model_path, self._device) if cfg.load_full_model else None
            self._enc_model = load_jit_model(self._enc_model_path, self._device) if cfg.load_enc_model else None
            self._dec_model = load_jit_model(self._dec_model_path, self._device) if cfg.load_dec_model else None

    @classmethod
    def from_pretrained(
        cls,
        tokenizer_type="Cosmos-Tokenizer-DV4x8x8",
        load_encoder=True,
        load_decoder=True,
        load_full_model=False,
        use_pytorch=False,
        dtype="bfloat16",
    ):
        cls._hf_model_name = f"nvidia/{tokenizer_type}"

        # Requires setting HF_TOKEN env variable
        hf_token = get_hf_token()

        full_model_path = hf_hub_download(
            repo_id=cls._hf_model_name,
            filename="autoencoder.jit",
            token=hf_token,
        )

        _ = hf_hub_download(
            repo_id=cls._hf_model_name,
            filename="encoder.jit",
            token=hf_token,
        )

        _ = hf_hub_download(
            repo_id=cls._hf_model_name,
            filename="decoder.jit",
            token=hf_token,
        )

        # No need to load in encoder and decoder with full model loaded
        if load_full_model:
            load_encoder = False
            load_decoder = False

        # Assumes HF downloads all files to same local dir
        ckpt_dir = str(Path(full_model_path).parent)
        cfg = DictConfig(
            {
                'checkpoint_dir': ckpt_dir,
                'dtype': dtype,
                'load_enc_model': load_encoder,
                'load_dec_model': load_decoder,
                'load_full_model': load_full_model,
                'tokenizer_type': tokenizer_type,
                'use_pytorch': use_pytorch,
            }
        )

        return cls(cfg)

    @torch.no_grad()
    def autoencode(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Reconstructs a batch of video tensors after embedding into a latent.

        Args:
            video: The input video Bx3xTxHxW layout, range [-1..1].
        Returns:
            The reconstructed video, layout Bx3xTxHxW, range [-1..1].
        """
        if self._full_model is not None:
            output_tensor = self._full_model(input_tensor)
            output_tensor = output_tensor[0] if isinstance(output_tensor, tuple) else output_tensor
        else:
            output_latent = self.encode(input_tensor)[0]
            output_tensor = self.decode(output_latent)
        return output_tensor

    @torch.no_grad()
    def encode(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor]:
        """Encodes a numpy video into a CausalVideo latent or code.

        Args:
            input_tensor: The input tensor Bx3xTxHxW layout, range [-1..1].
        Returns:
            For causal continuous video (CausalCV) tokenizer, the tuple contains:
                - The latent embedding, Bx16x(t)x(h)x(w), where the compression
                rate is (T/t x H/h x W/w), and channel dimension of 16.
            For causal discrete video (CausalDV) tokenizer, the tuple contains:
              1) The indices, Bx(t)x(h)x(w), from a codebook of size 64K, which
                is formed by FSQ levels of (8,8,8,5,5,5).
              2) The discrete code, Bx6x(t)x(h)x(w), where the compression rate
                is again (T/t x H/h x W/w), and channel dimension of 6.
        """
        assert input_tensor.ndim == 5, "input video should be of 5D."

        output_latent = self._enc_model(input_tensor)
        if isinstance(output_latent, torch.Tensor):
            return output_latent
        return output_latent[:-1]

    @torch.no_grad()
    def decode(self, input_latent: torch.Tensor) -> torch.Tensor:
        """Encodes a numpy video into a CausalVideo latent.

        Args:
            input_latent: The continuous latent Bx16xtxhxw for CausalCV,
                        or the discrete indices Bxtxhxw for CausalDV.
        Returns:
            The reconstructed tensor, layout [B,3,1+(T-1)*8,H*16,W*16] in range [-1..1].
        """
        assert input_latent.ndim >= 4, "input latent should be of 5D for continuous and 4D for discrete."
        return self._dec_model(input_latent)

    def forward(
        self,
        video: np.ndarray,
        temporal_window: int = 17,
    ) -> np.ndarray:
        """Reconstructs video using a pre-trained CausalTokenizer autoencoder.
        Given a video of arbitrary length, the forward invokes the CausalVideoTokenizer
        in a sliding manner with a `temporal_window` size.

        Args:
            video: The input video BxTxHxWx3 layout, range [0..255].
            temporal_window: The length of the temporal window to process, default=25.
        Returns:
            The reconstructed video in range [0..255], layout BxTxHxWx3.
        """
        assert video.ndim == 5, "input video should be of 5D."
        num_frames = video.shape[1]  # can be of any length.
        output_video_list = []
        for idx in tqdm(range(0, (num_frames - 1) // temporal_window + 1)):
            # Input video for the current window.
            start, end = idx * temporal_window, (idx + 1) * temporal_window
            input_video = video[:, start:end, ...]

            # Spatio-temporally pad input_video so it's evenly divisible.
            padded_input_video, crop_region = pad_video_batch(input_video)
            input_tensor = numpy2tensor(padded_input_video, dtype=self._dtype, device=self._device)
            output_tensor = self.autoencode(input_tensor)
            padded_output_video = tensor2numpy(output_tensor)
            output_video = unpad_video_batch(padded_output_video, crop_region)

            output_video_list.append(output_video)
        return np.concatenate(output_video_list, axis=1)

    def setup_training_data(self, train_data_config: Optional[DictConfig]):
        pass

    def setup_validation_data(self, val_data_config: Optional[DictConfig]):
        pass

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        pass
