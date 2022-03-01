# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import copy
import os
from typing import Dict, Optional

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict

from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer_bpe import WERBPE
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin
from nemo.collections.asr.parts.preprocessing.perturb import process_augmentations
from nemo.core.classes.common import PretrainedModelInfo
from nemo.utils import logging, model_utils

__all__ = ['EncDecCTCModelBPE']


class EncDecCTCModelBPE(EncDecCTCModel, ASRBPEMixin):
    """Encoder decoder CTC-based models with Byte Pair Encoding."""

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_256",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_256",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_256/versions/1.0.0rc1/files/stt_en_citrinet_256.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_512/versions/1.0.0rc1/files/stt_en_citrinet_512.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_1024/versions/1.0.0rc1/files/stt_en_citrinet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_256_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_256_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_256_gamma_0_25/versions/1.0.0/files/stt_en_citrinet_256_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_512_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_512_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_512_gamma_0_25/versions/1.0.0/files/stt_en_citrinet_512_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_citrinet_1024_gamma_0_25/versions/1.0.0/files/stt_en_citrinet_1024_gamma_0_25.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_es_citrinet_512",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_es_citrinet_512",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_es_citrinet_512/versions/1.0.0/files/stt_es_citrinet_512.nemo",
        )

        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_citrinet_1024",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_citrinet_1024",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_citrinet_1024/versions/1.3.2/files/stt_de_citrinet_1024.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_citrinet_1024_gamma_0_25/versions/1.5/files/stt_fr_citrinet_1024_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_no_hyphen_citrinet_1024_gamma_0_25",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_citrinet_1024_gamma_0_25",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_citrinet_1024_gamma_0_25/versions/1.5/files/stt_fr_no_hyphen_citrinet_1024_gamma_0_25.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_small",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_small",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small/versions/1.6.0/files/stt_en_conformer_ctc_small.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_medium",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_medium",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_medium/versions/1.6.0/files/stt_en_conformer_ctc_medium.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_large/versions/1.6.0/files/stt_en_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_small_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_small_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_small_ls/versions/1.0.0/files/stt_en_conformer_ctc_small_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_medium_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_medium_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_medium_ls/versions/1.0.0/files/stt_en_conformer_ctc_medium_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_en_conformer_ctc_large_ls",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_en_conformer_ctc_large_ls",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_conformer_ctc_large_ls/versions/1.0.0/files/stt_en_conformer_ctc_large_ls.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_conformer_ctc_large",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_conformer_ctc_large/versions/1.5.1/files/stt_fr_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_fr_no_hyphen_conformer_ctc_large",
            description="For details about this model, please visit https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_fr_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_fr_conformer_ctc_large/versions/1.5.1/files/stt_fr_no_hyphen_conformer_ctc_large.nemo",
        )
        results.append(model)

        model = PretrainedModelInfo(
            pretrained_model_name="stt_de_conformer_ctc_large",
            description="For details about this model, please visit https://ngc.nvidia.com/catalog/models/nvidia:nemo:stt_de_conformer_ctc_large",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_de_conformer_ctc_large/versions/1.5.0/files/stt_de_conformer_ctc_large.nemo",
        )
        results.append(model)

        return results

    def __init__(self, cfg: DictConfig, trainer=None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        if 'tokenizer' not in cfg:
            raise ValueError("`cfg` must have `tokenizer` config to create a tokenizer !")

        # Setup the tokenizer
        self._setup_tokenizer(cfg.tokenizer)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        with open_dict(cfg):
            cfg.decoder.vocabulary = ListConfig(list(vocabulary.keys()))

        # Override number of classes if placeholder provided
        num_classes = cfg.decoder["num_classes"]

        if num_classes < 1:
            logging.info(
                "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                    num_classes, len(vocabulary)
                )
            )
            cfg.decoder["num_classes"] = len(vocabulary)

        super().__init__(cfg=cfg, trainer=trainer)

        # Setup metric objects
        self._wer = WERBPE(
            tokenizer=self.tokenizer,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=True,
            dist_sync_on_step=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

    def _setup_dataloader_from_config(self, config: Optional[Dict]):
        if 'augmentor' in config:
            augmentor = process_augmentations(config['augmentor'])
        else:
            augmentor = None

        shuffle = config['shuffle']
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        if config.get('use_dali', False):
            device_id = self.local_rank if device == 'gpu' else None
            dataset = audio_to_text_dataset.get_dali_bpe_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle=shuffle,
                device_id=device_id,
                global_rank=self.global_rank,
                world_size=self.world_size,
                preprocessor_cfg=self._cfg.preprocessor,
            )
            return dataset

        # Instantiate tarred dataset loader or normal dataset loader
        if config.get('is_tarred', False):
            if ('tarred_audio_filepaths' in config and config['tarred_audio_filepaths'] is None) or (
                'manifest_filepath' in config and config['manifest_filepath'] is None
            ):
                logging.warning(
                    "Could not load dataset as `manifest_filepath` was None or "
                    f"`tarred_audio_filepaths` is None. Provided config : {config}"
                )
                return None

            shuffle_n = config.get('shuffle_n', 4 * config['batch_size']) if shuffle else 0
            dataset = audio_to_text_dataset.get_tarred_dataset(
                config=config,
                tokenizer=self.tokenizer,
                shuffle_n=shuffle_n,
                global_rank=self.global_rank,
                world_size=self.world_size,
                augmentor=augmentor,
            )
            shuffle = False
        else:
            if 'manifest_filepath' in config and config['manifest_filepath'] is None:
                logging.warning(f"Could not load dataset as `manifest_filepath` was None. Provided config : {config}")
                return None

            dataset = audio_to_text_dataset.get_bpe_dataset(
                config=config, tokenizer=self.tokenizer, augmentor=augmentor
            )
        if hasattr(dataset, 'collate_fn'):
            collate_fn = dataset.collate_fn
        else:
            collate_fn = dataset.datasets[0].collate_fn

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=config['batch_size'],
            collate_fn=collate_fn,
            drop_last=config.get('drop_last', False),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=config.get('pin_memory', False),
        )

    def _setup_transcribe_dataloader(self, config: Dict) -> 'torch.utils.data.DataLoader':
        """
        Setup function for a temporary data loader which wraps the provided audio file.

        Args:
            config: A python dictionary which contains the following keys:
            paths2audio_files: (a list) of paths to audio files. The files should be relatively short fragments. \
                Recommended length per file is between 5 and 25 seconds.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            temp_dir: (str) A temporary directory where the audio manifest is temporarily
                stored.
            num_workers: (int) number of workers. Depends of the batch_size and machine. \
                0 - only the main process will load batches, 1 - one worker (not main process)

        Returns:
            A pytorch DataLoader for the given audio file(s).
        """
        batch_size = min(config['batch_size'], len(config['paths2audio_files']))
        dl_config = {
            'manifest_filepath': os.path.join(config['temp_dir'], 'manifest.json'),
            'sample_rate': self.preprocessor._sample_rate,
            'batch_size': batch_size,
            'shuffle': False,
            'num_workers': config.get('num_workers', min(batch_size, os.cpu_count() - 1)),
            'pin_memory': True,
            'use_start_end_token': self.cfg.validation_ds.get('use_start_end_token', False),
        }

        temporary_datalayer = self._setup_dataloader_from_config(config=DictConfig(dl_config))
        return temporary_datalayer

    def change_vocabulary(self, new_tokenizer_dir: str, new_tokenizer_type: str):
        """
        Changes vocabulary of the tokenizer used during CTC decoding process.
        Use this method when fine-tuning on from pre-trained model.
        This method changes only decoder and leaves encoder and pre-processing modules unchanged. For example, you would
        use it if you want to use pretrained encoder when fine-tuning on a data in another language, or when you'd need
        model to learn capitalization, punctuation and/or special characters.

        Args:
            new_tokenizer_dir: Path to the new tokenizer directory.
            new_tokenizer_type: Either `bpe` or `wpe`. `bpe` is used for SentencePiece tokenizers,
                whereas `wpe` is used for `BertTokenizer`.

        Returns: None

        """
        if not os.path.isdir(new_tokenizer_dir):
            raise NotADirectoryError(
                f'New tokenizer dir must be non-empty path to a directory. But I got: {new_tokenizer_dir}'
            )

        if new_tokenizer_type.lower() not in ('bpe', 'wpe'):
            raise ValueError(f'New tokenizer type must be either `bpe` or `wpe`')

        tokenizer_cfg = OmegaConf.create({'dir': new_tokenizer_dir, 'type': new_tokenizer_type})

        # Setup the tokenizer
        self._setup_tokenizer(tokenizer_cfg)

        # Initialize a dummy vocabulary
        vocabulary = self.tokenizer.tokenizer.get_vocab()

        # Set the new vocabulary
        decoder_config = copy.deepcopy(self.decoder.to_config_dict())
        decoder_config.vocabulary = ListConfig(list(vocabulary.keys()))

        decoder_num_classes = decoder_config['num_classes']

        # Override number of classes if placeholder provided
        logging.info(
            "\nReplacing old number of classes ({}) with new number of classes - {}".format(
                decoder_num_classes, len(vocabulary)
            )
        )

        decoder_config['num_classes'] = len(vocabulary)

        del self.decoder
        self.decoder = EncDecCTCModelBPE.from_config_dict(decoder_config)
        del self.loss
        self.loss = CTCLoss(
            num_classes=self.decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self._cfg.get("ctc_reduction", "mean_batch"),
        )
        self._wer = WERBPE(
            tokenizer=self.tokenizer,
            batch_dim_index=0,
            use_cer=self._cfg.get('use_cer', False),
            ctc_decode=True,
            log_prediction=self._cfg.get("log_prediction", False),
        )

        # Update config
        OmegaConf.set_struct(self._cfg.decoder, False)
        self._cfg.decoder = decoder_config
        OmegaConf.set_struct(self._cfg.decoder, True)

        logging.info(f"Changed tokenizer to {self.decoder.vocabulary} vocabulary.")
