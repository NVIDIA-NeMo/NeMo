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


import os
from argparse import ArgumentParser

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from nemo.collections.nlp.data.language_modeling import TarredSentenceDataset
from nemo.collections.nlp.data.machine_translation import TarredTranslationDataset
from nemo.collections.nlp.models.machine_translation.mt_enc_dec_model import MTEncDecModel
from nemo.utils import logging


def get_args():
    parser = ArgumentParser(description='Batch translation of sentences from a pre-trained model on multiple GPUs')
    parser.add_argument("--model", type=str, required=True, help="Path to the .nemo translation model file")
    parser.add_argument(
        "--text2translate", type=str, required=True, help="Path to the pre-processed tarfiles for translation"
    )
    parser.add_argument("--result_dir", type=str, required=True, help="Folder to write translation results")
    parser.add_argument(
        "--twoside", action="store_true", help="Set flag when translating the source side of a parallel dataset"
    )
    parser.add_argument(
        '--metadata_path', type=str, required=True, help="Path to the JSON file that contains dataset info"
    )
    parser.add_argument('--topk', type=int, default=500, help="Value of k for topk sampling")
    parser.add_argument('--src_language', type=str, required=True, help="Source lang ID for detokenization")
    parser.add_argument('--tgt_language', type=str, required=True, help="Target lang ID for detokenization")
    parser.add_argument(
        '--reverse_lang_direction',
        action="store_true",
        help="Reverse source and target language direction for parallel dataset",
    )
    parser.add_argument('--n_gpus', type=int, default=1, help="Number of GPUs to use")
    args = parser.parse_args()
    return args


def translate(rank, world_size, args):
    if args.model.endswith(".nemo"):
        logging.info("Attempting to initialize from .nemo file")
        model = MTEncDecModel.restore_from(restore_path=args.model, map_location=f"cuda:{rank}")
    elif args.model.endswith(".ckpt"):
        logging.info("Attempting to initialize from .ckpt file")
        model = MTEncDecModel.load_from_checkpoint(checkpoint_path=args.model, map_location=f"cuda:{rank}")
    model.replace_beam_with_sampling(topk=args.topk)
    model.eval()
    if args.twoside:
        dataset = TarredTranslationDataset(
            text_tar_filepaths=args.text2translate,
            metadata_path=args.metadata_path,
            encoder_tokenizer=model.encoder_tokenizer,
            decoder_tokenizer=model.decoder_tokenizer,
            shuffle_n=100,
            shard_strategy="scatter",
            world_size=world_size,
            global_rank=rank,
            reverse_lang_direction=args.reverse_lang_direction,
        )
    else:
        dataset = TarredSentenceDataset(
            text_tar_filepaths=args.text2translate,
            metadata_path=args.metadata_path,
            tokenizer=model.encoder_tokenizer,
            shuffle_n=100,
            shard_strategy="scatter",
            world_size=world_size,
            global_rank=rank,
        )
    loader = DataLoader(dataset, batch_size=1)
    result_dir = os.path.join(args.result_dir, f'rank{rank}')
    os.makedirs(result_dir, exist_ok=True)
    originals_file_name = os.path.join(result_dir, 'originals.txt')
    translations_file_name = os.path.join(result_dir, 'translations.txt')
    num_translated_sentences = 0

    with open(originals_file_name, 'w') as of, open(translations_file_name, 'w') as tf:
        for batch_idx, batch in enumerate(loader):
            for i in range(len(batch)):
                if batch[i].ndim == 3:
                    batch[i] = batch[i].squeeze(dim=0)
                batch[i] = batch[i].to(rank)
            if args.twoside:
                src_ids, src_mask, _, _, _ = batch
            else:
                src_ids, src_mask = batch
            if batch_idx % 100 == 0:
                logging.info(
                    f"{batch_idx} batches ({num_translated_sentences} sentences) were translated by process with "
                    f"rank {rank}"
                )
            num_translated_sentences += len(src_ids)
            inputs, translations = model.batch_translate(src=src_ids, src_mask=src_mask)
            for src, translation in zip(inputs, translations):
                of.write(src + '\n')
                tf.write(translation + '\n')


def main() -> None:
    args = get_args()
    world_size = torch.cuda.device_count() if args.n_gpus == -1 else args.n_gpus
    mp.spawn(translate, args=(world_size, args), nprocs=world_size, join=True)


if __name__ == '__main__':
    main()
