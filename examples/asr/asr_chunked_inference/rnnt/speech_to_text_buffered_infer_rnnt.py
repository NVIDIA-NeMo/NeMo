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

"""
Script to perform buffered inference using RNNT models.

Buffered inference is the primary form of audio transcription when the audio segment is longer than 20-30 seconds.
This is especially useful for models such as Conformers, which have quadratic time and memory scaling with
audio duration.

The difference between streaming and buffered inference is the chunk size (or the latency of inference).
Buffered inference will use large chunk sizes (5-10 seconds) + some additional buffer for context.
Streaming inference will use small chunk sizes (0.1 to 0.25 seconds) + some additional buffer for context.

# Middle Token merge algorithm

python speech_to_text_buffered_infer_rnnt.py \
    --asr_model="<Path to a nemo model>" \
    --test_manifest="<Path to a JSON manifest>" \
    --model_stride=4 \
    --output_path="." \
    --total_buffer_in_secs=10.0 \
    --chunk_len_in_secs=8.0 \
    --device="cuda:0" \
    --batch_size=128

# Longer Common Subsequence (LCS) Merge algorithm

python speech_to_text_buffered_infer_rnnt.py \
    --asr_model="<Path to a nemo model>" \
    --test_manifest="<Path to a JSON manifest>" \
    --model_stride=4 \
    --output_path="." \
    --merge_algo="lcs" \
    --lcs_alignment_dir=<OPTIONAL: Some path to store the LCS alignments> \
    --total_buffer_in_secs=10.0 \
    --chunk_len_in_secs=8.0 \
    --device="cuda:0" \
    --batch_size=128

# NOTE:

    You can use `DEBUG=1 python speech_to_text_buffered_infer_rnnt.py ...` to print out the
    ground truth text and predictions of the model.

"""

import copy
import json
import math
import os
from argparse import ArgumentParser

import torch
import tqdm
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.streaming_utils import (
    BatchedFrameASRRNNT,
    LongestCommonSubsequenceBatchedFrameASRRNNT,
)
from nemo.utils import logging

can_gpu = torch.cuda.is_available()

# Common Arguments
parser = ArgumentParser()
parser.add_argument(
    "--asr_model", type=str, required=True, help="Path to asr model .nemo file",
)
parser.add_argument("--test_manifest", type=str, required=True, help="path to evaluation data")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--total_buffer_in_secs",
    type=float,
    default=4.0,
    help="Length of buffer (chunk + left and right padding) in seconds ",
)
parser.add_argument("--chunk_len_in_secs", type=float, default=1.6, help="Chunk length in seconds")
parser.add_argument("--output_path", type=str, help="path to output file", default=None)
parser.add_argument(
    "--model_stride",
    type=int,
    default=8,
    help="Model downsampling factor, 8 for Citrinet models and 4 for Conformer models",
)
parser.add_argument(
    '--max_steps_per_timestep', type=int, default=5, help='Maximum number of tokens decoded per acoustic timestep'
)
parser.add_argument('--stateful_decoding', action='store_true', help='Whether to perform stateful decoding')
parser.add_argument('--device', default=None, type=str, required=False)

# Merge algorithm for transducers
parser.add_argument(
    '--merge_algo',
    default='middle',
    type=str,
    required=False,
    choices=['middle', 'lcs'],
    help='Choice of algorithm to apply during inference.',
)

# LCS Merge Algorithm
parser.add_argument(
    '--lcs_alignment_dir', type=str, default=None, help='Path to a directory to store LCS algo alignments'
)


def get_wer_feat(mfst, asr, tokens_per_chunk, delay, model_stride_in_secs, batch_size):
    hyps = []
    refs = []
    audio_filepaths = []

    with open(mfst, "r") as mfst_f:
        print("Parsing manifest files...")
        for l in mfst_f:
            row = json.loads(l.strip())
            audio_filepaths.append(row['audio_filepath'])
            refs.append(row['text'])

    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            batch = []
            asr.sample_offset = 0
            for idx in tqdm.tqdm(range(len(audio_filepaths)), desc='Sample:', total=len(audio_filepaths)):
                batch.append((audio_filepaths[idx], refs[idx]))

                if len(batch) == batch_size:
                    audio_files = [sample[0] for sample in batch]

                    asr.reset()
                    asr.read_audio_file(audio_files, delay, model_stride_in_secs)
                    hyp_list = asr.transcribe(tokens_per_chunk, delay)
                    hyps.extend(hyp_list)

                    batch.clear()
                    asr.sample_offset += batch_size

            if len(batch) > 0:
                asr.batch_size = len(batch)
                asr.frame_bufferer.batch_size = len(batch)
                asr.reset()

                audio_files = [sample[0] for sample in batch]
                asr.read_audio_file(audio_files, delay, model_stride_in_secs)
                hyp_list = asr.transcribe(tokens_per_chunk, delay)
                hyps.extend(hyp_list)

                batch.clear()
                asr.sample_offset += len(batch)

    if os.environ.get('DEBUG', '0') in ('1', 'y', 't'):
        for hyp, ref in zip(hyps, refs):
            print("hyp:", hyp)
            print("ref:", ref)

    wer = word_error_rate(hypotheses=hyps, references=refs)
    return hyps, refs, wer


def main(args):
    torch.set_grad_enabled(False)
    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.asr_model)

    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0

    if cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecRNNTBPEModel models trained with per_feature normalization are supported currently")

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    logging.info(f"Inference will be done on device : {device}")

    # Disable config overwriting
    OmegaConf.set_struct(cfg.preprocessor, True)
    asr_model.freeze()
    asr_model = asr_model.to(device)

    # Change Decoding Config
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        if args.stateful_decoding:
            decoding_cfg.strategy = "greedy"
        else:
            decoding_cfg.strategy = "greedy_batch"

        decoding_cfg.preserve_alignments = True  # required to compute the middle token for transducers.
        decoding_cfg.fused_batch_size = -1  # temporarily stop fused batch during inference.

    asr_model.change_decoding_strategy(decoding_cfg)

    feature_stride = cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * args.model_stride
    total_buffer = args.total_buffer_in_secs

    chunk_len = float(args.chunk_len_in_secs)

    tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
    print("Tokens per chunk :", tokens_per_chunk, "Min Delay :", mid_delay)

    if args.merge_algo == 'middle':
        frame_asr = BatchedFrameASRRNNT(
            asr_model=asr_model,
            frame_len=chunk_len,
            total_buffer=args.total_buffer_in_secs,
            batch_size=args.batch_size,
            max_steps_per_timestep=args.max_steps_per_timestep,
            stateful_decoding=args.stateful_decoding,
        )

    elif args.merge_algo == 'lcs':
        frame_asr = LongestCommonSubsequenceBatchedFrameASRRNNT(
            asr_model=asr_model,
            frame_len=chunk_len,
            total_buffer=args.total_buffer_in_secs,
            batch_size=args.batch_size,
            max_steps_per_timestep=args.max_steps_per_timestep,
            stateful_decoding=args.stateful_decoding,
            alignment_basepath=args.lcs_alignment_dir,
        )
        # Set the LCS algorithm delay.
        frame_asr.lcs_delay = math.floor(((total_buffer - chunk_len)) / model_stride_in_secs)

    else:
        raise ValueError("Invalid choice of merge algorithm for transducer buffered inference.")

    hyps, refs, wer = get_wer_feat(
        mfst=args.test_manifest,
        asr=frame_asr,
        tokens_per_chunk=tokens_per_chunk,
        delay=mid_delay,
        model_stride_in_secs=model_stride_in_secs,
        batch_size=args.batch_size,
    )
    logging.info(f"WER is {round(wer, 4)} when decoded with a delay of {round(mid_delay*model_stride_in_secs, 2)}s")

    if args.output_path is not None:

        fname = (
            os.path.splitext(os.path.basename(args.asr_model))[0]
            + "_"
            + os.path.splitext(os.path.basename(args.test_manifest))[0]
            + "_"
            + str(args.chunk_len_in_secs)
            + "_"
            + str(int(total_buffer * 1000))
            + "_"
            + args.merge_algo
            + ".json"
        )

        hyp_json = os.path.join(args.output_path, fname)
        os.makedirs(args.output_path, exist_ok=True)
        with open(hyp_json, "w") as out_f:
            for i, hyp in enumerate(hyps):
                record = {
                    "pred_text": hyp,
                    "text": refs[i],
                    "wer": round(word_error_rate(hypotheses=[hyp], references=[refs[i]]) * 100, 2),
                }
                out_f.write(json.dumps(record) + '\n')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)  # noqa pylint: disable=no-value-for-parameter
