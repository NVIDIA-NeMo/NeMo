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
This script serves three goals:
    (1) Demonstrate how to use NeMo Models outside of PytorchLightning
    (2) Shows example of batch ASR inference
    (3) Serves as CI test for pre-trained checkpoint

python speech_to_text_buffered_infer_ctc.py \
    --asr_model="<Add model path here>" \
    --test_manifest="<Add test dataset here>" \
    --model_stride=4 \
    --batch_size=32 \
    --total_buffer_in_secs=4.0 \
    --chunk_len_in_ms=1000



"""

import copy
import json
import math
import os
from argparse import ArgumentParser

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.parts.utils.streaming_utils import FrameBatchASR
from nemo.utils import logging

can_gpu = torch.cuda.is_available()


def get_wer_feat(mfst, asr, frame_len, tokens_per_chunk, delay, preprocessor_cfg, model_stride_in_secs, device):
    # Create a preprocessor to convert audio samples into raw features,
    # Normalization will be done per buffer in frame_bufferer
    # Do not normalize whatever the model's preprocessor setting is
    preprocessor_cfg.normalize = "None"
    preprocessor = nemo_asr.models.EncDecCTCModelBPE.from_config_dict(preprocessor_cfg)
    preprocessor.to(device)
    hyps = []
    refs = []

    with open(mfst, "r") as mfst_f:
        for l in tqdm(mfst_f, desc="Sample:"):
            asr.reset()
            row = json.loads(l.strip())
            asr.read_audio_file(row['audio_filepath'], delay, model_stride_in_secs)
            hyp = asr.transcribe(tokens_per_chunk, delay)
            hyps.append(hyp)
            refs.append(row['text'])

    wer = word_error_rate(hypotheses=hyps, references=refs)
    return hyps, refs, wer


def main():
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
    parser.add_argument("--chunk_len_in_ms", type=int, default=1600, help="Chunk length in milliseconds")
    parser.add_argument("--output_path", type=str, help="path to output file", default=None)
    parser.add_argument(
        "--model_stride",
        type=int,
        default=8,
        help="Model downsampling factor, 8 for Citrinet models and 4 for Conformer models",
    )

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=args.asr_model)
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=args.asr_model)

    cfg = copy.deepcopy(asr_model._cfg)
    OmegaConf.set_struct(cfg.preprocessor, False)

    # some changes for streaming scenario
    cfg.preprocessor.dither = 0.0
    cfg.preprocessor.pad_to = 0

    if cfg.preprocessor.normalize != "per_feature":
        logging.error("Only EncDecCTCModelBPE models trained with per_feature normalization are supported currently")

    # Disable config overwriting
    OmegaConf.set_struct(cfg.preprocessor, True)
    asr_model.eval()
    asr_model = asr_model.to(asr_model.device)

    feature_stride = cfg.preprocessor['window_stride']
    model_stride_in_secs = feature_stride * args.model_stride
    total_buffer = args.total_buffer_in_secs

    chunk_len = args.chunk_len_in_ms / 1000

    tokens_per_chunk = math.ceil(chunk_len / model_stride_in_secs)
    mid_delay = math.ceil((chunk_len + (total_buffer - chunk_len) / 2) / model_stride_in_secs)
    print(tokens_per_chunk, mid_delay)

    frame_asr = FrameBatchASR(
        asr_model=asr_model, frame_len=chunk_len, total_buffer=args.total_buffer_in_secs, batch_size=args.batch_size,
    )

    hyps, refs, wer = get_wer_feat(
        args.test_manifest,
        frame_asr,
        chunk_len,
        tokens_per_chunk,
        mid_delay,
        cfg.preprocessor,
        model_stride_in_secs,
        asr_model.device,
    )
    logging.info(f"WER is {round(wer, 4)} when decoded with a delay of {round(mid_delay*model_stride_in_secs, 2)}s")

    if args.output_path is not None:

        fname = (
            os.path.splitext(os.path.basename(args.asr_model))[0]
            + "_"
            + os.path.splitext(os.path.basename(args.test_manifest))[0]
            + "_"
            + str(args.chunk_len_in_ms)
            + "_"
            + str(int(total_buffer * 1000))
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
    main()  # noqa pylint: disable=no-value-for-parameter
