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
This script is based on speech_to_text_eval.py and allows you to score the hypotheses
with sclite. A local installation from https://github.com/usnistgov/SCTK is required.
Hypotheses and references are first saved in trn format and are scored after applying a glm
file (if provided).

# Usage

python speech_to_text_sclite.py \
    --asr_model="<Path to ASR Model>" \
    --dataset="<Path to manifest file>" \
    --out_dir="<Path to output dir, should be unique per model evaluated>" \
    --sctk_dir="<Path to root directory where SCTK is installed>" \
    --glm="<OPTIONAL: Path to glm file>" \
    --batch_size=4

"""

import errno
import json
import os
import subprocess
from argparse import ArgumentParser

import torch

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.utils import logging

try:
    from torch.cuda.amp import autocast
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def autocast(enabled=None):
        yield


def score_with_sctk(sctk_dir, ref_fname, hyp_fname, out_dir, glm=""):
    sclite_path = os.path.join(sctk_dir, "bin", "sclite")
    if not os.path.exists(sclite_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), sclite_path)
    # apply glm
    if os.path.exists(glm):
        rfilter_path = os.path.join(sctk_dir, "bin", "rfilter1")
        if not os.path.exists(rfilter_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), rfilter_path)
        hypglm = os.path.join(out_dir, os.path.basename(hyp_fname)) + ".glm"
        rfilt_cmd = [rfilter_path] + [glm]
        with open(hypglm, "w", encoding='utf-8') as hypf, open(hyp_fname, "r", encoding='utf-8') as hyp_in:
            subprocess.run(rfilt_cmd, stdin=hyp_in, stdout=hypf)
        refglm = os.path.join(out_dir, os.path.basename(ref_fname)) + ".glm"
        with open(refglm, "w", encoding='utf-8') as reff, open(ref_fname, "r", encoding='utf-8') as ref_in:
            subprocess.run(rfilt_cmd, stdin=ref_in, stdout=reff)
    else:
        refglm = ref_fname
        hypglm = hyp_fname

    _ = subprocess.check_output(f"{sclite_path} -h {hypglm}  -r {refglm} -i wsj -o all", shell=True)


can_gpu = torch.cuda.is_available()


def get_utt_info(manifest_path):
    info_list = []
    with open(manifest_path, "r", encoding='utf-8') as utt_f:
        for line in utt_f:
            utt = json.loads(line)
            info_list.append(utt)

    return info_list


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--asr_model", type=str, default="QuartzNet15x5Base-En", required=False, help="Pass: 'QuartzNet15x5Base-En'",
    )
    parser.add_argument("--dataset", type=str, required=True, help="path to evaluation data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out_dir", type=str, required=True, help="Destination dir for output files")
    parser.add_argument("--sctk_dir", type=str, required=False, default="", help="Path to sctk root dir")
    parser.add_argument("--glm", type=str, required=False, default="", help="Path to glm file")
    args = parser.parse_args()
    torch.set_grad_enabled(False)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    use_sctk = os.path.exists(args.sctk_dir)

    if args.asr_model.endswith('.nemo'):
        logging.info(f"Using local ASR model from {args.asr_model}")
        asr_model = ASRModel.restore_from(restore_path=args.asr_model, map_location='cpu')
    else:
        logging.info(f"Using NGC cloud ASR model {args.asr_model}")
        asr_model = ASRModel.from_pretrained(model_name=args.asr_model, map_location='cpu')

    if can_gpu:
        asr_model = asr_model.cuda()

    asr_model.eval()

    manifest_data = read_manifest(args.dataset)

    references = [data['text'] for data in manifest_data]
    audio_filepaths = [data['audio_filepath'] for data in manifest_data]

    with autocast():
        hypotheses = asr_model.transcribe(audio_filepaths, batch_size=args.batch_size)

        # if transcriptions form a tuple (from RNNT), extract just "best" hypothesis
        if type(hypotheses) == tuple and len(hypotheses) == 2:
            hypotheses = hypotheses[0]

    info_list = get_utt_info(args.dataset)
    hypfile = os.path.join(args.out_dir, "hyp.trn")
    reffile = os.path.join(args.out_dir, "ref.trn")
    with open(hypfile, "w") as hyp_f, open(reffile, "w") as ref_f:
        for i in range(len(hypotheses)):
            utt_id = os.path.splitext(os.path.basename(info_list[i]['audio_filepath']))[0]
            # rfilter in sctk likes each transcript to have a space at the beginning
            hyp_f.write(" " + hypotheses[i] + " (" + utt_id + ")" + "\n")
            ref_f.write(" " + references[i] + " (" + utt_id + ")" + "\n")

    if use_sctk:
        score_with_sctk(args.sctk_dir, reffile, hypfile, args.out_dir, glm=args.glm)


if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
