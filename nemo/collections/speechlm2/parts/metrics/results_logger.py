# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import json
import torch
import torchaudio
from collections import defaultdict
from nemo.utils import logging
import shutil

class ResultsLogger:
    """
    Saves audios and a json file with the model outputs.
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.audio_save_path = os.path.join(save_path, "pred_wavs")
        os.makedirs(self.audio_save_path, exist_ok=True)
        self.matadata_save_path = os.path.join(save_path, "metadatas")
        os.makedirs(self.matadata_save_path, exist_ok=True)

    def reset(self):
        # ensures that the output directories is emptly
        if os.path.isdir(self.audio_save_path):
            shutil.rmtree(self.audio_save_path)
        os.makedirs(self.audio_save_path, exist_ok=True)
        if os.path.isdir(self.matadata_save_path):
            shutil.rmtree(self.matadata_save_path)
        os.makedirs(self.matadata_save_path, exist_ok=True)
        return self

    @staticmethod
    def merge_and_save_audio(out_audio_path: str, pred_audio: torch.Tensor, pred_audio_sr: int, user_audio: torch.Tensor, user_audio_sr: int) -> None:
        user_audio = torchaudio.functional.resample(user_audio.float(), user_audio_sr, pred_audio_sr)
        T1, T2 = pred_audio.shape[0], user_audio.shape[0]
        max_len = max(T1, T2)
        pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
        user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)

        # combine audio in a multichannel audio
        combined_wav = torch.cat([user_audio_padded.squeeze().unsqueeze(0).detach().cpu(), pred_audio_padded.squeeze().unsqueeze(0).detach().cpu()], dim=0)

        # save audio
        torchaudio.save(out_audio_path, combined_wav.squeeze(), pred_audio_sr)
        logging.info(f"Audio saved at: {out_audio_path}")

    def update(self, name: str, refs: list[str], hyps: list[str], asr_hyps: list[str], samples_id: list[str], pred_audio: torch.Tensor, pred_audio_sr: int, user_audio: torch.Tensor, user_audio_sr: int) -> None:
        out_json_path = os.path.join(self.matadata_save_path, f"{name}.json")
        out_dicts = []
        for i in range(len(refs)):
            # save audio
            sample_id = samples_id[i][:150] # make sure that sample id is not too big
            out_audio_path = os.path.join(self.audio_save_path, f"{name}_{sample_id}.wav")
            self.merge_and_save_audio(out_audio_path, pred_audio[i], pred_audio_sr, user_audio[i], user_audio_sr)

            # cache metadata
            out_dict = {"target_text": refs[i], "pred_text": hyps[i], "speech_pred_transcribed": asr_hyps[i], "audio_path": os.path.relpath(out_audio_path, self.save_path)}
            out_dicts.append(out_dict)

        with open(out_json_path, 'a+', encoding='utf-8') as fout:
            for out_dict in out_dicts:
                json.dump(out_dict, fout)

        logging.info(f"Metadata file for {name} dataset updated at: {out_json_path}")

