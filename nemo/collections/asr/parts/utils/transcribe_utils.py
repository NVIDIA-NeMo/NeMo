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
from typing import List

from tqdm.auto import tqdm

from nemo.collections.asr.models.ctc_models import EncDecCTCModel
from nemo.utils import logging


def transcribe_partial_audio(
    asr_model,
    path2manifest: str,
    batch_size: int = 4,
    logprobs: bool = False,
    return_hypotheses: bool = False,
    num_workers: int = 0,
) -> List[str]:

    assert isinstance(asr_model, EncDecCTCModel), "Currently support CTC model only."

    if return_hypotheses and logprobs:
        raise ValueError(
            "Either `return_hypotheses` or `logprobs` can be True at any given time."
            "Returned hypotheses will contain the logprobs."
        )
    if num_workers is None:
        num_workers = min(batch_size, os.cpu_count() - 1)

    # We will store transcriptions here
    hypotheses = []
    # Model's mode and device
    mode = asr_model.training
    device = next(asr_model.parameters()).device
    dither_value = asr_model.preprocessor.featurizer.dither
    pad_to_value = asr_model.preprocessor.featurizer.pad_to

    try:
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        # Switch model to evaluation mode
        asr_model.eval()
        # Freeze the encoder and decoder modules
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()
        logging_level = logging.get_verbosity()
        logging.set_verbosity(logging.WARNING)

        config = {
            'manifest_filepath': path2manifest,
            'batch_size': batch_size,
            'num_workers': num_workers,
        }

        temporary_datalayer = asr_model._setup_transcribe_dataloader(config)
        for test_batch in tqdm(temporary_datalayer, desc="Transcribing"):
            logits, logits_len, greedy_predictions = asr_model.forward(
                input_signal=test_batch[0].to(device), input_signal_length=test_batch[1].to(device)
            )
            if logprobs:
                # dump log probs per file
                for idx in range(logits.shape[0]):
                    lg = logits[idx][: logits_len[idx]]
                    hypotheses.append(lg.cpu().numpy())
            else:
                current_hypotheses, _ = asr_model._wer.decoding.ctc_decoder_predictions_tensor(
                    decoder_outputs=greedy_predictions,
                    decoder_lengths=logits_len,
                    return_hypotheses=return_hypotheses,
                )

                if return_hypotheses:
                    # dump log probs per file
                    for idx in range(logits.shape[0]):
                        current_hypotheses[idx].y_sequence = logits[idx][: logits_len[idx]]

                hypotheses += current_hypotheses

            del greedy_predictions
            del logits
            del test_batch

    finally:
        # set mode back to its original value
        asr_model.train(mode=mode)
        asr_model.preprocessor.featurizer.dither = dither_value
        asr_model.preprocessor.featurizer.pad_to = pad_to_value
        if mode is True:
            asr_model.encoder.unfreeze()
            asr_model.decoder.unfreeze()
        logging.set_verbosity(logging_level)
    return hypotheses
