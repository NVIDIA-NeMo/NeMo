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

import copy
import glob
import json
import os
import tempfile
from functools import lru_cache

import jiwer
import pytest
import torch
from omegaconf import open_dict

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.rnnt_beam_decoding import BeamBatchedInfer
from nemo.collections.asr.parts.submodules.tdt_beam_decoding import BeamBatchedTDTInfer
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.core.utils import numba_utils
from nemo.core.utils.cuda_python_utils import skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported
from nemo.core.utils.numba_utils import __NUMBA_MINIMUM_VERSION__
from nemo.core.utils.optional_libs import KENLM_AVAILABLE

RNNT_MODEL = "stt_en_conformer_transducer_small"
TDT_MODEL = "nvidia/parakeet-tdt_ctc-110m"
DEVICES = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device('cuda'))

NUMBA_RNNT_LOSS_AVAILABLE = numba_utils.numba_cpu_is_supported(
    __NUMBA_MINIMUM_VERSION__
) or numba_utils.numba_cuda_is_supported(__NUMBA_MINIMUM_VERSION__)


@pytest.fixture()
def test_audio_filenames(test_data_dir):
    return tuple(glob.glob(os.path.join(test_data_dir, "asr", "test", "an4", "wav", "*.wav")))


@lru_cache(maxsize=4)
def get_model(model_name: str, device: torch.device = torch.device("cpu")):
    return ASRModel.from_pretrained(model_name, map_location=device)


@lru_cache(maxsize=4)
def get_model_encoder_output(
    test_audio_filenames,
    num_samples: int,
    model_name: ASRModel,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    audio_filepaths = test_audio_filenames[:num_samples]
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, 'manifest.json'), 'w', encoding='utf-8') as fp:
            for audio_file in audio_filepaths:
                entry = {'audio_filepath': audio_file, 'duration': 100000, 'text': ''}
                fp.write(json.dumps(entry) + '\n')

        config = {
            'paths2audio_files': audio_filepaths,
            'batch_size': len(audio_filepaths),
            'temp_dir': tmpdir,
            'num_workers': 1,
        }

        with torch.no_grad():
            model = get_model(model_name, device)
            model.preprocessor.featurizer.dither = 0.0
            model.preprocessor.featurizer.pad_to = 0
            model.eval()

            temporary_datalayer = model._setup_transcribe_dataloader(config)
            for test_batch in temporary_datalayer:
                encoded, encoded_len = model.forward(
                    input_signal=test_batch[0].to(device, dtype=dtype), input_signal_length=test_batch[1].to(device)
                )
    return model, encoded, encoded_len


def decode_text_from_hypotheses(hyps, decoding):
    decoded_hyps = decoding.decode_hypothesis(hyps)  # type: List[str]

    return decoded_hyps


def decode_text_from_nbest_hypotheses(hyps, decoding):
    all_hypotheses = []

    for nbest_hyp in hyps:  # type: rnnt_utils.NBestHypotheses
        n_hyps = nbest_hyp.n_best_hypotheses  # Extract all hypotheses for this sample
        decoded_hyps = decoding.decode_hypothesis(n_hyps)  # type: List[str]

        all_hypotheses.append(decoded_hyps)

    return all_hypotheses


class TestRNNTDecoding:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
            {"search_type": "maes_batch", "allow_cuda_graphs": False},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("beam_size", [2, 4])
    def test_rnnt_beam_decoding_return_best_hypothesis(
        self, test_audio_filenames, beam_config, device, batch_size, beam_size
    ):
        num_samples = min(batch_size, len(test_audio_filenames))

        model, encoder_output, encoded_lengths = get_model_encoder_output(
            test_audio_filenames, num_samples, RNNT_MODEL, device, dtype=torch.float32
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            **beam_config,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]
            assert type(hyps) == list
            assert type(hyps[0]) == rnnt_utils.Hypothesis

            assert len(hyps) == num_samples
            assert hasattr(hyps[0], "y_sequence")
            assert hasattr(hyps[0], "score")
            assert hasattr(hyps[0], "timestamp")

            assert len(hyps[0].y_sequence) > 0
            assert len(hyps[0].timestamp) > 0

            hyps = decode_text_from_hypotheses(hyps, model.decoding)

            print()

            print(
                f"""Beam search algorithm: {beam_config['search_type']},
                    beam size: {beam_size},
                    Cuda Graphs: {beam_config.get('allow_cuda_graphs', True)}"""
            )
            for hyp_idx, hyp in enumerate(hyps):
                print("Sample: ", hyp_idx)
                print("Decoded text: ", hyp.text)
                print("Score: ", hyp.score)
                print("Transcript", hyp.y_sequence)
                print("Timesteps", hyp.timestamp)
                print()

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
            {"search_type": "maes_batch", "allow_cuda_graphs": False},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("beam_size", [2, 4])
    def test_rnnt_beam_decoding_return_nbest(self, test_audio_filenames, beam_config, device, beam_size, batch_size):
        num_samples = min(batch_size, len(test_audio_filenames))

        model, encoder_output, encoded_lengths = get_model_encoder_output(
            test_audio_filenames, num_samples, RNNT_MODEL, device, dtype=torch.float32
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=False,
            **beam_config,
        )

        with torch.no_grad():
            batch_nbest_hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]
            assert type(batch_nbest_hyps) == list
            assert type(batch_nbest_hyps[0]) == rnnt_utils.NBestHypotheses

            assert len(batch_nbest_hyps) == num_samples

            assert hasattr(batch_nbest_hyps[0].n_best_hypotheses[0], "y_sequence")
            assert hasattr(batch_nbest_hyps[0].n_best_hypotheses[0], "score")
            assert hasattr(batch_nbest_hyps[0].n_best_hypotheses[0], "timestamp")

            assert len(batch_nbest_hyps[0].n_best_hypotheses[0].y_sequence) > 0
            assert len(batch_nbest_hyps[0].n_best_hypotheses[0].timestamp) > 0

            batch_nbest_hyps = decode_text_from_nbest_hypotheses(batch_nbest_hyps, model.decoding)

            print()
            print(f"Decoding device: {encoder_output.device}")
            print(
                f"""Beam search algorithm: {beam_config['search_type']},
                    beam size: {beam_size},
                    Cuda Graphs: {beam_config.get('allow_cuda_graphs', True)}"""
            )
            for batch_idx, nbest_hyps in enumerate(batch_nbest_hyps):
                print(f"Batch idx: {batch_idx}")
                for idx, hyp in enumerate(nbest_hyps):
                    print(f"Hyp index: {idx + 1}")
                    print("Text: ", hyp.text)
                    print("Score: ", hyp.score)

                    assert len(hyp.timestamp) > 0
                    print("Transcripts: ", hyp.y_sequence)
                    print("Timesteps: ", hyp.timestamp)
                    print()

            print()

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False, "ngram_lm_alpha": 0.3},
            {"search_type": "maes_batch", "allow_cuda_graphs": False, "ngram_lm_alpha": 0.3},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True, "ngram_lm_alpha": 0.3},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("pruning_mode", ["late", "early"])
    @pytest.mark.parametrize("blank_lm_score_mode", ["no_score", "lm_weighted_full"])
    def test_rnnt_beam_decoding_kenlm(
        self,
        test_data_dir,
        test_audio_filenames,
        beam_config,
        device,
        batch_size,
        beam_size,
        pruning_mode,
        blank_lm_score_mode,
    ):
        kenlm_model_path = os.path.join(
            test_data_dir, "asr", "kenlm_ngram_lm", "parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        )
        beam_config["ngram_lm_model"] = kenlm_model_path

        num_samples = min(batch_size, len(test_audio_filenames))

        model, encoder_output, encoded_lengths = get_model_encoder_output(
            test_audio_filenames, num_samples, RNNT_MODEL, device, dtype=torch.float32
        )

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            pruning_mode=pruning_mode,
            blank_lm_score_mode=blank_lm_score_mode,
            **beam_config,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]
            assert type(hyps) == list
            assert type(hyps[0]) == rnnt_utils.Hypothesis

            assert len(hyps) == num_samples
            assert hasattr(hyps[0], "y_sequence")
            assert hasattr(hyps[0], "score")
            assert hasattr(hyps[0], "timestamp")

            assert len(hyps[0].y_sequence) > 0
            assert len(hyps[0].timestamp) > 0

            hyps = decode_text_from_hypotheses(hyps, model.decoding)

            print()

            print(
                f"""Beam search algorithm: {beam_config['search_type']},
                    beam size: {beam_size},
                    Cuda Graphs: {beam_config.get('allow_cuda_graphs', True)}"""
            )
            for hyp_idx, hyp in enumerate(hyps):
                print("Sample: ", hyp_idx)
                print("Decoded text: ", hyp.text)
                print("Score: ", hyp.score)
                print("Transcript", hyp.y_sequence)
                print("Timesteps", hyp.timestamp)
                print()

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.unit
    @pytest.mark.with_downloads
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("beam_size", [4, 8])
    def test_cuda_graph_rnnt_batched_alsd_decoder(self, test_audio_filenames, batch_size, beam_size):
        # Set device to CUDA
        device = torch.device("cuda")
        model = get_model(RNNT_MODEL, device=device)

        # Modify decoding config
        decoding_config = copy.deepcopy(model.cfg.decoding)
        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = beam_size
            decoding_config["beam"]["allow_cuda_graphs"] = False
            decoding_config["beam"]["return_best_hypothesis"] = False

        # Change decoding strategy
        model.change_decoding_strategy(decoding_config)

        # Transcribe without CUDA graphs
        actual_hypotheses = model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        actual_transcripts = [[hyp.text for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_scores = [[hyp.score for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_timestamps = [[hyp.timestamp for hyp in actual_beam] for actual_beam in actual_hypotheses]

        # Re-enable CUDA graphs
        decoding_config["beam"]["allow_cuda_graphs"] = True
        model.change_decoding_strategy(decoding_config)

        # Transcribe with CUDA graphs
        cudagraph_hypotheses = model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        cudagraph_transcripts = [[hyp.text for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]
        cudagraph_scores = [[hyp.score for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]
        cudagraph_timestamps = [[hyp.timestamp for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]

        for batch_idx in range(min(batch_size, len(test_audio_filenames))):
            assert len(actual_transcripts[batch_idx]) == len(cudagraph_transcripts[batch_idx])
            assert cudagraph_scores[batch_idx] == pytest.approx(
                actual_scores[batch_idx], abs=1e-2
            ), f"Scores mismatch for batch_idx {batch_idx}"
            assert (
                cudagraph_timestamps[batch_idx] == actual_timestamps[batch_idx]
            ), f"Timestamps mismatch for batch_idx {batch_idx}"

            # Calculate WER (Word Error Rate)
            wer_value = jiwer.wer(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx])

            # Assert WER is within tolerance
            assert wer_value <= 1e-3, "Cuda graph greedy decoder should match original decoder implementation."

            # Print erroneous samples if WER is high
            for actual, fast in zip(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx]):
                if actual != fast:
                    print("Erroneous samples in batch:", batch_idx)
                    print("Original transcript:", actual)
                    print("New transcript:", fast)

    @pytest.mark.with_downloads
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
    @pytest.mark.parametrize("force_mode", ["no_graphs", "no_while_loops", "full_graph"])
    def test_stated_stateless(self, test_audio_filenames, force_mode: str):
        '''Compares stated and stateless implementations'''
        if force_mode == "full_graph":
            skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

        batch_size = 16
        device = torch.device("cuda")
        model = get_model(RNNT_MODEL, device)
        decoding_config = copy.deepcopy(model.cfg.decoding)

        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypotheses"] = False
            decoding_config["beam"]["allow_cuda_graphs"] = False

        model.change_decoding_strategy(decoding_config)

        actual_hypotheses = model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        actual_transcripts = [[hyp.text for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_scores = [[hyp.score for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_timestamps = [[hyp.timestamp for hyp in actual_beam] for actual_beam in actual_hypotheses]

        # transcribe with use implementation with cuda graphs
        decoding_config["beam"]["allow_cuda_graphs"] = True
        model.change_decoding_strategy(decoding_config)
        model.decoding.decoding._decoding_computer.force_cuda_graphs_mode(mode=force_mode)

        cudagraph_hypotheses = model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        cudagraph_transcripts = [[hyp.text for hyp in cudagraphs_beam] for cudagraphs_beam in cudagraph_hypotheses]
        cudagraph_scores = [[hyp.score for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]
        cudagraph_timestamps = [[hyp.timestamp for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]

        for batch_idx in range(min(batch_size, len(test_audio_filenames))):
            assert len(actual_transcripts[batch_idx]) == len(cudagraph_transcripts[batch_idx])
            assert cudagraph_scores[batch_idx] == pytest.approx(
                actual_scores[batch_idx], abs=1e-2
            ), f"Scores mismatch for batch_idx {batch_idx}"
            assert (
                cudagraph_timestamps[batch_idx] == actual_timestamps[batch_idx]
            ), f"Timestamps mismatch for batch_idx {batch_idx}"

            wer = jiwer.wer(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx])

            assert wer <= 1e-3, "Cuda graph greedy decoder should match original decoder implementation."

            for actual, fast in zip(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx]):
                if actual != fast:
                    print("Erroneous samples in batch:", batch_idx)
                    print("Original transcript:", actual)
                    print("New transcript:", fast)

    @pytest.mark.with_downloads
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
    @pytest.mark.parametrize("force_mode", ["no_graphs", "no_while_loops", "full_graph"])
    def test_stated_stateless(self, test_audio_filenames, force_mode: str):
        '''Compares stated and stateless implementations with bfloat16'''
        # for bfloat16 computational errors accumulate, so just checking if algorithms run without errors
        if force_mode == "full_graph":
            skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

        batch_size = 16
        device = torch.device("cuda")
        model = get_model(RNNT_MODEL, device=device)
        decoding_config = copy.deepcopy(model.cfg.decoding)

        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypotheses"] = False
            decoding_config["beam"]["allow_cuda_graphs"] = False

        model.change_decoding_strategy(decoding_config)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)

        # transcribe with use implementation with cuda graphs
        decoding_config["beam"]["allow_cuda_graphs"] = True
        model.change_decoding_strategy(decoding_config)
        model.decoding.decoding._decoding_computer.force_cuda_graphs_mode(mode=force_mode)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)


class TestTDTDecoding:
    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("beam_size", [2, 4])
    def test_rnnt_beam_decoding_return_best_hypothesis(
        self, test_audio_filenames, beam_config, device, batch_size, beam_size
    ):
        num_samples = min(batch_size, len(test_audio_filenames))

        model, encoder_output, encoded_lengths = get_model_encoder_output(
            test_audio_filenames, num_samples, TDT_MODEL, device, dtype=torch.float32
        )

        model_config = model.to_config_dict()
        durations = list(model_config["model_defaults"]["tdt_durations"])

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedTDTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            durations=durations,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            **beam_config,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]
            assert type(hyps) == list
            assert type(hyps[0]) == rnnt_utils.Hypothesis

            assert len(hyps) == num_samples
            assert hasattr(hyps[0], "y_sequence")
            assert hasattr(hyps[0], "score")
            assert hasattr(hyps[0], "timestamp")

            assert len(hyps[0].y_sequence) > 0
            assert len(hyps[0].timestamp) > 0

            hyps = decode_text_from_hypotheses(hyps, model.decoding)

            print()

            print(
                f"""Beam search algorithm: {beam_config['search_type']},
                    beam size: {beam_size},
                    Cuda Graphs: {beam_config.get('allow_cuda_graphs', True)}"""
            )
            for hyp_idx, hyp in enumerate(hyps):
                print("Sample: ", hyp_idx)
                print("Decoded text: ", hyp.text)
                print("Score: ", hyp.score)
                print("Transcript", hyp.y_sequence)
                print("Timesteps", hyp.timestamp)
                print()

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "beam_config",
        [
            {"search_type": "malsd_batch", "allow_cuda_graphs": False},
            {"search_type": "malsd_batch", "allow_cuda_graphs": True},
        ],
    )
    @pytest.mark.parametrize("batch_size", [4, 16])
    @pytest.mark.parametrize("beam_size", [2, 4])
    def test_rnnt_beam_decoding_return_nbest(self, test_audio_filenames, beam_config, device, beam_size, batch_size):
        num_samples = min(batch_size, len(test_audio_filenames))

        model, encoder_output, encoded_lengths = get_model_encoder_output(
            test_audio_filenames, num_samples, TDT_MODEL, device, dtype=torch.float32
        )

        model_config = model.to_config_dict()
        durations = list(model_config["model_defaults"]["tdt_durations"])

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedTDTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            durations=durations,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=False,
            **beam_config,
        )

        with torch.no_grad():
            batch_nbest_hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]
            assert type(batch_nbest_hyps) == list
            assert type(batch_nbest_hyps[0]) == rnnt_utils.NBestHypotheses

            assert len(batch_nbest_hyps) == num_samples

            assert hasattr(batch_nbest_hyps[0].n_best_hypotheses[0], "y_sequence")
            assert hasattr(batch_nbest_hyps[0].n_best_hypotheses[0], "score")
            assert hasattr(batch_nbest_hyps[0].n_best_hypotheses[0], "timestamp")

            assert len(batch_nbest_hyps[0].n_best_hypotheses[0].y_sequence) > 0
            assert len(batch_nbest_hyps[0].n_best_hypotheses[0].timestamp) > 0

            batch_nbest_hyps = decode_text_from_nbest_hypotheses(batch_nbest_hyps, model.decoding)

            print()
            print(f"Decoding device: {encoder_output.device}")
            print(
                f"Beam search algorithm : {beam_config['search_type']}, \
                    beam size: {beam_size}, \
                    Cuda Graphs: {beam_config.get('allow_cuda_graphs', True)}"
            )
            for batch_idx, nbest_hyps in enumerate(batch_nbest_hyps):
                print(f"Batch idx: {batch_idx}")
                for idx, hyp in enumerate(nbest_hyps):
                    print(f"Hyp index: {idx + 1}")
                    print("Text: ", hyp.text)
                    print("Score: ", hyp.score)

                    assert len(hyp.timestamp) > 0
                    print("Transcripts: ", hyp.y_sequence)
                    print("Timesteps: ", hyp.timestamp)
                    print()

            print()

    @pytest.mark.skipif(
        not NUMBA_RNNT_LOSS_AVAILABLE,
        reason='RNNTLoss has not been compiled with appropriate numba version.',
    )
    @pytest.mark.skipif(not KENLM_AVAILABLE, reason="KenLM is not available")
    @pytest.mark.with_downloads
    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize(
        "beam_config",
        [
            {
                "search_type": "malsd_batch",
                "allow_cuda_graphs": False,
                "ngram_lm_alpha": 0.3,
            },
            {
                "search_type": "malsd_batch",
                "allow_cuda_graphs": True,
                "ngram_lm_alpha": 0.3,
            },
        ],
    )
    @pytest.mark.parametrize("batch_size", [4])
    @pytest.mark.parametrize("beam_size", [4])
    @pytest.mark.parametrize("pruning_mode", ["late", "early"])
    @pytest.mark.parametrize("blank_lm_score_mode", ["lm_weighted_full", "no_score"])
    def test_rnnt_beam_decoding_kenlm(
        self,
        test_data_dir,
        test_audio_filenames,
        beam_config,
        device,
        batch_size,
        beam_size,
        pruning_mode,
        blank_lm_score_mode,
    ):
        kenlm_model_path = os.path.join(
            test_data_dir, "asr", "kenlm_ngram_lm", "parakeet-tdt_ctc-110m-libri-1024.kenlm.tmp.arpa"
        )
        beam_config["ngram_lm_model"] = kenlm_model_path

        audio_filepaths = glob.glob(os.path.join(test_data_dir, "asr", "test", "an4", "wav", "*.wav"))
        num_samples = min(batch_size, len(audio_filepaths))

        model, encoder_output, encoded_lengths = get_model_encoder_output(
            test_audio_filenames, num_samples, TDT_MODEL, device, dtype=torch.float32
        )

        model_config = model.to_config_dict()
        durations = list(model_config["model_defaults"]["tdt_durations"])

        vocab_size = model.tokenizer.vocab_size
        decoding = BeamBatchedTDTInfer(
            model.decoder,
            model.joint,
            blank_index=vocab_size,
            durations=durations,
            beam_size=beam_size,
            score_norm=True,
            return_best_hypothesis=True,
            pruning_mode=pruning_mode,
            blank_lm_score_mode=blank_lm_score_mode,
            **beam_config,
        )

        with torch.no_grad():
            hyps = decoding(encoder_output=encoder_output, encoded_lengths=encoded_lengths)[0]
            assert type(hyps) == list
            assert type(hyps[0]) == rnnt_utils.Hypothesis

            assert len(hyps) == num_samples
            assert hasattr(hyps[0], "y_sequence")
            assert hasattr(hyps[0], "score")
            assert hasattr(hyps[0], "timestamp")

            assert len(hyps[0].y_sequence) > 0
            assert len(hyps[0].timestamp) > 0

            hyps = decode_text_from_hypotheses(hyps, model.decoding)

            print()

            print(
                f"Beam search algorithm : {beam_config['search_type']}, \
                    beam size: {beam_size}, \
                    Cuda Graphs: {beam_config.get('allow_cuda_graphs', True)}"
            )
            for hyp_idx, hyp in enumerate(hyps):
                print("Sample: ", hyp_idx)
                print("Decoded text: ", hyp.text)
                print("Score: ", hyp.score)
                print("Transcript", hyp.y_sequence)
                print("Timesteps", hyp.timestamp)
                print()

    @pytest.mark.with_downloads
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
    @pytest.mark.parametrize("force_mode", ["no_graphs", "no_while_loops", "full_graph"])
    def test_stated_stateless(self, test_audio_filenames, force_mode: str):
        '''Compares stated and stateless implementations with bfloat16'''
        if force_mode == "full_graph":
            skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

        batch_size = 16
        device = torch.device("cuda")
        nemo_model = get_model(TDT_MODEL, device)
        decoding_config = copy.deepcopy(nemo_model.cfg.decoding)

        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypotheses"] = False
            decoding_config["beam"]["allow_cuda_graphs"] = False

        nemo_model.change_decoding_strategy(decoding_config)

        actual_hypotheses = nemo_model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        actual_transcripts = [[hyp.text for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_scores = [[hyp.score for hyp in actual_beam] for actual_beam in actual_hypotheses]
        actual_timestamps = [[hyp.timestamp for hyp in actual_beam] for actual_beam in actual_hypotheses]

        # transcribe with use implementation with cuda graphs

        nemo_model.change_decoding_strategy(decoding_config)
        nemo_model.decoding.decoding._decoding_computer.force_cuda_graphs_mode(mode=force_mode)

        cudagraph_hypotheses = nemo_model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
        cudagraph_transcripts = [[hyp.text for hyp in cudagraphs_beam] for cudagraphs_beam in cudagraph_hypotheses]
        cudagraph_scores = [[hyp.score for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]
        cudagraph_timestamps = [[hyp.timestamp for hyp in cudagraph_beam] for cudagraph_beam in cudagraph_hypotheses]

        for batch_idx in range(min(batch_size, len(test_audio_filenames))):
            assert len(actual_transcripts[batch_idx]) == len(cudagraph_transcripts[batch_idx])
            assert cudagraph_scores[batch_idx] == pytest.approx(
                actual_scores[batch_idx], abs=abs
            ), f"Scores mismatch for batch_idx {batch_idx}"
            assert (
                cudagraph_timestamps[batch_idx] == actual_timestamps[batch_idx]
            ), f"Timestamps mismatch for batch_idx {batch_idx}"

            wer = jiwer.wer(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx])

            assert wer <= 1e-3, "Cuda graph greedy decoder should match original decoder implementation."

            for actual, fast in zip(actual_transcripts[batch_idx], cudagraph_transcripts[batch_idx]):

                print("Erroneous samples in batch:", batch_idx)
                print("Original transcript:", actual)
                print("New transcript:", fast)

    @pytest.mark.with_downloads
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA decoder can run only on CUDA")
    @pytest.mark.parametrize("force_mode", ["no_graphs", "no_while_loops", "full_graph"])
    def test_stated_stateless(self, test_audio_filenames, force_mode: str):
        '''Compares stated and stateless implementations with bfloat16'''
        # for bfloat16 computational errors accumulate, so just checking if algorithms run without errors
        if force_mode == "full_graph":
            skip_cuda_python_test_if_cuda_graphs_conditional_nodes_not_supported()

        batch_size = 16
        device = torch.device("cuda")
        nemo_model = get_model(TDT_MODEL, device)
        decoding_config = copy.deepcopy(nemo_model.cfg.decoding)

        with open_dict(decoding_config):
            decoding_config["strategy"] = "malsd_batch"
            decoding_config["beam"]["beam_size"] = 4
            decoding_config["beam"]["return_best_hypotheses"] = False
            decoding_config["beam"]["allow_cuda_graphs"] = False

        nemo_model.change_decoding_strategy(decoding_config)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            nemo_model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)

        # transcribe with use implementation with cuda graphs
        decoding_config["beam"]["allow_cuda_graphs"] = True
        nemo_model.change_decoding_strategy(decoding_config)
        nemo_model.decoding.decoding._decoding_computer.force_cuda_graphs_mode(mode=force_mode)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            nemo_model.transcribe(test_audio_filenames, batch_size=batch_size, num_workers=None)
