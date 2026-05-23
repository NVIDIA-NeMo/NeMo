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

import logging
import multiprocessing as mp
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from lhotse import CutSet, MonoCut, Seconds, SupervisionSegment

# Use NeMo's force alignment utilities instead of torchaudio
from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.parts.utils.aligner_utils import (
    add_t_start_end_to_utt_obj,
    get_batch_variables,
    viterbi_decoding,
)


class ForceAligner:
    """
    Force alignment utility using NeMo CTC-based ASR models for speech-to-text alignment.
    """

    def __init__(
        self,
        asr_model: Optional[ASRModel] = None,
        device: str = None,
        frame_length: float = 0.02,
        asr_model_name: str = "stt_en_fastconformer_ctc_large",
    ):
        """
        Initialize the ForceAligner.

        Args:
            asr_model: NeMo ASR model instance for alignment. If None, will load from asr_model_name
            device: Device to run alignment on (default: auto-detect)
            frame_length: Frame length in seconds for timestamp conversion
            asr_model_name: Name of the NeMo ASR model to load if asr_model is None
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_length = frame_length
        self.asr_model_name = asr_model_name

        self.asr_model = asr_model
        self.output_timestep_duration = None
        self._model_loaded = False

    def _load_asr_model(self):
        """Load the NeMo ASR model."""
        try:
            if self.device == 'cuda' and mp.get_start_method(allow_none=True) == 'fork':
                logging.warning(
                    "Detected 'fork' multiprocessing start method with CUDA device. "
                    "To avoid CUDA re-initialization errors in worker processes, "
                    "falling back to CPU for force alignment. "
                    "To use CUDA, set mp.set_start_method('spawn', force=True) in your main training script "
                    "before creating the DataLoader."
                )
                self.device = 'cpu'

            device = torch.device(self.device)
            logging.info(f"Loading NeMo ASR model '{self.asr_model_name}' for force alignment on device {device}")

            if self.asr_model is None:
                # Load ASR model from pretrained
                self.asr_model = ASRModel.from_pretrained(self.asr_model_name, map_location=device)
            else:
                self.asr_model = self.asr_model.to(device)

            self.asr_model.eval()

            # Calculate output timestep duration
            try:
                self.output_timestep_duration = (
                    self.asr_model.cfg['preprocessor']['window_stride'] * self.asr_model.encoder.subsampling_factor
                )
            except Exception as e:
                # Default fallback based on typical FastConformer settings
                self.output_timestep_duration = 0.04
                logging.warning(
                    f"Could not calculate output_timestep_duration from model config: {e}. "
                    f"Using default {self.output_timestep_duration}s"
                )

            logging.info(
                f"NeMo ASR model loaded successfully for force alignment. "
                f"Output timestep duration: {self.output_timestep_duration}s"
            )
        except Exception as e:
            logging.error(f"Failed to load NeMo ASR model for force alignment: {e}")
            self.asr_model = None
            raise

    def batch_force_align_user_audio(self, cuts: CutSet, source_sample_rate: int = 16000) -> CutSet:
        """
        Perform batch force alignment on all user audio segments.

        Args:
            cuts: CutSet containing all cuts to process
            source_sample_rate: Source sample rate of the audio

        Returns:
            CutSet containing only cuts where force alignment succeeded
        """
        if not self._model_loaded:
            self._load_asr_model()
            self._model_loaded = True

        if self.asr_model is None:
            logging.warning("ASR model not available for force alignment, returning empty cutset")
            return CutSet.from_cuts([])

        user_supervisions = []
        user_cuts = []
        cut_to_supervisions = {}

        for cut in cuts:
            user_sups_in_cut = []
            for supervision in cut.supervisions:
                if supervision.speaker.lower() == "user":
                    user_supervisions.append(supervision)
                    user_cuts.append(cut)
                    user_sups_in_cut.append(supervision)
            if user_sups_in_cut:
                cut_to_supervisions[cut.id] = user_sups_in_cut

        if not user_supervisions:
            logging.info("No user supervisions found for force alignment")
            return cuts

        logging.info(f"Performing force alignment on {len(user_supervisions)} user audio segments")

        # Process alignments
        success_count = 0
        failed_count = 0

        for i, (supervision, cut) in enumerate(zip(user_supervisions, user_cuts)):
            try:
                start_time = supervision.start
                duration = supervision.duration

                # Extract user segment audio
                user_cut = cut.truncate(offset=start_time, duration=duration)

                # Strip timestamps from text
                text = self._strip_timestamps(supervision.text)

                # Get alignment using NeMo utilities
                alignment_result = self._align_with_nemo(user_cut, text, source_sample_rate)

                if alignment_result is not None:
                    # Convert alignment to timestamped text
                    original_text = supervision.text
                    timestamped_text = self._convert_alignment_to_timestamped_text(alignment_result, original_text)
                    supervision.text = timestamped_text
                    success_count += 1
                else:
                    failed_count += 1

                # Periodic cleanup
                if self.device == 'cuda' and torch.cuda.is_available() and (i + 1) % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                logging.error(f"Failed to align segment {i}: {e}")
                failed_count += 1

                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if failed_count > 0:
            logging.warning(
                f"Force alignment failed for {failed_count}/{len(user_supervisions)} user segments. "
                f"Keeping all cuts with original text for failed alignments."
            )
        else:
            logging.info(f"Force alignment succeeded for all {success_count} user segments.")

        return cuts

    def _align_with_nemo(self, cut: MonoCut, text: str, source_sample_rate: int) -> Optional[List[Dict[str, Any]]]:
        """
        Perform forced alignment using NeMo ASR model and CTC-based alignment.

        Args:
            cut: Lhotse MonoCut containing audio segment
            text: Text transcript (without timestamps)
            source_sample_rate: Sample rate of the audio

        Returns:
            List of word segments with timing information, or None if alignment fails
        """
        if not text.strip():
            logging.warning("Empty text for alignment")
            return None

        # Normalize the text
        normalized_text = self._normalize_transcript(text)
        if not normalized_text.strip():
            logging.warning(f"Text became empty after normalization: {text}")
            return None

        try:
            # Load audio from cut
            audio = cut.load_audio()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)  # Convert to mono

            # Resample to 16kHz if needed
            target_sample_rate = 16000
            if source_sample_rate != target_sample_rate:
                # Use scipy for resampling to avoid torchaudio dependency
                from scipy import signal

                num_samples = int(len(audio) * target_sample_rate / source_sample_rate)
                audio = signal.resample(audio, num_samples)
                source_sample_rate = target_sample_rate

            # Add silence padding for better alignment at the end
            silence_duration = 0.64
            silence_samples = int(silence_duration * target_sample_rate)
            audio = np.concatenate([audio, np.zeros(silence_samples)])

            # Save audio to temporary file (NeMo's aligner expects file paths)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
                sf.write(tmp_path, audio, target_sample_rate)

            try:
                # Get alignment using NeMo's viterbi decoding
                (
                    log_probs_batch,
                    y_batch,
                    T_batch,
                    U_batch,
                    utt_obj_batch,
                    output_timestep_duration,
                ) = get_batch_variables(
                    audio=[tmp_path],
                    model=self.asr_model,
                    gt_text_batch=[normalized_text],
                    align_using_pred_text=False,
                    output_timestep_duration=self.output_timestep_duration,
                )

                if len(utt_obj_batch) == 0 or len(utt_obj_batch[0].token_ids_with_blanks) == 0:
                    logging.warning(f"Failed to tokenize text for alignment: {text}")
                    return None

                # Perform Viterbi decoding
                alignments_batch = viterbi_decoding(
                    log_probs_batch=log_probs_batch,
                    y_batch=y_batch,
                    T_batch=T_batch,
                    U_batch=U_batch,
                    viterbi_device=torch.device(self.device),
                )

                if len(alignments_batch) == 0:
                    logging.warning(f"Viterbi decoding returned no alignments for text: {text}")
                    return None

                # Add timing information to utterance object
                utt_obj = utt_obj_batch[0]
                alignment = alignments_batch[0]
                utt_obj = add_t_start_end_to_utt_obj(utt_obj, alignment, output_timestep_duration)

                # Extract word-level timestamps
                word_segments = self._extract_word_timestamps(utt_obj)

                return word_segments if word_segments else None

            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    logging.warning(f"Failed to delete temporary file {tmp_path}: {e}")

        except Exception as e:
            logging.error(f"Failed to align with NeMo: {e}")
            return None

    def _extract_word_timestamps(self, utt_obj) -> List[Dict[str, Any]]:
        """
        Extract word-level timestamps from the utterance object returned by NeMo aligner.

        Args:
            utt_obj: Utterance object with timing information

        Returns:
            List of word segments with timing information
        """
        word_segments = []

        for segment_or_token in utt_obj.segments_and_tokens:
            # Check if this is a Segment object (has words_and_tokens attribute)
            if hasattr(segment_or_token, 'words_and_tokens'):
                segment = segment_or_token
                for word_or_token in segment.words_and_tokens:
                    # Check if this is a Word object (has 'text' and timing attributes)
                    if hasattr(word_or_token, 'text') and hasattr(word_or_token, 't_start'):
                        word = word_or_token
                        # Only include words with valid timing (t_start and t_end >= 0)
                        if (
                            word.t_start is not None
                            and word.t_end is not None
                            and word.t_start >= 0
                            and word.t_end >= 0
                        ):
                            word_segments.append(
                                {
                                    'word': word.text,
                                    'start': word.t_start,
                                    'end': word.t_end,
                                    'score': 1.0,  # NeMo CTC alignment doesn't provide confidence scores
                                }
                            )

        return word_segments

    def _normalize_transcript(self, transcript: str) -> str:
        """
        Normalize transcript for the ASR model's tokenizer.
        Keeps it simple to match common ASR preprocessing.
        """
        text = transcript.lower()
        # Remove special characters except apostrophes and spaces
        text = re.sub(r"[^a-z' ]", " ", text)
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()

    def _convert_alignment_to_timestamped_text(
        self, alignment_result: List[Dict[str, Any]], original_text: str
    ) -> str:
        """
        Convert alignment results to timestamped text format.

        Args:
            alignment_result: List of word segments with timing information
            original_text: Original text without timestamps

        Returns:
            Text with timestamp tokens in the format <|start_frame|>word<|end_frame|>
        """
        timestamped_words = []

        for word_seg in alignment_result:
            word = word_seg["word"]
            start_frame = int(word_seg["start"] / self.frame_length)
            end_frame = int(word_seg["end"] / self.frame_length)
            timestamped_words.append(f"<|{start_frame}|> {word} <|{end_frame}|>")

        return " ".join(timestamped_words)

    def _strip_timestamps(self, text: str) -> str:
        """
        Strip timestamp tokens from text.

        Args:
            text: Text that may contain timestamp tokens

        Returns:
            Text with timestamp tokens removed
        """
        text = re.sub(r'<\|[0-9]+\|>', '', text)
        text = re.sub(r' +', ' ', text)

        return text.strip()
