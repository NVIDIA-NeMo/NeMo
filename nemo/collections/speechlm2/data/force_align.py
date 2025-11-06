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

import re
import random
import logging
import multiprocessing as mp
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torchaudio

from lhotse import CutSet, MonoCut, Seconds, SupervisionSegment


class ForceAligner:
    """
    Force alignment utility using wav2vec2-based models for speech-to-text alignment.
    """
    
    def __init__(self, device: str = None, frame_length: float = 0.02):
        """
        Initialize the ForceAligner.
        
        Args:
            device: Device to run alignment on (default: auto-detect)
            frame_length: Frame length in seconds for timestamp conversion
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.frame_length = frame_length
        
        # Initialize wav2vec2 model components (lazy initialization)
        self.wav2vec2_model = None
        self.wav2vec2_tokenizer = None
        self.wav2vec2_aligner = None
        self.wav2vec2_bundle = None
        self._model_loaded = False
    
    def _load_wav2vec2_model(self):
        """Load the wav2vec2 model and related components."""
        try:
            # Check if we're in a forked subprocess with CUDA
            # If using DataLoader with num_workers > 0, you need to set multiprocessing start method to 'spawn'
            # in your main training script before creating the DataLoader:
            #   import multiprocessing as mp
            #   mp.set_start_method('spawn', force=True)
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
            logging.info(f"Loading wav2vec2 model for force alignment on device {device}")

            # Load wav2vec2 model and bundle (using MMS_FA for multilingual support)
            from torchaudio.pipelines import MMS_FA as bundle
            self.wav2vec2_bundle = bundle
            
            # Try to load on the specified device, fall back to CPU on OOM
            try:
                self.wav2vec2_model = bundle.get_model().to(device)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logging.warning(
                        f"CUDA OOM when loading wav2vec2 model on {device}. "
                        f"Falling back to CPU for force alignment."
                    )
                    device = torch.device('cpu')
                    self.device = 'cpu'
                    # Clear CUDA cache and load on CPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.wav2vec2_model = bundle.get_model().to(device)
                else:
                    raise
            
            self.wav2vec2_tokenizer = bundle.get_tokenizer()
            self.wav2vec2_aligner = bundle.get_aligner()
            
            # Set model to evaluation mode
            self.wav2vec2_model.eval()
            
            logging.info("Wav2vec2 model loaded successfully for force alignment")
        except Exception as e:
            logging.error(f"Failed to load wav2vec2 model for force alignment: {e}")
            self.wav2vec2_model = None
    
    def batch_force_align_user_audio(self, cuts: CutSet, source_sample_rate: int = 16000) -> CutSet:
        """
        Perform batch force alignment on all user audio segments.
        
        Args:
            cuts: CutSet containing all cuts to process
            source_sample_rate: Source sample rate of the audio
            
        Returns:
            CutSet containing only cuts where force alignment succeeded
        """
        # Lazy load the model on first use (for multiprocessing compatibility)
        if not self._model_loaded:
            self._load_wav2vec2_model()
            self._model_loaded = True
        
        if self.wav2vec2_model is None:
            logging.warning("Wav2vec2 model not available for force alignment, returning empty cutset")
            return CutSet.from_cuts([])
        
        # Collect all user supervisions and their corresponding cuts
        user_supervisions = []
        user_cuts = []
        cut_to_supervisions = {}  # Map cut to its user supervisions for tracking
        
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
        
        # Prepare audio tensors and texts for batch processing
        audio_tensors = []
        texts = []
        
        for i, (supervision, cut) in enumerate(zip(user_supervisions, user_cuts)):
            # Extract user audio segment
            start_time = supervision.start
            duration = supervision.duration
            
            # Truncate the cut to get only the user segment
            user_cut = cut.truncate(offset=start_time, duration=duration)
            
            # Load audio and resample to model's expected sample rate
            audio = user_cut.load_audio()
            if audio.shape[0] > 1:  # Convert to mono if stereo
                audio = audio.mean(dim=0, keepdim=True)
            
            # Convert numpy array to torch tensor if needed
            if isinstance(audio, np.ndarray):
                audio = torch.from_numpy(audio)
            
            # Resample to wav2vec2's expected sample rate (16kHz)
            target_sample_rate = 16000
            if source_sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=source_sample_rate,
                    new_freq=target_sample_rate
                )
                audio = resampler(audio)
            
            # Add 0.64 seconds of trailing silence
            silence_duration = 0.64  # seconds
            silence_samples = int(silence_duration * target_sample_rate)
            silence = torch.zeros(1, silence_samples)
            audio = torch.cat([audio, silence], dim=1)
            
            # Store audio tensor and text for batch processing
            audio_tensors.append(audio)
            texts.append(self._strip_timestamps(supervision.text))
        
        # Use wav2vec2-based force alignment with in-memory audio tensors
        alignments_batch = self._wav2vec2_batch_align_tensors(audio_tensors, texts)
        
        # Process each alignment result and update supervision texts
        success_count = 0
        failed_count = 0
        for i, alignment_result in enumerate(alignments_batch):
            if alignment_result is not None:
                # Convert alignment to timestamped text
                original_text = user_supervisions[i].text
                timestamped_text = self._convert_wav2vec2_alignment_to_timestamped_text(alignment_result, original_text)
                if random.random() < 0.1:
                    print(f'original_text: {original_text}')
                    print(f'timestamped_text: {timestamped_text}')
                # Update the supervision text with force-aligned timestamps
                user_supervisions[i].text = timestamped_text
                success_count += 1
            else:
                # Alignment failed, keep the original text as-is
                failed_count += 1
        
        if failed_count > 0:
            logging.warning(
                f"Force alignment failed for {failed_count}/{len(user_supervisions)} user segments. "
                f"Keeping all cuts with original text for failed alignments."
            )
        else:
            logging.info(f"Force alignment succeeded for all {success_count} user segments.")
        
        return cuts
    
    def _wav2vec2_batch_align_tensors(self, audio_tensors: List[torch.Tensor], texts: List[str]) -> List[Optional[List[Dict[str, Any]]]]:
        """
        Perform batch force alignment using wav2vec2 with in-memory audio tensors.
        
        Args:
            audio_tensors: List of audio waveform tensors
            texts: List of text transcripts corresponding to each audio tensor
            
        Returns:
            List of alignment results for each audio tensor
        """
        alignments = []
        
        for idx, (audio_tensor, text) in enumerate(zip(audio_tensors, texts)):
            try:
                # Perform alignment directly with the audio tensor
                alignment_result = self._wav2vec2_align(
                    audio_tensor, 
                    16000,  # wav2vec2 expects 16kHz
                    text
                )
                alignments.append(alignment_result)
                
                # Clear CUDA cache periodically to avoid memory fragmentation
                if self.device == 'cuda' and torch.cuda.is_available() and (idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logging.error(f"Failed to align audio tensor: {e}")
                alignments.append(None)
                
                # Clear cache on error as well
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return alignments
    
    def _wav2vec2_align(self, waveform: torch.Tensor, sample_rate: int, transcript: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform forced alignment using wav2vec2.
        
        Args:
            waveform: Audio waveform tensor
            sample_rate: Sample rate of the audio
            transcript: Text transcript
            
        Returns:
            List of word segments with timing information
        """
        # Normalize transcript (following the documentation approach)
        normalized_transcript = self._normalize_transcript(transcript)
        
        # Split transcript into words for word-level alignment
        transcript_words = normalized_transcript.split()
        
        if not transcript_words:
            logging.warning(f"No valid words found in transcript: {transcript}")
            return None
        
        # Resample if needed (wav2vec2 expects 16kHz)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        
        # Tokenize transcript early to calculate target length before model forward pass
        tokens = self.wav2vec2_tokenizer(transcript_words)
        target_length = sum(len(token_list) for token_list in tokens)
        
        # Early check: Estimate emission length to avoid expensive model forward pass
        # wav2vec2/MMS_FA typically downsamples by a factor of 320 (20ms frames at 16kHz)
        DOWNSAMPLE_FACTOR = 320
        expected_emission_length = waveform.size(1) // DOWNSAMPLE_FACTOR
        
        if expected_emission_length < target_length:
            logging.warning(
                f"Audio too short for CTC alignment (pre-check). "
                f"Expected emission length: {expected_emission_length}, target length: {target_length}, "
                f"audio samples: {waveform.size(1)}, transcript: '{transcript[:100]}...'. Skipping alignment."
            )
            return None
        
        # Move to device
        device = torch.device(self.device)
        waveform = waveform.to(device)
        
        # Get emission from wav2vec2 model
        try:
            with torch.no_grad():
                emission, _ = self.wav2vec2_model(waveform)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logging.warning(
                    f"CUDA OOM during force alignment inference. "
                    f"Falling back to CPU for this sample. "
                    f"Consider setting device='cpu' for ForceAligner initialization."
                )
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Retry on CPU
                device = torch.device('cpu')
                waveform = waveform.to(device)
                
                # Move model to CPU if needed
                if next(self.wav2vec2_model.parameters()).device.type == 'cuda':
                    self.wav2vec2_model = self.wav2vec2_model.to(device)
                    self.device = 'cpu'
                    logging.info("Moved wav2vec2 model to CPU for subsequent alignments")
                
                with torch.no_grad():
                    emission, _ = self.wav2vec2_model(waveform)
            else:
                raise
        
        emission_length = emission.size(1)
        
        # Validate that we have enough frames for CTC alignment (safety check)
        # CTC requires at least as many frames as target tokens
        if emission_length < target_length:
            logging.warning(
                f"Audio too short for CTC alignment (actual check). "
                f"Emission length: {emission_length}, target length: {target_length}, "
                f"transcript: '{transcript[:100]}...'. Skipping alignment."
            )
            return None
        
        # Perform forced alignment using the bundle's aligner
        try:
            token_spans = self.wav2vec2_aligner(emission[0], tokens)
        except RuntimeError as e:
            if "targets length is too long for CTC" in str(e):
                logging.warning(
                    f"CTC alignment failed due to length mismatch: {e}. "
                    f"Emission length: {emission_length}, target length: {target_length}. "
                    f"Skipping alignment."
                )
                return None
            else:
                # Re-raise other RuntimeErrors
                raise
        
        if not token_spans:
            logging.warning(f"No alignment found for transcript: {transcript}")
            return None
        
        # Convert token spans to word segments
        word_segments = []
        ratio = waveform.size(1) / emission.size(1) / 16000  # Convert frames to seconds
        
        for word, spans in zip(transcript_words, token_spans):
            if spans:
                start_time = spans[0].start * ratio
                end_time = spans[-1].end * ratio
                # Calculate average score weighted by span length
                avg_score = sum(span.score * len(span) for span in spans) / sum(len(span) for span in spans)
                
                word_segments.append({
                    'word': word,
                    'start': start_time,
                    'end': end_time,
                    'score': avg_score
                })
        
        return word_segments
    
    def _normalize_transcript(self, transcript: str) -> str:
        """
        Normalize transcript following the documentation approach.
        This removes punctuation and converts to lowercase for the MMS_FA tokenizer.
        """
        # Convert to lowercase
        text = transcript.lower()
        
        # Replace apostrophes
        text = text.replace("'", "'")
        
        # Remove non-alphabetic characters except apostrophes and spaces
        text = re.sub(r"[^a-z' ]", " ", text)
        
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _convert_wav2vec2_alignment_to_timestamped_text(self, alignment_result: List[Dict[str, Any]], original_text: str) -> str:
        """
        Convert wav2vec2 alignment results to timestamped text format.
        
        Args:
            alignment_result: List of word segments with timing information
            original_text: Original text without timestamps
            
        Returns:
            Text with timestamp tokens in the format <|start_frame|>word<|end_frame|>
        """
        timestamped_words = []
        
        for word_seg in alignment_result:
            # Use the word from the alignment result as it represents what was actually aligned
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
        # Remove timestamp tokens in the format <|frame_number|>
        text = re.sub(r'<\|[0-9]+\|>', '', text)
        
        # Clean up extra spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
