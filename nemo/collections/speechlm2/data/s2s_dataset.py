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
import torch
import torch.utils.data
import torchaudio

from lhotse import CutSet, MonoCut, Recording, Seconds, SupervisionSegment, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.collections.speechlm2.data.force_align import ForceAligner
from nemo.utils import logging
from nemo.collections.common.data.lhotse.text_adapters import Formattable


class DuplexS2SDataset(torch.utils.data.Dataset):
    """
    A dataset for duplex speech-to-speech models that handles bidirectional conversations.

    This dataset processes Lhotse CutSet objects containing recordings with supervision segments
    from different speakers (roles). It creates aligned representations of audio and text for
    both source (input) and target (output) channels, preserving temporal alignment between
    audio frames and text tokens.

    Args:
        tokenizer (TokenizerSpec):
            Tokenizer for converting text to token IDs and vice versa. Must support BOS and EOS tokens.
            It's expected to support PAD token as well, otherwise we will use 0 as the pad token
            and emit a warning.

        frame_length (Seconds):
            Duration of a single frame in seconds. Used to calculate frame positions for token alignment.

        source_sample_rate (int):
            Sample rate for source audio (e.g., 16000 Hz).

        target_sample_rate (int):
            Sample rate for target audio (e.g., 22050 Hz).

        input_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as inputs. Defaults to ["user"].

        output_roles (list[str], optional):
            List of speaker roles (cut.supervisions[:].speaker) to consider as outputs. Defaults to ["agent"].

        force_align_user_text (bool, optional):
            If True, performs force alignment on user audio segments to generate word-level timestamps.
            Only applies to supervision turns where speaker.role is "user". Defaults to False.

    Returns:
        A dictionary with the following keys:
            - source_audio: Tensor of source waveform samples [B, T]
            - source_audio_lens: Tensor of source audio lengths [B]
            - target_audio: Tensor of target waveform samples [B, T]
            - target_audio_lens: Tensor of target audio lengths [B]
            - target_tokens: Tensor of target text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - target_token_lens: Tensor of target token sequence lengths [B]
            - source_tokens: Tensor of source text tokens [B, T], with special tokens (BOS/EOS/PAD)
                at positions aligned with audio frames
            - source_token_lens: Tensor of source token sequence lengths [B]
            - target_texts: List of full target texts joined from output_roles supervisions [B]
            - prompt_tokens: Tensor of prompt text tokens [B, T]
            - prompt_token_lens: Tensor of prompt token sequence lengths [B]
            - target_turn_texts: (Optional, if include_turn_metadata=True) List of lists of turn dictionaries [B]
                Each turn dict contains: start_time, duration, role, text
            - source_turn_texts: (Optional, if include_turn_metadata=True) List of lists of turn dictionaries [B]
                Each turn dict contains: start_time, duration, role, text
            - system_prompt: (Optional, if include_turn_metadata=True) List of system prompts [B]

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
        - When force_align_user_text is enabled, user audio segments are
          force-aligned using wav2vec2 to generate word-level timestamps, which are then
          converted to frame-level token positions for more precise alignment.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        aug_by_swap_role: bool = False,
        include_turn_metadata: bool = False,
        cfg: dict = None,
        model_cfg: dict = None,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.aug_by_swap_role = aug_by_swap_role
        self.include_turn_metadata = include_turn_metadata

        self.word_align_position = cfg.get("word_align_position", "left") if cfg is not None else "left"
        self.predict_user_text = model_cfg.get("predict_user_text", False) if model_cfg is not None else False
        self.force_align_user_text = model_cfg.get("force_align_user_text", False) if model_cfg is not None else None
        # Default to CPU for force alignment to avoid OOM during training/validation when main model is on GPU
        self.force_align_device = model_cfg.get("force_align_device", "cpu") if model_cfg is not None else "cpu"

        self.cfg = cfg
        self.model_cfg = model_cfg
        

        # Initialize force aligner if needed
        self.force_aligner = None
        if self.force_align_user_text:
            self.force_aligner = ForceAligner(device=self.force_align_device, frame_length=self.frame_length)
        
        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

    def _create_minimal_batch(self) -> dict:
        """Create a minimal valid batch when all cuts are filtered out."""
        # Create minimal tensors with batch size 1
        device = torch.device('cpu')  # Default device
        
        return {
            "sample_id": ["empty_batch"],
            "source_audio": torch.zeros((1, 1000), dtype=torch.float32),  # 1 second of silence at 16kHz
            "source_audio_lens": torch.tensor([1000], dtype=torch.long),
            "agent_bos_vad": None,
            "target_audio": torch.zeros((1, 22050), dtype=torch.float32),  # 1 second of silence at 22.05kHz
            "target_audio_lens": torch.tensor([22050], dtype=torch.long),
            "target_tokens": torch.full((1, 50), self.tokenizer.pad_id, dtype=torch.long),
            "target_token_lens": torch.tensor([1], dtype=torch.long),
            "source_tokens": torch.full((1, 50), self.tokenizer.pad_id, dtype=torch.long),
            "source_token_lens": torch.tensor([1], dtype=torch.long),
            "source_texts": [""],
            "target_texts": [""],
            "all_texts": [""],
            "target_first_turn_audio": torch.zeros((1, 22050), dtype=torch.float32),
            "target_first_turn_audio_lens": torch.tensor([22050], dtype=torch.long),
            "formatter": ["s2s_duplex"],
        }

    def __getitem__(self, all_cuts: CutSet) -> dict:
        # audio mini-batch
        cuts = all_cuts.filter(lambda c: isinstance(c, Cut))
        audio_data = None

        if cuts and hasattr(cuts[0], 'formatter') and cuts[0].formatter == 'nemo_tarred_to_duplex':
            filtered_cuts = []
            skipped_cuts = []
            for cut in cuts:
                if any(s.text.strip() for s in cut.supervisions if s.speaker in self.input_roles):
                    filtered_cuts.append(cut)
                else:
                    skipped_cuts.append(cut.id)
            if skipped_cuts:
                logging.info(f"Skipped {len(skipped_cuts)} cuts with empty input text. Skipped cut ids: {', '.join(skipped_cuts)}")
            if not filtered_cuts:
                logging.warning(f"All cuts were filtered out! Original batch size: {len(cuts)}. Returning minimal valid batch to continue training.")
                return self._create_minimal_batch()
            cuts = CutSet.from_cuts(filtered_cuts)

        if cuts:
            swapped_cuts = []

            if self.aug_by_swap_role:
                for cut in cuts:
                    total_turns = cut.custom.get('total_turns', len(cut.supervisions))

                    if total_turns > 4 and total_turns % 2 == 0:
                        swapped_cut = self._create_role_swapped_cut(cut)
                        if swapped_cut:
                            swapped_cuts.append(swapped_cut)

            if swapped_cuts:
                all_cuts_combined = CutSet.from_cuts(list(cuts) + swapped_cuts)
            else:
                all_cuts_combined = cuts
            
            prompt_tokens, prompt_token_lens = collate_system_prompt(
                all_cuts_combined, self.tokenizer
            )
            source_audio, source_audio_lens = collate_audio(all_cuts_combined.resample(self.source_sample_rate))
            target_audio, target_audio_lens = collate_audio(
                all_cuts_combined.resample(self.target_sample_rate), recording_field="target_audio"
            )

            target_tokens, target_token_lens = collate_token_channel(
                all_cuts_combined, self.tokenizer, self.frame_length, roles=self.output_roles, bos_id=self.tokenizer.bos, eos_id=self.tokenizer.eos, remove_timestamps=True
            )

            # Only run force alignment during training (when gradients are enabled)
            if self.force_align_user_text and torch.is_grad_enabled():
                logging.info(f"Force aligning user text for {len(all_cuts_combined)} cuts on device {self.force_align_device}")
                all_cuts_combined = self.force_aligner.batch_force_align_user_audio(all_cuts_combined, source_sample_rate=self.source_sample_rate)
                
                # Check if we have any cuts left after filtering
                if len(all_cuts_combined) == 0:
                    logging.warning("All cuts filtered out due to force alignment failures, returning minimal valid batch to continue training.")
                    return self._create_minimal_batch()

            source_tokens, source_token_lens = collate_token_channel(
                all_cuts_combined, self.tokenizer, self.frame_length,
                roles=self.input_roles,
                bos_id=self.tokenizer.text_to_ids('^')[0], 
                eos_id=self.tokenizer.text_to_ids('$')[0], 
                word_align_position=self.word_align_position, 
                remove_timestamps=not self.predict_user_text, 
                user_bos_id=self.tokenizer.text_to_ids('^')[0], 
                agent_bos_id=self.tokenizer.bos
            )
                
            try:
                target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
                    all_cuts_combined.resample(self.target_sample_rate), roles=self.output_roles,
                    recording_field="target_audio"
                )
            except Exception as e:
                target_first_turn_audio = None
                target_first_turn_audio_lens = None

            if self.model_cfg is not None and self.model_cfg.get("debug", False):
                print("source_tokens[0]:", source_tokens[0][:500]*(source_tokens[0][:500]!=self.tokenizer.pad_id))
                print("target_tokens[0]:", target_tokens[0][:500]*(target_tokens[0][:500]!=self.tokenizer.pad_id))
                print("cut.supervisions[0].duration:", int(cuts[0].supervisions[0].duration / 0.08))
                # Find the indices of the first non-pad tokens in target_tokens[0]
                first_non_pad_idx = (target_tokens[0] != self.tokenizer.pad_id).nonzero(as_tuple=True)[0][0].item() if (target_tokens[0] != self.tokenizer.pad_id).any() else None
                print("First non-pad token index in target_tokens[0]:", first_non_pad_idx)
                # print('Agent start timestamp: ', int(cuts[0].supervisions[1].start / 0.08))
                import pdb; pdb.set_trace()


            audio_data = {
                "sample_id": [str(cut.id) for cut in all_cuts_combined],
                "source_audio": source_audio,
                "source_audio_lens": source_audio_lens,
                "target_audio": target_audio,
                "target_audio_lens": target_audio_lens,
                "target_tokens": target_tokens,
                "target_token_lens": target_token_lens,
                "source_tokens": source_tokens,
                "source_token_lens": source_token_lens,
                "source_texts": [
                    " ".join(_strip_timestamps(s.text) for s in cut.supervisions if s.speaker in self.input_roles) for cut in all_cuts_combined
                ],
                "target_texts": [
                    " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles)
                    for cut in all_cuts_combined
                ],
                "all_texts": [
                    " ".join(_strip_timestamps(s.text) for s in cut.supervisions) for cut in all_cuts_combined
                ],
                "target_first_turn_audio": target_first_turn_audio,
                "target_first_turn_audio_lens": target_first_turn_audio_lens,
                "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in all_cuts_combined],
                "aug_by_noise": [getattr(cut, "aug_by_noise", True) for cut in all_cuts_combined]
            }
        
            if torch.sum(prompt_token_lens) > 0:
                audio_data['prompt_tokens'] = prompt_tokens
                audio_data['prompt_token_lens'] = prompt_token_lens
            
            # Optionally include detailed turn metadata for analysis
            if self.include_turn_metadata:
                audio_data["target_turn_texts"] = [
                    [
                        {
                            "start_time": s.start,
                            "duration": s.duration,
                            "role": s.speaker,
                            "text": s.text,
                        }
                        for s in cut.supervisions if s.speaker in self.output_roles
                    ]
                    for cut in all_cuts_combined
                ]
                audio_data["source_turn_texts"] = [
                    [
                        {
                            "start_time": s.start,
                            "duration": s.duration,
                            "role": s.speaker,
                            "text": s.text,
                        }
                        for s in cut.supervisions if s.speaker in self.input_roles
                    ]
                    for cut in all_cuts_combined
                ]
                audio_data["system_prompt"] = [
                    cut.custom.get('system_prompt', '') for cut in all_cuts_combined
                ]
                
        text_cuts = all_cuts.filter(lambda c: isinstance(c, Formattable))
        text_data = None
        if text_cuts:
            text_tokens = []
            text_token_lens = []
            for c in text_cuts:
                text_ids = c.input_ids
                text_tokens.append(text_ids)
                text_token_lens.append(text_ids.shape[0])

            text_tokens = collate_vectors(
                text_tokens, padding_value=get_pad_id(self.tokenizer)
            )
            text_token_lens = torch.tensor(text_token_lens, dtype=torch.long)
            text_data = {
                "text_tokens": text_tokens,
                "text_token_lens": text_token_lens,
            }

        return {
            "audio_data": audio_data,
            "text_data": text_data,
        }

    def _create_role_swapped_cut(self, cut):

        from lhotse import AudioSource
        from io import BytesIO
        import soundfile as sf
        import numpy as np

        swapped_supervisions = []
        for sup in cut.supervisions:
            if sup.speaker == 'User':
                new_speaker = 'Assistant'
            elif sup.speaker == 'Assistant':
                new_speaker = 'User'
            else:
                continue

            swapped_sup = SupervisionSegment(
                id=sup.id + "_swapped",
                recording_id=sup.recording_id,
                start=sup.start,
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
                language=sup.language,
                speaker=new_speaker,
                gender=sup.gender,
                custom=sup.custom,
                alignment=sup.alignment
            )
            swapped_supervisions.append(swapped_sup)

        swapped_supervisions = sorted(swapped_supervisions, key=lambda s: s.start)

        first_agent_idx = None
        last_user_idx = None

        for i, sup in enumerate(swapped_supervisions):
            if sup.speaker == 'Assistant' and first_agent_idx is None:
                first_agent_idx = i
            if sup.speaker == 'User':
                last_user_idx = i

        filtered_supervisions = []
        for i, sup in enumerate(swapped_supervisions):
            if i != first_agent_idx and i != last_user_idx:
                filtered_supervisions.append(sup)

        if not filtered_supervisions:
            return None

        first_remaining_start = filtered_supervisions[0].start
        last_remaining_end = max(s.start + s.duration for s in filtered_supervisions)
        new_duration = last_remaining_end - first_remaining_start

        adjusted_supervisions = []
        for sup in filtered_supervisions:
            adjusted_sup = SupervisionSegment(
                id=sup.id,
                recording_id=sup.recording_id,
                start=sup.start - first_remaining_start,  # 减去offset
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
                language=sup.language,
                speaker=sup.speaker,
                gender=sup.gender,
                custom=sup.custom,
                alignment=sup.alignment
            )
            adjusted_supervisions.append(adjusted_sup)

        total_duration = max(s.start + s.duration for s in adjusted_supervisions)
        total_samples = int(total_duration * cut.sampling_rate)

        new_source_audio = np.zeros(total_samples, dtype=np.float32)
        new_target_audio = np.zeros(total_samples, dtype=np.float32)

        for sup in adjusted_supervisions:
            start_sample = int(sup.start * cut.sampling_rate)
            end_sample = int((sup.start + sup.duration) * cut.sampling_rate)

            if sup.speaker == 'User':

                original_start = sup.start + first_remaining_start
                agent_audio = cut.custom['target_audio'].to_cut().truncate(
                    offset=original_start,
                    duration=sup.duration
                ).load_audio()
                if len(agent_audio.shape) > 1:
                    agent_audio = agent_audio.squeeze()
                actual_end = min(end_sample, start_sample + len(agent_audio))
                new_source_audio[start_sample:actual_end] = agent_audio[:actual_end - start_sample]

            elif sup.speaker == 'Assistant':
                original_start = sup.start + first_remaining_start
                user_audio = cut.recording.to_cut().truncate(
                    offset=original_start,
                    duration=sup.duration
                ).load_audio()
                if len(user_audio.shape) > 1:
                    user_audio = user_audio.squeeze()
                actual_end = min(end_sample, start_sample + len(user_audio))
                new_target_audio[start_sample:actual_end] = user_audio[:actual_end - start_sample]

        source_buffer = BytesIO()
        sf.write(source_buffer, new_source_audio, cut.sampling_rate, format='wav')
        source_buffer.seek(0)

        new_source_recording = Recording(
            id=f"{cut.id}_swapped_source",
            sampling_rate=cut.sampling_rate,
            num_samples=len(new_source_audio),
            duration=total_duration,
            sources=[AudioSource(
                type="memory",
                channels=[0],
                source=source_buffer.getvalue()
            )]
        )

        target_buffer = BytesIO()
        sf.write(target_buffer, new_target_audio, cut.sampling_rate, format='wav')
        target_buffer.seek(0)

        new_target_recording = Recording(
            id=f"{cut.id}_swapped_target",
            sampling_rate=cut.sampling_rate,
            num_samples=len(new_target_audio),
            duration=total_duration,
            sources=[AudioSource(
                type="memory",
                channels=[0],
                source=target_buffer.getvalue()
            )]
        )

        swapped_cut = MonoCut(
            id=f"{cut.id}_swapped",
            start=0,
            duration=total_duration,
            channel=0,
            supervisions=adjusted_supervisions,
            recording=new_source_recording,
            custom={
                **cut.custom,
                'total_turns': len(adjusted_supervisions),
                'role_swapped': True,
                'target_audio': new_target_recording,
            }
        )

        return swapped_cut


def collate_first_turn_audio(
        cuts: CutSet,
        roles: set[str],
        recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        # Find supervisions that match the specified roles
        matching_supervisions = [s for s in cut.supervisions if s.speaker in roles]
        
        if not matching_supervisions:
            # Log warning and skip this cut if no matching supervisions found
            logging.warning(f"No supervisions found with roles {roles} for cut {cut.id}. Available speakers: {[s.speaker for s in cut.supervisions]}")
            continue
            
        first_supervision = matching_supervisions[0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    if not first_turn_audios:
        # If no valid audio was found, return empty tensors
        logging.error(f"No valid audio found for any cuts with roles {roles}")
        return torch.empty(0), torch.empty(0)

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
    bos_id: int = None,
    eos_id: int = None,
    word_align_position: str = 'left',
    remove_timestamps: bool = False,
    user_bos_id: int = None,
    agent_bos_id: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id, word_align_position=word_align_position, remove_timestamps=remove_timestamps, user_bos_id=user_bos_id, agent_bos_id=agent_bos_id)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def collate_system_prompt(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate system prompts from cuts.
    System prompts should be stored in cut.custom['system_prompt'].
    """
    pad_id = get_pad_id(tokenizer)
    tokens = []
    for c in cuts:
        # Check if system prompt exists in custom field
        if c.custom and c.custom.get("system_prompt", None):
            prompt_text = c.custom["system_prompt"]
            tokens.append(torch.as_tensor(
                [tokenizer.bos] + tokenizer.text_to_ids(prompt_text) + [tokenizer.eos],
                dtype=torch.long
            ))
        else:
            # No system prompt for this cut
            tokens.append(torch.as_tensor([], dtype=torch.long))
    
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens

def build_token_channel(
        cut: Cut,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        roles: set[str],
        pad_id: int = -1,
        bos_id: int = None,
        eos_id: int = None,
        word_align_position: str = 'left',
        remove_timestamps: bool = False,
        user_bos_id: int = None,
        agent_bos_id: int = None,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id
    for supervision in cut.supervisions:
        if supervision.speaker in roles:

            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos >= len(tokens):  # Changed from > to >= for robustness
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than or equal to the example's length {len(tokens)}. {diagnostic}"
                )
                continue
            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            available_frames_for_text = eospos - pos

            text = supervision.text

            # Use different bos_id for user and agent
            text_ids = torch.as_tensor([bos_id] + _text_to_ids(text, tokenizer, available_frames_for_text=available_frames_for_text, word_align_position=word_align_position, remove_timestamps=remove_timestamps))

            if available_frames_for_text > 0 and len(text_ids) > available_frames_for_text:
                # Truncate text_ids to fit before the eos position.
                text_ids = text_ids[:available_frames_for_text]
            elif available_frames_for_text <= 0:
                # If there's no space for text (e.g., start >= end), use an empty sequence.
                text_ids = torch.tensor([], dtype=torch.long)

            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
                endpos = pos + len(text_ids)

            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            # Place EOS token - critical for turn-taking behavior
            if eospos < len(tokens) and eos_id is not None:
                # Normal case: place EOS at the intended position
                tokens[eospos] = eos_id
            else:
                # Interruption case: place EOS at the last valid position
                # This ensures the model learns to stop when interrupted by user
                if endpos < len(tokens):
                    # Case 1: text finished, interrupted during sil/audio generation
                    # Place EOS right after the last text token (or at sequence end if closer)
                    actual_eos_pos = min(endpos, len(tokens) - 1)
                    tokens[actual_eos_pos] = eos_id
                elif len(tokens) > 0:
                    # Case 2: text truncated due to interruption
                    # Place EOS at the very end of the sequence
                    tokens[-1] = eos_id
                logging.warning(
                    f"Supervision was likely interrupted: {eospos=} >= {len(tokens)=}. "
                    f"Placed EOS at fallback position to ensure proper turn-taking training. {diagnostic}"
                )

    return tokens


def _strip_timestamps(
        text: str, _TIMESTAMP_PATTERN=re.compile(r"<\|\d+\|>"), _SPACE_PATTERN=re.compile(r"\s+")
) -> str:
    """
    Strips timestamp tokens from text, e.g. turns:
      '<|0|> Hey <|3|> <|3|> how <|5|> <|7|> are <|8|> <|8|> <|10|> you? <|12|>'
      into:
      'Hey how are you?'
    """
    # Regexp pattern args are cached compiled patterns (micro-optimization).
    text = _TIMESTAMP_PATTERN.sub("", text)  # strip timestamp tokens if present
    return _SPACE_PATTERN.sub(" ", text).strip()  # strip multi-whitespaces

def _text_to_ids(text: str, tokenizer: TokenizerSpec,
                 _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
                 available_frames_for_text=None,
                 word_align_position='left',
                 remove_timestamps=False):
    if not remove_timestamps and re.compile(_TIMESTAMP_PATTERN_STR).search(text):
        text_ids = _text_with_timestamps_to_ids(text, tokenizer, _TIMESTAMP_PATTERN_STR, available_frames_for_text, word_align_position)
    else:
        _TIMESTAMP_PATTERN = re.compile(_TIMESTAMP_PATTERN_STR)
        text = _TIMESTAMP_PATTERN.sub("", text)
        # Remove extra spaces between words
        text = " ".join(text.strip().split())
        text_ids = tokenizer.text_to_ids(text)
    return text_ids


def _text_with_timestamps_to_ids(text: str, tokenizer: TokenizerSpec,
                                 _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>",
                                 available_frames_for_text=None,
                                 word_align_position='left') -> list[int]:
    text_ids = []
    text_ids, start_times, end_times, word_lens = _extract_text_and_time_tokens(text, tokenizer, _TIMESTAMP_PATTERN_STR)
    text_ids_with_timestamps = _expand_text_with_timestamps_and_word_lengths(text_ids, word_lens, start_times, end_times, available_frames_for_text, frame_rate=0.08, pad_id=get_pad_id(tokenizer), word_align_position=word_align_position)
    
    if random.random() < 0.1:
        logging.info(f'text_ids_with_timestamps: {text_ids_with_timestamps}')
        logging.info(f'text_ids: {text_ids}')
        logging.info(f'start_times: {start_times}')
        logging.info(f'end_times: {end_times}')
        logging.info(f'word_lens: {word_lens}')
    return text_ids_with_timestamps


def _extract_text_and_time_tokens(text, tokenizer: TokenizerSpec,
                                 _TIMESTAMP_PATTERN_STR=r"<\|(\d+)\|>"):
    # Find all time tokens
    time_tokens = re.findall(_TIMESTAMP_PATTERN_STR, text)
    start_time = [int(time_tokens[i]) for i in range(0, len(time_tokens), 2)]
    end_time = [int(time_tokens[i]) for i in range(1, len(time_tokens), 2)]
    # Remove all time tokens to isolate words
    words = re.sub(_TIMESTAMP_PATTERN_STR, '', text).split()
    # Process each word, tokenize it, and calculate token lengths
    text_ids = []
    word_lens = []
    for i, word in enumerate(words):
        word_with_space = word if i == 0 else ' ' + word
        word_ids = tokenizer.text_to_ids(word_with_space)
        word_len = len(word_ids)
        text_ids.extend(word_ids)
        word_lens.append(word_len)
    return text_ids, start_time, end_time, word_lens


def _expand_text_with_timestamps_and_word_lengths(
        text_ids, word_lens, start_time, end_time, available_frames_for_text, frame_rate=0.08, pad_id=None, word_align_position='left'
    ):    
    """
    Expand word tokens according to start time tokens and word lengths for a batch of sequences.

    Args:
    - word_tokens: List of text ids w/o timestamps
    - word_lens: List of word lengths
    - start_time: List of start times
    - end_time: List of end times
    - available_frames_for_text: Maximum number of frames for text
    - frame_rate: Frame rate resolution
    - pad_id: Padding ID to use for empty positions in the tensor

    Returns:
    - text ids with word-level timestamps
    """

    def discretize_time(start_token, speech_frame_rate=0.08, timestamp_frame_rate=0.08):
        return int(start_token * timestamp_frame_rate / speech_frame_rate)

    if pad_id is None:
        raise ValueError("pad_id must be provided.")

    max_length = available_frames_for_text

    # Create the empty tensor with pad_id as the default value
    text_ids_with_timestamps = [pad_id] * max_length

    # Populate ids of each word starting at start_idx and ending at end_idx
    cur_word_idx = 0  # Start frame index of current word
    for word_idx, word_len in enumerate(word_lens):
        start_idx = discretize_time(start_time[word_idx], speech_frame_rate=frame_rate)
        end_idx = discretize_time(end_time[word_idx], speech_frame_rate=frame_rate)
        if word_align_position == 'left':
            end_idx = min(start_idx + word_len, end_idx)
        elif word_align_position == 'right':
            start_idx = max(start_idx, end_idx - word_len)
        else:
            raise ValueError(f"Unknown word_align_position: {word_align_position}")

        # Get ids of a single word
        word_ids = text_ids[cur_word_idx : cur_word_idx + word_len]

        # Populate a single word
        for i in range(start_idx, end_idx + 1):  # End inclusive at word level
            if i - start_idx < len(word_ids) and i < max_length:
                token_id = word_ids[i - start_idx]
                text_ids_with_timestamps[i] = token_id

        # Move to the next word in the concatenated word tokens
        cur_word_idx += word_len

    return text_ids_with_timestamps