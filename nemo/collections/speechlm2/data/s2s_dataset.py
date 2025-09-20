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

import torch
import torch.utils.data
import torchaudio

from lhotse import CutSet, MonoCut, Recording, Seconds, SupervisionSegment, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
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

    Notes:
        - The dataset ensures frame-level alignment between audio and text by inserting tokens at
          specific frame positions based on the timing of supervision segments.
        - PAD tokens (typically 0) are used to fill gaps where there's no text.
        - BOS tokens mark the beginning of each speech segment.
        - EOS tokens mark the end of each speech segment.
        - Text tokens from each speaker are placed at frame positions corresponding to their
          timestamp in the original recording, preserving the temporal relationship.
          This is a segment-level alignment only, not word-level alignment.
    """

    def __init__(
        self,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        source_sample_rate: int,
        target_sample_rate: int,
        input_roles: list[str] = None,
        output_roles: list[str] = None,
        aug_by_swap_role: bool = True,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        self.aug_by_swap_role = aug_by_swap_role  # 保存标志
        
        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

    def __getitem__(self, all_cuts: CutSet) -> dict:
        # audio mini-batch
        cuts = all_cuts.filter(lambda c: isinstance(c, Cut))
        audio_data = None
        
        if cuts:
            cuts = cuts.transform_text(_strip_timestamps)
            

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
                
            source_audio, source_audio_lens = collate_audio(all_cuts_combined.resample(self.source_sample_rate))
            target_audio, target_audio_lens = collate_audio(
                all_cuts_combined.resample(self.target_sample_rate), recording_field="target_audio"
            )
            target_tokens, target_token_lens = collate_token_channel(
                all_cuts_combined, self.tokenizer, self.frame_length, roles=self.output_roles
            )
            source_tokens, source_token_lens = collate_token_channel(
                all_cuts_combined, self.tokenizer, self.frame_length, roles=self.input_roles
            )
            
            try:
                target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
                    all_cuts_combined.resample(self.target_sample_rate), roles=self.output_roles, recording_field="target_audio"
                )
            except Exception as e:
                target_first_turn_audio = None
                target_first_turn_audio_lens = None


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
                "target_texts": [
                    " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) 
                    for cut in all_cuts_combined
                ],
                "target_first_turn_audio": target_first_turn_audio,
                "target_first_turn_audio_lens": target_first_turn_audio_lens,
                "formatter": [getattr(cut, "formatter", "s2s_duplex") for cut in all_cuts_combined],
            }

        
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
        """创建单个角色交换的cut"""
        from lhotse import AudioSource
        from io import BytesIO
        import soundfile as sf
        import numpy as np
        
        # 3. 复制并完全交换supervisions
        swapped_supervisions = []
        for sup in cut.supervisions:
            if sup.speaker == 'User':
                new_speaker = 'Assistant'
            elif sup.speaker == 'Assistant':
                new_speaker = 'User'
            else:
                continue  # 跳过其他speaker
                
            swapped_sup = SupervisionSegment(
                id=sup.id + "_swapped",
                recording_id=sup.recording_id,
                start=sup.start,
                duration=sup.duration,
                channel=sup.channel,
                text=sup.text,
                language=sup.language,
                speaker=new_speaker,  # 交换角色
                gender=sup.gender,
                custom=sup.custom,  # 完全复制custom
                alignment=sup.alignment
            )
            swapped_supervisions.append(swapped_sup)
        
        # 4. 丢掉第一轮Agent和最后一轮User
        swapped_supervisions = sorted(swapped_supervisions, key=lambda s: s.start)
        
        # 找到第一轮Agent和最后一轮User
        first_agent_idx = None
        last_user_idx = None
        
        for i, sup in enumerate(swapped_supervisions):
            if sup.speaker == 'Assistant' and first_agent_idx is None:
                first_agent_idx = i
            if sup.speaker == 'User':
                last_user_idx = i
                
        # 移除第一轮Agent和最后一轮User
        filtered_supervisions = []
        for i, sup in enumerate(swapped_supervisions):
            if i != first_agent_idx and i != last_user_idx:
                filtered_supervisions.append(sup)
                
        if not filtered_supervisions:
            return None
            
        # 5. 计算时间偏移
        first_remaining_start = filtered_supervisions[0].start
        last_remaining_end = max(s.start + s.duration for s in filtered_supervisions)
        new_duration = last_remaining_end - first_remaining_start
        
        # 调整所有supervision的时间偏移
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
        
        # 6. 🔧 关键修复：根据调整后的supervisions重新构建音频
        
        # 计算新音频的总长度
        total_duration = max(s.start + s.duration for s in adjusted_supervisions)
        total_samples = int(total_duration * cut.sampling_rate)
        
        # 创建空音频轨道
        new_source_audio = np.zeros(total_samples, dtype=np.float32)
        new_target_audio = np.zeros(total_samples, dtype=np.float32)
        
        # 🔧 关键：只为adjusted_supervisions中的时间段填充音频
        for sup in adjusted_supervisions:
            start_sample = int(sup.start * cut.sampling_rate)
            end_sample = int((sup.start + sup.duration) * cut.sampling_rate)
            
            # 根据角色确定音频来源并填充到正确轨道
            if sup.speaker == 'User':  # 这是原Agent变成的User
                # 从原Agent音频提取，放到source轨道
                # 需要从原始时间戳提取（加上offset）
                original_start = sup.start + first_remaining_start  # 回到原始时间戳
                agent_audio = cut.custom['target_audio'].to_cut().truncate(
                    offset=original_start,
                    duration=sup.duration
                ).load_audio()
                if len(agent_audio.shape) > 1:
                    agent_audio = agent_audio.squeeze()
                actual_end = min(end_sample, start_sample + len(agent_audio))
                new_source_audio[start_sample:actual_end] = agent_audio[:actual_end-start_sample]
                
            elif sup.speaker == 'Assistant':  # 这是原User变成的Agent
                # 从原User音频提取，放到target轨道
                original_start = sup.start + first_remaining_start  # 回到原始时间戳
                user_audio = cut.recording.to_cut().truncate(
                    offset=original_start,
                    duration=sup.duration
                ).load_audio()
                if len(user_audio.shape) > 1:
                    user_audio = user_audio.squeeze()
                actual_end = min(end_sample, start_sample + len(user_audio))
                new_target_audio[start_sample:actual_end] = user_audio[:actual_end-start_sample]

        # 创建新的Recording对象
        # 为source audio创建Recording
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
        
        # 为target audio创建Recording
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

        # 创建新cut
        swapped_cut = MonoCut(
            id=f"{cut.id}_swapped",
            start=0,
            duration=total_duration,
            channel=0,
            supervisions=adjusted_supervisions,
            recording=new_source_recording,  # 重新构建的source音频
            custom={
                **cut.custom,
                'total_turns': len(adjusted_supervisions),
                'role_swapped': True,
                'target_audio': new_target_recording,  # 重新构建的target音频
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
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        truncated_audio = cut.truncate(offset=max(0, first_supervision.start), duration=first_supervision.duration).load_custom(recording_field)
        first_turn_audios.append(truncated_audio.squeeze(0))
        first_turn_audios_lens.append(truncated_audio.shape[-1])

    return collate_vectors(first_turn_audios, padding_value=0), torch.tensor(first_turn_audios_lens)


def collate_token_channel(
    cuts: CutSet,
    tokenizer: TokenizerSpec,
    frame_length: Seconds,
    roles: set[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    pad_id = get_pad_id(tokenizer)
    tokens = [
        build_token_channel(c, tokenizer=tokenizer, frame_length=frame_length, roles=roles, pad_id=pad_id)
        for c in cuts
    ]
    token_lens = torch.tensor([len(tt) for tt in tokens])
    tokens = collate_vectors(tokens, padding_value=pad_id)
    return tokens, token_lens


def build_token_channel(
        cut: Cut,
        tokenizer: TokenizerSpec,
        frame_length: Seconds,
        roles: set[str],
        pad_id: int = -1,
) -> torch.Tensor:
    diagnostic = f"Extra info: {cut.id=}"
    if getattr(cut, "shard_origin", None) is not None:
        diagnostic = f"{diagnostic} {cut.shard_origin=}"

    total = compute_num_frames(cut.duration, frame_length, cut.sampling_rate)
    tokens = torch.ones(total, dtype=torch.long) * pad_id

    for supervision in cut.supervisions:
        if supervision.speaker in roles:
            text_ids = torch.as_tensor([tokenizer.bos] + tokenizer.text_to_ids(supervision.text))

            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos >= len(tokens):  # Changed from > to >= for robustness
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than or equal to the example's length {len(tokens)}. {diagnostic}"
                )
                continue


            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)


            available_frames_for_text = eospos - pos


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

            if eospos < len(tokens):
                tokens[eospos] = tokenizer.eos

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
