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

from lhotse import CutSet, Seconds, compute_num_frames
from lhotse.cut import Cut
from lhotse.dataset.collation import collate_audio, collate_vectors
from lhotse.utils import ifnone

from nemo.collections.common.tokenizers import TokenizerSpec
from nemo.collections.speechlm2.data.utils import get_pad_id
from nemo.utils import logging


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
        prompt_audio_path: str = None,
    ):
        self.tokenizer = tokenizer
        self.frame_length = frame_length
        self.source_sample_rate = source_sample_rate
        self.target_sample_rate = target_sample_rate
        self.input_roles = set(ifnone(input_roles, ["user"]))
        self.output_roles = set(ifnone(output_roles, ["agent"]))
        
        assert tokenizer.bos is not None, "BOS support in the tokenizer is required for S2S models."
        assert tokenizer.eos is not None, "EOS support in the tokenizer is required for S2S models."

        if prompt_audio_path is not None:
            audio, sr = torchaudio.load(prompt_audio_path)
            self.prompt_audio = torchaudio.functional.resample(audio, sr, self.source_sample_rate)


    def __getitem__tts_repeat_after_me(self, cuts) -> dict[str, torch.Tensor | list[str] | dict]:

        # adapted from https://github.com/blisc/NeMo/blob/magpietts_2503/nemo/collections/tts/data/text_to_speech_dataset_lhotse.py
        cuts = cuts.sort_by_duration()
        source_audios = []
        source_audios_lens = []
        target_audios = []
        target_audio_lens = []
        target_first_turn_audio = []
        target_first_turn_audio_lens = []
        target_tokens = []
        target_token_lens = []
        source_tokens = []
        source_token_lens = []

        pad_id = get_pad_id(self.tokenizer)
        for i, cut in enumerate(cuts):
            # load target/answer audio
            cur_target_audio = torch.FloatTensor(cut.target_audio.resample(self.target_sample_rate).load_audio())
            # convert answer to the input sr
            cur_target_audio_input_sr = torchaudio.functional.resample(cur_target_audio, self.target_sample_rate, self.source_sample_rate)

            # define silences between turns
            turn_silence_sec = 0.32
            silence_padding_input = torch.zeros((1, int(turn_silence_sec * self.source_sample_rate))) # input sr silence
            silence_padding_output = torch.zeros((1, int(turn_silence_sec * self.target_sample_rate))) # output sr silence

            # generate target audio with end 32 ms padding
            cur_target_audio_with_padding = torch.cat([cur_target_audio, silence_padding_output], dim=1)
            # generates user audio with prompt and padding
            source_audio_with_prompt_and_padding = torch.cat([self.prompt_audio, silence_padding_input, cur_target_audio_input_sr, silence_padding_input], dim=1)

            # downsample it to the input/output sr
            cur_target_audio_with_padding_input_sr = torchaudio.functional.resample(cur_target_audio_with_padding, self.target_sample_rate, self.source_sample_rate)
            source_audio_with_prompt_and_padding_output_sr = torchaudio.functional.resample(source_audio_with_prompt_and_padding, self.source_sample_rate, self.target_sample_rate)

            # make the final source and target audios
            source_audio = torch.cat([source_audio_with_prompt_and_padding, torch.zeros_like(cur_target_audio_with_padding_input_sr)], dim=1)
            target_audio = torch.cat([torch.zeros_like(source_audio_with_prompt_and_padding_output_sr), cur_target_audio_with_padding], dim=1)

            # add source and target audios to the lists
            source_audios.append(source_audio)
            source_audios_lens.append(torch.tensor(source_audio.shape[1]).long())
            target_audios.append(target_audio)
            target_audio_lens.append(torch.tensor(target_audio.shape[1]).long())

            target_first_turn_audio.append(cur_target_audio)
            target_first_turn_audio_lens.append(torch.tensor(cur_target_audio.shape[1]).long())

            total_steps = compute_num_frames(
                duration=(target_audio.size(1) / self.target_sample_rate),
                frame_shift=self.frame_length,
                sampling_rate=self.target_sample_rate
            )

            # target text_start_step is the size of source_audio_with_prompt_and_padding
            text_start_step = compute_num_frames(
                duration=(source_audio_with_prompt_and_padding.size(1) / self.target_sample_rate),
                frame_shift=self.frame_length,
                sampling_rate=self.target_sample_rate
            ) - 1

            text_end_step = total_steps - 1

            cur_target_tokens = torch.full(
                [total_steps],
                pad_id,
            )
            # create emptly sorce text
            cur_source_tokens = torch.full(
                [total_steps],
                pad_id,
            )

            if cut.supervisions[1].speaker == "agent":

                target_text = torch.as_tensor([self.tokenizer.bos] + self.tokenizer.text_to_ids(cut.supervisions[1].text))
                source_text = torch.as_tensor([self.tokenizer.bos] + self.tokenizer.text_to_ids("Can you repeat after me? " + cut.supervisions[1].text))

                # added source and target text
                text_len = min(text_end_step - text_start_step, target_text.shape[0])
                cur_target_tokens[text_start_step : (text_start_step + text_len)] = target_text[:text_len]
                cur_target_tokens[text_end_step] = self.tokenizer.eos

                # for source text, the bos happens in the position and eos happens right before target text_start_step
                source_text_start_step = 0
                source_text_end_step = text_start_step - 1
        
                cur_source_tokens[source_text_end_step] = self.tokenizer.eos
                text_len = min(source_text_end_step - source_text_start_step, source_text.shape[0])
                cur_source_tokens[source_text_start_step : (source_text_start_step + text_len)] = source_text[:text_len]

                target_tokens.append(cur_target_tokens)
                target_token_lens.append(cur_target_tokens.size(0))
                source_tokens.append(cur_source_tokens)
                source_token_lens.append(cur_source_tokens.size(0))


        # collate_vectors target_audios
        target_audios = collate_vectors([a.squeeze(0) for a in target_audios], padding_value=0.0)
        target_audio_lens = torch.tensor(target_audio_lens).long()

        # collate_vectors source_audios
        source_audio = collate_vectors([a.squeeze(0) for a in source_audios], padding_value=0.0)
        source_audio_lens = torch.tensor(source_audios_lens).long()

        # prepare target_first_turn_audio that will be used for speaker conditioning
        target_first_turn_audio = collate_vectors([a.squeeze(0) for a in target_first_turn_audio], padding_value=0.0)
        target_first_turn_audio_lens = torch.tensor(target_first_turn_audio_lens).long()

        # collate_vectors text tokens
        target_tokens = collate_vectors(target_tokens, padding_value=pad_id)
        target_token_lens = torch.tensor(target_token_lens).long()
        source_tokens = collate_vectors(source_tokens, padding_value=pad_id)
        source_token_lens = torch.tensor(source_token_lens).long()

        return {
            "sample_id": ["-".join(s.id for s in cut.supervisions if s.speaker in ["user"]) for cut in cuts],
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_tokens": target_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "target_texts": [
                " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts
            ],
            "target_first_turn_audio": target_first_turn_audio,
            "target_first_turn_audio_lens": target_first_turn_audio_lens,
        }

        return return_batch

    def __getitem__(self, cuts: CutSet) -> dict:
        cuts = cuts.transform_text(_strip_timestamps)

        if getattr(cuts[0], "tts_repeat_after_me", False):
            return self.__getitem__tts_repeat_after_me(cuts)

        source_audio, source_audio_lens = collate_audio(cuts.resample(self.source_sample_rate))
        target_audio, target_audio_lens = collate_audio(
            cuts.resample(self.target_sample_rate), recording_field="target_audio"
        )
        target_tokens, target_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.output_roles
        )
        source_tokens, source_token_lens = collate_token_channel(
            cuts, self.tokenizer, self.frame_length, roles=self.input_roles
        )
        # extract target speaker first turn audio to uses for speaker conditioning
        target_first_turn_audio, target_first_turn_audio_lens = collate_first_turn_audio(
            cuts.resample(self.target_sample_rate), roles=self.output_roles, recording_field="target_audio"
        )

        return {
            "sample_id": ["-".join(s.id for s in cut.supervisions if s.speaker in ["user"]) for cut in cuts],
            "source_audio": source_audio,
            "source_audio_lens": source_audio_lens,
            "target_audio": target_audio,
            "target_audio_lens": target_audio_lens,
            "target_tokens": target_tokens,
            "target_token_lens": target_token_lens,
            "source_tokens": source_tokens,
            "source_token_lens": source_token_lens,
            "target_texts": [
                " ".join(s.text for s in cut.supervisions if s.speaker in self.output_roles) for cut in cuts
            ],
            "target_first_turn_audio": target_first_turn_audio,
            "target_first_turn_audio_lens": target_first_turn_audio_lens,
        }

def collate_first_turn_audio(
    cuts: CutSet,
    roles: set[str],
    recording_field: str = "target_audio",
) -> tuple[torch.Tensor, torch.Tensor]:
    first_turn_audios = []
    first_turn_audios_lens = []
    for cut in cuts:
        first_supervision = [s for s in cut.supervisions if s.speaker in roles][0]
        truncated_audio = cut.truncate(offset=first_supervision.start, duration=first_supervision.duration).load_custom(recording_field)
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

            # Determine the frame offset for the start of the supervision to insert the text tokens.
            pos = compute_num_frames(supervision.start, frame_length, cut.sampling_rate)
            if pos > len(tokens):
                logging.warning(
                    f"Ill-constructed example: the beginning offset of a supervision {pos} is larger than the example's length {len(tokens)}. {diagnostic}"
                )
                continue

            # Determine the frame offset for the last non-EOS text token to form a valid range for insertion;
            # Note that EOS will be placed possibly much later, at the frame that coincides with end of speech,
            # rather than end of text. The gap between last non-EOS token and EOS token will be filled with `pad_id`.
            endpos = pos + len(text_ids)
            if endpos > len(tokens):
                trunc_len = len(tokens) - pos
                logging.warning(
                    f"Truncating training example's text_ids of length {len(text_ids)} by {trunc_len} because {endpos=} > {len(tokens)=}. {diagnostic}"
                )
                text_ids = text_ids[:trunc_len]
            try:
                tokens[pos:endpos] = text_ids
            except Exception as e:
                raise RuntimeError(f"{tokens.shape=} {pos=} {endpos=} {text_ids.shape=} {diagnostic}") from e

            # Insert EOS at the end of the supervision segment.
            eospos = compute_num_frames(supervision.end, frame_length, cut.sampling_rate)
            if eospos < len(tokens):  # skip otherwise - unfinished turn
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
