# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Force-align multi-speaker t-SOT transcripts with PEE and TransformerCTC.

This specialized NeMo Forced Aligner tool consumes audio listed in a JSONL
manifest plus Nemotron-Transcribe t-SOT text (for example, ``<spk:0>`` tags).
It combines PEE Sortformer speaker probabilities with TransformerCTC
log-probabilities and emits file-relative word timestamps, ``.seglst``,
one CTM per session (numeric speaker ID in column two), and one RTTM per session.

Use ``--config`` to load stable model and alignment defaults from YAML. Explicit
CLI options take precedence; matching uppercase environment variables remain a
legacy fallback.
"""

import argparse
import inspect
import json
import math
import os
import sys
from pathlib import Path

import soundfile as sf
import torch
from omegaconf import OmegaConf

from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from nemo.collections.asr.modules.parallel_expert_encoder import (
    PEETransformerCTCTimestampExtractor,
    TransformerCTCDecoder,
)
from nemo.collections.asr.modules.parallel_expert_encoder_resolver import resolve_parallel_expert_encoder_pt
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer


_CONFIG_PATH_FIELDS = frozenset(
    {
        "input_jsonl",
        "output_root",
        "output_jsonl",
        "seglst_output",
        "ctm_output_dir",
        "rttm_output_dir",
        "pee_nemo_path",
        "ctc_head_path",
        "tokenizer_model",
    }
)
_CONFIG_FIELDS = _CONFIG_PATH_FIELDS | frozenset(
    {
        "sot_field",
        "alignment_mode",
        "speaker_assignment_mode",
        "speaker_logprob_weight",
        "max_records",
        "fail_fast",
        "segment_gap_seconds",
        "speaker_label_source",
        "device",
        "model_dtype",
    }
)


def _coerce_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be one of 1/0, true/false, or yes/no.")


def _load_config_defaults(config_path) -> dict:
    if not config_path:
        return {}

    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Timestamp config does not exist or is not a file: {path}")
    raw_config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw_config, dict):
        raise TypeError(f"Timestamp config must contain a top-level mapping: {path}")

    invalid_keys = sorted(str(key) for key in set(raw_config) - _CONFIG_FIELDS)
    if invalid_keys:
        raise ValueError(f"Unsupported timestamp config key(s) in {path}: {', '.join(invalid_keys)}")

    defaults = {key: value for key, value in raw_config.items() if value is not None}
    for key in _CONFIG_PATH_FIELDS:
        value = defaults.get(key)
        if isinstance(value, str) and value:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                defaults[key] = str((path.parent / candidate).resolve())
    return defaults


def _option_default(config_defaults: dict, config_key: str, environment_name: str, fallback=None):
    return os.environ.get(environment_name, config_defaults.get(config_key, fallback))


def configure_environment_from_cli() -> None:
    """Parse CLI/config options and populate the environment-backed runtime configuration."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=os.environ.get("TIMESTAMP_CONFIG"),
        metavar="PATH",
        help="YAML file providing model and alignment defaults.",
    )
    config_args, _ = config_parser.parse_known_args()
    config_defaults = _load_config_defaults(config_args.config)

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description=(
            "Force-align a t-SOT transcript using PEE Sortformer probabilities and "
            "TransformerCTC log-probabilities. Timestamps are relative to each input WAV."
        ),
    )
    path_options = {
        "input-jsonl": ("INPUT_JSONL", "input_jsonl", "PATH", "Input manifest with audio_filepath and t-SOT text."),
        "output-jsonl": ("OUTPUT_JSONL", "output_jsonl", "PATH", "Output word-timestamp JSONL."),
        "seglst-output": ("SEGLST_OUTPUT", "seglst_output", "PATH", "Output segment-list JSON."),
        "ctm-output-dir": ("CTM_OUTPUT_DIR", "ctm_output_dir", "DIR", "Directory for recording-level CTM files."),
        "rttm-output-dir": ("RTTM_OUTPUT_DIR", "rttm_output_dir", "DIR", "Directory for per-session RTTM files."),
        "pee-nemo-path": ("PEE_NEMO_PATH", "pee_nemo_path", "PATH", "PEE .nemo checkpoint."),
        "ctc-head-path": ("CTC_HEAD_PATH", "ctc_head_path", "PATH", "TransformerCTC decoder state dict."),
        "tokenizer-model": ("TOKENIZER_MODEL", "tokenizer_model", "PATH", "SentencePiece tokenizer model."),
    }
    for option, (environment_name, config_key, metavar, help_text) in path_options.items():
        parser.add_argument(
            f"--{option}",
            default=_option_default(config_defaults, config_key, environment_name),
            metavar=metavar,
            help=help_text,
        )
    parser.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_root",
        default=_option_default(config_defaults, "output_root", "TIMESTAMP_OUTPUT_ROOT"),
        metavar="DIR",
        help="Root for words/, segments/, ctm/, and rttm/ output directories.",
    )

    parser.add_argument(
        "--sot-field",
        default=_option_default(config_defaults, "sot_field", "SOT_FIELD", "generation"),
    )
    parser.add_argument(
        "--alignment-mode",
        choices=("serialized", "parallel"),
        default=_option_default(config_defaults, "alignment_mode", "ALIGNMENT_MODE", "serialized"),
    )
    parser.add_argument(
        "--speaker-assignment-mode",
        choices=("optimal", "identity"),
        default=_option_default(
            config_defaults,
            "speaker_assignment_mode",
            "SPEAKER_ASSIGNMENT_MODE",
            "optimal",
        ),
    )
    parser.add_argument(
        "--speaker-logprob-weight",
        type=float,
        default=_option_default(config_defaults, "speaker_logprob_weight", "SPEAKER_LOGPROB_WEIGHT", 0.25),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=_option_default(config_defaults, "max_records", "MAX_RECORDS", 0),
    )
    parser.add_argument(
        "--fail-fast",
        action=argparse.BooleanOptionalAction,
        default=_coerce_bool(_option_default(config_defaults, "fail_fast", "FAIL_FAST", True), "FAIL_FAST"),
        help="Stop after the first failed record (default: enabled).",
    )
    parser.add_argument(
        "--segment-gap-seconds",
        type=float,
        default=_option_default(
            config_defaults,
            "segment_gap_seconds",
            "SEGMENT_GAP_SECONDS",
            os.environ.get("MERGE_GAP_SECONDS", 0.1),
        ),
        help="Merge same-speaker words within this gap; -1 disables merging.",
    )
    parser.add_argument(
        "--speaker-label-source",
        choices=("tag", "global_ids", "tag_map", "auto"),
        default=_option_default(config_defaults, "speaker_label_source", "SPEAKER_LABEL_SOURCE", "tag"),
    )
    parser.add_argument("--device", default=_option_default(config_defaults, "device", "DEVICE", "cuda"))
    parser.add_argument(
        "--model-dtype",
        choices=("bf16", "bfloat16", "fp32", "float32"),
        default=_option_default(config_defaults, "model_dtype", "MODEL_DTYPE", "bf16"),
    )
    args = parser.parse_args()

    if args.output_root:
        if not args.input_jsonl:
            parser.error("--input-jsonl is required when --output-dir is used.")
        input_stem = Path(args.input_jsonl).stem
        output_root = Path(args.output_root).expanduser()
        if not args.output_jsonl:
            args.output_jsonl = str(output_root / "words" / f"{input_stem}.word_timestamps.jsonl")
        if not args.seglst_output:
            args.seglst_output = str(output_root / "segments" / f"{input_stem}.seglst")
        if not args.ctm_output_dir:
            args.ctm_output_dir = str(output_root / "ctm")
        if not args.rttm_output_dir:
            args.rttm_output_dir = str(output_root / "rttm")

    path_values = {
        environment_name: getattr(args, option.replace("-", "_"))
        for option, (environment_name, _, _, _) in path_options.items()
    }
    missing = [
        f"--{option}"
        for option, (environment_name, _, _, _) in path_options.items()
        if not path_values[environment_name]
    ]
    if missing:
        parser.error(f"Missing required option(s): {', '.join(missing)}")

    environment_values = {
        **path_values,
        "TIMESTAMP_OUTPUT_ROOT": args.output_root or "",
        "SOT_FIELD": args.sot_field,
        "ALIGNMENT_MODE": args.alignment_mode,
        "SPEAKER_ASSIGNMENT_MODE": args.speaker_assignment_mode,
        "SPEAKER_LOGPROB_WEIGHT": args.speaker_logprob_weight,
        "MAX_RECORDS": args.max_records,
        "FAIL_FAST": "1" if args.fail_fast else "0",
        "SEGMENT_GAP_SECONDS": args.segment_gap_seconds,
        "SPEAKER_LABEL_SOURCE": args.speaker_label_source,
        "DEVICE": args.device,
        "MODEL_DTYPE": args.model_dtype,
    }
    os.environ.update({name: str(value) for name, value in environment_values.items()})


def environment_path(name: str) -> Path:
    return Path(os.environ[name]).expanduser()


def resolve_audio_path(raw_path: str, manifest_path: Path) -> Path:
    """Resolve the WAV named by a manifest record's ``audio_filepath``."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("audio_filepath must be a non-empty string.")
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Audio file listed in audio_filepath is missing: {candidate}")
    return candidate


def _format_time(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Timestamp must be finite, got {value!r}.")
    if value == 0.0:
        return "0"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _coerce_speaker_tag(value):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("speaker_tag must be an integer or null, not a boolean.")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"speaker_tag must be an integer or null, got {value!r}.") from error


def _session_id(record: dict) -> str:
    value = record.get("audio_filepath")
    if isinstance(value, str) and value.strip():
        return Path(value).stem
    value = record.get("sample_id")
    if value is not None and str(value).strip():
        return str(value).strip()
    raise ValueError("Cannot derive session_id: provide audio_filepath or sample_id.")


def _speaker_label(record: dict, speaker_tag, source: str) -> str:
    if speaker_tag is None:
        return "spk:unknown"
    default = f"spk:{speaker_tag}"
    if source in {"global_ids", "auto"}:
        global_ids = record.get("global_speaker_ids")
        if isinstance(global_ids, list) and 0 <= speaker_tag < len(global_ids):
            value = global_ids[speaker_tag]
            if value is not None and str(value).strip():
                return str(value).strip()
    if source in {"tag_map", "auto"}:
        tag_map = record.get("speaker_tag_map")
        if isinstance(tag_map, dict):
            for key in (f"s{speaker_tag}", default, str(speaker_tag), speaker_tag):
                value = tag_map.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
    return default


def _segment_from_words(record: dict, words: list, speaker_tag, speaker_label_source: str, record_index: int) -> dict:
    start = words[0]["start"]
    end = max(word["end"] for word in words)
    columns = {word["sortformer_column"] for word in words}
    if len(columns) == 1:
        sortformer_column = columns.pop()
    else:
        sortformer_column = None
    return {
        "end_time": _format_time(end),
        "session_id": _session_id(record),
        "speaker": _speaker_label(record, speaker_tag, speaker_label_source),
        "start_time": _format_time(start),
        "words": " ".join(word["word"] for word in words),
        "speaker_tag": speaker_tag,
        "sortformer_column": sortformer_column,
        "sample_id": record.get("sample_id"),
        "record_index": record_index,
        "audio_filepath": record["audio_filepath"],
        "num_words": len(words),
    }


def build_seglst_segments(
    record: dict,
    speaker_word_timestamps: dict,
    *,
    merge_gap_seconds: float,
    speaker_label_source: str,
    record_index: int,
) -> list:
    grouped_words = {}
    for timestamp_key, word_rows in speaker_word_timestamps.items():
        if not isinstance(word_rows, list):
            raise TypeError("speaker_word_timestamps values must be lists.")
        for row_index, word_row in enumerate(word_rows):
            if not isinstance(word_row, dict):
                raise TypeError("Each speaker word timestamp must be a dictionary.")
            speaker_tag = _coerce_speaker_tag(word_row.get("speaker_tag", timestamp_key))
            word = str(word_row.get("word", "")).strip()
            if not word:
                raise ValueError("Cannot create a segment from an empty aligned word.")
            try:
                start = float(word_row["start"])
                end = float(word_row["end"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid aligned word timestamp for {word!r}.") from error
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise ValueError(f"Invalid aligned interval [{start!r}, {end!r}] for {word!r}.")
            grouped_words.setdefault(speaker_tag, []).append(
                {
                    "word": word,
                    "start": start,
                    "end": end,
                    "word_index": int(word_row.get("word_index", row_index)),
                    "sortformer_column": word_row.get("sortformer_column"),
                }
            )

    segments = []
    for speaker_tag, words in grouped_words.items():
        words.sort(key=lambda row: (row["start"], row["end"], row["word_index"]))
        current_segment = []
        current_end = None
        for word in words:
            if (
                current_segment
                and (merge_gap_seconds < 0.0 or word["start"] - current_end > merge_gap_seconds)
            ):
                segments.append(
                    _segment_from_words(record, current_segment, speaker_tag, speaker_label_source, record_index)
                )
                current_segment = []
            current_segment.append(word)
            current_end = word["end"] if current_end is None else max(current_end, word["end"])
        if current_segment:
            segments.append(
                _segment_from_words(record, current_segment, speaker_tag, speaker_label_source, record_index)
            )
    return segments


def _time_interval(start_value, end_value, description: str) -> tuple[float, float]:
    start = float(start_value)
    end = float(end_value)
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        raise ValueError(f"Invalid interval [{start_value!r}, {end_value!r}] for {description}.")
    return start, end


def build_ctm_lines(session_id: str, speaker_word_timestamps: dict) -> list[str]:
    """Build one CTM stream for a recording with a numeric speaker ID in column two."""
    if not session_id or len(session_id.split()) != 1:
        raise ValueError(f"CTM session ID must be a single non-empty token, got {session_id!r}.")
    if not isinstance(speaker_word_timestamps, dict):
        raise TypeError("speaker_word_timestamps must be a dictionary.")

    words = []
    for timestamp_key, word_rows in speaker_word_timestamps.items():
        if not isinstance(word_rows, list):
            raise TypeError("CTM speaker word rows must be lists.")
        for row_index, word_row in enumerate(word_rows):
            if not isinstance(word_row, dict):
                raise TypeError("Each speaker word timestamp must be a dictionary.")
            speaker_tag = _coerce_speaker_tag(word_row.get("speaker_tag", timestamp_key))
            if speaker_tag is None or speaker_tag < 0:
                raise ValueError("CTM output requires a non-negative integer speaker tag.")
            word = str(word_row.get("word", "")).strip()
            if len(word.split()) != 1:
                raise ValueError(f"CTM words must be one token, got {word!r}.")
            start, end = _time_interval(word_row["start"], word_row["end"], f"CTM word {word!r}")
            words.append((start, end, speaker_tag, int(word_row.get("word_index", row_index)), word))

    words.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return [
        f"{session_id} {speaker_tag} {start:08.2f} {end - start:.2f} {word} -1.00"
        for start, end, speaker_tag, _, word in words
    ]


def write_ctm_file(ctm_output_dir: Path, session_id: str, speaker_word_timestamps: dict) -> Path:
    """Write all speakers for one recording to a single session-named CTM file."""
    ctm_path = ctm_output_dir / f"{session_id}.ctm"
    write_timing_file(ctm_path, build_ctm_lines(session_id, speaker_word_timestamps))
    return ctm_path


def build_rttm_lines(segments: list[dict]) -> list[str]:
    entries = []
    for segment in segments:
        start, end = _time_interval(segment["start_time"], segment["end_time"], "RTTM segment")
        speaker = str(segment["speaker"]).strip()
        if len(speaker.split()) != 1:
            raise ValueError(f"RTTM speaker must be one token, got {speaker!r}.")
        entries.append((start, end, speaker))

    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return [
        f"SPEAKER <NA> <NA> {start:.2f} {end - start:.2f} <NA> <NA> {speaker} <NA> <NA>"
        for start, end, speaker in entries
    ]


def write_timing_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        for line in lines:
            output_file.write(line + "\n")
    os.replace(temporary_path, path)


def load_runtime(device: torch.device, dtype: torch.dtype):
    pee_path = environment_path("PEE_NEMO_PATH")
    head_path = environment_path("CTC_HEAD_PATH")
    tokenizer_path = environment_path("TOKENIZER_MODEL")

    pee_cls = resolve_parallel_expert_encoder_pt(str(pee_path))
    pee = pee_cls.load_from_nemo(str(pee_path), map_location="cpu", strict=True)
    pee = pee.to(device=device, dtype=dtype).eval()

    head_blob = torch.load(head_path, map_location="cpu", weights_only=False)
    accepted = set(inspect.signature(TransformerCTCDecoder.__init__).parameters)
    decoder_cfg = {key: value for key, value in head_blob["decoder_config"].items() if key in accepted}
    head = TransformerCTCDecoder(**decoder_cfg)
    head.load_state_dict(head_blob["state_dict"], strict=True)
    head = head.to(device=device, dtype=dtype).eval()

    preprocessor = AudioToMelSpectrogramPreprocessor(
        sample_rate=16000,
        normalize=None,  # PEE applies per-feature normalization internally.
        window_size=0.025,
        window_stride=0.01,
        window="hann",
        features=128,
        n_fft=512,
        log=True,
        frame_splicing=1,
        dither=0.0,
        pad_to=0,
        pad_value=0.0,
    ).to(device).eval()
    tokenizer = SentencePieceTokenizer(str(tokenizer_path))
    extractor = PEETransformerCTCTimestampExtractor(
        encoder=pee,
        ctc_decoder=head,
        tokenizer=tokenizer,
        alignment_mode=os.environ["ALIGNMENT_MODE"],
        speaker_assignment_mode=os.environ["SPEAKER_ASSIGNMENT_MODE"],
        speaker_logprob_weight=float(os.environ["SPEAKER_LOGPROB_WEIGHT"]),
    )
    return preprocessor, extractor


def main() -> int:
    input_path = environment_path("INPUT_JSONL")
    output_path = environment_path("OUTPUT_JSONL")
    seglst_path = environment_path("SEGLST_OUTPUT")
    ctm_output_dir = environment_path("CTM_OUTPUT_DIR")
    rttm_output_dir = environment_path("RTTM_OUTPUT_DIR")
    if output_path.resolve() == seglst_path.resolve():
        raise ValueError("OUTPUT_JSONL and SEGLST_OUTPUT must be different files.")
    if input_path.resolve() in {output_path.resolve(), seglst_path.resolve()}:
        raise ValueError("INPUT_JSONL must be different from OUTPUT_JSONL and SEGLST_OUTPUT.")

    required_file_paths = {
        "INPUT_JSONL": input_path,
        "PEE_NEMO_PATH": environment_path("PEE_NEMO_PATH"),
        "CTC_HEAD_PATH": environment_path("CTC_HEAD_PATH"),
        "TOKENIZER_MODEL": environment_path("TOKENIZER_MODEL"),
    }
    for environment_name, required_path in required_file_paths.items():
        if not required_path.is_file():
            raise FileNotFoundError(f"{environment_name} does not exist or is not a file: {required_path}")
    for output_directory in (output_path.parent, seglst_path.parent, ctm_output_dir, rttm_output_dir):
        output_directory.mkdir(parents=True, exist_ok=True)

    sot_field = os.environ["SOT_FIELD"]
    max_records = int(os.environ["MAX_RECORDS"])
    fail_fast = os.environ["FAIL_FAST"].strip().lower() not in {"0", "false", "no"}
    try:
        segment_gap_seconds = float(os.environ["SEGMENT_GAP_SECONDS"])
    except ValueError as error:
        raise ValueError("SEGMENT_GAP_SECONDS must be a number of seconds.") from error
    if not math.isfinite(segment_gap_seconds) or (segment_gap_seconds < 0.0 and segment_gap_seconds != -1.0):
        raise ValueError("SEGMENT_GAP_SECONDS must be non-negative, or exactly -1 to disable merging.")
    speaker_label_source = os.environ["SPEAKER_LABEL_SOURCE"].strip().lower()
    if speaker_label_source not in {"tag", "global_ids", "tag_map", "auto"}:
        raise ValueError("SPEAKER_LABEL_SOURCE must be tag, global_ids, tag_map, or auto.")
    requested_device = os.environ["DEVICE"].strip()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE requests CUDA, but torch.cuda.is_available() is false.")
    device = torch.device(requested_device)
    dtype_name = os.environ["MODEL_DTYPE"].strip().lower()
    if device.type == "cuda" and dtype_name in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif dtype_name in {"fp32", "float32"} or device.type != "cuda":
        dtype = torch.float32
    else:
        raise ValueError("MODEL_DTYPE must be bf16 or fp32; fp32 is required for CPU.")

    print(f"Loading PEE and CTC head on {device} with {dtype}.", flush=True)
    preprocessor, extractor = load_runtime(device, dtype)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    seglst_temporary_path = seglst_path.with_name(seglst_path.name + ".tmp")
    processed = 0
    failures = 0
    seglst_segments = []
    written_session_ids = set()

    with input_path.open(encoding="utf-8") as input_file, temporary_path.open("w", encoding="utf-8") as output_file:
        for record_index, line in enumerate(input_file):
            if max_records > 0 and processed >= max_records:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                sot_transcript = record[sot_field]
                if not isinstance(sot_transcript, str):
                    raise TypeError(f"{sot_field} must be a string, got {type(sot_transcript).__name__}.")
                audio_path = resolve_audio_path(record["audio_filepath"], input_path)
                record = dict(record)
                record["audio_filepath"] = str(audio_path)
                audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
                if sample_rate != 16000:
                    raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz for {audio_path}.")
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
                waveform_length = torch.tensor([waveform.shape[-1]], dtype=torch.long, device=device)
                duration = waveform.shape[-1] / float(sample_rate)
                result = extractor.extract_from_audio(
                    input_signal=waveform,
                    input_signal_length=waveform_length,
                    preprocessor=preprocessor,
                    sot_transcript=sot_transcript,
                    audio_duration=duration,
                    time_offset=0.0,
                )
                segments = build_seglst_segments(
                    record,
                    result["speaker_word_timestamps"],
                    merge_gap_seconds=segment_gap_seconds,
                    speaker_label_source=speaker_label_source,
                    record_index=record_index,
                )
                session_id = _session_id(record)
                if session_id in written_session_ids:
                    raise ValueError(f"Duplicate session_id would overwrite timing files: {session_id!r}.")
                ctm_path = write_ctm_file(ctm_output_dir, session_id, result["speaker_word_timestamps"])
                rttm_path = rttm_output_dir / f"{session_id}.rttm"
                write_timing_file(rttm_path, build_rttm_lines(segments))
                written_session_ids.add(session_id)
                output_record = {
                    "record_index": record_index,
                    "sample_id": record.get("sample_id"),
                    "session_id": session_id,
                    "audio_filepath": record["audio_filepath"],
                    "timestamp_reference": "chunk_relative",
                    "duration": duration,
                    "ctm_filepath": str(ctm_path),
                    "rttm_filepath": str(rttm_path),
                    "speaker_word_timestamps": result["speaker_word_timestamps"],
                    "speaker_tag_to_sortformer_column": result["speaker_tag_to_sortformer_column"],
                    "alignment_mode": result["alignment_mode"],
                    "speaker_assignment_mode": result["speaker_assignment_mode"],
                    "ctc_frame_seconds": result["ctc_frame_seconds"],
                    "sortformer_frame_seconds": result["sortformer_frame_seconds"],
                    "num_ctc_frames": result["num_ctc_frames"],
                    "num_sortformer_frames": result["num_sortformer_frames"],
                    "ctc_log_normalizer_error": result["ctc_log_normalizer_error"],
                    "alignment_diagnostics": result["alignment_diagnostics"],
                }
                output_file.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                output_file.flush()
                seglst_segments.extend(segments)
                processed += 1
                print(f"[{processed}] {record.get('sample_id', audio_path.name)}", flush=True)
            except Exception as error:
                failures += 1
                print(f"Failed record {record_index}: {error}", file=sys.stderr, flush=True)
                if fail_fast:
                    raise

    seglst_segments.sort(
        key=lambda segment: (
            segment["session_id"],
            float(segment["start_time"]),
            float(segment["end_time"]),
            segment["speaker"],
        )
    )
    with seglst_temporary_path.open("w", encoding="utf-8") as seglst_file:
        json.dump(seglst_segments, seglst_file, ensure_ascii=False, indent=2)
        seglst_file.write("\n")
    os.replace(temporary_path, output_path)
    os.replace(seglst_temporary_path, seglst_path)
    print(
        f"Wrote {processed} record(s) to {output_path}; "
        f"{len(seglst_segments)} segment(s) to {seglst_path}; failures={failures}.",
        flush=True,
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    configure_environment_from_cli()
    raise SystemExit(main())

