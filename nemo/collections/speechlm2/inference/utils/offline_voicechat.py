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

"""Offline Nemotron VoiceChat inference helpers (HF checkpoint layout)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Type
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F
import torchaudio

TARGET_SR = 22050
SOURCE_SR = 16000


def output_stem_from_wav(wav_path: str | os.PathLike) -> str:
    """Basename of the input wav without extension, used for output file prefixes."""
    return Path(wav_path).stem


def output_paths_from_wav(wav_path: str | os.PathLike, output_dir: str | os.PathLike) -> dict[str, Path]:
    """Derive output file paths from the input wav name (e.g. sample.wav -> sample_output.wav)."""
    stem = output_stem_from_wav(wav_path)
    out = Path(output_dir)
    return {
        "text": out / f"{stem}_output.txt",
        "output": out / f"{stem}_output.wav",
        "combined": out / f"{stem}_combined.wav",
        "fc_json": out / f"{stem}_fc.json",
    }


def _load_file(mod_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def import_nemotron_voicechat(code_dir: str) -> Type:
    """Load NemotronVoiceChat from a Speech source tree without heavy __init__ chains."""
    code_dir = code_dir.rstrip("/")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    for pkg_name, pkg_path in [
        ("nemo.collections.speechlm2", f"{code_dir}/nemo/collections/speechlm2"),
        ("nemo.collections.speechlm2.models", f"{code_dir}/nemo/collections/speechlm2/models"),
        ("nemo.collections.speechlm2.modules", f"{code_dir}/nemo/collections/speechlm2/modules"),
    ]:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [pkg_path]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

    _load_file(
        "nemo.collections.speechlm2.data.utils",
        f"{code_dir}/nemo/collections/speechlm2/data/utils.py",
    )
    _load_file(
        "nemo.collections.speechlm2.parts.precision",
        f"{code_dir}/nemo/collections/speechlm2/parts/precision.py",
    )
    perception = _load_file(
        "nemo.collections.speechlm2.modules.perception",
        f"{code_dir}/nemo/collections/speechlm2/modules/perception.py",
    )
    speech_gen = _load_file(
        "nemo.collections.speechlm2.modules.speech_generation",
        f"{code_dir}/nemo/collections/speechlm2/modules/speech_generation.py",
    )
    modules_pkg = sys.modules["nemo.collections.speechlm2.modules"]
    modules_pkg.AudioPerceptionModule = perception.AudioPerceptionModule
    modules_pkg.TransformerARSpeechDecoder = speech_gen.TransformerARSpeechDecoder

    _load_file(
        "nemo.collections.speechlm2.parts.hf_hub",
        f"{code_dir}/nemo/collections/speechlm2/parts/hf_hub.py",
    )
    _load_file(
        "nemo.collections.speechlm2.parts.pretrained",
        f"{code_dir}/nemo/collections/speechlm2/parts/pretrained.py",
    )
    _load_file(
        "nemo.collections.speechlm2.parts.optim_setup",
        f"{code_dir}/nemo/collections/speechlm2/parts/optim_setup.py",
    )

    for mod_name in [
        "nemo.collections.speechlm2.parts.metrics",
        "nemo.collections.speechlm2.parts.metrics.asr_bleu",
        "nemo.collections.speechlm2.parts.metrics.asr_cer_wer",
        "nemo.collections.speechlm2.parts.metrics.bleu",
        "nemo.collections.speechlm2.parts.metrics.empty_text",
        "nemo.collections.speechlm2.parts.metrics.results_logger",
        "nemo.collections.speechlm2.parts.metrics.secs",
        "nemo.collections.speechlm2.parts.metrics.text_wer",
        "nemo.collections.speechlm2.parts.metrics.token_accuracy",
        "nemo.collections.speechlm2.parts.lora",
    ]:
        sys.modules[mod_name] = MagicMock()

    _load_file(
        "nemo.collections.speechlm2.models.duplex_s2s_model",
        f"{code_dir}/nemo/collections/speechlm2/models/duplex_s2s_model.py",
    )
    _load_file(
        "nemo.collections.speechlm2.models.duplex_stt_model",
        f"{code_dir}/nemo/collections/speechlm2/models/duplex_stt_model.py",
    )
    _load_file(
        "nemo.collections.speechlm2.models.duplex_ear_tts",
        f"{code_dir}/nemo/collections/speechlm2/models/duplex_ear_tts.py",
    )
    nemotron_mod = _load_file(
        "nemo.collections.speechlm2.models.nemotron_voicechat",
        f"{code_dir}/nemo/collections/speechlm2/models/nemotron_voicechat.py",
    )
    return nemotron_mod.NemotronVoiceChat


def load_hf_config(ckpt_dir: str, speaker_name: str = "Aria") -> dict[str, Any]:
    """Load config.json from an HF-format checkpoint and apply inference overrides."""
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)

    cfg["model"]["pretrained_s2s_model"] = ckpt_dir
    cfg["model"]["stt"]["model"]["pretrained_weights"] = False
    cfg["model"]["inference_speaker_name"] = speaker_name
    cas_cfg = cfg["model"]["speech_generation"]["model"]["tts_config"]["cas_config"]
    cas_cfg.pop("pretrained_tokenizer_name", None)
    return cfg


def build_model(ckpt_dir: str, code_dir: str, device: str | torch.device = "cuda"):
    """Build NemotronVoiceChat from an HF checkpoint directory."""
    NemotronVoiceChat = import_nemotron_voicechat(code_dir)
    model = NemotronVoiceChat(load_hf_config(ckpt_dir))
    return model.to(device).eval()


def load_wav_16k_mono(wav_path: str, device: str | torch.device = "cuda"):
    """Load a wav file as 16 kHz mono; return (wav_1d, input_signal, input_signal_lens)."""
    wav, sr = torchaudio.load(wav_path)
    if sr != SOURCE_SR:
        wav = torchaudio.functional.resample(wav, sr, SOURCE_SR)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)
    input_signal = wav.unsqueeze(0).to(device)
    input_signal_lens = torch.tensor([wav.shape[0]], device=device)
    return wav, input_signal, input_signal_lens


def encode_system_prompt(model, system_prompt: str, device: str | torch.device = "cuda"):
    """Tokenize a system prompt; returns (prompt_tokens, prompt_token_lens) or (None, None)."""
    if not system_prompt.strip():
        return None, None

    tokenizer = model.stt_model.tokenizer
    prompt_ids = [tokenizer.bos_id] + tokenizer.text_to_ids(system_prompt) + [tokenizer.eos_id]
    prompt_tokens = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_token_lens = torch.tensor([len(prompt_ids)], dtype=torch.long, device=device)
    return prompt_tokens, prompt_token_lens


@torch.no_grad()
def run_offline_inference(
    model,
    input_signal: torch.Tensor,
    input_signal_lens: torch.Tensor,
    prompt_tokens: torch.Tensor | None = None,
    prompt_token_lens: torch.Tensor | None = None,
    decode_audio: bool = True,
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    **kwargs,
) -> dict[str, Any]:
    """Run model.offline_inference with common defaults."""
    return model.offline_inference(
        input_signal=input_signal,
        input_signal_lens=input_signal_lens,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_token_lens,
        decode_audio=decode_audio,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        **kwargs,
    )


def save_offline_outputs(
    result: dict[str, Any],
    wav_1d: torch.Tensor,
    output_dir: str | os.PathLike,
    wav_path: str | os.PathLike,
    target_sr: int = TARGET_SR,
) -> dict[str, str]:
    """Write text, agent mono wav, and combined stereo (user left / agent right) outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    out_paths = output_paths_from_wav(wav_path, output_dir)

    text = result.get("text", [""])[0]
    out_paths["text"].write_text(text)
    paths["text"] = str(out_paths["text"])

    if result.get("audio") is not None:
        agent_len = result["audio_len"][0].item()
        agent_audio = result["audio"][0, :agent_len].cpu().unsqueeze(0)
        torchaudio.save(str(out_paths["output"]), agent_audio, target_sr)
        paths["output"] = str(out_paths["output"])

        user_audio = torchaudio.functional.resample(wav_1d.unsqueeze(0), SOURCE_SR, target_sr)
        t_len = max(user_audio.shape[1], agent_audio.shape[1])
        combined = torch.cat(
            [
                F.pad(user_audio, (0, t_len - user_audio.shape[1])),
                F.pad(agent_audio, (0, t_len - agent_audio.shape[1])),
            ],
            dim=0,
        )
        torchaudio.save(str(out_paths["combined"]), combined, target_sr)
        paths["combined"] = str(out_paths["combined"])

    return paths


SAMPLES_PER_FRAME = int(0.08 * TARGET_SR)


def render_fc_system_prompt(template_path: str, system_message: str, tools: list[dict]) -> str:
    from jinja2 import Environment

    with open(template_path) as f:
        template = Environment().from_string(f.read())
    return template.render(system_message=system_message, tools=tools)


def run_fc_offline_inference(
    model,
    input_signal: torch.Tensor,
    input_signal_lens: torch.Tensor,
    prompt_tokens: torch.Tensor,
    prompt_token_lens: torch.Tensor,
    api_response: dict[str, Any],
    device: str | torch.device = "cuda",
) -> tuple[dict[str, Any], dict[str, Any], int, int, list[int]]:
    """
    Two-pass FC inference: detect tool call, inject API response, decode audio.

    Returns:
        result, fc_output, call_step, response_step, resp_token_ids
    """
    tokenizer = model.stt_model.tokenizer

    result1 = run_offline_inference(
        model,
        input_signal=input_signal,
        input_signal_lens=input_signal_lens,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_token_lens,
        decode_audio=False,
    )

    fc_output: dict[str, Any] = {"tool_calls": [], "tool_responses": []}
    call_step = -1
    call_tokens_ids: list[int] = []
    call_end_pos = 0

    func_tokens = result1.get("tokens_function_pred", result1.get("tokens_function", None))
    if func_tokens is not None:
        tokens_len = result1.get("tokens_len")
        tokens_text = result1.get("tokens_text")
        positions = model.stt_model._extract_function_call_positions(
            func_tokens, tokens_len, tokens_text
        )
        for b_info in positions:
            for call in b_info.get("function_calls", []):
                call_step = call["start_pos"]
                call_end_pos = call["end_pos"]
                call_text = call["call_text"]
                print(f"TOOL CALL at step {call_step}: {call_text}")

                clean = call_text.replace("<SPECIAL_20>", "").replace("<SPECIAL_21>", "").strip()
                if "<TOOLCALL>" in clean:
                    clean = clean.split("<TOOLCALL>")[1].split("</TOOLCALL>")[0].strip()

                try:
                    calls = json.loads(clean) if clean.startswith("[") else [json.loads(clean)]
                    for tc in calls:
                        name = tc.get("name", "")
                        args = tc.get("arguments", {})
                        if isinstance(args, str):
                            args = json.loads(args) if args.strip().startswith("{") else {}
                        print(f"  parsed tool: {name}  arguments: {args}")
                        fc_output["tool_calls"].append(
                            {"name": name, "arguments": args, "step": call_step, "raw": call_text}
                        )
                        call_tokens_ids = tokenizer.text_to_ids(clean)
                except Exception as exc:
                    print(f"Warning: parse error: {exc}")
                break

    resp_payload = json.dumps(api_response["response"])
    resp_str = f"<TOOL_RESPONSE>[{resp_payload}]</TOOL_RESPONSE>"
    resp_token_ids = tokenizer.text_to_ids(resp_str)
    response_step = call_end_pos + 1 if call_step >= 0 else 0
    print(
        f"TOOL RESPONSE at step {response_step} "
        f"({len(resp_token_ids)} tokens): {resp_str[:120]}{'...' if len(resp_str) > 120 else ''}"
    )

    fc_output["tool_responses"].append(
        {
            "name": api_response["tool_name"],
            "response": api_response["response"],
            "formatted_response": resp_str,
            "response_step": response_step,
        }
    )

    if call_step >= 0 and call_tokens_ids:
        print(f"=== Pass 2: inference with injected tool call + response ===")
        fc_calls = torch.tensor(call_tokens_ids, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
        fc_call_lens = torch.tensor([[len(call_tokens_ids)]], dtype=torch.long, device=device)
        fc_call_steps = torch.tensor([[call_step]], dtype=torch.long, device=device)

        fc_resps = torch.tensor(resp_token_ids, dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
        fc_resp_lens = torch.tensor([[len(resp_token_ids)]], dtype=torch.long, device=device)
        fc_resp_steps = torch.tensor([[response_step]], dtype=torch.long, device=device)

        result = run_offline_inference(
            model,
            input_signal=input_signal,
            input_signal_lens=input_signal_lens,
            prompt_tokens=prompt_tokens,
            prompt_token_lens=prompt_token_lens,
            decode_audio=True,
            function_calls=fc_calls,
            function_call_lengths=fc_call_lens,
            function_call_steps=fc_call_steps,
            function_responses=fc_resps,
            function_response_lengths=fc_resp_lens,
            function_response_steps=fc_resp_steps,
        )
    else:
        print("No function call detected in pass 1 — running audio decode without FC injection")
        result = run_offline_inference(
            model,
            input_signal=input_signal,
            input_signal_lens=input_signal_lens,
            prompt_tokens=prompt_tokens,
            prompt_token_lens=prompt_token_lens,
            decode_audio=True,
        )

    return result, fc_output, call_step, response_step, resp_token_ids


def splice_fc_audio_gap(
    agent_audio: torch.Tensor,
    call_step: int,
    response_step: int,
    resp_token_len: int,
    target_sr: int = TARGET_SR,
) -> torch.Tensor:
    """Remove tool-call gap from agent audio (SOTC through EOTR)."""
    if call_step < 0:
        return agent_audio

    gap_start_step = call_step
    gap_end_step = response_step + resp_token_len + 1
    gap_start_sample = gap_start_step * SAMPLES_PER_FRAME
    gap_end_sample = min(gap_end_step * SAMPLES_PER_FRAME, agent_audio.shape[1])

    if gap_start_sample < agent_audio.shape[1] and gap_start_sample < gap_end_sample:
        return torch.cat(
            [agent_audio[:, :gap_start_sample], agent_audio[:, gap_end_sample:]],
            dim=1,
        )
    return agent_audio


def save_fc_offline_outputs(
    result: dict[str, Any],
    wav_1d: torch.Tensor,
    output_dir: str | os.PathLike,
    fc_output: dict[str, Any],
    system_prompt: str,
    wav_path: str,
    sample_id: str = "sample_fc",
    call_step: int = -1,
    response_step: int = 0,
    resp_token_len: int = 0,
    target_sr: int = TARGET_SR,
) -> dict[str, str]:
    """Write FC JSON plus text/audio outputs (with tool-call gap removed from agent audio)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    out_paths = output_paths_from_wav(wav_path, output_dir)

    text = result.get("text", [""])[0]
    out_paths["text"].write_text(text)
    paths["text"] = str(out_paths["text"])

    fc_record = {
        "id": sample_id,
        "audio_file": wav_path,
        "predicted_agent_text": text,
        "system_prompt": system_prompt[:200] + "...",
        "function_calls": fc_output["tool_calls"],
        "tool_responses": fc_output["tool_responses"],
    }
    out_paths["fc_json"].write_text(json.dumps(fc_record, indent=2, ensure_ascii=False))
    paths["fc_json"] = str(out_paths["fc_json"])

    if result.get("audio") is not None:
        agent_len = result["audio_len"][0].item()
        agent_audio = result["audio"][0, :agent_len].cpu().unsqueeze(0)
        agent_audio = splice_fc_audio_gap(
            agent_audio, call_step, response_step, resp_token_len, target_sr=target_sr
        )

        torchaudio.save(str(out_paths["output"]), agent_audio, target_sr)
        paths["output"] = str(out_paths["output"])

        user_audio = torchaudio.functional.resample(wav_1d.unsqueeze(0), SOURCE_SR, target_sr)
        t_len = max(user_audio.shape[1], agent_audio.shape[1])
        combined = torch.cat(
            [
                F.pad(user_audio, (0, t_len - user_audio.shape[1])),
                F.pad(agent_audio, (0, t_len - agent_audio.shape[1])),
            ],
            dim=0,
        )
        torchaudio.save(str(out_paths["combined"]), combined, target_sr)
        paths["combined"] = str(out_paths["combined"])

    return paths
