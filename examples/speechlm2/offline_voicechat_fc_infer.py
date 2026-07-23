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

"""
Offline Nemotron VoiceChat inference with function calling (two-pass).

Usage:
    python offline_voicechat_fc_infer.py \\
        --checkpoint /path/to/hf_checkpoint \\
        --wav /path/to/input.wav \\
        --api-response-json /path/to/random_number_response.json \\
        --output-dir /path/to/output \\
        --code-dir /path/to/Speech
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

DEFAULT_TEMPLATE = os.path.join(os.path.dirname(__file__), "function_calling", "template.jinja")

DEFAULT_SYSTEM_MESSAGE = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Your job is to be helpful and harmless and have engaging conversations in English. "
    "Maintain a warm and friendly tone. "
    "Keep the dialogue open and ongoing. "
    "Be clear and direct, especially when answering yes or no questions and multiple-choice questions. "
    "Avoid long answers unless the user asks you to provide details or context. "
    "You must provide diverse responses and rephrase answers if the user asks the same question. "
    "DO NOT interrupt the user when they are speaking, let them finish their turn before answering."
    "\n\nWhen you receive a request, follow this decision process:\n"
    "1. Does the request match one of your available tools below? If yes, you MUST call that tool - "
    "never answer it directly from your own knowledge, even if you think you know the answer.\n"
    "2. Is it a general knowledge question (history, science, geography, math, facts, etc.)? "
    "If yes, answer directly from your own knowledge - do not call any tool.\n"
    "3. Does it require an external action or live data that none of your tools cover "
    "(e.g. ordering food, sending email)? If yes, politely say you don't have that capability."
    "\n\nNEVER say \"I don't have a tool for that\" for general knowledge questions you can answer yourself."
    "\n\nDO NOT use any tools when not needed to answer the user's requests, under no circumstance."
    "\n\nYou are an expert across history, geography, science, math, literature, biographies, languages, "
    "recipes, programming, current affairs, and general knowledge. When the user asks about any of these, "
    "answer directly and conversationally from your own knowledge — no <TOOLCALL>."
    "\n\nCall a tool ONLY when the user's request matches one of the tools listed in <AVAILABLE_TOOLS> below. "
    "For every other request, do not call any tool — just answer from your knowledge. "
    "Never invent or call a tool name that is not literally in <AVAILABLE_TOOLS>."
    "\n\nTool-call arguments must be values the user spoke. "
    "If a required argument is missing, ask the user; never guess."
)

DEFAULT_TOOLS = [
    {
        "function": {
            "name": "generate_random_number",
            "description": "Generate a random integer between min and max (inclusive).",
            "parameters": {
                "type": "object",
                "properties": {
                    "min": {"type": "integer", "description": "Minimum value (inclusive)"},
                    "max": {"type": "integer", "description": "Maximum value (inclusive)"},
                },
                "required": ["min", "max"],
            },
        }
    }
]


def _load_offline_voicechat_module(code_dir: str):
    module_path = os.path.join(
        code_dir,
        "nemo/collections/speechlm2/inference/utils/offline_voicechat.py",
    )
    spec = importlib.util.spec_from_file_location("offline_voicechat", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["offline_voicechat"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Nemotron VoiceChat FC inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help="Jinja chat template (defaults to function_calling/template.jinja)",
    )
    parser.add_argument("--api-response-json", required=True, help="Pre-built tool response JSON")
    parser.add_argument("--tools-json", default=None, help="Tool definitions JSON (defaults to generate_random_number)")
    parser.add_argument("--system-message", default=DEFAULT_SYSTEM_MESSAGE)
    parser.add_argument("--sample-id", default="sample_fc")
    parser.add_argument("--code-dir", default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_dir = args.code_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    ov = _load_offline_voicechat_module(code_dir)

    tools = DEFAULT_TOOLS
    if args.tools_json:
        with open(args.tools_json) as f:
            tools = json.load(f)

    with open(args.api_response_json) as f:
        api_response = json.load(f)

    print(f"Checkpoint : {args.checkpoint}")
    print(f"WAV        : {args.wav}")
    print(f"Template   : {args.template}")
    print(f"API JSON   : {args.api_response_json}")
    print(f"Output dir : {args.output_dir}")
    print(f"Code dir   : {code_dir}")
    print(f"API response: {api_response['tool_name']} -> {api_response['response']}")
    print()

    print("Building model...")
    model = ov.build_model(args.checkpoint, code_dir, device=args.device)
    print("Model ready.")

    system_prompt = ov.render_fc_system_prompt(args.template, args.system_message, tools)
    print(f"System prompt ({len(system_prompt)} chars):\n{system_prompt}\n")

    wav_1d, input_signal, input_signal_lens = ov.load_wav_16k_mono(args.wav, device=args.device)
    prompt_tokens, prompt_token_lens = ov.encode_system_prompt(model, system_prompt, device=args.device)

    print(f"=== Pass 1: detect function call ({wav_1d.shape[0] / 16000:.2f}s audio) ===")
    result, fc_output, call_step, response_step, resp_token_ids = ov.run_fc_offline_inference(
        model,
        input_signal=input_signal,
        input_signal_lens=input_signal_lens,
        prompt_tokens=prompt_tokens,
        prompt_token_lens=prompt_token_lens,
        api_response=api_response,
        device=args.device,
    )

    text = result.get("text", [""])[0]
    print(f"\n=== Generated text ===\n{text}\n")

    paths = ov.save_fc_offline_outputs(
        result,
        wav_1d,
        args.output_dir,
        fc_output,
        system_prompt,
        args.wav,
        sample_id=args.sample_id,
        call_step=call_step,
        response_step=response_step,
        resp_token_len=len(resp_token_ids),
    )
    for name, path in paths.items():
        print(f"{name:8} -> {path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
