# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Streaming Triton client for the S2S voicechat model.

Usage:
    python client_streaming.py \
        --host localhost --port 8001 \
        --audio_filename /path/to/input.wav \
        --dur_test_audio 30
"""

import argparse
import uuid
import random
import sys

import librosa
import numpy as np
import soundfile as sf
import time
import tritonclient.grpc as grpcclient
from tqdm import tqdm
from tritonclient.utils import *

# Use Python's built-in logging so this script can run without NeMo installed
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8001
DEFAULT_NUM_FRAMES_PER_CHUNK = 1
DEFAULT_DUR_TEST_AUDIO = 30

parser = argparse.ArgumentParser(description="Streaming client for voicechat model")
parser.add_argument("--host", type=str, default=DEFAULT_HOST, help=f"Triton server host (default: {DEFAULT_HOST})")
parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Triton server port (default: {DEFAULT_PORT})")
parser.add_argument("--num_frames_per_chunk", type=int, default=DEFAULT_NUM_FRAMES_PER_CHUNK, help=f"Number of 80ms frames per inference step (default: {DEFAULT_NUM_FRAMES_PER_CHUNK})")
parser.add_argument("--audio_filename", type=str, required=True, help="Path to input audio file")
parser.add_argument("--dur_test_audio", type=int, default=DEFAULT_DUR_TEST_AUDIO, help=f"Duration of test audio in seconds; audio will be padded or trimmed to this length (default: {DEFAULT_DUR_TEST_AUDIO})")
parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output audio files (default: current directory)")
parser.add_argument("--system_prompt", type=str, default=None, help="System prompt to send to the model on the first request (overrides server default)")
args = parser.parse_args()

model_name = "voicechat"
audio_file = args.audio_filename

NUM_FRAMES_PER_CHUNK = args.num_frames_per_chunk
DUR_TEST_AUDIO = args.dur_test_audio
INPUT_CHUNK_SIZE_SAMPLES = int(16000 * 0.08) * NUM_FRAMES_PER_CHUNK  # number of samples per input chunk
NUM_CHUNKS_TEST_AUDIO = int(DUR_TEST_AUDIO / (0.08 * NUM_FRAMES_PER_CHUNK))
print(f"{NUM_CHUNKS_TEST_AUDIO=}")

times_spend_on_inference = []


def get_audio_as_chunks(audio_file):
    audio_signal, sr = librosa.load(audio_file, sr=16000)
    audio_signal = np.expand_dims(audio_signal, axis=0)

    padded_len_samples = int(NUM_CHUNKS_TEST_AUDIO * INPUT_CHUNK_SIZE_SAMPLES)
    audio_signal_padded = np.zeros((1, padded_len_samples), dtype=np.float32)

    if padded_len_samples > audio_signal.shape[1]:  # actually doing padding
        audio_signal_padded[:, : audio_signal.shape[1]] = audio_signal
    else:  # actually need to trim (because audio is longer than maxlen)
        audio_signal_padded = audio_signal[:, :padded_len_samples]

    audio_signal_chunks = [
        audio_signal_padded[:, i : i + INPUT_CHUNK_SIZE_SAMPLES]
        for i in range(0, audio_signal_padded.shape[1], INPUT_CHUNK_SIZE_SAMPLES)
    ]

    return audio_signal_chunks


def send_sequence_end(client, sequence_id):
    """Send a final request with sequence_end=True to properly clean up the sequence"""
    try:
        logger.info(f"Sending sequence_end=True for sequence_id={sequence_id}")
        
        # Send empty audio chunk with sequence_end=True
        empty_audio = np.zeros((1, INPUT_CHUNK_SIZE_SAMPLES), dtype=np.float32)
        
        inputs = [
            grpcclient.InferInput(
                "audio_signal", empty_audio.shape, np_to_triton_dtype(empty_audio.dtype)
            ),
        ]
        inputs[0].set_data_from_numpy(empty_audio)

        outputs = [
            grpcclient.InferRequestedOutput("output_text"),
            grpcclient.InferRequestedOutput("output_audio"),
        ]

        response = client.infer(
            model_name,
            inputs,
            request_id=str(uuid.uuid4()),
            outputs=outputs,
            sequence_id=sequence_id,
            sequence_start=False,
            sequence_end=True,  # This is the key - properly end the sequence
        )
        logger.info("Sequence ended successfully")
        
    except Exception as e:
        logger.error(f"Error ending sequence: {e}")

with grpcclient.InferenceServerClient(f"{args.host}:{args.port}") as client:
    audio_signal_chunks = get_audio_as_chunks(audio_file)

    generated_text = []
    generated_asr_text = []
    generated_audio = []

    # Generate a numeric sequence ID instead of string UUID to match UINT64 type
    sequence_id = random.randint(1, 2**63 - 1)  # Generate random uint64 value
    
    try:
        # If a system prompt is provided, send a separate prefill request first:
        # zero-length audio + system_prompt, with sequence_start=True.
        prefill_sent = False
        if args.system_prompt is not None:
            logger.info(f"Sending prefill request with system_prompt ({len(args.system_prompt)} chars)")
            empty_audio = np.zeros((1, 0), dtype=np.float32)
            prefill_inputs = [
                grpcclient.InferInput(
                    "audio_signal", empty_audio.shape, np_to_triton_dtype(empty_audio.dtype)
                ),
            ]
            prefill_inputs[0].set_data_from_numpy(empty_audio)

            prompt_np = np.array([args.system_prompt.encode("utf-8")], dtype=object)
            prompt_input = grpcclient.InferInput("system_prompt", prompt_np.shape, "BYTES")
            prompt_input.set_data_from_numpy(prompt_np)
            prefill_inputs.append(prompt_input)

            prefill_outputs = [
                grpcclient.InferRequestedOutput("output_text"),
                grpcclient.InferRequestedOutput("output_asr_text"),
                grpcclient.InferRequestedOutput("output_audio"),
            ]

            prefill_start = time.time()
            client.infer(
                model_name,
                prefill_inputs,
                request_id=str(uuid.uuid4()),
                outputs=prefill_outputs,
                sequence_id=sequence_id,
                sequence_start=True,
                sequence_end=False,
            )
            logger.info(f"Prefill completed in {time.time() - prefill_start:.3f}s")
            prefill_sent = True

        for idx, audio_chunk in tqdm(enumerate(audio_signal_chunks)):
            inputs = [
                grpcclient.InferInput(
                    "audio_signal", audio_chunk.shape, np_to_triton_dtype(audio_chunk.dtype)
                ),
            ]

            inputs[0].set_data_from_numpy(audio_chunk)

            outputs = [
                grpcclient.InferRequestedOutput("output_text"),
                grpcclient.InferRequestedOutput("output_asr_text"),
                grpcclient.InferRequestedOutput("output_audio"),
            ]

            start_time = time.time()
            response = client.infer(
                model_name,
                inputs,
                request_id=str(uuid.uuid4()),
                outputs=outputs,
                sequence_id=sequence_id,
                sequence_start=(idx == 0 and not prefill_sent),
                sequence_end=idx == len(audio_signal_chunks) - 1,
            )
            end_time = time.time()

            result = response.get_response()
            output_text = response.as_numpy("output_text")
            output_asr_text = response.as_numpy("output_asr_text")
            output_audio = response.as_numpy("output_audio")

            generated_text.extend([i.decode("utf-8") for i in output_text])
            generated_asr_text.extend([i.decode("utf-8") for i in output_asr_text])

            if output_audio.shape[1] > 0:
                times_spend_on_inference.append(end_time - start_time)
                generated_audio.append(output_audio)
                
    except KeyboardInterrupt:
        logger.info("\nKeyboardInterrupt received! Calling send_sequence_end...")
        send_sequence_end(client, sequence_id)
        logger.info("Sequence cleanup completed. Exiting...")
        sys.exit(0)

    logger.info("Agent text: " + "".join([str(i) for i in generated_text]))
    logger.info("ASR text (user's speech): " + "".join([str(i) for i in generated_asr_text]))
    generated_audio = np.concatenate(generated_audio, axis=1)

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    output_audio_path = os.path.join(args.output_dir, "output_audio.wav")
    sf.write(output_audio_path, generated_audio.squeeze(0), 22050)
    logger.info(f"Agent audio written to {output_audio_path}")

    # Save audio file with both input and output in each channel
    # Resample input to 22050 Hz, and pad shorter file to the same length as the longer one
    input_audio, sr = librosa.load(audio_file, sr=22050)
    generated_audio_1d = generated_audio.squeeze(0)  # Convert from [1, T] to [T]
    maxlen = max(input_audio.shape[0], generated_audio_1d.shape[0])
    input_audio = np.pad(input_audio, (0, maxlen - input_audio.shape[0]), mode="constant")
    generated_audio_1d = np.pad(generated_audio_1d, (0, maxlen - generated_audio_1d.shape[0]), mode="constant")
    both_audio = np.column_stack([input_audio, generated_audio_1d])  # Create stereo: [T, 2]
    combined_path = os.path.join(args.output_dir, "input_and_output_combined.wav")
    sf.write(combined_path, both_audio, 22050)
    logger.info(f"Input and output combined audio written to {combined_path}")

    logger.info(f"Average time spend on inference: {np.mean(times_spend_on_inference)}")
    logger.info(f"std of time spend on inference: {np.std(times_spend_on_inference)}")
    logger.info(f"Median time spend on inference: {np.median(times_spend_on_inference)}")
    logger.info(f"Min time spend on inference: {np.min(times_spend_on_inference)}")
    logger.info(f"Max time spend on inference: {np.max(times_spend_on_inference)}")
    logger.info(f"All times spend on inference: {[round(i, 4) for i in times_spend_on_inference]}")
    logger.info(f"Number of chunks: {len(times_spend_on_inference)}")
