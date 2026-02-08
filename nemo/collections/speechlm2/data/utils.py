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
import warnings
import torch
from torch.nn.utils.rnn import pad_sequence


def get_pad_id(tokenizer) -> int:
    pad_id = tokenizer.pad
    if pad_id is not None:
        return pad_id
    pad_id = tokenizer.unk_id
    if pad_id is not None:
        return pad_id
    warnings.warn(
        "The text tokenizer has no <pad> or <unk> tokens available, using ID 0 for padding (this may lead to silent bugs)."
    )
    return 0


def collate_and_pad_1d(data, pad_id=-1):
    """
    Collate and pad 1D sequences for batch processing.
    
    Args:
        data (List[List[int]] or List[torch.Tensor]): List of sequences.
        pad_id (int, optional): Padding value. Defaults to -1.
    
    Returns:
        torch.Tensor: Padded tensor of shape [batch_size, max_sequence_length]
    """
    if not data:
        return torch.tensor([])
    
    # Fast path: if all are already tensors, skip conversion
    if all(isinstance(seq, torch.Tensor) for seq in data):
        tensor_data = data
    else:
        # Convert to tensors using list comprehension (faster than loop)
        tensor_data = [seq if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.long) for seq in data]
    
    # Pad sequences
    padded = pad_sequence(tensor_data, batch_first=True, padding_value=pad_id)
    return padded


def collate_and_pad_2d(tensors, pad_id):
    """
    Collate and pad 2D tensors for batch processing.
    
    Args:
        tensors (List[torch.Tensor]): List of 2D tensors with varying shapes.
        pad_id (int): Padding value.
    
    Returns:
        torch.Tensor: Padded tensor of shape [batch_size, max_rows, max_cols]
    """
    if not tensors:
        return torch.tensor([])
    
    # Find max dimensions in a single pass
    batch_size = len(tensors)
    max_rows = 0
    max_cols = 0
    for t in tensors:
        max_rows = max(max_rows, t.shape[0])
        max_cols = max(max_cols, t.shape[1])
    
    # Create padded tensor on the same device as input tensors
    first_tensor = tensors[0]
    padded = torch.full(
        (batch_size, max_rows, max_cols), 
        pad_id, 
        dtype=first_tensor.dtype,
        device=first_tensor.device
    )
    
    # Fill with actual data (in-place copy is fast)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0], :t.shape[1]] = t
    
    return padded


def collate_and_pad(inputs, text_pad_id):
    """
    Collate and pad sequences.
    
    Args:
        inputs: List of tensors to collate
        text_pad_id: Padding ID for text
    
    Returns:
        Tuple of (padded_tokens, token_lengths)
    """
    if not inputs:
        return torch.tensor([]), torch.tensor([])
    
    # Compute lengths and pad in a more efficient way
    # Use tensor instead of list for better memory and GPU transfer
    token_lengths = torch.tensor([len(seq) for seq in inputs], dtype=torch.long)
    tokens = pad_sequence(inputs, batch_first=True, padding_value=text_pad_id)
    return tokens, token_lengths


def create_one_second_silence_template(
    perception_module,
    sample_rate: int,
    device: torch.device = None,
) -> tuple:
    """
    Create a 1-second silence template by encoding through perception module.
    This template can be reused by repeating and slicing as needed.
    
    Args:
        perception_module: The audio encoder module (e.g., ASR encoder)
        sample_rate: Audio sampling rate (e.g., 16000 Hz)
        device: Device to create tensors on (optional, auto-detected from perception_module if not provided)
        
    Returns:
        tuple: (silence_embeddings, num_frames)
            - silence_embeddings: [num_frames, hidden_size] - embeddings for 1 second of silence
            - num_frames: Number of frames produced by 1 second of audio
    """
    # Create 1 second of silence audio
    audio_length = sample_rate  # 1 second at the given sample rate
    
    # Use the same device and dtype as regular user audio
    # Audio is always float32 by default, same as user audio from dataset
    if device is None:
        device = next(perception_module.parameters()).device
    perception_dtype = next(perception_module.parameters()).dtype
    audio_dtype = torch.float32  # Standard dtype for audio
    
    silence_audio = torch.zeros((1, audio_length), device=device, dtype=audio_dtype)
    silence_audio_lens = torch.tensor([audio_length], device=device, dtype=torch.long)
    
    # Encode through perception module with autocast if needed
    # During training, autocast is enabled which handles float32->bfloat16 conversion
    # During init, we need to explicitly enable it to match training behavior
    with torch.no_grad():
        # Enable autocast if perception module uses bfloat16 (to match training behavior)
        if perception_dtype == torch.bfloat16 and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                silence_encoded, silence_lens, _ = perception_module(
                    input_signal=silence_audio,
                    input_signal_length=silence_audio_lens,
                    return_encoder_emb=True,
                )
        else:
            silence_encoded, silence_lens, _ = perception_module(
                input_signal=silence_audio,
                input_signal_length=silence_audio_lens,
                return_encoder_emb=True,
            )
    
    # Extract [1, T, H] -> [T, H]
    silence_encoded = silence_encoded[0]  # Remove batch dimension
    num_frames = silence_encoded.shape[0]
    
    return silence_encoded, num_frames


def get_silence_embeddings_from_ratio(
    perception_module,
    frames_per_second: float,
    sample_rate: int,
    length: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Generate silence embeddings by calculating the required audio duration from the frame ratio.
    
    Strategy:
    1. Use frames_per_second (from 1-second template) to calculate duration needed
    2. Add 10% buffer to handle rounding
    3. Create that much silence audio
    4. Encode through perception module
    5. Slice to exact length
    
    Args:
        perception_module: The audio encoder module (e.g., ASR encoder)
        frames_per_second: Number of frames produced by 1 second of audio
        sample_rate: Audio sampling rate (e.g., 16000 Hz)
        length: Number of embedding frames needed
        device: Device to create tensors on (optional, auto-detected from perception_module if not provided)
        dtype: Target data type for the embeddings (optional, auto-detected from perception_module if not provided)
        
    Returns:
        silence_embeddings: [length, hidden_size] - silence embeddings of exact length
    """
    # Step 1: Calculate how many seconds of audio we need
    # If 1 second = frames_per_second frames, then length frames = length/frames_per_second seconds
    seconds_needed = length / frames_per_second
    
    # Step 2: Add 10% buffer to ensure we get enough frames after encoding
    seconds_with_buffer = seconds_needed * 1.1
    
    # Step 3: Convert to audio samples
    audio_length = int(seconds_with_buffer * sample_rate)
    
    # Step 4: Create silence audio - use same dtype as regular user audio
    if device is None:
        device = next(perception_module.parameters()).device
    perception_dtype = next(perception_module.parameters()).dtype
    if dtype is None:
        dtype = torch.float32  # Standard dtype for audio, PyTorch handles conversion
    
    # Audio input should always be float32 (same as user audio from dataset)
    audio_dtype = torch.float32
    silence_audio = torch.zeros((1, audio_length), device=device, dtype=audio_dtype)
    silence_audio_lens = torch.tensor([audio_length], device=device, dtype=torch.long)
    
    # Step 5: Encode through perception module with autocast if needed
    # During training, autocast is enabled which handles float32->bfloat16 conversion
    # During init, we need to explicitly enable it to match training behavior
    with torch.no_grad():
        # Enable autocast if perception module uses bfloat16 (to match training behavior)
        if perception_dtype == torch.bfloat16 and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                silence_encoded, silence_lens, _ = perception_module(
                    input_signal=silence_audio,
                    input_signal_length=silence_audio_lens,
                    return_encoder_emb=True,
                )
        else:
            silence_encoded, silence_lens, _ = perception_module(
                input_signal=silence_audio,
                input_signal_length=silence_audio_lens,
                return_encoder_emb=True,
            )
    
    # Extract [1, T, H] -> [T, H]
    silence_encoded = silence_encoded[0]  # Remove batch dimension
    
    # Safeguard: If we still didn't get enough frames (should be rare with 10% buffer),
    # repeat to reach desired length
    if silence_encoded.shape[0] < length:
        num_repeats = (length // silence_encoded.shape[0]) + 1
        silence_encoded = silence_encoded.repeat(num_repeats, 1)
    
    # Step 6: Slice to exactly the number of frames we need
    return silence_encoded[:length].to(dtype=dtype)


def get_silence_embeddings(
    perception_module,
    length: int,
    device: torch.device,
    dtype: torch.dtype = None,
    subsampling_factor: float = None
) -> torch.Tensor:
    """
    Generate proper silence embeddings by encoding actual silence audio through the perception module.
    
    DEPRECATED: This function encodes silence from scratch each time. 
    Consider using create_one_second_silence_template() once and then 
    get_silence_embeddings_from_template() for better performance.
    
    Strategy to handle rounding issues:
    1. Create audio longer than needed (with buffer) in time domain
    2. Encode through perception module
    3. Slice to exact length needed
    
    Args:
        perception_module: The audio encoder module (e.g., ASR encoder)
        length: Number of embedding frames needed
        device: Device to create tensors on (optional, auto-detected from perception_module if not provided)
        dtype: Data type for the embeddings (optional, auto-detected from perception_module if not provided)
        subsampling_factor: The ratio of audio samples to embedding frames
        
    Returns:
        silence_embeddings: [length, hidden_size] - proper silence embeddings of exact length
    """
    # Step 1: Reverse calculation with buffer to handle rounding issues
    base_audio_length = int(length * subsampling_factor)
    audio_length = int(base_audio_length * 1.1)  # 10% buffer for safety
    
    # Create silence in time domain (zeros = no sound) - use same dtype as regular user audio
    if device is None:
        device = next(perception_module.parameters()).device
    perception_dtype = next(perception_module.parameters()).dtype
    if dtype is None:
        dtype = torch.float32  # Standard dtype for audio, PyTorch handles conversion
    
    # Audio input should always be float32 (same as user audio from dataset)
    audio_dtype = torch.float32
    silence_audio = torch.zeros((1, audio_length), device=device, dtype=audio_dtype)
    silence_audio_lens = torch.tensor([audio_length], device=device, dtype=torch.long)
    
    # Step 2: Encode silence through perception module with autocast if needed
    # During training, autocast is enabled which handles float32->bfloat16 conversion
    # During init, we need to explicitly enable it to match training behavior
    with torch.no_grad():
        # Enable autocast if perception module uses bfloat16 (to match training behavior)
        if perception_dtype == torch.bfloat16 and device.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                silence_encoded, silence_lens, _ = perception_module(
                    input_signal=silence_audio,
                    input_signal_length=silence_audio_lens,
                    return_encoder_emb=True,
                )
        else:
            silence_encoded, silence_lens, _ = perception_module(
                input_signal=silence_audio,
                input_signal_length=silence_audio_lens,
                return_encoder_emb=True,
            )
    
    # Extract the required number of frames [1, T, H] -> [T, H]
    silence_encoded = silence_encoded[0]  # Remove batch dimension
    
    # Safeguard: If we still didn't get enough frames, repeat to reach desired length
    if silence_encoded.shape[0] < length:
        num_repeats = (length // silence_encoded.shape[0]) + 1
        silence_encoded = silence_encoded.repeat(num_repeats, 1)
    
    # Step 3: Slice to exactly the number of frames we need
    return silence_encoded[:length].to(dtype=dtype)
