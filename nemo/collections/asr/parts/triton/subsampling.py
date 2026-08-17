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

"""Fused conv -> ReLU -> depthwise for FastConformer's ``dw_striding`` subsampling.

The first three stages of the pre-encoder are a convolution, a ReLU, and a depthwise
convolution, both convolutions 3x3 stride 2 over (time, frequency). Writing their kernel taps as
(dt, df) and the left padding as P, they compute

    middle[c, t, f] = relu( conv_bias[c] + sum over (dt, df) of W_conv[c, dt, df] * mel[2t + dt - P, 2f + df - P] )
    out[c, t, f]    = depth_bias[c] + sum over (dt, df) of W_depth[c, dt, df] * middle[c, 2t + dt - P, 2f + df - P]

The first expands 1 channel to many; the second keeps `c` on both sides, so it is depthwise.
``middle`` is never written to memory -- each program recomputes the slice it needs -- and the
output is channels-last, the layout the following 1x1 convolution reads.

conv0 is a GEMM
---------------
Each output reads a 3x3 patch of ``middle``, and a program emits two adjacent frequency bins at
once, so together they span a 3x5 window of it: three time rows by five frequency columns, the
15 positions ``WINDOW_SIZE`` counts. Filling that window is where conv0 becomes a matrix product,
because every channel contracts its nine taps against the same mel patch:

    taps  [CHANNEL_BLOCK, PAD]  x  feats [PAD, PAD]  ->  middle [CHANNEL_BLOCK, PAD]

one tensor-core MMA per output row. Both axes are padded to PAD, 16, rather than the 9 taps and
15 window positions they carry, because ``tl.arange`` and ``tl.dot`` require powers of two; the
surplus lanes are masked.

The depthwise is not a GEMM -- its weights differ per channel, so there is nothing to
contract. It needs no gathers either: ``sel_low`` and ``sel_high`` hold the depthwise taps
pre-placed at the window positions each output frequency bin reads, built once per step on the
host, so each output is a masked reduction over registers already loaded.

The backward reads the same two blocks in reverse
-------------------------------------------------
    dmiddle           = (grad_low*sel_low + grad_high*sel_high) * (middle > 0)
    grad_conv_weight += dot(dmiddle, feats^T)            a second GEMM, feats already loaded
    acc_low/high     += grad_low/high * relu(middle)     outer products -> grad depthwise

No gradient is computed for ``mel``: the features are a leaf. ``grad_conv_bias`` rides in lane
TAPS of ``grad_conv_weight`` rather than accumulating separately; see the backward kernel.

Both kernels skip tiles starting past the utterance length.

``tl.dot`` silently uses TF32 for fp32 inputs, so these kernels pass ``input_precision="ieee"``.
It is ignored for bf16, where the MMA is used regardless.
"""

from __future__ import annotations

import torch

from nemo.core.utils.optional_libs import TRITON_AVAILABLE

# The geometry both kernels are written for. Their index arithmetic hardcodes it -- the window
# walk, the two frequency bins per program, the tap decode -- so these are the single place it is
# named, not knobs.
_KERNEL = 3
_STRIDE = 2
_BINS = 2  # output frequency bins one program emits together

# Plain ints for host code. The kernels need ``tl.constexpr`` versions instead, and those cannot
# be defined unless triton is importable -- hence the pair.
_WINDOW_COLS = _KERNEL + (_BINS - 1) * _STRIDE  # middle columns those bins span between them
_WINDOW_SIZE = _KERNEL * _WINDOW_COLS
_TAPS = _KERNEL * _KERNEL
_PAD = 1 << (max(_TAPS, _WINDOW_SIZE) - 1).bit_length()  # tl.arange / tl.dot need a power of two


if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    KERNEL = tl.constexpr(_KERNEL)
    STRIDE = tl.constexpr(_STRIDE)
    BINS = tl.constexpr(_BINS)
    WINDOW_COLS = tl.constexpr(_WINDOW_COLS)
    WINDOW_SIZE = tl.constexpr(_WINDOW_SIZE)
    TAPS = tl.constexpr(_TAPS)
    PAD = tl.constexpr(_PAD)

    # The grid is deliberately small: runtime is nearly flat across launch geometry here, so a
    # wider search buys only autotune time. For the same reason the autotune keys exclude the time
    # extent, which varies per batch with the longest utterance: each unseen value would trigger a
    # full sweep, while the winning tile depends only on the channel count and frequency extent.
    def _forward_configs():
        return [
            triton.Config({"CHANNEL_BLOCK": block, "TIME_ROWS": rows}, num_warps=warps)
            for block in (32, 64, 128)
            for rows in (1, 2, 4, 8)
            for warps in (2, 4)
        ]

    def _backward_configs():
        return [
            triton.Config({"CHANNEL_BLOCK": block, "TIME_ROWS": rows, "TILE_SPLITS": splits}, num_warps=warps)
            for block in (32, 64, 128)
            for rows in (1, 2, 4)
            for splits in (1024, 4096)
            for warps in (2, 4)
        ]

    @triton.jit
    def _window_geometry(first_out_freq, middle_freq, pad_left):
        """Each window position's (middle row offset, middle column) and whether that column exists."""
        window_pos = tl.arange(0, PAD)
        middle_col = STRIDE * first_out_freq + window_pos % WINDOW_COLS - pad_left
        return (
            window_pos // WINDOW_COLS,
            middle_col,
            (middle_col >= 0) & (middle_col < middle_freq) & (window_pos < WINDOW_SIZE),
        )

    @triton.jit
    def _load_params(conv_weight_ptr, conv_bias_ptr, sel_low_ptr, sel_high_ptr, channel, channel_mask):
        """The per-channel constants both kernels open with: conv0's taps and bias, both selections."""
        tap = tl.arange(0, PAD)
        window_pos = tl.arange(0, PAD)
        conv_taps = tl.load(
            conv_weight_ptr + channel[:, None] * TAPS + tap[None, :],
            mask=(tap < TAPS)[None, :] & channel_mask[:, None],
            other=0.0,
        )
        conv_bias = tl.load(conv_bias_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
        sel_low = tl.load(
            sel_low_ptr + channel[:, None] * PAD + window_pos[None, :], mask=channel_mask[:, None], other=0.0
        ).to(tl.float32)
        sel_high = tl.load(
            sel_high_ptr + channel[:, None] * PAD + window_pos[None, :], mask=channel_mask[:, None], other=0.0
        ).to(tl.float32)
        return conv_taps, conv_bias, sel_low, sel_high

    @triton.jit
    def _load_window(
        mel_ptr, batch_base, mel_time_stride, middle_row, middle_col, pos_ok, valid_time, mel_freq, pad_left
    ):
        """The window's mel patch as [tap, window position], both padded to PAD -- the GEMM's rhs.

        Indexed straight out of the unpadded mel. ``valid_time`` is this utterance's own length,
        so the mask that keeps reads in bounds also applies NeMo's pre-conv0 length masking.
        """
        tap = tl.arange(0, PAD)
        mel_row = STRIDE * middle_row[None, :] + (tap // KERNEL)[:, None] - pad_left
        mel_col = STRIDE * middle_col[None, :] + (tap % KERNEL)[:, None] - pad_left
        in_range = (mel_row >= 0) & (mel_row < valid_time) & (mel_col >= 0) & (mel_col < mel_freq)
        return tl.load(
            mel_ptr + batch_base + mel_row * mel_time_stride + mel_col,
            mask=(tap < TAPS)[:, None] & pos_ok[None, :] & in_range,
            other=0.0,
        )

    # ----------------------------------------------------------------------------- forward

    @triton.autotune(configs=_forward_configs(), key=["channels", "out_freq"])
    @triton.jit
    def _forward_kernel(
        mel_ptr,
        conv_weight_ptr,
        conv_bias_ptr,
        sel_low_ptr,
        sel_high_ptr,
        depth_bias_ptr,
        output_ptr,
        mel_len_ptr,
        middle_len_ptr,
        out_len_ptr,
        pad_left,
        channels,
        out_time,
        out_freq,
        mel_freq,
        middle_freq,
        mel_batch_stride,
        mel_time_stride,
        out_batch_stride,
        out_time_stride,
        out_freq_stride,
        TIME_ROWS: tl.constexpr,
        CHANNEL_BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0)
        freq_tiles = tl.cdiv(out_freq, BINS)
        first_out_time = (tl.program_id(2) // freq_tiles) * TIME_ROWS
        first_out_freq = (tl.program_id(2) % freq_tiles) * BINS
        channel = tl.program_id(1) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
        channel_mask = channel < channels

        conv_taps, conv_bias, sel_low, sel_high = _load_params(
            conv_weight_ptr, conv_bias_ptr, sel_low_ptr, sel_high_ptr, channel, channel_mask
        )
        depth_bias = tl.load(depth_bias_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)

        window_row, middle_col, freq_ok = _window_geometry(first_out_freq, middle_freq, pad_left)
        batch_base = batch * mel_batch_stride
        # NeMo masks between every stage: mel at its own length, `middle` at the post-conv0
        # length, the output at the post-depthwise length. relu(conv_bias) is non-zero, so
        # without the middle mask the bias leaks into padded time.
        this_mel_len = tl.load(mel_len_ptr + batch)
        this_middle_len = tl.load(middle_len_ptr + batch)
        this_out_len = tl.load(out_len_ptr + batch)

        # A batch is padded to its longest utterance, so whole tiles fall past the end of a short
        # one. Their gather and both dots are skipped, but the zeros must still be stored: the
        # downstream `pw1` Linear contracts over every time position for its weight gradient, so an
        # undefined tail corrupts grad(pw1.weight) even though the output itself stays correct.
        tile_live = first_out_time < this_out_len

        for step in tl.static_range(TIME_ROWS):
            if tile_live:
                middle_row = STRIDE * (first_out_time + step) + window_row - pad_left
                pos_ok = freq_ok & (middle_row >= 0) & (middle_row < this_middle_len)
                feats = _load_window(
                    mel_ptr,
                    batch_base,
                    mel_time_stride,
                    middle_row,
                    middle_col,
                    pos_ok,
                    this_mel_len,
                    mel_freq,
                    pad_left,
                )
                middle = tl.dot(conv_taps, feats, input_precision="ieee") + conv_bias[:, None]
                middle = tl.where(pos_ok[None, :], tl.maximum(middle, 0.0), 0.0)

                output_low = depth_bias + tl.sum(middle * sel_low, 1)
                output_high = depth_bias + tl.sum(middle * sel_high, 1)
            else:
                output_low = tl.zeros([CHANNEL_BLOCK], tl.float32)
                output_high = tl.zeros([CHANNEL_BLOCK], tl.float32)
            # beyond the utterance NeMo stores zeros, so mask the value, not the store
            in_length = first_out_time + step < this_out_len
            output_low = tl.where(in_length, output_low, 0.0)
            output_high = tl.where(in_length, output_high, 0.0)
            time_valid = channel_mask & (first_out_time + step < out_time)
            out_base = (
                output_ptr
                + batch * out_batch_stride
                + (first_out_time + step) * out_time_stride
                + first_out_freq * out_freq_stride
                + channel
            )
            # The low bin always exists -- the grid is cdiv(out_freq, BINS) tiles. The high bin is
            # the one that falls off the end when out_freq is odd.
            tl.store(out_base, output_low.to(output_ptr.dtype.element_ty), mask=time_valid)
            tl.store(
                out_base + out_freq_stride,
                output_high.to(output_ptr.dtype.element_ty),
                mask=time_valid & (first_out_freq + 1 < out_freq),
            )

    # ---------------------------------------------------------------------------- backward

    # reset_to_zero is essential: this kernel accumulates with atomic_add and autotune
    # benchmarks every config several times. Without it the trial runs sum into the same
    # buffers and the gradients come out orders of magnitude too large.
    @triton.autotune(
        configs=_backward_configs(),
        key=["channels", "out_freq"],
        reset_to_zero=[
            "grad_conv_weight_ptr",
            "grad_conv_bias_ptr",
            "acc_low_ptr",
            "acc_high_ptr",
            "grad_depth_bias_ptr",
        ],
    )
    @triton.jit
    def _backward_kernel(
        mel_ptr,
        conv_weight_ptr,
        conv_bias_ptr,
        sel_low_ptr,
        sel_high_ptr,
        grad_output_ptr,
        grad_conv_weight_ptr,
        grad_conv_bias_ptr,
        acc_low_ptr,
        acc_high_ptr,
        grad_depth_bias_ptr,
        mel_len_ptr,
        middle_len_ptr,
        out_len_ptr,
        pad_left,
        channels,
        out_time,
        out_freq,
        mel_freq,
        middle_freq,
        batch_size,
        mel_batch_stride,
        mel_time_stride,
        grad_batch_stride,
        grad_time_stride,
        grad_freq_stride,
        TIME_ROWS: tl.constexpr,
        CHANNEL_BLOCK: tl.constexpr,
        TILE_SPLITS: tl.constexpr,
    ):
        channel = tl.program_id(0) * CHANNEL_BLOCK + tl.arange(0, CHANNEL_BLOCK)
        channel_mask = channel < channels
        tap = tl.arange(0, PAD)
        window_pos = tl.arange(0, PAD)
        conv_taps, conv_bias, sel_low, sel_high = _load_params(
            conv_weight_ptr, conv_bias_ptr, sel_low_ptr, sel_high_ptr, channel, channel_mask
        )

        # grad_conv_bias is not accumulated separately -- it rides in lane TAPS of grad_conv_weight.
        grad_conv_weight = tl.zeros([CHANNEL_BLOCK, PAD], tl.float32)
        acc_low = tl.zeros([CHANNEL_BLOCK, PAD], tl.float32)
        acc_high = tl.zeros([CHANNEL_BLOCK, PAD], tl.float32)
        grad_depth_bias = tl.zeros([CHANNEL_BLOCK], tl.float32)

        freq_tiles = tl.cdiv(out_freq, BINS)
        tiles_per_batch = tl.cdiv(out_time, TIME_ROWS) * freq_tiles
        for tile in range(tl.program_id(1), batch_size * tiles_per_batch, TILE_SPLITS):
            batch = tile // tiles_per_batch
            tile_in_batch = tile % tiles_per_batch
            first_out_time = (tile_in_batch // freq_tiles) * TIME_ROWS
            first_out_freq = (tile_in_batch % freq_tiles) * BINS
            window_row, middle_col, freq_ok = _window_geometry(first_out_freq, middle_freq, pad_left)
            batch_base = batch * mel_batch_stride
            this_mel_len = tl.load(mel_len_ptr + batch)
            this_middle_len = tl.load(middle_len_ptr + batch)
            this_out_len = tl.load(out_len_ptr + batch)

            # Tiles past the end of this utterance contribute nothing to any gradient and are
            # skipped outright. Triton has no `continue`, hence a guard block rather than an exit.
            if first_out_time < this_out_len:
                for step in tl.static_range(TIME_ROWS):
                    middle_row = STRIDE * (first_out_time + step) + window_row - pad_left
                    pos_ok = freq_ok & (middle_row >= 0) & (middle_row < this_middle_len)
                    feats = _load_window(
                        mel_ptr,
                        batch_base,
                        mel_time_stride,
                        middle_row,
                        middle_col,
                        pos_ok,
                        this_mel_len,
                        mel_freq,
                        pad_left,
                    )
                    # Lanes TAPS..PAD-1 exist only to make the MMA a power of two and conv_taps
                    # masks them to zero. Filling lane TAPS with ones makes the same dot that
                    # produces grad_conv_weight also produce sum(dmiddle) there, which is exactly
                    # grad_conv_bias -- trading a [PAD,PAD] select for a per-step cross-lane
                    # reduction. The conv0 recompute below is unaffected: conv_taps is zero there.
                    feats = tl.where(
                        (tap == TAPS)[:, None] & pos_ok[None, :], tl.full((1, 1), 1.0, feats.dtype), feats
                    )
                    pre_activation = tl.dot(conv_taps, feats, input_precision="ieee") + conv_bias[:, None]
                    live = pos_ok[None, :] & (pre_activation > 0.0)
                    middle = tl.where(live, pre_activation, 0.0)

                    time_valid = (
                        channel_mask & (first_out_time + step < out_time) & (first_out_time + step < this_out_len)
                    )
                    grad_base = (
                        grad_output_ptr
                        + batch * grad_batch_stride
                        + (first_out_time + step) * grad_time_stride
                        + first_out_freq * grad_freq_stride
                        + channel
                    )
                    # The low bin always exists; only the high one falls off an odd out_freq.
                    grad_low = tl.load(grad_base, mask=time_valid, other=0.0).to(tl.float32)
                    grad_high = tl.load(
                        grad_base + grad_freq_stride, mask=time_valid & (first_out_freq + 1 < out_freq), other=0.0
                    ).to(tl.float32)

                    grad_depth_bias += grad_low + grad_high
                    acc_low += grad_low[:, None] * middle
                    acc_high += grad_high[:, None] * middle

                    dmiddle = tl.where(live, grad_low[:, None] * sel_low + grad_high[:, None] * sel_high, 0.0)
                    grad_conv_weight += tl.dot(dmiddle.to(feats.dtype), tl.trans(feats), input_precision="ieee")

        # one reduction at the end, instead of one per step
        grad_conv_bias = tl.sum(tl.where((tap == TAPS)[None, :], grad_conv_weight, 0.0), 1)
        tl.atomic_add(
            grad_conv_weight_ptr + channel[:, None] * PAD + tap[None, :],
            grad_conv_weight,
            mask=channel_mask[:, None] & (tap < TAPS)[None, :],
        )
        tl.atomic_add(acc_low_ptr + channel[:, None] * PAD + window_pos[None, :], acc_low, mask=channel_mask[:, None])
        tl.atomic_add(
            acc_high_ptr + channel[:, None] * PAD + window_pos[None, :], acc_high, mask=channel_mask[:, None]
        )
        tl.atomic_add(grad_conv_bias_ptr + channel, grad_conv_bias, mask=channel_mask)
        tl.atomic_add(grad_depth_bias_ptr + channel, grad_depth_bias, mask=channel_mask)


def _downsampled_length(length, pad_total):
    """Output extent of one strided convolution stage, floor mode -- NeMo's calc_length.

    Works on ints and on int tensors, so the same expression gives the tensor extents and
    the per-utterance lengths.
    """
    return (length + pad_total - _KERNEL) // _STRIDE + 1


def _as_window(row):
    """Read a flat PAD-wide row back as the KERNEL x WINDOW_COLS window it holds."""
    return row[:, :_WINDOW_SIZE].reshape(-1, _KERNEL, _WINDOW_COLS)


def _as_row(window):
    """Flatten a window into the PAD-wide row the kernel loads; PAD is the next power of two."""
    row = window.new_zeros(window.shape[0], _PAD)
    row[:, :_WINDOW_SIZE] = window.reshape(window.shape[0], _WINDOW_SIZE)
    return row


def _build_selection(depth_weight):
    """The depthwise weights placed where each output frequency bin reads them, zero elsewhere.

    One window spans two output bins: the low bin reads its first KERNEL columns, the high bin
    its last. Placing the taps at those two offsets turns the depthwise into a multiply of the
    whole window -- already in registers from conv0 -- by one vector, plus a sum. The zeros do
    the selecting, so the kernel needs no gather and no index arithmetic.
    """
    channels = depth_weight.shape[0]
    taps = depth_weight.reshape(channels, _KERNEL, _KERNEL)
    low = depth_weight.new_zeros(channels, _KERNEL, _WINDOW_COLS)
    high = depth_weight.new_zeros(channels, _KERNEL, _WINDOW_COLS)
    low[..., :_KERNEL] = taps
    high[..., _STRIDE:] = taps
    return _as_row(low), _as_row(high)


class _FusedSubsampling(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mel,
        conv_weight,
        conv_bias,
        depth_weight,
        depth_bias,
        mel_lengths,
        middle_lengths,
        out_lengths,
        pad_left,
        dims,
        dtype,
    ):
        out_time, out_freq, middle_freq = dims
        batch_size, _, _, mel_freq = mel.shape
        channels = conv_weight.shape[0]
        mel = mel.contiguous().to(dtype)
        sel_low, sel_high = _build_selection(depth_weight)
        conv_weight_cast, conv_bias_cast = conv_weight.to(dtype), conv_bias.to(dtype)
        sel_low, sel_high = sel_low.to(dtype), sel_high.to(dtype)
        output = torch.empty((batch_size, out_time, out_freq, channels), device=mel.device, dtype=dtype)

        def grid(meta):
            return (
                batch_size,
                triton.cdiv(channels, meta["CHANNEL_BLOCK"]),
                triton.cdiv(out_time, meta["TIME_ROWS"]) * triton.cdiv(out_freq, _BINS),
            )

        _forward_kernel[grid](
            mel,
            conv_weight_cast,
            conv_bias_cast,
            sel_low,
            sel_high,
            depth_bias.to(dtype),
            output,
            mel_lengths,
            middle_lengths,
            out_lengths,
            pad_left,
            channels,
            out_time,
            out_freq,
            mel_freq,
            middle_freq,
            mel.stride(0),
            mel.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
        )
        ctx.save_for_backward(
            mel, conv_weight_cast, conv_bias_cast, sel_low, sel_high, mel_lengths, middle_lengths, out_lengths
        )
        ctx.shapes = (batch_size, channels, mel_freq, middle_freq, out_time, out_freq, pad_left)
        ctx.param_dtypes = (conv_weight.dtype, conv_bias.dtype, depth_weight.dtype, depth_bias.dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (mel, conv_weight, conv_bias, sel_low, sel_high, mel_lengths, middle_lengths, out_lengths) = ctx.saved_tensors
        (batch_size, channels, mel_freq, middle_freq, out_time, out_freq, pad_left) = ctx.shapes
        grad_output = grad_output.contiguous()
        device = grad_output.device
        grad_conv_weight = torch.zeros((channels, _PAD), device=device, dtype=torch.float32)
        grad_conv_bias = torch.zeros((channels,), device=device, dtype=torch.float32)
        acc_low = torch.zeros((channels, _PAD), device=device, dtype=torch.float32)
        acc_high = torch.zeros((channels, _PAD), device=device, dtype=torch.float32)
        grad_depth_bias = torch.zeros((channels,), device=device, dtype=torch.float32)

        def grid(meta):
            tiles = batch_size * triton.cdiv(out_time, meta["TIME_ROWS"]) * triton.cdiv(out_freq, _BINS)
            return triton.cdiv(channels, meta["CHANNEL_BLOCK"]), min(meta["TILE_SPLITS"], tiles)

        _backward_kernel[grid](
            mel,
            conv_weight,
            conv_bias,
            sel_low,
            sel_high,
            grad_output,
            grad_conv_weight,
            grad_conv_bias,
            acc_low,
            acc_high,
            grad_depth_bias,
            mel_lengths,
            middle_lengths,
            out_lengths,
            pad_left,
            channels,
            out_time,
            out_freq,
            mel_freq,
            middle_freq,
            batch_size,
            mel.stride(0),
            mel.stride(2),
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
        )
        # Undo `_build_selection`: each bin accumulated into the window columns it read from.
        grad_depth_weight = (_as_window(acc_low)[..., :_KERNEL] + _as_window(acc_high)[..., _STRIDE:]).reshape(
            channels, _TAPS
        )
        conv_w_dtype, conv_b_dtype, depth_w_dtype, depth_b_dtype = ctx.param_dtypes
        return (
            None,  # mel
            grad_conv_weight[:, :_TAPS].view(channels, 1, _KERNEL, _KERNEL).to(conv_w_dtype),
            grad_conv_bias.to(conv_b_dtype),
            grad_depth_weight.view(channels, 1, _KERNEL, _KERNEL).to(depth_w_dtype),
            grad_depth_bias.to(depth_b_dtype),
            None,  # mel_lengths
            None,  # middle_lengths
            None,  # out_lengths
            None,  # pad_left
            None,  # dims
            None,  # dtype
        )


def fused_conv_relu_dw(
    mel: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    depth_weight: torch.Tensor,
    depth_bias: torch.Tensor,
    left_padding: int,
    right_padding: int,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse ``conv -> ReLU -> depthwise``, with the between-stage length masking folded in.

    Both convolutions are 3x3 stride 2. The intermediate is never written to memory, and padding
    is index arithmetic inside the kernel rather than a materialised copy, so symmetric and
    causal (asymmetric ``(2, 1)``) padding take the same path.

    Args:
        mel: ``(batch, 1, time, freq)`` features, channel-first and unpadded.
        conv_weight: first convolution weight, ``(channels, 1, 3, 3)``.
        conv_bias: first convolution bias, ``(channels,)``.
        depth_weight: depthwise convolution weight, ``(channels, 1, 3, 3)``.
        depth_bias: depthwise convolution bias, ``(channels,)``.
        left_padding: padding applied to the low edge of both axes.
        right_padding: padding applied to the high edge; only affects the output extent.
        lengths: ``(batch,)`` valid time steps.

    Returns:
        ``(output, out_lengths)`` where output is ``(batch, out_time, out_freq, channels)``
        channels-last -- the layout the following 1x1 convolution wants.

    Raises:
        RuntimeError: if triton is unavailable or the inputs are not on a CUDA device.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is required for the fused subsampling kernel")
    if not mel.is_cuda:
        raise RuntimeError("The fused subsampling kernel requires CUDA tensors")
    batch_size, _, mel_time, mel_freq = mel.shape
    pad_total = left_padding + right_padding
    # Both convolutions downsample both axes, so every extent is two stages deep.
    middle_time = _downsampled_length(mel_time, pad_total)
    middle_freq = _downsampled_length(mel_freq, pad_total)
    dims = (
        _downsampled_length(middle_time, pad_total),
        _downsampled_length(middle_freq, pad_total),
        middle_freq,
    )
    middle_lengths = _downsampled_length(lengths, pad_total)
    out_lengths = _downsampled_length(middle_lengths, pad_total)
    # Autocast only rewrites aten ops, never a Triton launch, so the cast is done by hand.
    dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else mel.dtype
    with torch.autocast("cuda", enabled=False):
        output = _FusedSubsampling.apply(
            mel,
            conv_weight,
            conv_bias,
            depth_weight,
            depth_bias,
            lengths,
            middle_lengths,
            out_lengths,
            left_padding,
            dims,
            dtype,
        )
    return output, out_lengths
