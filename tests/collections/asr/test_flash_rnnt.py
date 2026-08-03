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

import pytest
import torch

from nemo.collections.asr.losses.flash_rnnt import FlashRNNTLoss, _balanced_time_tile_size, _validate_joint
from nemo.collections.asr.losses.rnnt import NUMBA_RNNT_AVAILABLE, RNNT_LOSS_RESOLVER, RNNTLoss
from nemo.collections.asr.modules.hybrid_autoregressive_transducer import HATJoint
from nemo.collections.asr.modules.rnnt import RNNTJoint
from nemo.collections.asr.parts.triton.rnnt_joint import joint_hidden_state
from nemo.collections.asr.parts.triton.rnnt_loss import MAX_TARGET_TOKENS, rnnt_loss_triton
from nemo.core.utils.optional_libs import TRITON_AVAILABLE, TRITON_INSTALLATION_MESSAGE

CUDA_TRITON_AVAILABLE = TRITON_AVAILABLE and torch.cuda.is_available()

if TRITON_AVAILABLE:
    # This module imports Triton at the top level, so only reach for it once Triton is known good.
    from nemo.collections.asr.parts.triton.rnnt_logprobs import rnnt_logprobs_triton


def _dense_rnnt_loss(logits, labels, source_lengths, target_lengths, blank, fastemit_lambda=0.0, clamp=-1.0):
    """Reference loss over dense logits, using the kernels the flash path composes.

    Flash trades the dense ``[B, T, U + 1, V]`` tensor for chunking, tiling and recomputation.
    Running the same extraction and dynamic programming over materialized logits isolates those
    mechanics, so a mismatch points at the chunking rather than at the kernels.
    """
    # Clamping needs the unit scale that autograd folds into the score gradients; the loss
    # backward publishes it here for the extraction backward to divide out.
    upstream_scale = torch.zeros(logits.shape[0], device=logits.device) if clamp > 0.0 else None
    target_scores, blank_scores = rnnt_logprobs_triton(
        logits,
        labels,
        blank,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        clamp=clamp,
        upstream_scale=upstream_scale,
    )
    return rnnt_loss_triton(
        target_scores[..., :-1],
        blank_scores,
        source_lengths,
        target_lengths,
        fastemit_lambda=fastemit_lambda,
        upstream_scale=upstream_scale,
    )


@pytest.mark.unit
def test_flash_rnnt_resolver_tracks_triton_availability():
    config = RNNT_LOSS_RESOLVER["flash_rnnt"]
    assert config.is_available is TRITON_AVAILABLE
    assert config.installation_msg == TRITON_INSTALLATION_MESSAGE


@pytest.mark.unit
@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is required")
def test_flash_rnnt_configuration():
    loss = RNNTLoss(
        num_classes=1023,
        loss_name="flash_rnnt",
        loss_kwargs={
            "fastemit_lambda": 0.01,
            "clamp": -1.0,
            "max_target_tokens": 2047,
            "max_joint_rows": 12345,
        },
    )

    assert loss.requires_factorized_joint
    assert loss._loss.blank == 1023
    assert loss._loss.fastemit_lambda == 0.01
    assert loss._loss.clamp == 0.0
    assert loss._loss.max_target_tokens == 2047
    assert loss._loss.max_joint_rows == 12345


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_target_tokens": MAX_TARGET_TOKENS + 1}, "max_target_tokens must be in"),
        ({"max_joint_rows": 0}, "max_joint_rows must be positive"),
        ({"fastemit_lambda": -0.01}, "fastemit_lambda must be nonnegative"),
    ],
)
def test_flash_rnnt_rejects_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FlashRNNTLoss(blank=1023, **kwargs)


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton is required")
@pytest.mark.unit
def test_flash_rnnt_rejects_dense_path():
    loss = RNNTLoss(num_classes=1023, loss_name="flash_rnnt")

    with pytest.raises(RuntimeError, match="fuse_loss_wer=true"):
        loss(
            log_probs=torch.empty(1, 1, 1, 1024),
            targets=torch.empty(1, 0, dtype=torch.long),
            input_lengths=torch.ones(1, dtype=torch.long),
            target_lengths=torch.zeros(1, dtype=torch.long),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("source_steps", "chunk_batch", "target_states", "max_joint_rows", "expected"),
    [
        (430, 16, 129, 206400, 86),
        (20, 4, 5, 200_000, 20),
        (400, 32, 129, 100, 1),
    ],
)
def test_flash_rnnt_balances_time_tiles(
    source_steps,
    chunk_batch,
    target_states,
    max_joint_rows,
    expected,
):
    assert _balanced_time_tile_size(source_steps, chunk_batch, target_states, max_joint_rows) == expected


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_rnnt_logprobs_backward_accepts_noncontiguous_score_gradients():
    torch.manual_seed(5)
    expected_logits = torch.randn(2, 6, 5, 8, device="cuda", requires_grad=True)
    actual_logits = expected_logits.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 7, (2, 4), device="cuda")
    source_lengths = torch.tensor([6, 4], device="cuda")
    target_lengths = torch.tensor([4, 2], device="cuda")

    expected_scores = rnnt_logprobs_triton(
        expected_logits,
        targets,
        blank_id=7,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
    )
    actual_scores = rnnt_logprobs_triton(
        actual_logits,
        targets,
        blank_id=7,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
    )
    gradient_storage = tuple(
        torch.randn(score.shape + (2,), device=score.device, dtype=score.dtype) for score in actual_scores
    )
    score_gradients = tuple(storage[..., 0] for storage in gradient_storage)
    assert all(not gradient.is_contiguous() for gradient in score_gradients)

    expected_gradient = torch.autograd.grad(
        expected_scores,
        expected_logits,
        tuple(gradient.contiguous() for gradient in score_gradients),
    )[0]
    actual_gradient = torch.autograd.grad(actual_scores, actual_logits, score_gradients)[0]

    torch.testing.assert_close(actual_gradient, expected_gradient, atol=0.0, rtol=0.0)


@pytest.mark.unit
def test_flash_rnnt_rejects_hat_joint_without_reading_blank_out_of_bounds():
    jointnet = {
        "encoder_hidden": 6,
        "pred_hidden": 7,
        "joint_hidden": 8,
        "activation": "relu",
    }
    standard_joint = RNNTJoint(jointnet=jointnet, num_classes=7, log_softmax=False)
    _validate_joint(standard_joint, blank=7)

    hat_joint = HATJoint(jointnet=jointnet, num_classes=7, log_softmax=False)
    assert hat_joint.joint_net[-1].out_features == 7
    assert hat_joint.num_classes_with_blank == 8
    with pytest.raises(ValueError, match="include every label and the blank"):
        _validate_joint(hat_joint, blank=7)


@pytest.mark.unit
def test_rnnt_logprobs_rejects_invalid_pointer_layout_before_launch():
    logits = torch.empty(2, 3, 5, 8)
    targets = torch.zeros(2, 4, dtype=torch.int64)
    with pytest.raises(ValueError, match="blank_id=8"):
        rnnt_logprobs_triton(logits, targets, blank_id=8)
    with pytest.raises(ValueError, match="targets must have shape"):
        rnnt_logprobs_triton(logits, torch.zeros(2, 5, dtype=torch.int64), blank_id=7)
    with pytest.raises(ValueError, match="source_lengths must have shape"):
        rnnt_logprobs_triton(logits, targets, blank_id=7, source_lengths=torch.ones(3, dtype=torch.int64))

    noncontiguous_logits = torch.empty(2, 5, 3, 8).transpose(1, 2)
    with pytest.raises(ValueError, match="logits must be contiguous"):
        rnnt_logprobs_triton(noncontiguous_logits, targets, blank_id=7)


@pytest.mark.unit
def test_flash_rnnt_early_return_preserves_requested_empty_hypotheses():
    class FlashLoss:
        requires_factorized_joint = True

    joint = RNNTJoint(
        jointnet={
            "encoder_hidden": 6,
            "pred_hidden": 7,
            "joint_hidden": 8,
            "activation": "relu",
        },
        num_classes=7,
        log_softmax=False,
        fuse_loss_wer=True,
        fused_batch_size=1,
    )
    joint.set_loss(FlashLoss())
    joint.set_wer(object())
    result = joint(
        encoder_outputs=torch.zeros(2, 6, 3),
        decoder_outputs=None,
        encoder_lengths=torch.tensor([3, 2]),
        transcript_lengths=torch.tensor([0, 0]),
        keep_hypotheses=True,
    )

    assert result == (None, None, None, None)
    assert joint.get_hypotheses() == []


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
def test_flash_rnnt_join_takes_one_code_path_for_every_chunk_shape(activation):
    """Chunk views carry padded-batch strides and every step brings new extents.

    A shape-specialized ``torch.compile`` cache would exhaust Dynamo's recompile limit here
    and then silently fall back to eager, which changes reduced-precision rounding part-way
    through a run. Assert the join compiles nothing, so there is only ever one code path.
    """
    recompile_limit = torch._dynamo.config.recompile_limit
    torch._dynamo.reset()
    torch._dynamo.config.recompile_limit = 1
    try:
        for padded_target in range(4, 13):
            used_target = max(4, padded_target - 1)
            encoder = torch.randn(2, padded_target - 1, 8, device="cuda", requires_grad=True)
            predictor_storage = torch.randn(2, padded_target, 8, device="cuda", requires_grad=True)
            predictor = predictor_storage[:, :used_target]
            hidden = torch.utils.checkpoint.checkpoint(
                joint_hidden_state,
                encoder,
                predictor,
                activation,
                use_reentrant=False,
            )
            hidden.sum().backward()
            assert encoder.grad.shape == encoder.shape
            assert predictor_storage.grad.shape == predictor_storage.shape
        assert torch._dynamo.utils.counters["frames"]["ok"] == 0, "the joint activation must not invoke Dynamo"
    finally:
        torch._dynamo.config.recompile_limit = recompile_limit
        torch._dynamo.reset()


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_rnnt_join_matches_the_eager_broadcast_add(activation, dtype):
    """The fused kernel must compute the same function as an eager broadcast-add plus activation.

    It keeps the pre-activation in FP32 where eager rounds it to the input dtype, so in BF16 the
    two agree only to about one unit in the last place -- and for ReLU a rounded pre-activation can
    cross zero, flipping the derivative mask outright. Both effects are inherent to the precision
    rather than to the kernel, so compare at matched dtype with a precision-appropriate tolerance.
    """
    torch.manual_seed(7)
    encoder = torch.randn(3, 7, 129, device="cuda", dtype=dtype)
    predictor = torch.randn(3, 5, 129, device="cuda", dtype=dtype)
    upstream = torch.randn(3, 7, 5, 129, device="cuda", dtype=dtype)

    def forward_and_gradients(join):
        leaf_encoder = encoder.detach().clone().requires_grad_(True)
        leaf_predictor = predictor.detach().clone().requires_grad_(True)
        hidden = join(leaf_encoder, leaf_predictor)
        return hidden, torch.autograd.grad(hidden, (leaf_encoder, leaf_predictor), upstream)

    def eager(leaf_encoder, leaf_predictor):
        activate = {"relu": torch.relu, "sigmoid": torch.sigmoid, "tanh": torch.tanh}[activation]
        return activate(leaf_encoder.unsqueeze(2) + leaf_predictor.unsqueeze(1))

    fused_hidden, fused_gradients = forward_and_gradients(lambda e, p: joint_hidden_state(e, p, activation))
    eager_hidden, eager_gradients = forward_and_gradients(eager)

    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (3e-2, 3e-2)
    torch.testing.assert_close(fused_hidden, eager_hidden, atol=atol, rtol=rtol)
    for actual, expected in zip(fused_gradients, eager_gradients):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def _rnnt_loss_torch(logits, labels, source_lengths, target_lengths, blank):
    log_probs = logits.float().log_softmax(dim=-1)
    losses = []
    for batch_idx in range(logits.shape[0]):
        source_len = int(source_lengths[batch_idx])
        target_len = int(target_lengths[batch_idx])
        alpha = log_probs.new_full((target_len + 1,), -torch.inf)
        alpha[0] = 0.0
        for time_idx in range(source_len):
            if time_idx:
                alpha = alpha + log_probs[batch_idx, time_idx - 1, : target_len + 1, blank]
            values = [alpha[0]]
            for target_idx in range(target_len):
                target_score = log_probs[batch_idx, time_idx, target_idx, labels[batch_idx, target_idx]]
                values.append(torch.logaddexp(alpha[target_idx + 1], values[-1] + target_score))
            alpha = torch.stack(values)
        losses.append(-(alpha[-1] + log_probs[batch_idx, source_len - 1, target_len, blank]))
    return torch.stack(losses)


def _inputs(blank, dtype=torch.float32, vocab=8):
    torch.manual_seed(7)
    batch, source, target = 3, 6, 4
    logits = torch.randn(batch, source, target + 1, vocab, device="cuda", dtype=dtype, requires_grad=True)
    low = 1 if blank == 0 else 0
    high = vocab if blank == 0 else vocab - 1
    labels = torch.randint(low, high, (batch, target), device="cuda")
    source_lengths = torch.tensor([6, 4, 3], device="cuda", dtype=torch.int64)
    target_lengths = torch.tensor([4, 2, 0], device="cuda", dtype=torch.int64)
    return logits, labels, source_lengths, target_lengths


@pytest.mark.unit
def test_flash_rnnt_requires_triton(monkeypatch):
    monkeypatch.setattr("nemo.collections.asr.losses.flash_rnnt.TRITON_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="Triton is required"):
        FlashRNNTLoss(blank=7)(
            joint=object(),
            encoder=torch.zeros(1, 2, 3),
            predictor=torch.zeros(1, 2, 3),
            targets=torch.zeros(1, 1, dtype=torch.long),
            source_lengths=torch.ones(1, dtype=torch.long),
            target_lengths=torch.ones(1, dtype=torch.long),
            max_samples_per_chunk=1,
        )


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize(("blank", "vocab"), [(0, 8), (7, 8), (0, 9), (8, 9)])
def test_dense_rnnt_matches_torch_loss_and_gradient(blank, vocab):
    logits, labels, source_lengths, target_lengths = _inputs(blank, vocab=vocab)
    original_logits = logits.detach().clone()
    reference_logits = logits.detach().clone().requires_grad_(True)

    native_loss = _dense_rnnt_loss(logits, labels, source_lengths, target_lengths, blank)
    reference_loss = _rnnt_loss_torch(reference_logits, labels, source_lengths, target_lengths, blank)
    native_grad = torch.autograd.grad(native_loss.sum(), logits)[0]
    reference_grad = torch.autograd.grad(reference_loss.sum(), reference_logits)[0]

    torch.testing.assert_close(native_loss, reference_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(native_grad, reference_grad, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(logits, original_logits, atol=0.0, rtol=0.0)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_rnnt_dp_supports_target_length_above_1023():
    torch.manual_seed(23)
    source, target, vocab, blank = 2, 1024, 8, 7
    logits = torch.randn(1, source, target + 1, vocab, device="cuda", requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_(True)
    labels = torch.randint(0, blank, (1, target), device="cuda")
    source_lengths = torch.tensor([source], device="cuda")
    target_lengths = torch.tensor([target], device="cuda")

    actual = _dense_rnnt_loss(logits, labels, source_lengths, target_lengths, blank)
    expected = _rnnt_loss_torch(reference_logits, labels, source_lengths, target_lengths, blank)
    actual_grad = torch.autograd.grad(actual.sum(), logits)[0]
    expected_grad = torch.autograd.grad(expected.sum(), reference_logits)[0]
    relative_grad_error = torch.linalg.vector_norm(actual_grad - expected_grad) / torch.linalg.vector_norm(
        expected_grad
    )

    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-6)
    assert relative_grad_error < 1e-3


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_rnnt_logprobs_can_reuse_private_logits_for_gradient():
    logits, labels, source_lengths, target_lengths = _inputs(blank=7)
    reference_logits = logits.detach().clone().requires_grad_(True)
    target_scores, blank_scores = rnnt_logprobs_triton(
        logits,
        labels,
        blank_id=7,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        reuse_logits_for_grad=True,
    )
    loss = rnnt_loss_triton(target_scores[..., :-1], blank_scores, source_lengths, target_lengths)
    reference_loss = _dense_rnnt_loss(reference_logits, labels, source_lengths, target_lengths, 7)
    grad = torch.autograd.grad(loss.sum(), logits)[0]
    reference_grad = torch.autograd.grad(reference_loss.sum(), reference_logits)[0]

    torch.testing.assert_close(loss, reference_loss, atol=0.0, rtol=0.0)
    torch.testing.assert_close(grad, reference_grad, atol=0.0, rtol=0.0)
    assert logits.data_ptr() == grad.data_ptr()


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_rnnt_logprobs_reused_logits_reject_second_backward():
    logits, labels, source_lengths, target_lengths = _inputs(blank=7)
    target_scores, blank_scores = rnnt_logprobs_triton(
        logits,
        labels,
        blank_id=7,
        source_lengths=source_lengths,
        target_lengths=target_lengths,
        reuse_logits_for_grad=True,
    )
    loss = rnnt_loss_triton(target_scores[..., :-1], blank_scores, source_lengths, target_lengths).sum()

    torch.autograd.grad(loss, logits, retain_graph=True)
    with pytest.raises(RuntimeError, match="only supports one backward pass"):
        torch.autograd.grad(loss, logits)


@pytest.mark.unit
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE or not NUMBA_RNNT_AVAILABLE,
    reason="CUDA, Triton, and Numba RNN-T are required",
)
@pytest.mark.parametrize(("fastemit_lambda", "clamp"), [(0.0, -1.0), (0.01, -1.0), (0.01, 0.02)])
@pytest.mark.parametrize("blank", [0, 7])
def test_dense_rnnt_matches_numba_fastemit_and_clamp(fastemit_lambda, clamp, blank):
    from nemo.collections.asr.parts.numba.rnnt_loss import RNNTLossNumba

    logits, labels, source_lengths, target_lengths = _inputs(blank=blank)
    reference_logits = logits.detach().clone().requires_grad_(True)
    reference = RNNTLossNumba(
        blank=blank,
        reduction="none",
        fastemit_lambda=fastemit_lambda,
        clamp=clamp,
    )

    native_loss = _dense_rnnt_loss(
        logits, labels, source_lengths, target_lengths, blank, fastemit_lambda=fastemit_lambda, clamp=clamp
    )
    reference_loss = reference(reference_logits, labels, source_lengths, target_lengths)
    native_grad = torch.autograd.grad(native_loss.sum(), logits)[0]
    reference_grad = torch.autograd.grad(reference_loss.sum(), reference_logits)[0]

    torch.testing.assert_close(native_loss, reference_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(native_grad, reference_grad, atol=2e-5, rtol=2e-4)


@pytest.mark.unit
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE or not NUMBA_RNNT_AVAILABLE,
    reason="CUDA, Triton, and Numba RNN-T are required",
)
@pytest.mark.parametrize(("upstream", "amp_scale"), [("mean", 1.0), ("mean", 1024.0), ("weighted", 1.0)])
def test_dense_rnnt_clamp_precedes_upstream_scaling(upstream, amp_scale):
    from nemo.collections.asr.parts.numba.rnnt_loss import RNNTLossNumba

    logits, labels, source_lengths, target_lengths = _inputs(blank=7)
    reference_logits = logits.detach().clone().requires_grad_(True)
    native_loss = _dense_rnnt_loss(logits, labels, source_lengths, target_lengths, 7, fastemit_lambda=0.01, clamp=0.02)
    reference_loss = RNNTLossNumba(
        blank=7,
        reduction="none",
        fastemit_lambda=0.01,
        clamp=0.02,
    )(reference_logits, labels, source_lengths, target_lengths)

    if upstream == "mean":
        native_objective = native_loss.mean()
        reference_objective = reference_loss.mean()
    else:
        weights = torch.tensor([0.5, 0.0, -2.0], device="cuda")
        native_objective = (native_loss * weights).sum()
        reference_objective = (reference_loss * weights).sum()

    native_grad = torch.autograd.grad(native_objective * amp_scale, logits)[0] / amp_scale
    reference_grad = torch.autograd.grad(reference_objective * amp_scale, reference_logits)[0] / amp_scale

    torch.testing.assert_close(native_grad, reference_grad, atol=2e-5, rtol=2e-4)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_rnnt_mixed_precision_matches_float32(dtype):
    logits, labels, source_lengths, target_lengths = _inputs(blank=7, dtype=dtype)
    reference_logits = logits.detach().float().requires_grad_(True)

    loss = _dense_rnnt_loss(logits, labels, source_lengths, target_lengths, 7)
    reference_loss = _dense_rnnt_loss(reference_logits, labels, source_lengths, target_lengths, 7)
    grad = torch.autograd.grad(loss.sum(), logits)[0].float()
    reference_grad = torch.autograd.grad(reference_loss.sum(), reference_logits)[0]

    torch.testing.assert_close(loss, reference_loss, atol=2e-3, rtol=2e-3)
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
    relative_error = torch.linalg.vector_norm(grad - reference_grad) / torch.linalg.vector_norm(reference_grad)
    assert relative_error < 2e-3


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_dense_rnnt_target_limit_fails_before_extraction():
    logits = torch.empty(1, 1, MAX_TARGET_TOKENS + 2, 3, device="cuda")
    labels = torch.zeros(1, MAX_TARGET_TOKENS + 1, dtype=torch.long, device="cuda")
    lengths = torch.ones(1, dtype=torch.long, device="cuda")
    with pytest.raises(ValueError, match=str(MAX_TARGET_TOKENS)):
        _dense_rnnt_loss(logits, labels, lengths, torch.full_like(lengths, MAX_TARGET_TOKENS + 1), blank=2)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("reduction", [None, "sum", "mean_batch", "mean", "mean_volume"])
def test_flash_rnnt_reductions(reduction):
    logits, labels, source_lengths, _ = _inputs(blank=7)
    target_lengths = torch.tensor([4, 2, 1], device="cuda")
    per_sample = _dense_rnnt_loss(logits, labels, source_lengths, target_lengths, 7)
    if reduction is None:
        expected = per_sample
    elif reduction == "sum":
        expected = per_sample.sum()
    elif reduction == "mean_batch":
        expected = per_sample.mean()
    elif reduction == "mean":
        expected = (per_sample / target_lengths).mean()
    else:
        expected = per_sample.sum() / target_lengths.sum()

    loss = RNNTLoss(num_classes=7, reduction=reduction, loss_name="flash_rnnt")
    actual = loss.reduce(per_sample, target_lengths)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _make_joint(fused_batch_size, activation="relu", log_softmax=False, dropout=0.0):
    return RNNTJoint(
        jointnet={
            "encoder_hidden": 6,
            "pred_hidden": 7,
            "joint_hidden": 8,
            "activation": activation,
            "dropout": dropout,
        },
        num_classes=7,
        log_softmax=log_softmax,
        fuse_loss_wer=True,
        fused_batch_size=fused_batch_size,
    ).cuda()


@pytest.mark.unit
@pytest.mark.skipif(
    not CUDA_TRITON_AVAILABLE or not NUMBA_RNNT_AVAILABLE,
    reason="CUDA, Triton, and Numba RNN-T are required",
)
@pytest.mark.parametrize("fused_batch_size", [2, 3])
@pytest.mark.parametrize(("upstream", "amp_scale"), [("mean_batch", 1.0), ("mean_batch", 1024.0), ("weighted", 1.0)])
def test_flash_rnnt_clamp_matches_numba_across_chunks(fused_batch_size, upstream, amp_scale):
    """Clamping applies to the unit-scale gradient, which autograd has already scaled.

    Dividing that scale back out has to survive batch chunking, source-time tiling, the loss
    reduction and the AMP scale at once, so compare against warprnnt_numba, which clamps the same
    unit-scale gradient, rather than against another Triton path sharing the same plumbing.
    """
    from nemo.collections.asr.parts.numba.rnnt_loss import RNNTLossNumba

    torch.manual_seed(23)
    joint = _make_joint(fused_batch_size)
    for parameter in joint.parameters():
        torch.nn.init.normal_(parameter, std=0.3)
    max_joint_rows = 4
    flash_loss = RNNTLoss(
        num_classes=7,
        reduction=None,
        loss_name="flash_rnnt",
        loss_kwargs={"fastemit_lambda": 0.01, "clamp": 0.02, "max_joint_rows": max_joint_rows},
    )
    # A budget that stopped splitting source time would leave the scale plumbing untested.
    assert _balanced_time_tile_size(9, fused_batch_size, 6, max_joint_rows) < 9
    joint.set_loss(flash_loss)
    joint.set_wer(object())

    encoder = torch.randn(6, 6, 9, device="cuda")
    predictor = torch.randn(6, 7, 6, device="cuda")
    source_lengths = torch.tensor([9, 8, 6, 5, 3, 2], device="cuda")
    target_lengths = torch.tensor([5, 4, 3, 2, 1, 1], device="cuda")
    labels = torch.randint(0, 7, (6, 5), device="cuda")

    def objective(per_sample):
        if upstream == "mean_batch":
            return per_sample.mean() * amp_scale
        weights = torch.tensor([0.5, 0.0, -2.0, 1.0, 0.25, -0.75], device="cuda")
        return (per_sample * weights).sum() * amp_scale

    flash_encoder = encoder.clone().requires_grad_(True)
    flash_predictor = predictor.clone().requires_grad_(True)
    flash_costs = joint(
        encoder_outputs=flash_encoder,
        decoder_outputs=flash_predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    flash_gradients = torch.autograd.grad(
        objective(flash_costs), (flash_encoder, flash_predictor, *joint.parameters())
    )

    dense_encoder = encoder.clone().requires_grad_(True)
    dense_predictor = predictor.clone().requires_grad_(True)
    logits = joint.joint(dense_encoder.transpose(1, 2), dense_predictor.transpose(1, 2))
    dense_costs = RNNTLossNumba(blank=7, reduction="none", fastemit_lambda=0.01, clamp=0.02)(
        logits, labels, source_lengths, target_lengths
    )
    dense_gradients = torch.autograd.grad(
        objective(dense_costs), (dense_encoder, dense_predictor, *joint.parameters())
    )

    torch.testing.assert_close(flash_costs, dense_costs, atol=1e-5, rtol=1e-5)
    for actual, expected in zip(flash_gradients, dense_gradients):
        torch.testing.assert_close(actual / amp_scale, expected / amp_scale, atol=2e-5, rtol=2e-4)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("hidden_size", [6, 8, 1536])
def test_joint_hidden_state_dropout_mask_agrees_across_kernels(hidden_size):
    """The forward and both backward kernels must reach the same verdict on every element.

    Nothing stores the mask and the three kernels walk the joint grid along different axes, so each
    redraws it from the element's position. Under relu with a positive pre-activation the activation
    derivative is exactly 1, which reduces each backward under a grad_output of ones to the
    mask-weighted sum of ones along its axis -- that is, the sum of the forward output. One element
    the kernels disagree on breaks the identity exactly.

    ``hidden_size`` covers a size that is not a multiple of four, one block, and several blocks.
    """
    torch.manual_seed(53)
    dropout_p = 0.25
    encoder = torch.ones(2, 5, hidden_size, device="cuda", requires_grad=True)
    predictor = torch.zeros(2, 4, hidden_size, device="cuda", requires_grad=True)

    hidden = joint_hidden_state(encoder, predictor, "relu", dropout_p)
    hidden.backward(torch.ones_like(hidden))

    torch.testing.assert_close(encoder.grad, hidden.sum(dim=2))
    torch.testing.assert_close(predictor.grad, hidden.sum(dim=1))

    survivors = hidden[hidden != 0.0]
    torch.testing.assert_close(survivors, torch.full_like(survivors, 1.0 / (1.0 - dropout_p)))
    assert 0.6 < survivors.numel() / hidden.numel() < 0.9


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_joint_hidden_state_rejects_invalid_dropout():
    encoder = torch.randn(1, 2, 8, device="cuda")
    predictor = torch.randn(1, 3, 8, device="cuda")
    with pytest.raises(ValueError, match=r"dropout_p must be in \[0, 1\)"):
        joint_hidden_state(encoder, predictor, "relu", 1.0)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_flash_rnnt_joint_dropout_is_deterministic_across_recomputation():
    torch.manual_seed(131)
    joint = _make_joint(1, activation="tanh", dropout=0.25)
    loss = RNNTLoss(
        num_classes=7,
        reduction="mean_batch",
        loss_name="flash_rnnt",
        loss_kwargs={"max_joint_rows": 4},
    )
    joint.set_loss(loss)
    joint.set_wer(object())
    encoder = torch.randn(3, 6, 5, device="cuda")
    predictor = torch.randn(3, 7, 4, device="cuda")
    source_lengths = torch.tensor([5, 4, 3], device="cuda")
    target_lengths = torch.tensor([3, 2, 1], device="cuda")
    labels = torch.randint(0, 7, (3, 3), device="cuda")

    def run():
        run_encoder = encoder.detach().clone().requires_grad_(True)
        run_predictor = predictor.detach().clone().requires_grad_(True)
        torch.manual_seed(149)
        value = joint(
            encoder_outputs=run_encoder,
            decoder_outputs=run_predictor,
            encoder_lengths=source_lengths,
            transcripts=labels,
            transcript_lengths=target_lengths,
        )[0]
        gradients = torch.autograd.grad(value, (run_encoder, run_predictor, *joint.parameters()))
        return value, gradients

    first_value, first_gradients = run()
    second_value, second_gradients = run()
    torch.testing.assert_close(first_value, second_value, atol=0.0, rtol=0.0)
    for first, second in zip(first_gradients, second_gradients):
        torch.testing.assert_close(first, second, atol=0.0, rtol=0.0)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("fused_batch_size", [1, 3])
def test_flash_rnnt_workspace_batch_equivalence(fused_batch_size):
    torch.manual_seed(11)
    reference_joint = _make_joint(4)
    joint = _make_joint(fused_batch_size)
    joint.load_state_dict(reference_joint.state_dict())
    loss = RNNTLoss(num_classes=7, reduction="mean_batch", loss_name="flash_rnnt")
    reference_loss = RNNTLoss(num_classes=7, reduction="mean_batch", loss_name="flash_rnnt")
    joint.set_loss(loss)
    reference_joint.set_loss(reference_loss)
    joint.set_wer(object())
    reference_joint.set_wer(object())

    encoder = torch.randn(4, 6, 7, device="cuda", requires_grad=True)
    predictor = torch.randn(4, 7, 5, device="cuda", requires_grad=True)
    reference_encoder = encoder.detach().clone().requires_grad_(True)
    reference_predictor = predictor.detach().clone().requires_grad_(True)
    source_lengths = torch.tensor([7, 6, 4, 3], device="cuda")
    target_lengths = torch.tensor([4, 3, 2, 0], device="cuda")
    labels = torch.randint(0, 7, (4, 4), device="cuda")

    value = joint(
        encoder_outputs=encoder,
        decoder_outputs=predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    reference_value = reference_joint(
        encoder_outputs=reference_encoder,
        decoder_outputs=reference_predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    gradients = torch.autograd.grad(value, (encoder, predictor, *joint.parameters()))
    reference_gradients = torch.autograd.grad(
        reference_value,
        (reference_encoder, reference_predictor, *reference_joint.parameters()),
    )

    torch.testing.assert_close(value, reference_value, atol=1e-5, rtol=1e-5)
    for actual, expected in zip(gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_rnnt_workspace_time_equivalence(dtype):
    torch.manual_seed(17)
    reference_joint = _make_joint(4).to(dtype)
    tiled_joint = _make_joint(4).to(dtype)
    tiled_joint.load_state_dict(reference_joint.state_dict())
    reference_loss = RNNTLoss(
        num_classes=7,
        reduction="mean_batch",
        loss_name="flash_rnnt",
        loss_kwargs={"max_joint_rows": 10_000},
    )
    tiled_loss = RNNTLoss(
        num_classes=7,
        reduction="mean_batch",
        loss_name="flash_rnnt",
        loss_kwargs={"max_joint_rows": 40},
    )
    reference_joint.set_loss(reference_loss)
    tiled_joint.set_loss(tiled_loss)
    reference_joint.set_wer(object())
    tiled_joint.set_wer(object())

    encoder = torch.randn(4, 6, 7, device="cuda", dtype=dtype, requires_grad=True)
    predictor = torch.randn(4, 7, 5, device="cuda", dtype=dtype, requires_grad=True)
    tiled_encoder = encoder.detach().clone().requires_grad_(True)
    tiled_predictor = predictor.detach().clone().requires_grad_(True)
    source_lengths = torch.tensor([7, 6, 4, 3], device="cuda")
    target_lengths = torch.tensor([4, 3, 2, 0], device="cuda")
    labels = torch.randint(0, 7, (4, 4), device="cuda")

    reference_value = reference_joint(
        encoder_outputs=encoder,
        decoder_outputs=predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    tiled_value = tiled_joint(
        encoder_outputs=tiled_encoder,
        decoder_outputs=tiled_predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    reference_gradients = torch.autograd.grad(reference_value, (encoder, predictor, *reference_joint.parameters()))
    tiled_gradients = torch.autograd.grad(
        tiled_value,
        (tiled_encoder, tiled_predictor, *tiled_joint.parameters()),
    )

    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(tiled_value, reference_value, atol=atol, rtol=rtol)
    for actual, expected in zip(tiled_gradients, reference_gradients):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
@pytest.mark.parametrize("activation", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flash_rnnt_matches_dense_native_loss_and_gradients(dtype, activation):
    torch.manual_seed(19)
    dense_joint = _make_joint(4, activation=activation).to(dtype)
    flash_joint = _make_joint(4, activation=activation).to(dtype)
    flash_joint.load_state_dict(dense_joint.state_dict())
    flash_loss = RNNTLoss(
        num_classes=7,
        reduction="mean_batch",
        loss_name="flash_rnnt",
        loss_kwargs={"fastemit_lambda": 0.01},
    )
    flash_joint.set_loss(flash_loss)
    flash_joint.set_wer(object())
    encoder = torch.randn(4, 6, 7, device="cuda", dtype=dtype, requires_grad=True)
    predictor = torch.randn(4, 7, 5, device="cuda", dtype=dtype, requires_grad=True)
    flash_encoder = encoder.detach().clone().requires_grad_(True)
    flash_predictor = predictor.detach().clone().requires_grad_(True)
    source_lengths = torch.tensor([7, 6, 4, 3], device="cuda")
    target_lengths = torch.tensor([4, 3, 2, 0], device="cuda")
    labels = torch.randint(0, 7, (4, 4), device="cuda")

    logits = dense_joint.joint(encoder.transpose(1, 2), predictor.transpose(1, 2))
    dense_value = _dense_rnnt_loss(logits, labels, source_lengths, target_lengths, 7, fastemit_lambda=0.01).mean()
    flash_value = flash_joint(
        encoder_outputs=flash_encoder,
        decoder_outputs=flash_predictor,
        encoder_lengths=source_lengths,
        transcripts=labels,
        transcript_lengths=target_lengths,
    )[0]
    dense_gradients = torch.autograd.grad(dense_value, (encoder, predictor, *dense_joint.parameters()))
    flash_gradients = torch.autograd.grad(flash_value, (flash_encoder, flash_predictor, *flash_joint.parameters()))

    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(flash_value, dense_value, atol=atol, rtol=rtol)
    for actual, expected in zip(flash_gradients, dense_gradients):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.unit
@pytest.mark.skipif(not CUDA_TRITON_AVAILABLE, reason="CUDA and Triton are required")
def test_flash_rnnt_does_not_materialize_dense_joint_logits(monkeypatch):
    joint = _make_joint(1, activation="tanh")
    loss = RNNTLoss(num_classes=7, reduction="mean_batch", loss_name="flash_rnnt")

    def reject_dense_joint(*args, **kwargs):
        raise AssertionError("Flash RNN-T must not call RNNTJoint.joint")

    monkeypatch.setattr(joint, "joint", reject_dense_joint)
    joint.set_loss(loss)
    joint.set_wer(object())
    value = joint(
        encoder_outputs=torch.randn(1, 6, 3, device="cuda"),
        decoder_outputs=torch.randn(1, 7, 3, device="cuda"),
        encoder_lengths=torch.tensor([3], device="cuda"),
        transcripts=torch.randint(0, 7, (1, 2), device="cuda"),
        transcript_lengths=torch.tensor([2], device="cuda"),
    )[0]
    assert value.isfinite()
