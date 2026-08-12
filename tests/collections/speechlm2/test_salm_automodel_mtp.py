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

import nemo.collections.speechlm2.models.salm_automodel as salm_module

SALMAutomodel = salm_module.SALMAutomodel


def _bare_model():
    '''Create a SALMAutomodel instance without loading any weights.'''
    model = SALMAutomodel.__new__(SALMAutomodel)
    torch.nn.Module.__init__(model)
    return model


# ---------------------------------------------------------------------------
# _mtp_enabled property
# ---------------------------------------------------------------------------


def test_mtp_enabled_false_when_llm_missing():
    model = _bare_model()
    assert not model._mtp_enabled


def test_mtp_enabled_false_when_mtp_attr_missing():
    model = _bare_model()
    model.llm = torch.nn.Module()
    assert not model._mtp_enabled


def test_mtp_enabled_false_when_mtp_is_none():
    model = _bare_model()
    model.llm = torch.nn.Module()
    model.llm.mtp = None
    assert not model._mtp_enabled


def test_mtp_enabled_true_when_mtp_attached():
    model = _bare_model()
    model.llm = torch.nn.Module()
    model.llm.mtp = torch.nn.Linear(4, 4)
    assert model._mtp_enabled


def test_disabled_mtp_overrides_native_checkpoint_head(monkeypatch):
    """Native checkpoint MTP weights are not instantiated when MTP is disabled."""
    from omegaconf import DictConfig

    captured_kwargs = {}

    class _Perception(torch.nn.Module):
        def set_activation_checkpointing(self, _enabled):
            return None

    class _LLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": 8, "num_nextn_predict_layers": 1})()
            self.mtp = None

    def _load_llm(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return _LLM()

    model = _bare_model()
    model.cfg = DictConfig(
        {
            "pretrained_llm": "native-mtp-checkpoint",
            "pretrained_asr": "unused-in-test",
            "pretrained_weights": True,
            "mtp": {"enabled": False},
        }
    )
    model._trainer = None
    model._use_fsdp = False
    model._use_tp = False
    model.setup_moe_options = lambda: None

    monkeypatch.setattr(salm_module, "load_pretrained_automodel_llm", _load_llm)
    monkeypatch.setattr(
        salm_module, "setup_speech_encoder", lambda model, **_kwargs: setattr(model, "perception", _Perception())
    )
    monkeypatch.setattr(salm_module, "update_perception_output_dim", lambda _model: None)
    monkeypatch.setattr(salm_module, "maybe_load_pretrained_models", lambda _model: None)

    SALMAutomodel.configure_model(model)

    assert captured_kwargs["num_nextn_predict_layers"] == 0
    assert "mtp_config_overrides" not in captured_kwargs
    assert model.llm.config.num_nextn_predict_layers == 0
    assert not model._mtp_enabled


@pytest.mark.parametrize(
    ("training_mode", "replace_existing_head"),
    [("joint", False), ("head_only", False), ("joint", True)],
)
def test_configure_model_requests_mtp_and_applies_training_mode(monkeypatch, training_mode, replace_existing_head):
    """MTP loader options and parameter freezing follow the recipe configuration."""
    from omegaconf import DictConfig

    captured_kwargs = {}

    class _Perception(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Linear(4, 4)

        def set_activation_checkpointing(self, _enabled):
            return None

    class _LLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.config = type("Config", (), {"hidden_size": 4})()
            self.mtp = None

    model = _bare_model()
    model.cfg = DictConfig(
        {
            "pretrained_llm": "base-checkpoint-without-mtp",
            "pretrained_asr": "unused-in-test",
            "pretrained_weights": True,
            "freeze_params": ["^.+$"] if training_mode == "head_only" else [],
            "prevent_freeze_params": [],
            "mtp": {
                "enabled": True,
                "training_mode": training_mode,
                "replace_existing_head": replace_existing_head,
            },
        }
    )
    model._trainer = None
    model._use_fsdp = False
    model._use_tp = False
    model.setup_moe_options = lambda: None

    def _load_llm(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        llm = _LLM()
        llm.mtp = torch.nn.Linear(4, 4)
        return llm

    monkeypatch.setattr(salm_module, "load_pretrained_automodel_llm", _load_llm)
    monkeypatch.setattr(
        salm_module, "setup_speech_encoder", lambda model, **_kwargs: setattr(model, "perception", _Perception())
    )
    monkeypatch.setattr(salm_module, "update_perception_output_dim", lambda _model: None)
    monkeypatch.setattr(salm_module, "maybe_load_pretrained_models", lambda _model: None)

    SALMAutomodel.configure_model(model)

    assert captured_kwargs["mtp_config_overrides"] == {
        "num_nextn_predict_layers": 1,
        "mtp_hybrid_override_pattern": "*",
        "mtp_layers_block_type": None,
    }
    assert captured_kwargs["replace_mtp_config"] is replace_existing_head
    assert model._mtp_enabled
    assert all(param.requires_grad for param in model.llm.mtp.parameters())
    if training_mode == "head_only":
        assert all(not param.requires_grad for param in model.llm.backbone.parameters())
        assert all(not param.requires_grad for param in model.perception.parameters())
        assert r"^llm\.mtp\..+$" in model.cfg.prevent_freeze_params
        from nemo.collections.speechlm2.parts.optim_setup import freeze_and_subset

        optimizer_params = list(
            freeze_and_subset(model.named_parameters(), model.cfg.freeze_params, model.cfg.prevent_freeze_params)
        )
        assert {id(param) for param in optimizer_params} == {id(param) for param in model.llm.mtp.parameters()}
    else:
        assert all(param.requires_grad for param in model.llm.backbone.parameters())
        assert all(param.requires_grad for param in model.perception.parameters())


def test_invalid_mtp_training_mode_fails_before_loading(monkeypatch):
    from omegaconf import DictConfig

    model = _bare_model()
    model.cfg = DictConfig(
        {
            "pretrained_llm": "unused",
            "pretrained_asr": "unused",
            "mtp": {"enabled": True, "training_mode": "backbone_only"},
        }
    )
    model._trainer = None
    model._use_fsdp = False
    model._use_tp = False
    monkeypatch.setattr(
        salm_module,
        "load_pretrained_automodel_llm",
        lambda *_args, **_kwargs: pytest.fail("invalid mode must fail before loading"),
    )

    with pytest.raises(ValueError, match="mtp.training_mode"):
        SALMAutomodel.configure_model(model)


def test_repeated_layer_settings_reach_native_mtp_constructor(monkeypatch):
    """Reload preserves logical MTP depth when the checkpoint stores one physical repeated layer."""
    from omegaconf import DictConfig

    captured_kwargs = {}

    class _Perception(torch.nn.Module):
        def set_activation_checkpointing(self, _enabled):
            return None

    class _LLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Automodel consumes the logical constructor override into the HF
            # config before constructing the repeated physical head.
            self.config = type("Config", (), {"hidden_size": 4, "num_nextn_predict_layers": 3})()
            self.mtp_config = type("MTPConfig", (), {"num_layers": 3, "use_repeated_layer": True})()
            self.mtp = torch.nn.Linear(4, 4)

    def _load_llm(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return _LLM()

    model = _bare_model()
    model.cfg = DictConfig(
        {
            "pretrained_llm": "repeated-mtp-checkpoint",
            "pretrained_asr": "unused-in-test",
            "pretrained_weights": True,
            "freeze_params": [],
            "prevent_freeze_params": [],
            "mtp": {
                "enabled": True,
                "training_mode": "joint",
                "num_nextn_predict_layers": 3,
                "use_repeated_layer": True,
            },
        }
    )
    model._trainer = None
    model._use_fsdp = False
    model._use_tp = False
    model.setup_moe_options = lambda: None

    monkeypatch.setattr(salm_module, "load_pretrained_automodel_llm", _load_llm)
    monkeypatch.setattr(salm_module, "_build_mtp_loss_fn", lambda: torch.nn.Identity())
    monkeypatch.setattr(
        salm_module, "setup_speech_encoder", lambda model, **_kwargs: setattr(model, "perception", _Perception())
    )
    monkeypatch.setattr(salm_module, "update_perception_output_dim", lambda _model: None)
    monkeypatch.setattr(salm_module, "maybe_load_pretrained_models", lambda _model: None)

    SALMAutomodel.configure_model(model)

    assert captured_kwargs["mtp_config_overrides"] == {
        "num_nextn_predict_layers": 1,
        "mtp_hybrid_override_pattern": "*",
        "mtp_layers_block_type": None,
    }
    assert captured_kwargs["replace_mtp_config"] is False
    assert captured_kwargs["num_nextn_predict_layers"] == 3
    assert captured_kwargs["mtp_use_repeated_layer"] is True
    assert model.llm.config.num_nextn_predict_layers == 1
    assert model.llm.mtp_config.num_layers == 3


# ---------------------------------------------------------------------------
# forward: mtp_per_depth_h extraction
# ---------------------------------------------------------------------------


class _DictLikeOutput(dict):
    '''Mimics a HuggingFace ModelOutput — dict keys are also accessible as attributes.'''

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class _FakeLLM(torch.nn.Module):
    def __init__(self, out):
        super().__init__()
        self._out = out
        self.mtp = None

    def forward(self, **_kwargs):
        return self._out


def _make_forward_model(llm_out):
    '''Minimal model that can run the forward() method with a mocked LLM.'''
    model = _bare_model()
    model.llm = _FakeLLM(llm_out)
    model._use_tp = False
    return model


def test_forward_extracts_mtp_per_depth_h_when_present():
    mtp_h = torch.randn(1, 4, 8)
    fake_out = _DictLikeOutput(logits=torch.randn(1, 4, 32), mtp_per_depth_h=mtp_h)
    model = _make_forward_model(fake_out)

    result = model.forward(
        input_embeds=torch.randn(1, 4, 32),
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    assert 'mtp_per_depth_h' in result
    assert result['mtp_per_depth_h'] is mtp_h


def test_forward_omits_mtp_per_depth_h_when_absent():
    fake_out = _DictLikeOutput(logits=torch.randn(1, 4, 32))
    model = _make_forward_model(fake_out)

    result = model.forward(
        input_embeds=torch.randn(1, 4, 32),
        attention_mask=torch.ones(1, 4, dtype=torch.bool),
    )

    assert 'mtp_per_depth_h' not in result


# ---------------------------------------------------------------------------
# validation MTP teacher-forced agreement metrics
# ---------------------------------------------------------------------------


def test_validation_epoch_end_logs_mtp_teacher_forced_agreement():
    '''on_validation_epoch_end reports teacher-forced prefix agreement
    probabilities and the corresponding prefix length.'''
    from collections import defaultdict

    model = _bare_model()
    model._get_moe_dp_group = lambda: None  # single-rank: _reduce_validation_metric_sums is a no-op
    model.lss_loss = None

    logged = {}
    model.log = lambda name, value, **kwargs: logged.__setitem__(name, float(value))

    # Standard val metrics still need populating — the method aggregates them first.
    model._partial_val_loss_sums = defaultdict(list, {'ds': [torch.tensor(4.0)]})
    model._partial_val_corrects = defaultdict(list, {'ds': [torch.tensor(8.0)]})
    model._partial_val_num_frames = defaultdict(list, {'ds': [torch.tensor(10.0)]})
    model._partial_val_lss = defaultdict(list)

    # Depth-1 prefix: 8/10 agree; depth-2 prefix: 5/10 agree.
    model._partial_val_mtp_correct = defaultdict(list, {'ds': [torch.tensor([8.0, 5.0])]})
    model._partial_val_mtp_valid = defaultdict(list, {'ds': [torch.tensor([10.0, 10.0])]})

    SALMAutomodel.on_validation_epoch_end(model)

    assert logged['val_mtp_teacher_forced_agreement_ds/head_1'] == pytest.approx(0.8)
    assert logged['val_mtp_teacher_forced_agreement_ds/head_2'] == pytest.approx(0.5)
    # 1 (main verifier token) + P(A1) + P(A1 and A2) = 2.3.
    assert logged['val_mtp_teacher_forced_prefix_length_ds'] == pytest.approx(2.3)
    assert logged['val_mtp_teacher_forced_prefix_length'] == pytest.approx(2.3)


def test_validation_step_preserves_mtp_counter_precision_with_bf16_logits(monkeypatch):
    """MTP metric counters stay exact even when the validation loss is BF16."""
    model = _bare_model()
    model.llm = torch.nn.Module()
    model.llm.mtp = torch.nn.Identity()
    model.llm.compute_mtp_in_eval = False
    model.lss_loss = None
    SALMAutomodel.on_validation_epoch_start(model)

    inputs = {
        "input_embeds": torch.zeros(1, 2, 4, dtype=torch.bfloat16),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "target_ids": torch.tensor([[0, 1]]),
        "llm_kwargs": {},
    }
    logits = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    model.prepare_inputs = lambda _batch: inputs
    model.forward = lambda *_args, **_kwargs: {
        "logits": logits,
        "mtp_per_depth_h": [torch.zeros(1, 2, 4)],
    }
    monkeypatch.setattr(
        salm_module,
        "_calculate_mtp_teacher_forced_agreement_with_heads",
        lambda **_kwargs: ([torch.tensor(10_001)], [torch.tensor(10_003)]),
    )

    SALMAutomodel.validation_step(model, {"ds": {}}, batch_idx=0)

    correct = model._partial_val_mtp_correct["ds"][0]
    valid = model._partial_val_mtp_valid["ds"][0]
    assert correct.dtype == torch.int64
    assert valid.dtype == torch.int64
    assert correct.item() == 10_001
    assert valid.item() == 10_003


def test_calculate_mtp_teacher_forced_agreement_with_heads_counts(monkeypatch):
    '''Agreement compares drafts with teacher-forced verifier predictions and requires a matched prefix.'''
    # The helper imports these specific Automodel submodules; importorskip each so the test
    # skips cleanly when the installed Automodel rev predates them (rather than hard-failing).
    pytest.importorskip('nemo_automodel.components.loss.utils', reason='needs Automodel _get_lm_head_module')
    pytest.importorskip('nemo_automodel.components.models.common.mtp', reason='needs Automodel roll_tensor')

    # Identity lm_head over a V==H one-hot space so argmax(hidden) == predicted token id.
    V = 4
    lm_head = torch.nn.Linear(V, V, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(torch.eye(V))
    model = torch.nn.Module()
    model.lm_head = lm_head
    # Avoid depending on Automodel's lm-head discovery internals.
    monkeypatch.setattr('nemo_automodel.components.loss.utils._get_lm_head_module', lambda m: m.lm_head)

    labels = torch.tensor([[3, 3, 3, 3, 3]])  # Used only to define valid positions.
    verifier_predictions = torch.tensor([[0, 1, 2, 3, 0]])
    # Depth 1's verifier targets are [1, 2, 3, 0]. Drafts match positions 0, 2, 3.
    depth_1_ids = torch.tensor([[1, 0, 3, 0, 0]])
    # Depth 2 matches all three verifier targets [2, 3, 0], but position 1 cannot
    # agree at depth 2 because depth 1 already mismatched there.
    depth_2_ids = torch.tensor([[2, 3, 0, 0, 0]])
    mtp_h = [torch.nn.functional.one_hot(ids, num_classes=V).float() for ids in (depth_1_ids, depth_2_ids)]

    correct, valid = salm_module._calculate_mtp_teacher_forced_agreement_with_heads(
        mtp_per_depth_h=mtp_h,
        labels=labels,
        model=model,
        verifier_predictions=verifier_predictions,
    )

    assert [int(value) for value in correct] == [3, 2]
    assert [int(value) for value in valid] == [4, 3]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (torch.zeros(5, dtype=torch.long), torch.tensor([0, 0, 0, 1, 1])),
        (torch.zeros(2, 5, dtype=torch.long), torch.tensor([[0, 0, 0, 1, 1], [0, 0, 0, 1, 1]])),
    ],
)
def test_resolve_mtp_seq_idx_from_cu_seqlens(labels, expected):
    actual = salm_module._resolve_mtp_seq_idx(
        labels,
        cu_seqlens=torch.tensor([[0, 3, 5]], dtype=torch.int32),
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("labels", "seq_idx", "expected"),
    [
        (
            torch.zeros(2, 4, dtype=torch.long),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        ),
        (
            torch.zeros(4, dtype=torch.long),
            torch.tensor([[0, 0, 1, 1]]),
            torch.tensor([0, 0, 1, 1]),
        ),
    ],
)
def test_resolve_mtp_seq_idx_normalizes_explicit_shape(labels, seq_idx, expected):
    actual = salm_module._resolve_mtp_seq_idx(labels, seq_idx=seq_idx)

    torch.testing.assert_close(actual, expected)


def test_resolve_mtp_seq_idx_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="does not match labels shape"):
        salm_module._resolve_mtp_seq_idx(
            torch.zeros(2, 4, dtype=torch.long),
            seq_idx=torch.zeros(3, dtype=torch.long),
        )


def test_iter_mtp_depth_targets_masks_trailing_and_packed_boundaries():
    targets = list(
        salm_module._iter_mtp_depth_targets(
            torch.tensor([10, 11, 12, 20, 21]),
            2,
            cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
        )
    )

    torch.testing.assert_close(targets[0], torch.tensor([11, 12, -100, 21, -100]))
    torch.testing.assert_close(targets[1], torch.tensor([12, -100, -100, -100, -100]))


def test_training_step_forwards_packed_cu_seqlens_to_mtp_loss(monkeypatch):
    class _Perception(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.preprocessor = torch.nn.Identity()
            self.encoder = torch.nn.Identity()

    model = _bare_model()
    model.perception = _Perception()
    model.llm = torch.nn.Identity()
    model.lss_loss = None
    model._mtp_loss_fn = object()
    model._mtp_loss_scaling_factor = 0.1
    model._trainer = None
    model.tokenizer = type("Tokenizer", (), {"pad": -1, "unk_id": None})()
    model._get_moe_dp_group = lambda: None
    model.log = lambda *_args, **_kwargs: None
    model.log_dict = lambda *_args, **_kwargs: None
    model.maybe_log_moe_metrics = lambda _batch_idx: None

    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    inputs = {
        "input_embeds": torch.zeros(5, 4),
        "attention_mask": None,
        "target_ids": torch.tensor([0, 1, 2, 3, 4]),
        "llm_kwargs": {"cu_seqlens": cu_seqlens},
        "num_tokens": 5,
        "num_examples": 2,
    }
    model.prepare_inputs = lambda _batch: inputs
    model.forward = lambda *_args, **_kwargs: {
        "logits": torch.zeros(1, 5, 8),
        "mtp_per_depth_h": [torch.zeros(1, 5, 4)],
    }
    captured_kwargs = {}

    def _calculate_mtp_loss(*_args, **kwargs):
        captured_kwargs.update(kwargs)
        return torch.tensor(0.25), [torch.tensor(2.5)]

    monkeypatch.setattr(salm_module, "_calculate_mtp_loss_with_heads", _calculate_mtp_loss)

    model._training_step_batch({"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}, batch_idx=0)

    assert captured_kwargs["cu_seqlens"] is cu_seqlens
    assert captured_kwargs["labels"] is inputs["target_ids"]


def test_mtp_validation_forward_legacy_gate_keeps_children_in_eval():
    llm = torch.nn.Module()
    llm.child = torch.nn.Dropout()
    llm.eval()

    with salm_module._mtp_validation_forward(llm, enabled=True):
        assert llm.training
        assert not llm.child.training

    assert not llm.training
    assert not llm.child.training


def test_mtp_validation_forward_uses_and_restores_native_gate():
    llm = torch.nn.Module()
    llm.eval()
    llm.compute_mtp_in_eval = False

    with salm_module._mtp_validation_forward(llm, enabled=True):
        assert llm.compute_mtp_in_eval
        assert not llm.training

    assert not llm.compute_mtp_in_eval
    assert not llm.training


@pytest.mark.parametrize("native_gate", [False, True], ids=["legacy-training-flag", "compute-mtp-in-eval"])
def test_mtp_validation_forward_restores_gate_after_error(native_gate):
    def fail_forward():
        raise RuntimeError("forward failed")

    llm = torch.nn.Module()
    llm.eval()
    if native_gate:
        llm.compute_mtp_in_eval = False

    with pytest.raises(RuntimeError, match="forward failed"):
        with salm_module._mtp_validation_forward(llm, enabled=True):
            if native_gate:
                assert llm.compute_mtp_in_eval
                assert not llm.training
            else:
                assert llm.training
            fail_forward()

    assert not llm.training
    if native_gate:
        assert not llm.compute_mtp_in_eval


@pytest.mark.parametrize("hook_name", ["on_validation_start", "on_test_start"])
def test_standalone_evaluation_rejects_mtp_context_parallelism(monkeypatch, hook_name):
    from omegaconf import DictConfig

    import nemo.collections.speechlm2.parts.parallel as parallel_module

    class _Submesh:
        def size(self):
            return 2

    class _Mesh:
        mesh_dim_names = ("cp",)

        def __getitem__(self, name):
            assert name == "cp"
            return _Submesh()

    model = _bare_model()
    model.cfg = DictConfig({"mtp": {"enabled": True}, "packed_sequences": True})
    model._device_mesh = _Mesh()
    monkeypatch.setattr(parallel_module, "validate_parallelism_compatibility", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="requires cp_size=1"):
        getattr(model, hook_name)()


def test_mtp_rejects_tensor_parallelism_before_training(monkeypatch):
    from omegaconf import DictConfig

    import nemo.collections.speechlm2.parts.parallel as parallel_module

    class _Submesh:
        def size(self):
            return 2

    class _Mesh:
        mesh_dim_names = ("tp",)

        def __getitem__(self, name):
            assert name == "tp"
            return _Submesh()

    model = _bare_model()
    model.cfg = DictConfig({"mtp": {"enabled": True}, "packed_sequences": False})
    model._device_mesh = _Mesh()
    monkeypatch.setattr(parallel_module, "validate_parallelism_compatibility", lambda **_kwargs: None)

    with pytest.raises(ValueError, match="requires tp_size=1"):
        model._validate_parallelism_compatibility()


def test_vocab_argmax_does_not_materialize_full_dtensor_logits(monkeypatch):
    expected = torch.tensor([[3, 1]])

    class _FakeDTensor:
        def __init__(self, *, is_logits):
            self.is_logits = is_logits

        def argmax(self, dim):
            assert self.is_logits
            assert dim == -1
            return _FakeDTensor(is_logits=False)

        def full_tensor(self):
            if self.is_logits:
                pytest.fail("vocabulary-sharded logits must not be fully materialized")
            return expected

    monkeypatch.setattr(salm_module, "DTensor", _FakeDTensor)

    predictions = salm_module._vocab_parallel_argmax(_FakeDTensor(is_logits=True))

    assert predictions is expected


def test_fused_mtp_loss_reuses_lm_weight_without_projecting_logits(monkeypatch):
    pytest.importorskip('nemo_automodel.components.models.common.mtp', reason='needs Automodel roll_tensor')
    loss_module = pytest.importorskip('nemo_automodel.components.loss.linear_ce', reason='needs fused linear CE')
    loss_utils = pytest.importorskip('nemo_automodel.components.loss.utils', reason='needs Automodel loss utilities')

    class _FailingLMHead(torch.nn.Linear):
        def forward(self, _inputs):
            pytest.fail("fused MTP loss must not materialize logits through lm_head.forward")

    model = torch.nn.Module()
    model.lm_head = _FailingLMHead(4, 4, bias=False)
    monkeypatch.setattr(loss_utils, "_get_lm_head_module", lambda _model: model.lm_head)

    calls = []

    def _calculate_loss(_loss_fn, **kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"].sum() * 0 + 2.0

    monkeypatch.setattr(loss_utils, "calculate_loss", _calculate_loss)
    mtp_h = [torch.randn(1, 5, 4, requires_grad=True) for _ in range(2)]
    grad_reduce_group = object()
    materialize_calls = []

    def _materialize(weight, *, grad_reduce_group=None):
        materialize_calls.append((weight, grad_reduce_group))
        return weight

    monkeypatch.setattr(salm_module, "DTensor", torch.nn.Parameter)
    monkeypatch.setattr(
        loss_module.FusedLinearCrossEntropy,
        "materialize_lm_weight",
        staticmethod(_materialize),
        raising=False,
    )

    total, raw_head_losses = salm_module._calculate_mtp_loss_with_heads(
        loss_fn=loss_module.FusedLinearCrossEntropy(reduction="sum"),
        mtp_per_depth_h=mtp_h,
        labels=torch.tensor([[0, 1, 2, 3, 0]]),
        model=model,
        scaling_factor=0.1,
        grad_reduce_group=grad_reduce_group,
        cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
    )

    assert len(calls) == 2
    torch.testing.assert_close(calls[0]["labels"], torch.tensor([[1, 2, -100, 0, -100]]))
    torch.testing.assert_close(calls[1]["labels"], torch.tensor([[2, -100, -100, -100, -100]]))
    assert calls[0]["lm_weight"] is model.lm_head.weight
    assert calls[1]["lm_weight"] is calls[0]["lm_weight"]
    assert materialize_calls == [(model.lm_head.weight, grad_reduce_group)]
    assert all(call["grad_reduce_group"] is grad_reduce_group for call in calls)
    assert float(total.detach()) == pytest.approx(0.2)
    assert [float(value.detach()) for value in raw_head_losses] == pytest.approx([2.0, 2.0])
    total.backward()
    assert all(hidden.grad is not None for hidden in mtp_h)


def test_unfused_mtp_loss_resolves_lm_head_once(monkeypatch):
    pytest.importorskip('nemo_automodel.components.models.common.mtp', reason='needs Automodel roll_tensor')
    loss_utils = pytest.importorskip('nemo_automodel.components.loss.utils', reason='needs Automodel loss utilities')
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(4, 4, bias=False)
    lookup_calls = []

    def _get_lm_head(_model):
        lookup_calls.append(_model)
        return model.lm_head

    monkeypatch.setattr(loss_utils, "_get_lm_head_module", _get_lm_head)

    salm_module._calculate_mtp_loss_with_heads(
        loss_fn=MaskedCrossEntropy(reduction="sum", fp32_upcast=False),
        mtp_per_depth_h=[torch.randn(1, 4, 4) for _ in range(3)],
        labels=torch.tensor([[0, 1, 2, 3]]),
        model=model,
    )

    assert lookup_calls == [model]


@pytest.mark.parametrize(
    ("cut_ce_available", "gradient_safe_api", "expect_fused"),
    [
        pytest.param(False, False, False, id="cut-ce-unavailable"),
        pytest.param(False, True, False, id="cut-ce-unavailable-with-api"),
        pytest.param(True, False, False, id="legacy-automodel-api"),
        pytest.param(True, True, True, id="fused-supported"),
    ],
)
def test_mtp_loss_selection_handles_optional_cut_cross_entropy(
    monkeypatch, cut_ce_available, gradient_safe_api, expect_fused
):
    from nemo_automodel.components.loss import linear_ce
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

    monkeypatch.setattr(linear_ce, "HAVE_CUT_CROSS_ENTROPY", cut_ce_available)
    if gradient_safe_api:
        monkeypatch.setattr(
            linear_ce.FusedLinearCrossEntropy,
            "materialize_lm_weight",
            staticmethod(lambda weight, **_kwargs: weight),
            raising=False,
        )
    else:
        monkeypatch.delattr(linear_ce.FusedLinearCrossEntropy, "materialize_lm_weight", raising=False)

    loss_fn = salm_module._build_mtp_loss_fn()

    expected_type = linear_ce.FusedLinearCrossEntropy if expect_fused else MaskedCrossEntropy
    assert isinstance(loss_fn, expected_type)


def test_head_only_keep_pattern_follows_wrapped_mtp_namespace():
    from omegaconf import DictConfig

    from nemo.collections.speechlm2.parts.optim_setup import freeze_and_subset

    class _LLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.mtp = torch.nn.Linear(4, 4)

    class _CompiledWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

        @property
        def mtp(self):
            return self._orig_mod.mtp

    model = _bare_model()
    model.llm = _CompiledWrapper(_LLM())
    model.perception = torch.nn.Linear(4, 4)
    model.cfg = DictConfig({"freeze_params": [r"^llm\..+$"], "prevent_freeze_params": []})

    model._apply_mtp_training_mode("head_only")

    assert r"^llm\._orig_mod\.mtp\..+$" in model.cfg.prevent_freeze_params
    optimizer_params = list(
        freeze_and_subset(model.named_parameters(), model.cfg.freeze_params, model.cfg.prevent_freeze_params)
    )
    assert {id(param) for param in optimizer_params} == {id(param) for param in model.llm.mtp.parameters()}


def test_joint_keep_pattern_follows_wrapped_mtp_namespace():
    from omegaconf import DictConfig

    from nemo.collections.speechlm2.parts.optim_setup import freeze_and_subset

    class _LLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)
            self.mtp = torch.nn.Linear(4, 4)

    class _CompiledWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self._orig_mod = module

        @property
        def mtp(self):
            return self._orig_mod.mtp

    model = _bare_model()
    model.llm = _CompiledWrapper(_LLM())
    model.perception = torch.nn.Linear(4, 4)
    model.cfg = DictConfig(
        {
            "freeze_params": [r"^llm\..+$"],
            "prevent_freeze_params": [r"^llm\.mtp\..+$"],
        }
    )

    model._apply_mtp_training_mode("joint")

    assert r"^llm\._orig_mod\.mtp\..+$" in model.cfg.prevent_freeze_params
    optimizer_params = list(
        freeze_and_subset(model.named_parameters(), model.cfg.freeze_params, model.cfg.prevent_freeze_params)
    )
    optimizer_param_ids = {id(param) for param in optimizer_params}
    assert {id(param) for param in model.llm.mtp.parameters()} <= optimizer_param_ids
    assert all(param.requires_grad for param in model.llm.mtp.parameters())
    assert all(not param.requires_grad for param in model.llm._orig_mod.backbone.parameters())
