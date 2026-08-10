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
    assert model.llm.config.num_nextn_predict_layers == 0
    assert not model._mtp_enabled


@pytest.mark.parametrize("training_mode", ["joint", "head_only"])
def test_base_checkpoint_attaches_fresh_mtp_for_both_training_modes(monkeypatch, training_mode):
    """A checkpoint with no native head gets a fresh head in either enabled mode."""
    from omegaconf import DictConfig

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
            "mtp": {"enabled": True, "training_mode": training_mode},
        }
    )
    model._trainer = None
    model._use_fsdp = False
    model._use_tp = False
    model.setup_moe_options = lambda: None

    monkeypatch.setattr(salm_module, "load_pretrained_automodel_llm", lambda *_args, **_kwargs: _LLM())
    monkeypatch.setattr(
        salm_module, "setup_speech_encoder", lambda model, **_kwargs: setattr(model, "perception", _Perception())
    )
    monkeypatch.setattr(salm_module, "update_perception_output_dim", lambda _model: None)
    monkeypatch.setattr(salm_module, "maybe_load_pretrained_models", lambda _model: None)

    def _attach_fresh_head(self, _mtp_cfg, _dtype):
        self.llm.mtp = torch.nn.Linear(4, 4)

    monkeypatch.setattr(SALMAutomodel, "_build_and_attach_mtp_head", _attach_fresh_head)

    SALMAutomodel.configure_model(model)

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
# validation MTP acceptance metrics
# ---------------------------------------------------------------------------


def test_validation_epoch_end_logs_mtp_acceptance():
    '''on_validation_epoch_end reports verifier-prefix acceptance probabilities
    and their expected acceptance length.'''
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

    # Depth-1 prefix: 8/10 accepted; depth-2 prefix: 5/10 accepted.
    model._partial_val_mtp_correct = defaultdict(list, {'ds': [torch.tensor([8.0, 5.0])]})
    model._partial_val_mtp_valid = defaultdict(list, {'ds': [torch.tensor([10.0, 10.0])]})

    SALMAutomodel.on_validation_epoch_end(model)

    assert logged['val_mtp_acc_ds/head_1'] == pytest.approx(0.8)
    assert logged['val_mtp_acc_ds/head_2'] == pytest.approx(0.5)
    # 1 (main verifier token) + P(A1) + P(A1 and A2) = 2.3.
    assert logged['val_mtp_accept_length_ds'] == pytest.approx(2.3)
    assert logged['val_mtp_accept_length'] == pytest.approx(2.3)


def test_calculate_mtp_acceptance_with_heads_counts(monkeypatch):
    '''Acceptance compares drafts with verifier predictions and requires a matched prefix.'''
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
    # be accepted because depth 1 already mismatched there.
    depth_2_ids = torch.tensor([[2, 3, 0, 0, 0]])
    mtp_h = [torch.nn.functional.one_hot(ids, num_classes=V).float() for ids in (depth_1_ids, depth_2_ids)]

    correct, valid = salm_module._calculate_mtp_acceptance_with_heads(
        mtp_per_depth_h=mtp_h,
        labels=labels,
        model=model,
        verifier_predictions=verifier_predictions,
    )

    assert [int(value) for value in correct] == [3, 2]
    assert [int(value) for value in valid] == [4, 3]


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


def test_mtp_rejects_context_parallelism(monkeypatch):
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
        model._validate_parallelism_compatibility()


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
    mtp_h = [torch.randn(1, 4, 4, requires_grad=True) for _ in range(2)]

    total, head_losses, raw_head_losses = salm_module._calculate_mtp_loss_with_heads(
        loss_fn=loss_module.FusedLinearCrossEntropy(reduction="sum"),
        mtp_per_depth_h=mtp_h,
        labels=torch.tensor([[0, 1, 2, 3]]),
        model=model,
        scaling_factor=0.1,
    )

    assert len(calls) == 2
    assert calls[0]["lm_weight"] is model.lm_head.weight
    assert calls[1]["lm_weight"] is calls[0]["lm_weight"]
    assert float(total.detach()) == pytest.approx(0.2)
    assert [float(value.detach()) for value in head_losses] == pytest.approx([0.1, 0.1])
    assert [float(value.detach()) for value in raw_head_losses] == pytest.approx([2.0, 2.0])
    total.backward()
    assert all(hidden.grad is not None for hidden in mtp_h)
