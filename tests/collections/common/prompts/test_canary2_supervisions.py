import pytest
import torch
from unittest.mock import patch

from nemo.collections.common.prompts import canary2 as canary2_mod
from nemo.collections.common.prompts.canary2 import canary2


class _MonoCut: ...
class _MixedCut: ...


class _FakePrompt:
    PROMPT_LANGUAGE_SLOT = "prompt_language"
    tokenizer = None

    def encode_dialog(self, turns):
        return {"answer_ids": torch.tensor([1])}


def test_canary2_raises_on_empty_supervisions():
    with patch.object(canary2_mod, "MonoCut", _MonoCut), patch.object(
        canary2_mod, "MixedCut", _MixedCut
    ):
        cut = _MonoCut()
        cut.id = "empty"
        cut.custom = {"source_lang": "en", "target_lang": "en"}
        cut.supervisions = []
        with pytest.raises(RuntimeError, match="has no supervisions"):
            canary2(cut, _FakePrompt())
