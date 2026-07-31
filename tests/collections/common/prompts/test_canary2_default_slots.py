from nemo.collections.common.prompts.canary2 import Canary2PromptFormatter


class _TestFormatter(Canary2PromptFormatter):
    def __init__(self):
        self._defaults = [{"role": "user", "slots": {"target_lang": "en"}}]

    def get_slots(self, role):
        return ["decodercontext", "target_lang"]


def test_get_default_dialog_slots_uses_single_default_lookup():
    formatter = _TestFormatter()
    assert formatter.get_default_dialog_slots() == [
        {"role": "user", "slots": {"decodercontext": None, "target_lang": "en"}}
    ]
