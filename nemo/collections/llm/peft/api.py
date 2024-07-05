from nemo.collections.llm.peft.lora import LoRA
from nemo.collections.llm.utils import factory
from nemo.lightning.pytorch.callbacks.peft import PEFT


@factory
def gpt_lora() -> PEFT:
    return LoRA()


__all__ = ["gpt_lora"]
