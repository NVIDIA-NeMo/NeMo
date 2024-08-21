import torch

from nemo.collections.llm.utils import Config
from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision


def bf16_mixed_plugin() -> Config[MegatronMixedPrecision]:
    return Config(
        MegatronMixedPrecision,
        precision="bf16-mixed",
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )


def fp16_mixed_plugin() -> Config[MegatronMixedPrecision]:
    return Config(
        MegatronMixedPrecision,
        precision="16-mixed",
        params_dtype=torch.half,
        pipeline_dtype=torch.half,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
