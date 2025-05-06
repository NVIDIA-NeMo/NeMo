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

from os.path import basename, splitext

import nemo_run as run

from nemo.collections.diffusion.recipes.flux_12b import pretrain_recipe
from nemo.lightning.run.plugins import NsysPlugin, PerfEnvPlugin

from ..argument_parser import parse_cli_args
from ..utils import (
    args_sanity_check,
    get_user_configs,
    set_exp_logging_configs,
    set_primary_perf_configs,
    slurm_executor,
)


def override_recipe_configs(
    args: str,
    num_nodes: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    enable_cuda_graphs: bool,
):
    """
    flux 12b pre-train recipe aimed at achieving best possible performance and faster
    overall runtime.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """
    recipe = pretrain_recipe()
    recipe = set_primary_perf_configs(
        recipe,
        "pre_train",
        num_nodes,
        args.gpus_per_node,
        mbs,
        gbs,
        args.max_steps,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        enable_cuda_graphs=enable_cuda_graphs,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
    )
    recipe = set_exp_logging_configs(
        recipe,
        "pre_train",
        "diffusion",
        "flux",
        args.tensorboard,
        args.wandb,
        args.wandb_prj_name,
        args.wandb_job_name,
    )

    recipe.model.flux_params.t5_params = None  # run.Config(T5Config, version='/ckpts/text_encoder_2')
    recipe.model.flux_params.clip_params = None  # run.Config(ClipConfig, version='/ckpts/text_encoder')
    recipe.model.flux_params.vae_config = (
        None  # run.Config(AutoEncoderConfig, ckpt='/ckpts/ae.safetensors', ch_mult=[1,2,4,4], attn_resolutions=[])
    )
    from nemo.collections.diffusion.models.flux.model import FluxConfig
    recipe.model.flux_params.flux_config = run.Config(FluxConfig, num_joint_layers=11, num_single_layers=22)
    #recipe.model.flux_params.flux_config.enable_cuda_graph = True
    #recipe.model.flux_params.flux_config.use_te_rng_tracker = True
    #recipe.model.flux_params.flux_config.cuda_graph_warmup_steps = 2
    recipe.model.flux_params.flux_config.apply_rope_fusion = True
    recipe.model.flux_params.flux_config.rotary_interleaved = False
    recipe.trainer.strategy.use_te_rng_tracker=True
    from megatron.core.distributed import DistributedDataParallelConfig
    recipe.trainer.strategy.ddp = run.Config(
        DistributedDataParallelConfig,
        use_custom_fsdp=False,
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
    )

    return recipe


if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "flux", "12b", args)
    num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, _, enable_cuda_graphs = kwargs[:10]

    recipe = override_recipe_configs(
        args, num_nodes, mbs, gbs, tp_size, pp_size, cp_size, vp_size, ep_size, enable_cuda_graphs
    )

    exp_config = f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_{mbs}mbs_{gbs}gbs"
    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"

    executor = slurm_executor(
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        args.gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars={},
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
    )

    plugins = [
        PerfEnvPlugin(
            enable_vboost=True,
            nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
            gpu_sm100_or_newer=(args.gpu.lower() in ['b200', 'gb200']),
        ),
    ]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))

    with run.Experiment(exp_name) as exp:
        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=plugins,
        )

        if not args.dryrun:
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()
