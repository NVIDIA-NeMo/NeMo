# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
from typing import Optional

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator


def get_results(
    base_config=None,
    train_config=None,
    path_to_save: str = None,
    output_top_n: Optional[int] = 10,
):
    """Generates performance results.

    Args:
        config (AutoConfigurator): auto configurator runner config.
        path_to_save (str): path where to save performance results.
        output_top_n (Optional[int]): Number of configs to be printed out as best configs.
    """

    # Define needed variables
    model_name = train_config.model_type
    model_size = train_config.model_size_in_b
    global_batch_size = base_config.data.global_batch_size
    seq_length = base_config.data.seq_length

    vocab_size = train_config.vocab_size
    num_nodes = train_config.num_nodes
    gpus_per_node = train_config.gpus_per_node

    layers = base_config.model.num_layers
    hs = base_config.model.hidden_size
    ffn_hs = base_config.model.ffn_hidden_size

    training_logs = path_to_save
    final_result_logs = path_to_save

    result_columns = [
        "Model Name",
        "Model Size",
        "Seq Length",
        "TP",
        "PP",
        "CP",
        "EP",
        "MBS",
        "Act Ckpt Layers",
        "Act Ckpt Micro Bathes",
        "Act Ckpt Layers per Pipeline",
        "Num Layers",
        "Hidden Size",
        "FFN Hidden Size",
        "GBS",
        "Nodes",
        "GPUs per Node",
        "Time per Step",
        "Samples per Second",
        "Model TFLOPS / GPU",
        "Model TFLOPS Aggregate",
    ]
    error_columns = [
        "Model Name",
        "Model Size",
        "Seq Length",
        "TP",
        "PP",
        "CP",
        "EP",
        "MBS",
        "Act Ckpt Layers",
        "Act Ckpt Micro Bathes",
        "Act Ckpt Layers per Pipeline",
        "Num Layers",
        "Hidden Size",
        "FFN Hidden Size",
        "GBS",
        "Nodes",
        "GPUs per Node",
        "Error Message",
    ]
    result = []
    errors = []
    dirs = [f.path for f in os.scandir(training_logs) if f.is_dir()]

    for candidate_dir in dirs:
        logs_dir = os.path.join(training_logs, candidate_dir, "tb_logs/lightning_logs")
        logs_folder = [f.path for f in os.scandir(logs_dir) if f.is_dir()][0]
        tp, pp, cp, ep, mbs, act_ckpt, num_mbs_act, act_per_pipe = get_config(candidate_dir)

        for f in os.listdir(logs_folder):
            if f.endswith("0.txt"):
                error_file = os.path.join(logs_folder, f)
                error = find_error(error_file)
                if error:
                    errors.append(
                        [
                            model_name,
                            model_size,
                            seq_length,
                            tp,
                            pp,
                            cp,
                            ep,
                            mbs,
                            act_ckpt,
                            num_mbs_act,
                            act_per_pipe,
                            layers,
                            hs,
                            ffn_hs,
                            global_batch_size,
                            num_nodes,
                            gpus_per_node,
                            error,
                        ]
                    )

        files = os.listdir(logs_folder)
        for f in files:
            if f.startswith("events"):
                event_file = os.path.join(logs_folder, f)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()
                try:
                    timing_list = ea.Scalars("train_step_timing in s")
                    if len(timing_list) <= 6:
                        continue
                    timing_list = [x.value for x in timing_list[5:]]
                    avg_global_step_time = round(sum(timing_list) / len(timing_list), 4)
                    samples_per_s = round(global_batch_size / avg_global_step_time, 2)
                    m_tflops, m_tflops_gpu = calculate_tflops(
                        model_name=model_name,
                        gbs=global_batch_size,
                        enc_seq_len=seq_length,
                        dec_seq_len=seq_length,
                        hs=hs,
                        ffn_hs=ffn_hs,
                        layers=layers,
                        vocab=vocab_size,
                        nodes=num_nodes,
                        gpus_per_node=gpus_per_node,
                        time_per_step=avg_global_step_time,
                    )
                    config_name = f"tp{tp}_pp{pp}_cp{cp}_ep{ep}_mbs{mbs}_act_{act_ckpt}_num_mbs_act_{num_mbs_act}_act_per_pipe_{act_per_pipe}"
                    result.append(
                        [
                            model_name,
                            model_size,
                            seq_length,
                            tp,
                            pp,
                            cp,
                            ep,
                            mbs,
                            act_ckpt,
                            num_mbs_act,
                            act_per_pipe,
                            layers,
                            hs,
                            ffn_hs,
                            global_batch_size,
                            num_nodes,
                            gpus_per_node,
                            avg_global_step_time,
                            samples_per_s,
                            m_tflops_gpu,
                            m_tflops,
                        ]
                    )
                finally:
                    continue
    result.sort(key=lambda x: x[17])
    print(f"Top {min(output_top_n, len(result))} configs sorted from fastest to slowest:")
    for i, res in enumerate(result):
        print(f"Config #{i+1}: {res[-1]} with {res[17]:.4f}s per global step.")
        if i + 1 == output_top_n:
            break

    top_config = f"{model_name}_{model_size}b_{num_nodes}nodes_tp_{result[0][3]}_pp_{result[0][4]}_cp_{result[0][5]}_ep_{result[0][6]}_mbs_{result[0][7]}_act_ckpt_{result[0][8]}_num_mbs_act_{result[0][9]}_act_per_pipe_{result[0][10]}"
    print("\n==================================================")
    print(f"Optimal config: {top_config} with {result[0][17]:.4f}s per global step.")
    print("==================================================\n")

    # Save results as a CSV file.
    os.makedirs(final_result_logs, exist_ok=True)
    result_df = pd.DataFrame(result, columns=result_columns)
    result_df.to_csv(os.path.join(final_result_logs, f"final_summary_{num_nodes}nodes.csv"), index=False)

    error_df = pd.DataFrame(errors, columns=error_columns)
    error_df.to_csv(os.path.join(final_result_logs, f"failed_jobs_{num_nodes}nodes.csv"), index=False)


def calculate_tflops(
    model_name,
    gbs,
    enc_seq_len,
    dec_seq_len,
    hs,
    ffn_hs,
    layers,
    vocab,
    nodes,
    gpus_per_node,
    time_per_step,
):
    """Calculates model and hardware TFLOPS for each model.

    GPT-3 Formulas:
        Model FLOPs = (24𝐵𝑠ℎ^2 + 4𝐵��^2ℎ) x (3 x num_layers) + 6𝐵𝑠ℎ
    T5/mT5 Formula:
        Model FLOPs =
    Bert Formula:
        Model FLOPs = 72BLsh^2 * ( 1 + (s/6h) + (v/12hL))
    """

    if model_name in ["gpt3", "llama", "baichuan2", "chatglm", "qwen2", "mixtral"]:
        # Model FLOPS calculation
        model_flops = (
            (24 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs) * (3 * layers)
            + (6 * gbs * enc_seq_len * hs * vocab)
        ) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)

        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12

    elif model_name == "bert":
        model_flops = (
            72 * gbs * layers * enc_seq_len * hs * hs * (1 + (enc_seq_len / (6 * hs)) + (vocab / (12 * hs * layers)))
        ) / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12

    elif model_name in ["t5", "mt5"]:
        # Encoder Layer FLOPS: include self attention + MLP
        flops_self_attn_enc = 8 * gbs * enc_seq_len * hs * hs + 4 * gbs * enc_seq_len * enc_seq_len * hs
        flops_mlp_enc = 6 * gbs * enc_seq_len * hs * ffn_hs  # geglu needs two gemms for h -> ffn_h
        flops_enc_layer = flops_self_attn_enc + flops_mlp_enc

        # Decoder Layer FLOPS: inlcude self_attn + cross_attn + MLP
        flops_self_attn_dec = 8 * gbs * dec_seq_len * hs * hs + 4 * gbs * dec_seq_len * dec_seq_len * hs
        flops_cross_attn_dec = (
            4 * gbs * enc_seq_len * hs * hs
            + 4 * gbs * dec_seq_len * hs * hs
            + 4 * gbs * enc_seq_len * dec_seq_len * hs
        )
        flops_mlp_dec = 6 * gbs * dec_seq_len * hs * ffn_hs  # geglu needs two gemms for h -> ffn_h
        flops_dec_layer = flops_self_attn_dec + flops_cross_attn_dec + flops_mlp_dec

        # FLOPs of logits layer in the head
        flops_logits = 2 * gbs * dec_seq_len * hs * vocab

        # FLOPs of fprop
        flops_fprop = (flops_enc_layer + flops_dec_layer) * (layers // 2) + flops_logits

        # FLOPs of each train step (FLOPs of bprop is 2*fprop)
        model_flops = 3 * flops_fprop / time_per_step
        model_flops_per_gpu = model_flops / (nodes * gpus_per_node)
        model_tflops = model_flops / 1e12
        model_tflops_per_gpu = model_flops_per_gpu / 1e12

    else:
        raise NotImplementedError("Model type not supported.")
    return round(model_tflops, 2), round(model_tflops_per_gpu, 2)


def find_error(error_file: str, errors: list = ["CUDA out of memory"]):
    """Function that finds the error among job output.

    Args:
        errors (list): list of "popular" errors.
        error_file (str): path to the job output.

    Returns:
        str: serror message if job has been failed because of one of listed errors or None if not.
    """

    error = None
    with open(error_file, "r") as f:
        output = f.read()
    for e in errors:
        if e in output:
            error = e
    return error


def get_config(run_name: str) -> tuple:
    """Function that extract model parallelism parameters

    Args:
        run_name (str): name of the run.

    Returns:
        tuple: model parallelism parameters.
    """
    pattern = r'_(tp|pp|cp|ep|mbs|act_ckpt|num_mbs_act|act_per_pipe)_([^_]+)'

    # Find all matches in the input string
    matches = re.findall(pattern, run_name)

    # Convert matches to a dictionary
    params = {param: value for param, value in matches}

    return (
        params["tp"],
        params["pp"],
        params["cp"],
        params["ep"],
        params["mbs"],
        params["act_ckpt"],
        params["num_mbs_act"],
        params["act_per_pipe"],
    )


if __name__ == "__main__":
    main()
