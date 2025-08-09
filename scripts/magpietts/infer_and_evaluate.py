# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import argparse
import copy
import glob
import json
import os
import shutil
import time
from typing import List
from pathlib import Path
from functools import partial

import scripts.magpietts.evalset_config as evalset_config
import scripts.magpietts.evaluate_generated_audio as evaluate_generated_audio
import numpy as np
import scipy.stats as stats
import soundfile as sf
import torch
from omegaconf.omegaconf import OmegaConf, open_dict
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

from nemo.collections.asr.parts.utils.manifest_utils import read_manifest
from nemo.collections.tts.data.text_to_speech_dataset import MagpieTTSDataset
from nemo.collections.tts.models import MagpieTTSModel
from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import AggregatedTTSTokenizer, IPATokenizer

def compute_mean_and_confidence_interval(metrics_list, metric_keys, confidence=0.90):
    metrics = {}
    for key in metric_keys:
        measurements = [m[key] for m in metrics_list]
        mean = np.mean(measurements)
        std_err = stats.sem(measurements)

        confidence_interval = std_err * stats.t.ppf((1 + confidence) / 2, len(measurements) - 1)
        print(f"{key}: {mean} +/- {confidence_interval}")
        metrics[key] = "{:.4f} +/- {:.4f}".format(mean, confidence_interval)
    return metrics

def update_config(model_cfg, codecmodel_path, legacy_codebooks=False):
    ''' helper function to rename older yamls from t5 to magpie '''
    model_cfg.codecmodel_path = codecmodel_path
    if hasattr(model_cfg, 'text_tokenizer'):
        # Backward compatibility for models trained with absolute paths in text_tokenizer
        model_cfg.text_tokenizer.g2p.phoneme_dict = "scripts/tts_dataset_files/ipa_cmudict-0.7b_nv23.01.txt"
        model_cfg.text_tokenizer.g2p.heteronyms = "scripts/tts_dataset_files/heteronyms-052722"
        model_cfg.text_tokenizer.g2p.phoneme_probability = 1.0
    model_cfg.train_ds = None
    model_cfg.validation_ds = None
    if "t5_encoder" in model_cfg:
        model_cfg.encoder = model_cfg.t5_encoder
        del model_cfg.t5_encoder
    if "t5_decoder" in model_cfg:
        model_cfg.decoder = model_cfg.t5_decoder
        del model_cfg.t5_decoder
    if hasattr(model_cfg, 'decoder') and hasattr(model_cfg.decoder, 'prior_eps'):
        # Added to prevent crash after removing arg from transformer_2501.py in https://github.com/blisc/NeMo/pull/56
        del model_cfg.decoder.prior_eps
    if hasattr(model_cfg, 'use_local_transformer') and model_cfg.use_local_transformer:
        # For older checkpoints trained with a different parameter name
        model_cfg.local_transformer_type = "autoregressive"
        del model_cfg.use_local_transformer
    if hasattr(model_cfg, 'downsample_factor'):
        # Backward compatibility for models trained with the config option`downsample_factor` which was later renamed to `frame_stacking_factor`
        model_cfg.frame_stacking_factor = model_cfg.downsample_factor
        del model_cfg.downsample_factor
    if legacy_codebooks:
        # Added to address backward compatibility arising from
        #  https://github.com/blisc/NeMo/pull/64
        print("WARNING: Using legacy codebook indices for backward compatibility. Should only be used with old checkpoints.")
        num_audio_tokens_per_codebook = model_cfg.num_audio_tokens_per_codebook
        model_cfg.forced_num_all_tokens_per_codebook = num_audio_tokens_per_codebook
        model_cfg.forced_audio_eos_id = num_audio_tokens_per_codebook - 1
        model_cfg.forced_audio_bos_id = num_audio_tokens_per_codebook - 2
        if model_cfg.model_type == 'decoder_context_tts':
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 3
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 4
            model_cfg.forced_mask_token_id = num_audio_tokens_per_codebook - 5
        else:
            model_cfg.forced_context_audio_eos_id = num_audio_tokens_per_codebook - 1
            model_cfg.forced_context_audio_bos_id = num_audio_tokens_per_codebook - 2
    if hasattr(model_cfg, 'sample_rate'):
        # This was removed from the config and is now in the model class
        sample_rate = model_cfg.sample_rate
        del model_cfg.sample_rate
    else:
        sample_rate = None
    return model_cfg, sample_rate

def update_ckpt(state_dict):
    new_state_dict = {}
    for key in state_dict.keys():
        if 't5_encoder' in key:
            new_key = key.replace('t5_encoder', 'encoder')
            new_state_dict[new_key] = state_dict[key]
        elif 't5_decoder' in key:
            new_key = key.replace('t5_decoder', 'decoder')
            new_state_dict[new_key] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict


def delete_old_generated_files(output_dir):
    # Delete any leftover generated files from previous runs as these can confuse the evaluation
    print(f"Deleting old generated files in: {output_dir} ...")
    for f in glob.glob(f"{output_dir}/predicted_codes*.pt"):
        os.remove(f)
    for f in glob.glob(f"{output_dir}/predicted_audio*.wav"):
        os.remove(f)

def create_violin_plots(metrics: List[dict], metric_keys: List[str], output_png: str):
    # Create dataframe from list of dicts
    df = pd.DataFrame(metrics)

    # Plot the violin plots for all DataFrames side by side
    num_columns = len(metric_keys)
    width = num_columns * 5
    fig, axs = plt.subplots(1, num_columns, figsize=(width, 4))

    for i, column in enumerate(metric_keys):
        assert column in df
        # Create empty lists to store the parts objects for each DataFrame
        # Plot the violin plots for each DataFrame
        axs[i].violinplot(
            df[column], showmedians=True, positions=[i], widths=0.5
        )

        axs[i].set_title(column)
        axs[i].set_xticks([i])
        axs[i].set_xticklabels([column])
        axs[i].grid(True, linestyle="dotted")

        # Calculate and display the mean value for each DataFrame
        mean = df[column].mean()
        sem = df[column].sem()
        axs[i].plot(
            i,
            mean,
            "o",
            color="red",
            markersize=4,
            label="Mean (95%CI)"
        )

        label_numeric = f"{mean:.2f}±{1.96 * sem:.2f}"
        axs[i].text(i + 0.06, mean, label_numeric, ha="center", va="top")

    # Create a single legend for all subplots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_png, format="png", bbox_inches="tight")


def run_inference(
        hparams_file,
        checkpoint_file,
        nemo_file,
        datasets,
        out_dir,
        temperature,
        topk,
        codecmodel_path,
        use_cfg,
        cfg_scale,
        batch_size,
        sv_model,
        asr_model_name,
        num_repeats=1,
        apply_attention_prior=False,
        attention_prior_epsilon=1e-3,
        attention_prior_lookahead_window=10,
        estimate_alignment_from_layers=None,
        apply_prior_to_layers=None,
        start_prior_after_n_audio_steps=10,
        confidence_level=0.95,
        use_local_transformer=False,
        maskgit_n_steps=3,
        maskgit_noise_scale=0.0,
        maskgit_fixed_schedule=None,
        maskgit_sampling_type=None,
        legacy_codebooks=False,
        clean_up_disk=False,
        hparams_file_from_wandb=False,
        log_exp_name=False,
        compute_fcd=False,
        violin_plot_metrics=['cer', 'pred_context_ssim']
    ):
    # Load model
    if hparams_file is not None and checkpoint_file is not None:
        model_cfg = OmegaConf.load(hparams_file)
        if "cfg" in model_cfg:
            model_cfg = model_cfg.cfg

        if hparams_file_from_wandb:
            model_cfg = model_cfg.value

        with open_dict(model_cfg):
            model_cfg, cfg_sample_rate = update_config(model_cfg, codecmodel_path, legacy_codebooks)

        model = MagpieTTSModel(cfg=model_cfg)
        model.use_kv_cache_for_inference = True

        # Load weights from checkpoint file
        print("Loading weights from checkpoint")
        ckpt = torch.load(checkpoint_file, weights_only=False)
        state_dict = update_ckpt(ckpt['state_dict'])
        model.load_state_dict(state_dict)
        checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0]
    elif nemo_file is not None:
        model_cfg = MagpieTTSModel.restore_from(nemo_file, return_config=True)
        with open_dict(model_cfg):
            model_cfg, cfg_sample_rate = update_config(model_cfg, codecmodel_path, legacy_codebooks)
        model = MagpieTTSModel.restore_from(nemo_file, override_config_path=model_cfg)
        model.use_kv_cache_for_inference = True
        checkpoint_name = nemo_file.split("/")[-1].split(".nemo")[0]
    else:
        raise ValueError("Need either a checkpoint and hparams file, or a nemo file.")

    if cfg_sample_rate is not None and cfg_sample_rate != model.sample_rate:
        raise ValueError("Sample rate in config and model do not match")

    print("Loaded weights.")
    model.cuda()
    model.eval()

    if log_exp_name:
        # the experiment name is the name of the directory two above the checkpoint path,
        # since training produces directories of the form `exp_name/checkpoints/checkpoint_name.ckpt`.
        exp_name = f"{os.path.basename(os.path.dirname(os.path.dirname(checkpoint_file)))}__"
    else:
        exp_name = ""

    # Build checkpoint name
    checkpoint_name = checkpoint_file.split("/")[-1].split(".ckpt")[0]
    checkpoint_name = (
        "{exp_name}{checkpoint_name}_Temp{temperature}_Topk{topk}_Cfg_{use_cfg}_{cfg_scale}_"
        f"Prior_{apply_attention_prior}_"
    )
    if apply_attention_prior:
        # Only add prior config details if prior is enabled
        checkpoint_name += (
            f"{attention_prior_epsilon}_{attention_prior_lookahead_window}_{start_prior_after_n_audio_steps}_"
            f"{''.join([str(l) for l in estimate_alignment_from_layers]) if estimate_alignment_from_layers is not None else 'None'}_"
            f"{''.join([str(l) for l in apply_prior_to_layers]) if apply_prior_to_layers is not None else 'None'}_"
    )
    checkpoint_name += (
        f"LT_{use_local_transformer}_"
        f"MaskGit_{maskgit_n_steps}_{maskgit_sampling_type}_{''.join([str(l) for l in maskgit_fixed_schedule]) if maskgit_fixed_schedule is not None else 'None'}_"
        f"SV_{sv_model}"
    )

    dataset_meta_info = evalset_config.dataset_meta_info
    ssim_per_dataset = []
    cer_per_dataset = []
    for dataset in datasets:
        print(f"Evaluating dataset {dataset}")
        metrics_n_repeated = []
        manifest_records = read_manifest(dataset_meta_info[dataset]['manifest_path'])
        language = dataset_meta_info[dataset].get('whisper_language', 'en')
        dataset_meta_for_dl = copy.deepcopy(dataset_meta_info[dataset])
        for key in ["whisper_language", "load_cached_codes_if_available"]:
            if key in dataset_meta_for_dl:
                del dataset_meta_for_dl[key]

        dataset_meta = {dataset: dataset_meta_for_dl}

        eval_dir = os.path.join(out_dir, f"{checkpoint_name}_{dataset}")
        audio_dir = os.path.join(eval_dir, "audio")
        all_experiment_csv = os.path.join(eval_dir, "all_experiment_metrics.csv")
        os.makedirs(eval_dir, exist_ok=True)

        if not os.path.exists(all_experiment_csv):
            with open(all_experiment_csv, "w") as f:
                header = "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative"
                if compute_fcd:
                    header += ",frechet_codec_distance"
                header += "\n"
                f.write(header)

        context_duration_min = model.cfg.get('context_duration_min', 5.0)
        context_duration_max = model.cfg.get('context_duration_max', 5.0)
        if context_duration_min < 5.0 and context_duration_max > 5.0:
            context_duration_min = 5.0
            context_duration_max = 5.0 # @pneekhara - For multiencoder models, I want fixed size contexts for fair eval. Not too important though.

        for repeat_idx in range(num_repeats):
            pred_audio_dir = os.path.join(audio_dir, f"repeat_{repeat_idx}")
            os.makedirs(pred_audio_dir, exist_ok=True)
            delete_old_generated_files(pred_audio_dir)

            test_dataset = MagpieTTSDataset(
                dataset_meta=dataset_meta,
                sample_rate=model.sample_rate,
                min_duration=0.5,
                max_duration=20,
                codec_model_samples_per_frame=model.codec_model_samples_per_frame,
                bos_id=model.bos_id,
                eos_id=model.eos_id,
                context_audio_bos_id=model.context_audio_bos_id,
                context_audio_eos_id=model.context_audio_eos_id,
                audio_bos_id=model.audio_bos_id,
                audio_eos_id=model.audio_eos_id,
                num_audio_codebooks=model.num_audio_codebooks,
                prior_scaling_factor=None,
                load_cached_codes_if_available=False,
                dataset_type='test',
                tokenizer_config=None,
                load_16khz_audio=model.model_type == 'single_encoder_sv_tts',
                use_text_conditioning_tokenizer=model.use_text_conditioning_encoder,
                pad_context_text_to_max_duration=model.pad_context_text_to_max_duration,
                context_duration_min=context_duration_min,
                context_duration_max=context_duration_max,
            )
            assert len(test_dataset) == len(manifest_records), f"Dataset length and manifest length should be the same. Dataset length: {len(test_dataset)}, Manifest length: {len(manifest_records)}"

            test_dataset.text_tokenizer = model.tokenizer
            # Set phoneme prob = 1 for g2p
            g2p = None
            if isinstance(model.tokenizer, AggregatedTTSTokenizer):
                g2p = model.tokenizer.tokenizers["english_phoneme"].g2p
            elif isinstance(model.tokenizer, IPATokenizer):
                g2p = model.tokenizer.g2p
            if g2p is not None:
                g2p.phoneme_probability = 1.0
            test_dataset.text_conditioning_tokenizer = model.text_conditioning_tokenizer

            test_data_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=test_dataset.collate_fn,
                num_workers=2,
                shuffle=False,
            )

            item_idx = 0
            all_rtf_metrics = []
            codec_file_paths = []
            for bidx, batch in enumerate(test_data_loader):
                print(f"Processing batch {bidx} out of {len(test_data_loader)} of dataset {dataset}")
                batch_cuda = {}
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch_cuda[key] = batch[key].cuda()
                    else:
                        batch_cuda[key] = batch[key]

                st = time.time()
                predicted_audio, predicted_audio_lens, predicted_codes, predicted_codes_lens, rtf_metrics, cross_attention_maps, _ = model.infer_batch(
                    batch_cuda,
                    max_decoder_steps=440,
                    temperature=temperature,
                    topk=topk,
                    use_cfg=use_cfg,
                    cfg_scale=cfg_scale,
                    return_cross_attn_probs=True,
                    apply_attention_prior=apply_attention_prior,
                    prior_epsilon=attention_prior_epsilon,
                    lookahead_window_size=attention_prior_lookahead_window,
                    estimate_alignment_from_layers=estimate_alignment_from_layers,
                    apply_prior_to_layers=apply_prior_to_layers,
                    start_prior_after_n_audio_steps=start_prior_after_n_audio_steps,
                    use_local_transformer_for_inference=use_local_transformer,
                    maskgit_n_steps=maskgit_n_steps,
                    maskgit_noise_scale=maskgit_noise_scale,
                    maskgit_fixed_schedule=maskgit_fixed_schedule,
                    maskgit_sampling_type=maskgit_sampling_type
                )

                all_rtf_metrics.append(rtf_metrics)
                et = time.time()
                print(f"Time taken for inference: {et-st}", predicted_audio.size())
                for idx in range(predicted_audio.size(0)):
                    cross_attn_map_image = Image.fromarray(cross_attention_maps[idx])
                    cross_attn_map_image.save(os.path.join(audio_dir, f"cross_attn_map_{item_idx}.png"))

                    predicted_audio_np = predicted_audio[idx].float().detach().cpu().numpy()
                    predicted_audio_np = predicted_audio_np[:predicted_audio_lens[idx]]
                    audio_path = os.path.join(pred_audio_dir, f"predicted_audio_{item_idx}.wav")
                    sf.write(audio_path, predicted_audio_np, model.sample_rate)
                    codes_path = os.path.join(pred_audio_dir, f"predicted_codes_{item_idx}.pt")
                    predicted_codes_current = predicted_codes[idx, :, :predicted_codes_lens[idx]] # C, T'
                    torch.save(predicted_codes_current, codes_path)
                    codec_file_paths.append(codes_path)
                    context_audio_path = manifest_records[item_idx].get('context_audio_filepath', None)
                    target_audio_path = manifest_records[item_idx].get('audio_filepath', None)
                    if context_audio_path is not None:
                        context_audio_path = os.path.join(dataset_meta_info[dataset]['audio_dir'], context_audio_path)
                    if target_audio_path is not None:
                        target_audio_path = os.path.join(dataset_meta_info[dataset]['audio_dir'], target_audio_path)
                    if os.path.exists(context_audio_path):
                        shutil.copy(context_audio_path, os.path.join(audio_dir, f"context_audio_{item_idx}.wav"))
                    if os.path.exists(target_audio_path):
                        shutil.copy(target_audio_path, os.path.join(audio_dir, f"target_audio_{item_idx}.wav"))
                    item_idx += 1

            mean_rtf_metrics = {}
            for key in all_rtf_metrics[0]:
                mean_rtf_metrics[key] = float(np.mean([m[key] for m in all_rtf_metrics]))

            metrics, filewise_metrics = evaluate_generated_audio.evaluate(
                dataset_meta[dataset]['manifest_path'],
                dataset_meta[dataset]['audio_dir'],
                pred_audio_dir,
                language=language,
                sv_model_type=sv_model,
                asr_model_name=asr_model_name,
                codecmodel_path=codecmodel_path if compute_fcd else None
            )
            metrics_n_repeated.append(metrics)
            with open(os.path.join(eval_dir, f"{dataset}_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            with open(os.path.join(eval_dir, f"{dataset}_filewise_metrics_{repeat_idx}.json"), "w") as f:
                # Indent for better readability
                json.dump(filewise_metrics, f, indent=4)

            with open(os.path.join(eval_dir, f"{dataset}_rtf_metrics_{repeat_idx}.json"), "w") as f:
                json.dump(mean_rtf_metrics, f, indent=4)

            with open(all_experiment_csv, "a") as f:
                data = f"{checkpoint_name},{dataset},{metrics['cer_filewise_avg']},{metrics['wer_filewise_avg']},{metrics['cer_cumulative']},{metrics['wer_cumulative']},{metrics['ssim_pred_gt_avg']},{metrics['ssim_pred_context_avg']},{metrics['ssim_gt_context_avg']},{metrics['ssim_pred_gt_avg_alternate']},{metrics['ssim_pred_context_avg_alternate']},{metrics['ssim_gt_context_avg_alternate']},{metrics['cer_gt_audio_cumulative']},{metrics['wer_gt_audio_cumulative']}"
                if compute_fcd:
                    data += f",{metrics['frechet_codec_distance']}"
                data += "\n"
                f.write(data)
                print(f"Wrote metrics for {checkpoint_name} and {dataset} to {all_experiment_csv}")

            output_png_file = Path(eval_dir) / f"{dataset}_violin_{repeat_idx}.png"
            create_violin_plots(filewise_metrics, violin_plot_metrics, output_png_file)

            # Clean up temporary codec files
            for codes_file in codec_file_paths:
                os.remove(codes_file)

        metric_keys = ['cer_filewise_avg', 'wer_filewise_avg', 'cer_cumulative', 'wer_cumulative',
                       'ssim_pred_gt_avg', 'ssim_pred_context_avg', 'ssim_gt_context_avg',
                       'ssim_pred_gt_avg_alternate', 'ssim_pred_context_avg_alternate', 'ssim_gt_context_avg_alternate',
                       'cer_gt_audio_cumulative', 'wer_gt_audio_cumulative'
                       ]
        if compute_fcd:
            metric_keys.append('frechet_codec_distance')
        metrics_mean_ci = compute_mean_and_confidence_interval(metrics_n_repeated, metric_keys, confidence=confidence_level)
        all_experiment_csv_with_ci = os.path.join(out_dir, "all_experiment_metrics_with_ci.csv")
        if not os.path.exists(all_experiment_csv_with_ci):
            with open(all_experiment_csv_with_ci, "w") as f:
                header = "checkpoint_name,dataset,cer_filewise_avg,wer_filewise_avg,cer_cumulative,wer_cumulative,ssim_pred_gt_avg,ssim_pred_context_avg,ssim_gt_context_avg,ssim_pred_gt_avg_alternate,ssim_pred_context_avg_alternate,ssim_gt_context_avg_alternate,cer_gt_audio_cumulative,wer_gt_audio_cumulative"
                if compute_fcd:
                    header += ",frechet_codec_distance"
                header += "\n"
                f.write(header)
        with open(all_experiment_csv_with_ci, "a") as f:
            data = f"{checkpoint_name},{dataset},{metrics_mean_ci['cer_filewise_avg']},{metrics_mean_ci['wer_filewise_avg']},{metrics_mean_ci['cer_cumulative']},{metrics_mean_ci['wer_cumulative']},{metrics_mean_ci['ssim_pred_gt_avg']},{metrics_mean_ci['ssim_pred_context_avg']},{metrics_mean_ci['ssim_gt_context_avg']},{metrics_mean_ci['ssim_pred_gt_avg_alternate']},{metrics_mean_ci['ssim_pred_context_avg_alternate']},{metrics_mean_ci['ssim_gt_context_avg_alternate']},{metrics_mean_ci['cer_gt_audio_cumulative']},{metrics_mean_ci['wer_gt_audio_cumulative']}"
            if compute_fcd:
                data += f",{metrics_mean_ci['frechet_codec_distance']}"
            data += "\n"
            f.write(data)
            print(f"Wrote metrics with CI for {checkpoint_name} and {dataset} to {all_experiment_csv_with_ci}")


        measurements = [m['ssim_pred_context_avg'] for m in metrics_n_repeated]
        ssim_current = np.mean(measurements)
        ssim_per_dataset.append(ssim_current)
        measurements = [m['cer_cumulative'] for m in metrics_n_repeated]
        cer_current = np.mean(measurements)
        cer_per_dataset.append(cer_current)

    # Average across datasets
    ssim = np.mean(ssim_per_dataset)
    cer = np.mean(cer_per_dataset)
    if clean_up_disk:
        shutil.rmtree(out_dir)
    return cer, ssim


def main():
    parser = argparse.ArgumentParser(description='Experiment Evaluation')
    parser.add_argument('--hparams_files', type=str, default="/datap/misc/continuouscheckpoints_ks3ks3/multiencoder_small_sp_ks3_hparams.yaml,/datap/misc/continuouscheckpoints_ks3ks3/decodercontext_small_sp_ks3Correct_hparams.yaml")
    parser.add_argument('--hparams_file_from_wandb', action='store_true')
    parser.add_argument('--checkpoint_files', type=str, default="/datap/misc/continuouscheckpoints_ks3ks3/multiencoder_small_sp_ks3_epoch302.ckpt,/datap/misc/continuouscheckpoints_ks3ks3/decodercontext_small_sp_ks3Correct_epoch305.ckpt")
    parser.add_argument('--nemo_files', type=str, default=None)
    parser.add_argument('--codecmodel_path', type=str, default="/datap/misc/checkpoints/12.5_FPS_causal_13codebooks_codecmodel.nemo", help="Path to codec model (used for FCD computation unless --disable_fcd is specified)")
    parser.add_argument('--datasets', type=str, default="libri_unseen_test_12.5")
    parser.add_argument('--base_exp_dir', type=str, default="/datap/misc/eosmountedresson/")
    parser.add_argument('--draco_exp_dir', type=str, default="/lustre/fsw/llmservice_nemo_speechlm/users/pneekhara/gitrepos/experiments/NewT5TTS_FixedPosEmb/AllKernselSize3/EdressonCodecExperiments/")
    parser.add_argument('--server_address', type=str, default="pneekhara@login-eos02.eos.clusters.nvidia.com")
    parser.add_argument('--exp_names', type=str, default="koel_12.5_FPS_causal_13codebooks_codecmodel_context5sec_LTN1,koel_12.5_FPS_causal_13codebooks_codecmodel_context5sec_LTN3")
    parser.add_argument('--local_ckpt_dir', type=str, default="/datap/misc/experiment_checkpoints/localtransformer")
    parser.add_argument('--out_dir', type=str, default="/datap/misc/Evals/LocalTransformerAblations2")
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--use_cfg', action='store_true')
    parser.add_argument('--use_local_transformer', action='store_true', help="Enables use of local transformer for inference; applies to both Autoregressive and MaskGit sampling.")
    parser.add_argument('--maskgit_n_steps', type=int, default=3)
    parser.add_argument('--maskgit_noise_scale', type=float, default=0.0)
    parser.add_argument('--maskgit_fixed_schedule', type=int, nargs='+', default=None)
    parser.add_argument('--maskgit_sampling_type', default=None, choices=["default", "alternate", "causal", "purity_causal", "purity_default"])    
    parser.add_argument('--cfg_scale', type=float, default=2.5)
    parser.add_argument('--apply_attention_prior', action='store_true')
    parser.add_argument('--attention_prior_epsilon', type=float, default=1e-3)
    parser.add_argument('--attention_prior_lookahead_window', type=int, default=10)
    parser.add_argument('--estimate_alignment_from_layers', type=str, default=None)
    parser.add_argument('--apply_prior_to_layers', type=str, default=None)
    parser.add_argument('--start_prior_after_n_audio_steps', type=int, default=10)
    parser.add_argument('--topk', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sv_model', type=str, default="titanet") # titanet, wavlm
    parser.add_argument('--asr_model_name', type=str, default="nvidia/parakeet-tdt-1.1b") # stt_en_conformer_transducer_large, nvidia/parakeet-ctc-0.6b
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--confidence_level', type=float, default=0.95)
    parser.add_argument('--legacy_codebooks', action='store_true')
    parser.add_argument('--clean_up_disk', action='store_true')
    parser.add_argument('--cer_target', type=float, default=None)
    parser.add_argument('--ssim_target', type=float, default=None)
    parser.add_argument('--log_exp_name', action='store_true', help="Include the experiment name (derived from the checkpoint path) in the output folder name.")
    parser.add_argument('--disable_fcd', action='store_true', help="Disable Frechet Codec Distance computation")
    parser.add_argument('--violin_plot_metrics', type=str, nargs='*', default=['cer','pred_context_ssim'], help="Which metrics to add the violin plot.")
    args = parser.parse_args()

    # FCD computation is enabled by default, disabled only when --disable_fcd is specified
    compute_fcd = not args.disable_fcd

    estimate_alignment_from_layers = None
    if args.estimate_alignment_from_layers is not None:
        estimate_alignment_from_layers = [int(l.strip()) for l in args.estimate_alignment_from_layers.split(",")]
    apply_prior_to_layers = None
    if args.apply_prior_to_layers is not None:
        apply_prior_to_layers = [int(l.strip()) for l in args.apply_prior_to_layers.split(",")]

    run_inference_w_args = partial(
        run_inference,
        datasets=args.datasets.split(","),
        out_dir=args.out_dir,
        temperature=args.temperature,
        topk=args.topk,
        codecmodel_path=args.codecmodel_path,
        use_cfg=args.use_cfg,
        cfg_scale=args.cfg_scale,
        batch_size=args.batch_size,
        sv_model=args.sv_model,
        asr_model_name=args.asr_model_name,
        num_repeats=args.num_repeats,
        apply_attention_prior=args.apply_attention_prior,
        attention_prior_epsilon=args.attention_prior_epsilon,
        attention_prior_lookahead_window=args.attention_prior_lookahead_window,
        estimate_alignment_from_layers=estimate_alignment_from_layers,
        apply_prior_to_layers=apply_prior_to_layers,
        start_prior_after_n_audio_steps=args.start_prior_after_n_audio_steps,
        confidence_level=args.confidence_level,
        use_local_transformer=args.use_local_transformer,
        maskgit_n_steps=args.maskgit_n_steps,
        maskgit_noise_scale=args.maskgit_noise_scale,
        maskgit_fixed_schedule=args.maskgit_fixed_schedule,
        maskgit_sampling_type=args.maskgit_sampling_type,
        legacy_codebooks=args.legacy_codebooks,
        clean_up_disk=args.clean_up_disk,
        hparams_file_from_wandb=args.hparams_file_from_wandb,
        log_exp_name=args.log_exp_name,
        compute_fcd=compute_fcd,
        violin_plot_metrics=args.violin_plot_metrics
    )

    # Mode 1: Run inference from provided hparams and checkpoint files
    if (args.hparams_files is not None) and (args.checkpoint_files is not None) and (args.hparams_files != "null") and (args.checkpoint_files != "null"):
        hparam_files = args.hparams_files.split(",")
        checkpoint_files = args.checkpoint_files.split(",")
        print("Running inference for hparams files: ", hparam_files)
        print("Running inference for checkpoint files: ", checkpoint_files)
        assert len(hparam_files) == len(checkpoint_files), "Number of hparams files and checkpoint files should be the same."
        for hparams_file, checkpoint_file in zip(hparam_files, checkpoint_files):
            cer, ssim = run_inference_w_args(
                hparams_file=hparams_file,
                checkpoint_file=checkpoint_file,
                nemo_file=None,
            )
        return
    # Mode 2: Run inference from a .nemo file
    elif args.nemo_files:
        print(f"Running inference for nemo file: {args.nemo_files}")
        for nemo_file in args.nemo_files.split(","):
            cer, ssim = run_inference_w_args(
                hparams_file=None,
                checkpoint_file=None,
                nemo_file=nemo_file,
            )
    # Mode 3: Discover and run experiments from a base directory
    #   Mount DRACO_EXP_DIR to BASE_EXP_DIR as follows:
    #   sshfs -o allow_other pneekhara@draco-oci-dc-02.draco-oci-iad.nvidia.com:/lustre/fsw/portfolios/llmservice/users/pneekhara/gitrepos/experiments/NewT5AllFixedFresh /datap/misc/dracomount/
    elif args.base_exp_dir:
        BASE_EXP_DIR = args.base_exp_dir
        DRACO_EXP_DIR = args.draco_exp_dir
        if args.exp_names is None:
            exp_names = os.listdir(BASE_EXP_DIR)
        else:
            exp_names = args.exp_names.split(",")

        for exp_name in exp_names:
            exp_dir = os.path.join(BASE_EXP_DIR, exp_name)
            # recurisvely look for hparams.yaml
            try:
                hparams_file = glob.glob(f"{exp_dir}/**/hparams.yaml", recursive=True)[0]
                checkpoints_dir = glob.glob(f"{exp_dir}/**/checkpoints", recursive=True)[0]
                last_checkpoint = (glob.glob(f"{checkpoints_dir}/*last.ckpt"))[0]
            except:
                print(f"Skipping experiment {exp_name} as hparams or last checkpoint not found.")
                continue
            last_checkpoint_path_draco = last_checkpoint.replace(BASE_EXP_DIR, DRACO_EXP_DIR)
            epoch_num = last_checkpoint.split("epoch=")[1].split("-")[0]

            checkpoint_copy_path = os.path.join(args.local_ckpt_dir, f"{exp_name}_epoch_{epoch_num}.ckpt")
            hparams_copy_path = os.path.join(args.local_ckpt_dir, f"{exp_name}_hparams.yaml")

            scp_command = f"scp {args.server_address}:{last_checkpoint_path_draco} {checkpoint_copy_path}"
            print(f"Running command: {scp_command}")
            os.system(scp_command)
            print("Copied checkpoint.")
            hparams_path_draco = hparams_file.replace(BASE_EXP_DIR, DRACO_EXP_DIR)
            scp_command_hparams = f"scp {args.server_address}:{hparams_path_draco} {hparams_copy_path}"
            print(f"Running command: {scp_command_hparams}")
            os.system(scp_command_hparams)
            print("Copied hparams file.")
            print("Hparams file path: ", hparams_copy_path)
            print("Checkpoint file path: ", checkpoint_copy_path)
            run_inference_w_args(
                hparams_copy_path,
                checkpoint_copy_path,
                nemo_file=None,
            )
    else:
        parser.error(
            "You must provide a model to run. Please specify either:\n"
            "1. --hparams_files and --checkpoint_files\n"
            "2. --nemo_file\n"
            "3. --base_exp_dir to discover experiments"
        )
    if args.cer_target is not None and cer > float(args.cer_target):
        raise ValueError()
    if args.ssim_target is not None and ssim < float(args.ssim_target):
        raise ValueError()


if __name__ == '__main__':
    main()
