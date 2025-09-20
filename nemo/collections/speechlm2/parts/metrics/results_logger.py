from whisper_normalizer.english import EnglishTextNormalizer
import json
import os
import shutil
from collections import defaultdict
from typing import List, Optional
import glob
import time

import torch
import torchaudio

from nemo.utils import logging


def safe_remove_path(path):
    try:
        shutil.rmtree(path)
    except:
        pass


def get_rank():
    """Get the current rank in distributed training"""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0
    except:
        return 0


def get_world_size():
    """Get the world size in distributed training"""
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        else:
            return 1
    except:
        return 1


class ResultsLogger:
    """
    Saves audios and a json file with the model outputs.
    Now supports distributed training with result merging across ranks.
    """

    def __init__(self, save_path):
        self.save_path = save_path
        self.audio_save_path = os.path.join(save_path, "pred_wavs")
        os.makedirs(self.audio_save_path, exist_ok=True)
        self.matadata_save_path = os.path.join(save_path, "metadatas")
        os.makedirs(self.matadata_save_path, exist_ok=True)
        self.cached_results = defaultdict(list)
        self.normalizer = EnglishTextNormalizer()

    def reset(self):
        metadata_files = os.listdir(self.matadata_save_path)
        for f in metadata_files:
            open(os.path.join(self.matadata_save_path, f), 'w').close()
        self.cached_results = defaultdict(list)
        return self

    @staticmethod
    def merge_and_save_audio(
            out_audio_path: str, pred_audio: torch.Tensor, pred_audio_sr: int, user_audio: torch.Tensor,
            user_audio_sr: int
    ) -> None:
        user_audio = torchaudio.functional.resample(user_audio.float(), user_audio_sr, pred_audio_sr)
        T1, T2 = pred_audio.shape[0], user_audio.shape[0]
        max_len = max(T1, T2)
        pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
        user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)
        combined_wav = torch.cat(
            [
                user_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
                pred_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
            ],
            dim=0,
        )
        torchaudio.save(out_audio_path, combined_wav.squeeze(), pred_audio_sr)
        logging.info(f"Audio saved at: {out_audio_path}")

    def update(
            self,
            name,
            refs,
            hyps,
            asr_hyps,
            samples_id,
            pred_audio,
            pred_audio_sr,
            user_audio,
            user_audio_sr,
    ):
        rank = get_rank()

        for i in range(len(refs)):
            sample_id = samples_id[i][:150]
            # Add rank info to audio filename to avoid conflicts
            if pred_audio is not None:
                out_audio_path = os.path.join(self.audio_save_path, f"{name}_{sample_id}_rank{rank}.wav")
                self.merge_and_save_audio(out_audio_path, pred_audio[i], pred_audio_sr, user_audio[i], user_audio_sr)

            out_dict = {
                "id": sample_id,
                "target_text": refs[i],
                "pred_text": hyps[i],
                "pred_audio": asr_hyps[i] if asr_hyps is not None else None,
            }
            self.cached_results[name].append(out_dict)

        logging.info(f"Metadata for {name} dataset cached (rank {rank}).")

    def _merge_rank_files(self, dataset_name: str) -> List[dict]:
        """
        Merge results from all ranks for a given dataset.
        Only executed by rank 0.
        """
        rank = get_rank()
        world_size = get_world_size()

        if rank != 0:
            return []

        all_results = []

        # Wait a bit for all ranks to finish writing their files
        time.sleep(2)

        # Collect results from all ranks
        for r in range(world_size):
            rank_file = os.path.join(self.matadata_save_path, f"{dataset_name}_rank{r}.json")

            # Wait for the file to exist (with timeout)
            wait_time = 0
            max_wait = 30  # seconds
            while not os.path.exists(rank_file) and wait_time < max_wait:
                time.sleep(1)
                wait_time += 1

            if os.path.exists(rank_file):
                try:
                    with open(rank_file, 'r', encoding='utf-8') as fin:
                        rank_results = json.load(fin)
                        if isinstance(rank_results, list):
                            all_results.extend(rank_results)
                        else:
                            logging.warning(f"Unexpected format in {rank_file}: {type(rank_results)}")
                except Exception as e:
                    logging.warning(f"Failed to read {rank_file}: {e}")
            else:
                logging.warning(f"Rank file {rank_file} not found after waiting {max_wait} seconds")

        # logging.info(f"Total merged results for {dataset_name}: {len(all_results)} items")
        return all_results

    def compute_and_save(self, special_subset_names: Optional[List[str]] = None):
        """
        Saves all cached results. Now supports distributed training:
        1. Each rank saves its own results with rank suffix
        2. Rank 0 collects and merges all results into final files
        3. Computes metrics on the merged results

        Args:
            special_subset_names: A list of validation subset names to compute accuracy for.

        Returns:
            A dictionary of calculated metrics (accuracy and empty_rate) for the special subsets.
            E.g., {'web-qa': {'acc': 0.8, 'empty_rate': 0.1}, ...}
        """
        if special_subset_names is None:
            special_subset_names = ['web-qa', 'llama-qa', 'trivia-qa']

        rank = get_rank()
        world_size = get_world_size()
        metrics_results = {}

        # Step 1: Each rank saves its own results with rank suffix
        for name, results_list in self.cached_results.items():
            rank_json_path = os.path.join(self.matadata_save_path, f"{name}_rank{rank}.json")
            with open(rank_json_path, 'w', encoding='utf-8') as fout:
                json.dump(results_list, fout, indent=4, ensure_ascii=False)
            logging.info(f"Rank {rank} metadata file for {name} dataset saved at: {rank_json_path}")

        # Step 2: Synchronize all ranks before merging
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Step 3: Only rank 0 merges all results and computes final metrics
        if rank == 0:
            for name in self.cached_results.keys():
                # Merge results from all ranks
                merged_results = self._merge_rank_files(name)

                # Save merged results
                final_json_path = os.path.join(self.matadata_save_path, f"{name}.json")
                with open(final_json_path, 'w', encoding='utf-8') as fout:
                    json.dump(merged_results, fout, indent=4, ensure_ascii=False)
                logging.info(f"Final merged metadata file for {name} dataset saved at: {final_json_path}")

                # Compute metrics on merged results
                if name in special_subset_names and merged_results:
                    correct_count = 0
                    empty_count = 0
                    total_count = len(merged_results)

                    for item in merged_results:
                        pred_text = item["pred_text"].strip()
                        normalized_pred = self.normalizer(pred_text)

                        if not normalized_pred:
                            empty_count += 1
                            continue

                        pred_words = set(normalized_pred.split())

                        target_text = item["target_text"]
                        possible_targets = target_text.split(';')

                        is_correct = False
                        for target_option in possible_targets:
                            normalized_target_option = self.normalizer(target_option.strip())
                            target_words = set(normalized_target_option.split())

                            if not target_words or target_words.issubset(pred_words):
                                is_correct = True
                                break

                        if is_correct:
                            correct_count += 1

                    acc = correct_count / total_count if total_count > 0 else 0.0
                    empty_rate = empty_count / total_count if total_count > 0 else 0.0

                    metrics_results[name] = {'acc': torch.tensor(acc), 'empty_rate': torch.tensor(empty_rate)}
                    logging.info(
                        f"Metrics for special subset '{name}': Accuracy={acc}, Empty Rate={empty_rate} (total samples: {total_count})")

        # Step 4: Broadcast metrics from rank 0 to all other ranks
        if torch.distributed.is_available() and torch.distributed.is_initialized() and world_size > 1:
            # Convert metrics to a format that can be broadcasted
            if rank == 0:
                metrics_to_broadcast = {}
                for name, metrics in metrics_results.items():
                    metrics_to_broadcast[name] = {
                        'acc': metrics['acc'].item(),
                        'empty_rate': metrics['empty_rate'].item()
                    }
            else:
                metrics_to_broadcast = {}

            # Broadcast the metrics
            broadcast_list = [metrics_to_broadcast]
            torch.distributed.broadcast_object_list(broadcast_list, src=0)

            # Reconstruct metrics_results on all ranks
            if rank != 0:
                metrics_results = {}
                for name, metrics in broadcast_list[0].items():
                    metrics_results[name] = {
                        'acc': torch.tensor(metrics['acc']),
                        'empty_rate': torch.tensor(metrics['empty_rate'])
                    }

        return metrics_results