import json
import torch
import torchaudio
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter

from nemo.collections.audio.parts.utils.resampling import resample


def merge_audio(pred_audio: torch.Tensor, user_audio: torch.Tensor,):
    T1, T2 = pred_audio.shape[0], user_audio.shape[0]
    max_len = max(T1, T2)
    pred_audio_padded = torch.nn.functional.pad(pred_audio, (0, max_len - T1), mode='constant', value=0)
    user_audio_padded = torch.nn.functional.pad(user_audio, (0, max_len - T2), mode='constant', value=0)

    # combine audio in a multichannel audio
    combined_wav = torch.cat(
        [
            user_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
            pred_audio_padded.squeeze().unsqueeze(0).detach().cpu(),
        ],
        dim=0,
    )
    return combined_wav


class S2SDuplexPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, save_sample_rate, save_prefix=False, write_interval="batch_and_epoch"):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.save_sample_rate = save_sample_rate
        self.save_prefix = save_prefix

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        """Write predictions for a single batch to disk.
        
        This method processes predictions from a single batch and writes them to disk.
        For audio predictions, it saves individual audio files in the appropriate directory.
        For text predictions, it stores them for later writing in write_on_epoch_end.
        
        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module (model).
            prediction: Dictionary containing prediction results for the current batch.
            batch_indices: List of batch indices for the current batch.
            batch: The input batch that was used to generate predictions.
            batch_idx: Index of the current batch.
            dataloader_idx: Index of the current dataloader.
        """

        dataset_name = list(trainer.datamodule.cfg.predict_ds.datasets.keys())[dataloader_idx]

        if "audio" in prediction:
            for cur_audio, cur_audio_len, cur_sample_id, source_audio, source_audio_len in zip(
                prediction["audio"].to(torch.float32),
                prediction["audio_len"],
                prediction["sample_id"],
                prediction["source_audio"],
                prediction["source_audio_len"],
            ):
                audio_dir = self.output_dir / dataset_name / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)

                cur_audio = cur_audio[:cur_audio_len]
                if trainer.datamodule.cfg.target_sample_rate != self.save_sample_rate:
                    cur_audio = resample(
                        cur_audio,
                        trainer.datamodule.cfg.target_sample_rate,
                        self.save_sample_rate,
                    )

                source_audio = source_audio[:source_audio_len]
                if trainer.datamodule.cfg.source_sample_rate != self.save_sample_rate:
                    source_audio = resample(
                        source_audio,
                        trainer.datamodule.cfg.source_sample_rate,
                        self.save_sample_rate,
                    )

                merged_audio = merge_audio(pred_audio=cur_audio, user_audio=source_audio)

                if self.save_prefix:
                    save_path = audio_dir / f"{dataset_name}_{cur_sample_id}.wav"
                else:
                    save_path = audio_dir / f"{cur_sample_id}.wav"
                torchaudio.save(
                    save_path,
                    merged_audio.float().cpu(),
                    self.save_sample_rate,
                )

            del prediction["audio"]
            del prediction["audio_len"]

        del prediction["tokens_audio"]
        del prediction["source_audio"]
        del prediction["source_audio_len"]

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """Write predictions to JSONL files at the end of each epoch.
        
        This method aggregates all predictions from the epoch and writes them to JSONL files,
        organized by dataset name.
        
        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module (model).
            predictions: List of lists of predictions from each dataloader.
            batch_indices: List of lists of batch indices from each dataloader.
        """
        if len(list(trainer.datamodule.cfg.predict_ds.datasets.keys())) == 1:
            predictions = [predictions]
        for dataloader_idx, cur_predictions in enumerate(predictions):
            dataset_name = list(trainer.datamodule.cfg.predict_ds.datasets.keys())[dataloader_idx]
            out_file = self.output_dir / dataset_name / f"predictions_{trainer.global_rank}.jsonl"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                for batch_prediction in cur_predictions:
                    for cur_sample_id, cur_text, cur_text_token, cur_token_len in zip(
                        batch_prediction["sample_id"],
                        batch_prediction["text"],
                        batch_prediction["tokens_text"],
                        batch_prediction["tokens_len"],
                    ):
                        result = {
                            "sample_id": cur_sample_id,
                            "text": cur_text,
                            "tokens_text": cur_text_token.tolist()[:cur_token_len],
                        }
                        f.write(json.dumps(result) + "\n")
