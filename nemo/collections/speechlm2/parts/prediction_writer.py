import json
import torch
import torchaudio
from pathlib import Path
from lightning.pytorch.callbacks import BasePredictionWriter

from nemo.collections.audio.parts.utils.resampling import resample


class S2SDuplexPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, save_sample_rate, write_interval):
        super().__init__(write_interval)
        self.output_dir = Path(output_dir)
        self.save_sample_rate = save_sample_rate

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
            for cur_audio, cur_audio_len, cur_sample_id in zip(
                prediction["audio"].to(torch.float32),
                prediction["audio_len"],
                prediction["sample_id"]
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
                torchaudio.save(
                    audio_dir / f"{cur_sample_id}.wav",
                    cur_audio.unsqueeze(0).float().cpu(),
                    self.save_sample_rate,
                )
            
            del prediction["audio"]
            del prediction["audio_len"]
        
        del prediction["tokens_audio"]

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
