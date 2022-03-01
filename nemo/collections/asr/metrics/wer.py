# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import List

import editdistance
import torch
from torchmetrics import Metric

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.utils import logging

__all__ = ['word_error_rate', 'WER', 'move_dimension_to_the_front']


def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string. Hypotheses and references must have same
    length.
    Args:
      hypotheses: list of hypotheses
      references: list of references
      use_cer: bool, set True to enable cer
    Returns:
      (float) average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer


def move_dimension_to_the_front(tensor, dim_index):
    all_dims = list(range(tensor.ndim))
    return tensor.permute(*([dim_index] + all_dims[:dim_index] + all_dims[dim_index + 1 :]))


class WER(Metric):
    """
    This metric computes numerator and denominator for Overall Word Error Rate (WER) between prediction and reference
    texts. When doing distributed training/evaluation the result of ``res=WER(predictions, targets, target_lengths)``
    calls will be all-reduced between all workers using SUM operations. Here ``res`` contains three numbers
    ``res=[wer, total_levenstein_distance, total_number_of_words]``.

    If used with PytorchLightning LightningModule, include wer_numerator and wer_denominators inside validation_step
    results. Then aggregate (sum) then at the end of validation epoch to correctly compute validation WER.

    Example:
        def validation_step(self, batch, batch_idx):
            ...
            wer_num, wer_denom = self.__wer(predictions, transcript, transcript_len)
            return {'val_loss': loss_value, 'val_wer_num': wer_num, 'val_wer_denom': wer_denom}

        def validation_epoch_end(self, outputs):
            ...
            wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
            wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
            tensorboard_logs = {'validation_loss': val_loss_mean, 'validation_avg_wer': wer_num / wer_denom}
            return {'val_loss': val_loss_mean, 'log': tensorboard_logs}

    Args:
        vocabulary: List of strings that describes the vocabulary of the dataset.
        batch_dim_index: Index of the batch dimension of ``targets`` and ``predictions`` parameters of ``__call__``,
            ``forward``, ``update``, ``ctc_decoder_predictions_tensor`` methods. Can be either 0 or 1.
        use_cer: Whether to use Character Error Rate instead of Word Error Rate.
        ctc_decode: Whether to use CTC decoding or not. Currently, must be set.
        log_prediction: Whether to log a single decoded sample per call.
        fold_consecutive: Whether repeated consecutive characters should be folded into one when decoding.

    Returns:
        res: a tuple of 3 zero dimensional float32 ``torch.Tensor` objects: a WER score, a sum of Levenstein's
            distances for all prediction - reference pairs, total number of words in all references.
    """

    def __init__(
        self,
        vocabulary,
        batch_dim_index=0,
        use_cer=False,
        ctc_decode=True,
        log_prediction=True,
        dist_sync_on_step=False,
        fold_consecutive=True,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.batch_dim_index = batch_dim_index
        self.blank_id = len(vocabulary)
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])
        self.use_cer = use_cer
        self.ctc_decode = ctc_decode
        self.log_prediction = log_prediction
        self.fold_consecutive = fold_consecutive

        self.add_state("scores", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)
        self.add_state("words", default=torch.tensor(0), dist_reduce_fx='sum', persistent=False)

    def ctc_decoder_predictions_tensor(
        self, predictions: torch.Tensor, predictions_len: torch.Tensor = None, return_hypotheses: bool = False,
    ) -> List[str]:
        """
        Decodes a sequence of labels to words

        Args:
            predictions: An integer torch.Tensor of shape [Batch, Time] (if ``batch_index_dim == 0``) or [Time, Batch]
                (if ``batch_index_dim == 1``) of integer indices that correspond to the index of some character in the
                label set.
            predictions_len: Optional tensor of length `Batch` which contains the integer lengths
                of the sequence in the padded `predictions` tensor.
            return_hypotheses: Bool flag whether to return just the decoding predictions of the model
                or a Hypothesis object that holds information such as the decoded `text`,
                the `alignment` of emited by the CTC Model, and the `length` of the sequence (if available).
                May also contain the log-probabilities of the decoder (if this method is called via
                transcribe())

        Returns:
            Either a list of str which represent the CTC decoded strings per sample,
            or a list of Hypothesis objects containing additional information.
        """
        hypotheses = []
        # Drop predictions to CPU
        predictions = move_dimension_to_the_front(predictions, self.batch_dim_index)
        prediction_cpu_tensor = predictions.long().cpu()
        # iterate over batch
        for ind in range(prediction_cpu_tensor.shape[0]):
            if self.fold_consecutive:
                prediction = prediction_cpu_tensor[ind].detach().numpy().tolist()
                if predictions_len is not None:
                    prediction = prediction[: predictions_len[ind]]
                # CTC decoding procedure
                decoded_prediction = []
                previous = self.blank_id
                for p in prediction:
                    if (p != previous or previous == self.blank_id) and p != self.blank_id:
                        decoded_prediction.append(p)
                    previous = p
            else:
                prediction = prediction_cpu_tensor[ind].detach()
                if predictions_len is not None:
                    prediction = prediction[: predictions_len[ind]]
                decoded_prediction = prediction[prediction != self.blank_id].tolist()

            text = self.decode_tokens_to_str(decoded_prediction)

            if not return_hypotheses:
                hypothesis = text
            else:
                hypothesis = Hypothesis(
                    y_sequence=None,
                    score=-1.0,
                    text=text,
                    alignments=prediction,
                    length=predictions_len[ind] if predictions_len is not None else 0,
                )

            hypotheses.append(hypothesis)
        return hypotheses

    def decode_tokens_to_str(self, tokens: List[int]) -> str:
        """
        Implemented in order to decoder a token list into a string.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A decoded string.
        """
        hypothesis = ''.join(self.decode_ids_to_tokens(tokens))
        return hypothesis

    def decode_ids_to_tokens(self, tokens: List[int]) -> List[str]:
        """
        Implemented in order to decode a token id list into a token list.
        A token list is the string representation of each token id.

        Args:
            tokens: List of int representing the token ids.

        Returns:
            A list of decoded tokens.
        """
        token_list = [self.labels_map[c] for c in tokens if c != self.blank_id]
        return token_list

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
        predictions_lengths: torch.Tensor = None,
    ):
        """
        Updates metric state.
        Args:
            predictions: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            targets: an integer torch.Tensor of shape ``[Batch, Time]`` (if ``batch_dim_index == 0``) or
                ``[Time, Batch]`` (if ``batch_dim_index == 1``)
            target_lengths: an integer torch.Tensor of shape ``[Batch]``
            predictions_lengths: an integer torch.Tensor of shape ``[Batch]``
        """
        words = 0.0
        scores = 0.0
        references = []
        with torch.no_grad():
            # prediction_cpu_tensor = tensors[0].long().cpu()
            targets_cpu_tensor = targets.long().cpu()
            targets_cpu_tensor = move_dimension_to_the_front(targets_cpu_tensor, self.batch_dim_index)
            tgt_lenths_cpu_tensor = target_lengths.long().cpu()

            # iterate over batch
            for ind in range(targets_cpu_tensor.shape[0]):
                tgt_len = tgt_lenths_cpu_tensor[ind].item()
                target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
                reference = self.decode_tokens_to_str(target)
                references.append(reference)
            if self.ctc_decode:
                hypotheses = self.ctc_decoder_predictions_tensor(predictions, predictions_lengths)
            else:
                raise NotImplementedError("Implement me if you need non-CTC decode on predictions")

        if self.log_prediction:
            logging.info(f"\n")
            logging.info(f"reference:{references[0]}")
            logging.info(f"predicted:{hypotheses[0]}")

        for h, r in zip(hypotheses, references):
            if self.use_cer:
                h_list = list(h)
                r_list = list(r)
            else:
                h_list = h.split()
                r_list = r.split()
            words += len(r_list)
            # Compute Levenstein's distance
            scores += editdistance.eval(h_list, r_list)

        self.scores = torch.tensor(scores, device=self.scores.device, dtype=self.scores.dtype)
        self.words = torch.tensor(words, device=self.words.device, dtype=self.words.dtype)
        # return torch.tensor([scores, words]).to(predictions.device)

    def compute(self):
        scores = self.scores.detach().float()
        words = self.words.detach().float()
        return scores / words, scores, words
