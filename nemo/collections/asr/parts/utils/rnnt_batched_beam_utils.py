from typing import Optional

import torch

from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis, NBestHypotheses
from nemo.utils.enum import PrettyStrEnum

# Constants used for hashing text sequences.
MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2**64

# Constants used for initializing and managing beam search hypotheses.
INACTIVE_SCORE = -float("inf")  # Represents the score of inactive hypotheses.
INIT_POINTER_VALUE = -1  # Initial value for pointers in the hypothesis tree structure.
INIT_HASH_VALUE = 0  # Initial hash value for transcript hashes.
INIT_PREFIX_HASH_VALUE = 0  # Initial hash value for prefix hashes.
NON_EXISTENT_LABEL_VALUE = -1  # Placeholder value for non-existent labels in hypotheses. Needs to be negative.


def hash_text(prev_hash: torch.Tensor, add_labels: torch.Tensor) -> torch.Tensor:
    """
    Computes a new hash value by updating previous hash tensor with added labels tensor.
    Reference: https://stackoverflow.com/a/77213071

    Args:
        prev_hash (torch.Tensor): A tensor representing the previous hash value.
        add_labels (torch.Tensor): A tensor containing added labels.

    Returns:
        torch.Tensor: A tensor representing the updated hash value.
    """
    return prev_hash * MULTIPLIER + INCREMENT + add_labels


class BlankLMScoreMode(PrettyStrEnum):
    """
    Defines the strategies for handling blank token scores in a external Ngram LM
    when combined with an automatic speech recognition (ASR) model.
    """

    NO_SCORE = "no_score"
    """No score for blank."""
    LM_WEIGHTED_FULL = "lm_weighted_full"
    """Blank score for LM is set equal to blank score from ASR model; non-blank LM scores are reweighted to sum to 1."""


class PruningMode(PrettyStrEnum):
    """Specifies when pruning is applied external Ngram LM shallow fusion.."""

    EARLY = "early"
    """Hyps are pruned based on ASR probs, then rescored with LM"""
    LATE = "late"
    """Hyps are scored based on combined ASR and LM probs., then pruned"""


class BatchedBeamHyps:
    """Class to store batch of beam hypotheses (labels, time_indices, scores) for efficient batched beam decoding"""

    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        init_length: int,
        blank_index: int,
        device: Optional[torch.device] = None,
        float_dtype: Optional[torch.dtype] = None,
        store_prefix_hashes: Optional[bool] = False,
    ):
        """
        Initializes the batched beam search utility for RNN-T decoding.
        Args:
            batch_size (int): Batch size.
            beam_size (int): Beam size.
            init_length (int): The initial maximum length of the hypotheses.
            blank_index (int): The index representing the blank token in the vocabulary.
            device (Optional[torch.device]): The device on which tensors will be allocated. Defaults to None.
            float_dtype (Optional[torch.dtype]): The floating-point data type for scores. Defaults to None.
            store_prefix_hashes (Optional[bool]): Whether to store prefix hashes for hypotheses. Defaults to False.
        """

        if beam_size <= 0:
            raise ValueError("Beam size must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        if init_length <= 0:
            raise ValueError("Initial hypothesis lengths must be greater than 0.")

        self.INACTIVE_SCORE_TENSOR = torch.tensor(INACTIVE_SCORE, device=device, dtype=float_dtype)

        self.store_prefix_hashes = store_prefix_hashes
        self._max_length = init_length
        self.beam_size = beam_size
        self.blank_index = blank_index
        self.batch_size = batch_size
        self.batch_indices = torch.arange(self.batch_size, device=device)

        self.current_lengths_nb = torch.zeros(
            [batch_size, self.beam_size], device=device, dtype=torch.long
        )  # non-blank lengths
        self.current_lengths_wb = torch.zeros(
            [batch_size, self.beam_size], device=device, dtype=torch.long
        )  # full lengths

        # Initializing tree structure for hypothesis storing
        self.transcript_wb = torch.full(
            (batch_size, self.beam_size, self._max_length),
            fill_value=NON_EXISTENT_LABEL_VALUE,
            device=device,
            dtype=torch.long,
        )  # current labels
        self.transcript_wb_prev_ptr = torch.full(
            (batch_size, self.beam_size, self._max_length),
            fill_value=INIT_POINTER_VALUE,
            device=device,
            dtype=torch.long,
        )  # links to prefices

        # Initializing beam scores: Initially, only a single hypothesis is active within the beam.
        self.scores = torch.full(
            [batch_size, self.beam_size], device=device, dtype=float_dtype, fill_value=INACTIVE_SCORE
        )
        self.scores[:, 0].fill_(0.0)

        self.last_label = torch.full(
            (batch_size, self.beam_size), fill_value=NON_EXISTENT_LABEL_VALUE, device=device, dtype=torch.long
        )
        self.next_timestep = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)
        self.last_timestep_lasts = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)

        self.transcript_hash = torch.full(
            [batch_size, self.beam_size], device=device, dtype=torch.long, fill_value=INIT_HASH_VALUE
        )
        if store_prefix_hashes:
            self.transcript_prefix_hash = torch.full(
                [batch_size, self.beam_size], device=device, dtype=torch.long, fill_value=INIT_PREFIX_HASH_VALUE
            )

    def clear_(self):
        """
        Clears and resets the internal state of the object.
        """

        self.current_lengths_nb.fill_(0)
        self.current_lengths_wb.fill_(0)

        self.transcript_wb.fill_(0)
        self.transcript_wb_prev_ptr.fill_(INIT_POINTER_VALUE)

        self.scores.fill_(INACTIVE_SCORE)
        self.scores[:, 0].fill_(0.0)

        self.last_label.fill_(NON_EXISTENT_LABEL_VALUE)
        self.next_timestep.fill_(0)
        self.last_timestep_lasts.fill_(0)

        self.transcript_hash.fill_(INIT_HASH_VALUE)
        if self.store_prefix_hashes:
            self.transcript_prefix_hash.fill_(INIT_PREFIX_HASH_VALUE)

    def _allocate_more(self):
        """
        Dynamically allocates more memory for the internal buffers.
        This method doubles the size of the following tensors: `transcript_wb`, `transcript_wb_prev_ptr`.
        """
        self.transcript_wb = torch.cat(
            (self.transcript_wb, torch.full_like(self.transcript_wb, fill_value=NON_EXISTENT_LABEL_VALUE)), dim=-1
        )
        self.transcript_wb_prev_ptr = torch.cat(
            (self.transcript_wb_prev_ptr, torch.full_like(self.transcript_wb_prev_ptr, fill_value=INIT_POINTER_VALUE)),
            dim=-1,
        )

        self._max_length *= 2

    def add_results_(
        self,
        hyps_indices,
        next_labels,
        next_hyps_prob,
    ):
        """
        Updated batch of beam hypotheses with labels. If the maximum allowed length
        is exceeded, underlying memory is doubled.
        Args:
            hyps_indices (torch.Tensor): Indices of the hypotheses to be updated.
            next_labels (torch.Tensor): Labels corresponding to the next step in the beam search.
            next_hyps_prob (torch.Tensor): Probabilities of the next hypotheses.
        """

        if (self.current_lengths_wb + 1).max() >= self._max_length:
            self._allocate_more()

        self.add_results_no_checks_(
            hyps_indices=hyps_indices,
            next_labels=next_labels,
            next_hyps_prob=next_hyps_prob,
        )

    def add_results_no_checks_(
        self,
        hyps_indices,
        next_labels,
        next_hyps_prob,
    ):
        """
        Updated batch of beam hypotheses with labels.
        Args:
            hyps_indices (torch.Tensor): Indices of the hypotheses to be updated.
            next_labels (torch.Tensor): Labels corresponding to the next step in the beam search.
            next_hyps_prob (torch.Tensor): Probabilities of the next hypotheses.
        """

        self.transcript_wb.scatter_(dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=next_labels.unsqueeze(-1))
        self.transcript_wb_prev_ptr.scatter_(
            dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=hyps_indices.unsqueeze(-1)
        )

        extended_with_blank = next_labels == self.blank_index
        extended_with_label = (~extended_with_blank) & (next_labels >= 0)
        self.current_lengths_nb.copy_(
            torch.gather(self.current_lengths_nb, dim=-1, index=hyps_indices) + extended_with_label
        )
        torch.add(self.current_lengths_wb, 1, out=self.current_lengths_wb)

        torch.where(extended_with_blank | extended_with_label, next_hyps_prob, self.scores, out=self.scores)

        self.next_timestep.copy_(self.current_lengths_wb - self.current_lengths_nb)
        self.last_timestep_lasts.copy_(
            torch.where(
                extended_with_blank,
                0,
                torch.gather(self.last_timestep_lasts, dim=-1, index=hyps_indices) + extended_with_label,
            )
        )

        prev_transcript_hash = torch.gather(self.transcript_hash, dim=-1, index=hyps_indices)
        prev_transcript_hash = None
        if self.store_prefix_hashes:
            prev_transcript_prefix_hash = torch.gather(self.transcript_prefix_hash, dim=-1, index=hyps_indices)

        last_labels = torch.gather(self.last_label, dim=-1, index=hyps_indices)
        # track last label
        torch.where(
            extended_with_label,
            next_labels,
            last_labels,
            out=self.last_label,
        )

        # update hashes and prefix hashes
        torch.where(
            extended_with_label,
            hash_text(prev_transcript_hash, next_labels),
            prev_transcript_hash,
            out=self.transcript_hash,
        )
        if self.store_prefix_hashes and prev_transcript_prefix_hash is not None:
            torch.where(
                extended_with_label, prev_transcript_hash, prev_transcript_prefix_hash, out=self.transcript_prefix_hash
            )

    def recombine_hyps(self):
        """
        Recombines hypotheses in the beam search by merging equivalent hypotheses and updating their scores.
        This method identifies hypotheses that are equivalent based on their transcript hash, last label,
        and current lengths. It then merges these equivalent hypotheses by computing a new score using
        log-sum-exp over their scores and updates the scores tensor accordingly.
        Returns:
            None: The method modifies the `self.scores` tensor in place to reflect the recombined hypotheses.
        """

        if self.beam_size <= 1:
            return

        hyps_equal = (
            (self.transcript_hash[:, :, None] == self.transcript_hash[:, None, :])
            & (self.last_label[:, :, None] == self.last_label[:, None, :])
            & (self.current_lengths_nb[:, :, None] == self.current_lengths_nb[:, None, :])
        )

        scores_matrix = torch.where(
            hyps_equal,
            self.scores[:, None, :].expand(self.batch_size, self.beam_size, self.beam_size),
            self.INACTIVE_SCORE_TENSOR,
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size, device=scores_argmax.device, dtype=torch.long)[None, :] == scores_argmax
        )
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        torch.where(scores_to_keep, new_scores.to(self.scores.dtype), self.INACTIVE_SCORE_TENSOR, out=self.scores)

    def remove_duplicates(self, labels, total_logps):
        """
        Removes duplicate hypotheses that may arise after updating beam hypotheses with labels during the beam search process.
        Args:
            labels (torch.Tensor): A tensor containing the labels for the current beam
                search step. Shape: [batch_size, beam_size, ...].
            total_logps (torch.Tensor): A tensor containing the total log probabilities
                for the current beam search step. Shape: [batch_size, beam_size, ...].
        Returns:
            torch.Tensor: Updated total log probabilities with duplicates removed.
                Shape: [batch_size, beam_size, ...].
        """

        if self.beam_size <= 1:
            return total_logps

        # updating hashes for label expansions
        non_blank_mask = labels != self.blank_index
        expansion_hashes = hash_text(self.transcript_hash.unsqueeze(-1), labels)
        expansion_hashes = torch.where(non_blank_mask, expansion_hashes, self.transcript_hash.unsqueeze(-1)).view(
            self.batch_size, -1
        )

        # masking inactive hypotheses
        inactive_hyps_mask = self.scores != INACTIVE_SCORE
        masked_hashes = torch.where(inactive_hyps_mask, self.transcript_hash, -1)

        init_expansions_equal = (expansion_hashes[:, :, None] == masked_hashes[:, None, :]).any(dim=-1)

        init_expansions_equal = torch.logical_and(non_blank_mask.view(self.batch_size, -1), init_expansions_equal)
        expansions_equal = expansion_hashes[:, :, None] == expansion_hashes[:, None, :]
        expansion_scores = total_logps.view(self.batch_size, -1)
        expansion_scores = torch.where(init_expansions_equal, INACTIVE_SCORE, expansion_scores)
        expansion_scores = expansion_scores[:, None, :].expand(expansions_equal.shape)

        expansion_scores = torch.where(expansions_equal, expansion_scores, INACTIVE_SCORE)
        expansion_scores, expansion_scores_argmax = expansion_scores.max(dim=-1)

        scores_range = torch.arange(
            expansion_scores_argmax.shape[-1], device=expansion_scores_argmax.device, dtype=torch.long
        )
        scores_to_keep = scores_range[None, :] == expansion_scores_argmax
        total_logps = torch.where(scores_to_keep, expansion_scores, INACTIVE_SCORE).view(
            self.batch_size, self.beam_size, -1
        )

        return total_logps

    def recombine_prefixes(self, label_logps: torch.Tensor, active_mask: torch.Tensor):
        """
        Recombines prefixes (prefix search) in the beam search process by updating scores for hypotheses
        that share common prefixes.
        Args:
            label_logps (torch.Tensor): A tensor of shape (batch_size, beam_size, vocab_size)
                containing the log probabilities of the labels for each beam.
            active_mask (torch.Tensor): A boolean tensor of shape (batch_size, beam_size)
                indicating which beams are active.
        """

        if self.beam_size <= 1:
            return

        # if hypotheses are empty skip
        if (self.current_lengths_wb == 0).any():
            return

        # mask prefix hashes if hypotheses of the beam do not have prefixes (e.g. no non-blank labels were appended)
        prefix_hashes = torch.where(self.current_lengths_nb == 0, -2, self.transcript_prefix_hash)

        prefix_equal = self.transcript_hash[:, None, :] == prefix_hashes[:, :, None]

        last_labels = torch.where(self.last_label == NON_EXISTENT_LABEL_VALUE, self.blank_index, self.last_label)
        prefix_labels = last_labels.unsqueeze(1).repeat((1, self.beam_size, 1))
        prefix_scores = self.scores.unsqueeze(1).repeat((1, self.beam_size, 1))

        prefix_label_logps = torch.gather(label_logps, dim=-1, index=prefix_labels)
        prefix_label_logps = prefix_scores + prefix_label_logps.transpose(dim0=-1, dim1=-2)
        prefix_label_logps = torch.where(prefix_equal, prefix_label_logps, INACTIVE_SCORE)
        prefix_label_logps = torch.logsumexp(prefix_label_logps, dim=-1)

        to_update_mask = torch.logical_and(active_mask, self.scores != INACTIVE_SCORE)
        self.scores = torch.where(to_update_mask, torch.logaddexp(self.scores, prefix_label_logps), self.scores)

    def to_hyps_list(self, score_norm: bool = True) -> list[Hypothesis]:
        """
        Converts the batched beam search results into a list of signle best hypotheses for each batch.
        Args:
            score_norm (bool):  If True, normalize the scores before sorting. Defaults to True.
        Returns:
            list[Hypothesis]: A list where each element corresponds to a batch and contains
            best hypothesis.
        """

        self.flatten_sort(score_norm)

        scores = self.scores[self.batch_indices, 0].tolist()

        max_idx = self.current_lengths_wb.max() - 1
        transcripts = self.transcript_wb[..., 0, : max_idx + 1]
        timestamps = torch.cumsum(transcripts == self.blank_index, dim=-1).cpu().numpy()
        transcripts = transcripts.cpu().detach().numpy()
        hypotheses = [
            Hypothesis(
                score=scores[batch_idx],
                y_sequence=transcripts[batch_idx][
                    mask := (transcripts[batch_idx] >= 0) & (transcripts[batch_idx] != self.blank_index)
                ],
                timestamp=timestamps[batch_idx][mask],
                alignments=None,
                dec_state=None,
            )
            for batch_idx in range(self.batch_size)
        ]
        return hypotheses

    def to_nbest_hyps_list(self, score_norm: bool = True) -> list[NBestHypotheses]:
        """
        Converts the batched beam search results into a list of N-best hypotheses for each batch.
        Args:
            score_norm (bool, optional): If True, normalize the scores before sorting. Defaults to True.
        Returns:
            list[NBestHypotheses]: A list where each element corresponds to a batch and contains
            N-best hypotheses.
        """

        self.flatten_sort(score_norm)

        scores = self.scores.tolist()

        max_idx = self.current_lengths_wb.max() - 1
        transcripts = self.transcript_wb[..., : max_idx + 1]
        timestamps = torch.cumsum(transcripts == self.blank_index, dim=-1).cpu().numpy()
        transcripts = transcripts.cpu().detach().numpy()
        hypotheses = [
            NBestHypotheses(
                [
                    Hypothesis(
                        score=scores[batch_idx][beam_idx],
                        y_sequence=transcripts[batch_idx][beam_idx][
                            mask := (transcripts[batch_idx][beam_idx] >= 0)
                            & (transcripts[batch_idx][beam_idx] != self.blank_index)
                        ],
                        timestamp=timestamps[batch_idx][beam_idx][mask],
                        alignments=None,
                        dec_state=None,
                    )
                    for beam_idx in range(self.beam_size)
                    if scores[batch_idx][beam_idx] > float('-inf')
                ]
            )
            for batch_idx in range(self.batch_size)
        ]
        return hypotheses

    def flatten_sort(self, score_norm: bool = True) -> list[Hypothesis]:
        """
        Sorts and flattens the tree structure of hypotheses in a batched beam search decoding process.
        Args:
            score_norm (bool, optional): If True, normalizes the scores by dividing
                them by the current lengths of the hypotheses plus one. Defaults to True.
        Returns:
            list[Hypothesis]: A list of sorted and flattened hypotheses.
        This method performs the following steps:
        1. Normalizes the scores if `score_norm` is True.
        2. Sorts the normalized scores in descending order and retrieves the corresponding indices.
        3. Iteratively reconstructs the tokens and timestamps for each hypothesis in reverse order.
        4. Updates the internal state of the object, including transcripts, timestamps, scores,
           lengths, labels, and other metadata, based on the sorted order.
        """

        # add one for consistency with non-batched decodings, that use SOS.
        normalized_scores = (
            self.scores / (self.current_lengths_nb.to(self.scores.dtype) + 1) if score_norm else self.scores
        )
        normalized_scores, indices = torch.sort(normalized_scores, dim=-1, descending=True)

        max_idx = self.current_lengths_wb.max() - 1
        tokens_list = []
        ptrs = indices
        for idx in range(max_idx, -1, -1):
            tokens = self.transcript_wb[self.batch_indices.unsqueeze(-1), ptrs, idx]
            ptrs = self.transcript_wb_prev_ptr[self.batch_indices.unsqueeze(-1), ptrs, idx]

            tokens_list.insert(0, tokens)

        self.transcript_wb[..., : max_idx + 1] = torch.stack(tokens_list, dim=-1)
        self.transcript_wb_prev_ptr[..., : max_idx + 1] = torch.arange(self.beam_size)[None, :, None].expand(
            self.batch_size, self.beam_size, max_idx + 1
        )

        self.scores.copy_(torch.gather(self.scores, dim=-1, index=indices))
        self.current_lengths_nb.copy_(torch.gather(self.current_lengths_nb, dim=-1, index=indices))
        self.current_lengths_wb.copy_(torch.gather(self.current_lengths_wb, dim=-1, index=indices))

        self.last_label.copy_(torch.gather(self.last_label, dim=-1, index=indices))
        self.next_timestep.copy_(torch.gather(self.next_timestep, dim=-1, index=indices))
        self.last_timestep_lasts.copy_(torch.gather(self.last_timestep_lasts, dim=-1, index=indices))

        self.transcript_hash.copy_(torch.gather(self.transcript_hash, dim=-1, index=indices))
        if self.store_prefix_hashes:
            self.transcript_prefix_hash.copy_(torch.gather(self.transcript_prefix_hash, dim=-1, index=indices))


class BatchedBeamHypsTDT:
    """
    Class to store batched hypotheses (labels, time_indices, scores) for efficient RNNT decoding
    """

    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        init_length: int,
        blank_index: int,
        device: Optional[torch.device] = None,
        float_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initializes the batched beam search utility for RNN-T decoding.
        Args:
            batch_size (int): Batch size.
            beam_size (int): Beam size.
            init_length (int): The initial maximum length of the hypotheses.
            blank_index (int): The index representing the blank token in the vocabulary.
            device (Optional[torch.device]): The device on which tensors will be allocated. Defaults to None.
            float_dtype (Optional[torch.dtype]): The floating-point data type for scores. Defaults to None.
        """

        if beam_size <= 0:
            raise ValueError("Beam size must be greater than 0.")
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        if init_length <= 0:
            raise ValueError("Initial hypothesis lengths must be greater than 0.")

        self.device = device
        self.INACTIVE_SCORE_TENSOR = torch.tensor(INACTIVE_SCORE, device=device, dtype=float_dtype)
        self.ZERO_TENSOR = torch.tensor(0, device=device, dtype=torch.long)

        self._max_length = init_length
        self.beam_size = beam_size
        self.blank_index = blank_index
        self.batch_size = batch_size
        self.batch_indices = torch.arange(self.batch_size, device=device)

        self.current_lengths_nb = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)
        self.current_lengths_wb = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)

        # Initializing tree structure for hypothesis storing
        self.transcript_wb = torch.full(
            (batch_size, self.beam_size, self._max_length),
            fill_value=NON_EXISTENT_LABEL_VALUE,
            device=device,
            dtype=torch.long,
        )
        self.timestamps = torch.zeros((batch_size, self.beam_size, self._max_length), device=device, dtype=torch.long)
        self.transcript_wb_prev_ptr = torch.full(
            (batch_size, self.beam_size, self._max_length),
            fill_value=INIT_POINTER_VALUE,
            device=device,
            dtype=torch.long,
        )

        # Initializing beam scores: Initially, only a single hypothesis is active within the beam.
        self.scores = torch.full(
            [batch_size, self.beam_size], device=device, dtype=float_dtype, fill_value=INACTIVE_SCORE
        )
        self.scores[:, 0].fill_(0.0)

        self.last_label = torch.full(
            [batch_size, self.beam_size], fill_value=NON_EXISTENT_LABEL_VALUE, device=device, dtype=torch.long
        )
        self.next_timestamp = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)
        self.last_timestep_lasts = torch.zeros((batch_size, self.beam_size), device=device, dtype=torch.long)

        self.transcript_hash = torch.zeros([batch_size, self.beam_size], device=device, dtype=torch.long)

    def clear_(self):
        """
        Clears and resets the internal state of the object.
        """

        self.current_lengths_nb.fill_(0)
        self.current_lengths_wb.fill_(0)
        self.last_label.fill_(NON_EXISTENT_LABEL_VALUE)
        self.scores.fill_(INACTIVE_SCORE)
        self.scores[:, 0].fill_(0.0)
        self.next_timestamp.fill_(0)
        self.last_timestep_lasts.fill_(0)

        self.transcript_wb.fill_(0)
        self.timestamps.fill_(0)
        self.transcript_wb_prev_ptr.fill_(INIT_POINTER_VALUE)

        self.transcript_hash.fill_(0)

    def _allocate_more(self):
        """
        Dynamically allocates more memory for the internal buffers used in the beam search process.
        This method doubles the size of the following tensors: `transcript_wb`, `timestamps`, `transcript_wb_prev_ptr`
        """

        self.transcript_wb = torch.cat(
            (self.transcript_wb, torch.full_like(self.transcript_wb, fill_value=NON_EXISTENT_LABEL_VALUE)), dim=-1
        )
        self.timestamps = torch.cat((self.timestamps, torch.zeros_like(self.timestamps)), dim=-1)
        self.transcript_wb_prev_ptr = torch.cat(
            (self.transcript_wb_prev_ptr, torch.full_like(self.transcript_wb_prev_ptr, fill_value=INIT_POINTER_VALUE)),
            dim=-1,
        )

        self._max_length *= 2

    def add_results_(
        self,
        hyps_indices,
        next_labels,
        next_hyps_prob,
        next_label_durations,
    ):
        """
        Updated batch of beam hypotheses with labels. If the maximum allowed length
        is exceeded, underlying memory is doubled.
        Args:
            hyps_indices (torch.Tensor): Indices of the hypotheses to be updated.
            next_labels (torch.Tensor): Labels corresponding to the next step in the beam search.
            next_hyps_prob (torch.Tensor): Probabilities of the next hypotheses.
            next_label_durations (torch.Tensor): Durations associated with the next labels.
        """

        if (self.current_lengths_wb + 1).max() >= self._max_length:
            self._allocate_more()

        self.add_results_no_checks_(
            hyps_indices=hyps_indices,
            next_labels=next_labels,
            next_hyps_prob=next_hyps_prob,
            next_label_durations=next_label_durations,
        )

    def add_results_no_checks_(self, hyps_indices, next_labels, next_hyps_prob, next_label_durations):
        """
        Updated batch of beam hypotheses with labels.
        Args:
            hyps_indices (torch.Tensor): Indices of the hypotheses to be updated.
            next_labels (torch.Tensor): Labels corresponding to the next step in the beam search.
            next_hyps_prob (torch.Tensor): Probabilities of the next hypotheses.
            next_label_durations (torch.Tensor): Durations associated with the next labels.
        """

        next_label_durations = torch.where(next_labels < 0, 0, next_label_durations)
        timesteps = torch.gather(self.next_timestamp, dim=-1, index=hyps_indices)

        self.transcript_wb.scatter_(dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=next_labels.unsqueeze(-1))
        self.timestamps.scatter_(
            dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=(timesteps + next_label_durations).unsqueeze(-1)
        )
        self.transcript_wb_prev_ptr.scatter_(
            dim=-1, index=self.current_lengths_wb.unsqueeze(-1), src=hyps_indices.unsqueeze(-1)
        )

        extended_with_blank = next_labels == self.blank_index
        extended_with_label = (~extended_with_blank) & (next_labels >= 0)
        self.current_lengths_nb.copy_(
            torch.gather(self.current_lengths_nb, dim=-1, index=hyps_indices) + extended_with_label
        )
        torch.add(self.current_lengths_wb, 1, out=self.current_lengths_wb)

        torch.where(extended_with_blank | extended_with_label, next_hyps_prob, self.scores, out=self.scores)

        torch.where(next_labels >= 0, timesteps + next_label_durations, timesteps, out=self.next_timestamp)
        torch.where(
            (extended_with_blank | extended_with_label) & next_label_durations > 0,
            self.ZERO_TENSOR,
            torch.gather(self.last_timestep_lasts, dim=-1, index=hyps_indices) + extended_with_label,
            out=self.last_timestep_lasts,
        )

        prev_transcript_hash = torch.gather(self.transcript_hash, dim=-1, index=hyps_indices)
        last_labels = torch.gather(self.last_label, dim=-1, index=hyps_indices)
        # track last label
        torch.where(
            extended_with_label,
            next_labels,
            last_labels,
            out=self.last_label,
        )

        # update hashes and prefix hashes
        torch.where(
            extended_with_label,
            hash_text(prev_transcript_hash, next_labels),
            prev_transcript_hash,
            out=self.transcript_hash,
        )

    def recombine_hyps(self):
        """
        Recombines hypotheses in the beam search by merging equivalent hypotheses and updating their scores.
        This method identifies hypotheses that are equivalent based on their transcript hash, last label,
        current lengths and current timestamps. It then merges these equivalent hypotheses by computing a new score using
        log-sum-exp over their scores and updates the scores tensor accordingly.
        Returns:
            None: The method modifies the `self.scores` tensor in place to reflect the recombined hypotheses.
        """

        if self.beam_size <= 1:
            return

        hyps_equal = (
            (self.transcript_hash[:, :, None] == self.transcript_hash[:, None, :])
            & (self.last_label[:, :, None] == self.last_label[:, None, :])
            & (self.current_lengths_nb[:, :, None] == self.current_lengths_nb[:, None, :])
            & (self.next_timestamp[:, :, None] == self.next_timestamp[:, None, :])
        )

        scores_matrix = torch.where(
            hyps_equal,
            self.scores[:, None, :].expand(self.batch_size, self.beam_size, self.beam_size),
            self.INACTIVE_SCORE_TENSOR,
        )
        scores_argmax = scores_matrix.argmax(-1, keepdim=False)
        scores_to_keep = (
            torch.arange(self.beam_size, device=scores_argmax.device, dtype=torch.long)[None, :] == scores_argmax
        )
        new_scores = torch.logsumexp(scores_matrix, dim=-1, keepdim=False)
        torch.where(scores_to_keep, new_scores.to(self.scores.dtype), self.INACTIVE_SCORE_TENSOR, out=self.scores)

    def to_hyps_list(self, score_norm: bool = True) -> list[Hypothesis]:
        """
        Converts the batched beam search results into a list of signle best hypotheses for each batch.
        Args:
            score_norm (bool):  If True, normalize the scores before sorting. Defaults to True.
        Returns:
            list[Hypothesis]: A list where each element corresponds to a batch and contains
            best hypothesis.
        """

        self.flatten_sort(score_norm)

        scores = self.scores[self.batch_indices, 0].tolist()

        max_idx = self.current_lengths_wb.max() - 1
        timestamps = self.timestamps[..., 0, : max_idx + 1].cpu().detach().numpy()
        transcripts = self.transcript_wb[..., 0, : max_idx + 1].cpu().detach().numpy()
        hypotheses = [
            Hypothesis(
                score=scores[batch_idx],
                y_sequence=transcripts[batch_idx][
                    mask := (transcripts[batch_idx] >= 0) & (transcripts[batch_idx] != self.blank_index)
                ],
                timestamp=timestamps[batch_idx][mask],
                alignments=None,
                dec_state=None,
            )
            for batch_idx in range(self.batch_size)
        ]
        return hypotheses

    def to_nbest_hyps_list(self, score_norm: bool = True) -> list[Hypothesis]:
        """
        Converts the batched beam search results into a list of N-best hypotheses for each batch.
        Args:
            score_norm (bool, optional): If True, normalize the scores before sorting. Defaults to True.
        Returns:
            list[NBestHypotheses]: A list where each element corresponds to a batch and contains
            N-best hypotheses.
        """

        self.flatten_sort(score_norm)

        scores = self.scores.tolist()

        max_idx = self.current_lengths_wb.max() - 1
        transcripts = self.transcript_wb[..., : max_idx + 1].cpu().detach().numpy()
        timestamps = self.timestamps[..., : max_idx + 1].cpu().detach().numpy()
        hypotheses = [
            NBestHypotheses(
                [
                    Hypothesis(
                        score=scores[batch_idx][beam_idx],
                        y_sequence=transcripts[batch_idx][beam_idx][
                            mask := (transcripts[batch_idx][beam_idx] >= 0)
                            & (transcripts[batch_idx][beam_idx] != self.blank_index)
                        ],
                        timestamp=timestamps[batch_idx][beam_idx][mask],
                        alignments=None,
                        dec_state=None,
                    )
                    for beam_idx in range(self.beam_size)
                    if scores[batch_idx][beam_idx] > float('-inf')
                ]
            )
            for batch_idx in range(self.batch_size)
        ]
        return hypotheses

    def flatten_sort(self, score_norm: bool = True):
        """
        Sorts and flattens the tree structure of hypotheses in a batched beam search decoding process.
        Args:
            score_norm (bool, optional): If True, normalizes the scores by dividing
                them by the current lengths of the hypotheses plus one. Defaults to True.
        Returns:
            list[Hypothesis]: A list of sorted and flattened hypotheses.
        This method performs the following steps:
        1. Normalizes the scores if `score_norm` is True.
        2. Sorts the normalized scores in descending order and retrieves the corresponding indices.
        3. Iteratively reconstructs the tokens and timestamps for each hypothesis in reverse order.
        4. Updates the internal state of the object, including transcripts, timestamps, scores,
           lengths, labels, and other metadata, based on the sorted order.
        """

        # add one for consistency with non-batched decodings, that use SOS.
        normalized_scores = (
            self.scores / (self.current_lengths_nb.to(self.scores.dtype) + 1) if score_norm else self.scores
        )
        normalized_scores, indices = torch.sort(normalized_scores, dim=-1, descending=True)

        max_idx = self.current_lengths_wb.max() - 1
        tokens_list = []
        timestamps_list = []
        ptrs = indices
        for idx in range(max_idx, -1, -1):
            tokens = self.transcript_wb[self.batch_indices.unsqueeze(-1), ptrs, idx]
            timestamps = self.timestamps[self.batch_indices.unsqueeze(-1), ptrs, idx]
            ptrs = self.transcript_wb_prev_ptr[self.batch_indices.unsqueeze(-1), ptrs, idx]

            tokens_list.insert(0, tokens)
            timestamps_list.insert(0, timestamps)

        self.transcript_wb[..., : max_idx + 1] = torch.stack(tokens_list, dim=-1)
        self.timestamps[..., : max_idx + 1] = torch.stack(timestamps_list, dim=-1)
        self.transcript_wb_prev_ptr[..., : max_idx + 1] = torch.arange(self.beam_size)[None, :, None].expand(
            self.batch_size, self.beam_size, max_idx + 1
        )

        self.scores.copy_(torch.gather(self.scores, dim=-1, index=indices))
        self.current_lengths_nb.copy_(torch.gather(self.current_lengths_nb, dim=-1, index=indices))
        self.current_lengths_wb.copy_(torch.gather(self.current_lengths_wb, dim=-1, index=indices))

        self.last_label.copy_(torch.gather(self.last_label, dim=-1, index=indices))
        self.next_timestamp.copy_(torch.gather(self.next_timestamp, dim=-1, index=indices))
        self.last_timestep_lasts.copy_(torch.gather(self.last_timestep_lasts, dim=-1, index=indices))

        self.transcript_hash.copy_(torch.gather(self.transcript_hash, dim=-1, index=indices))
