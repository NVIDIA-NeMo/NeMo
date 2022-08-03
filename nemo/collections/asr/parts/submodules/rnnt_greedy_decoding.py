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

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.common.parts.rnn import label_collate
from nemo.core.classes import Typing, typecheck
from nemo.core.neural_types import AcousticEncodedRepresentation, ElementType, HypothesisType, LengthsType, NeuralType
from nemo.utils import logging


def pack_hypotheses(hypotheses: List[rnnt_utils.Hypothesis], logitlen: torch.Tensor,) -> List[rnnt_utils.Hypothesis]:

    if hasattr(logitlen, 'cpu'):
        logitlen_cpu = logitlen.to('cpu')
    else:
        logitlen_cpu = logitlen

    for idx, hyp in enumerate(hypotheses):  # type: rnnt_utils.Hypothesis
        hyp.y_sequence = torch.tensor(hyp.y_sequence, dtype=torch.long)
        hyp.length = logitlen_cpu[idx]

        if hyp.dec_state is not None:
            hyp.dec_state = _states_to_device(hyp.dec_state)

    return hypotheses


def _states_to_device(dec_state, device='cpu'):
    if torch.is_tensor(dec_state):
        dec_state = dec_state.to(device)

    elif isinstance(dec_state, (list, tuple)):
        dec_state = tuple(_states_to_device(dec_i, device) for dec_i in dec_state)

    return dec_state


class _GreedyRNNTInfer(Typing):
    """A greedy transducer decoder.

    Provides a common abstraction for sample level and batch level greedy decoding.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of
            Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
    """

    @property
    def input_types(self):
        """Returns definitions of module input ports.
        """
        return {
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "partial_hypotheses": [NeuralType(elements_type=HypothesisType(), optional=True)],  # must always be last
        }

    @property
    def output_types(self):
        """Returns definitions of module output ports.
        """
        return {"predictions": [NeuralType(elements_type=HypothesisType())]}

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
    ):
        super().__init__()
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = blank_index  # Start of single index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @torch.no_grad()
    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        add_sos: bool = False,
        batch_size: Optional[int] = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Common prediction step based on the AbstractRNNTDecoder implementation.

        Args:
            label: (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden: (Optional torch.Tensor): RNN State vector
            add_sos (bool): Whether to add a zero vector at the begging as "start of sentence" token.
            batch_size: Batch size of the output tensor.

        Returns:
            g: (B, U, H) if add_sos is false, else (B, U + 1, H)
            hid: (h, c) where h is the final sequence hidden state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if isinstance(label, torch.Tensor):
            # label: [batch, 1]
            if label.dtype != torch.long:
                label = label.long()

        else:
            # Label is an integer
            if label == self._SOS:
                return self.decoder.predict(None, hidden, add_sos=add_sos, batch_size=batch_size)

            label = label_collate([[label]])

        # output: [B, 1, K]
        return self.decoder.predict(label, hidden, add_sos=add_sos, batch_size=batch_size)

    def _joint_step(self, enc, pred, log_normalize: Optional[bool] = None):
        """
        Common joint step based on AbstractRNNTJoint implementation.

        Args:
            enc: Output of the Encoder model. A torch.Tensor of shape [B, 1, H1]
            pred: Output of the Decoder model. A torch.Tensor of shape [B, 1, H2]
            log_normalize: Whether to log normalize or not. None will log normalize only for CPU.

        Returns:
             logits of shape (B, T=1, U=1, V + 1)
        """
        with torch.no_grad():
            logits = self.joint.joint(enc, pred)

            if log_normalize is None:
                if not logits.is_cuda:  # Use log softmax only if on CPU
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)
            else:
                if log_normalize:
                    logits = logits.log_softmax(dim=len(logits.shape) - 1)

        return logits


class GreedyRNNTInfer(_GreedyRNNTInfer):
    """A greedy transducer decoder.

    Sequence level greedy decoding, performed auto-repressively.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of
            Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
    """

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
        )

    @typecheck()
    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        with torch.inference_mode():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)

            self.decoder.eval()
            self.joint.eval()

            hypotheses = []
            # Process each sequence independently
            with self.decoder.as_frozen(), self.joint.as_frozen():
                for batch_idx in range(encoder_output.size(0)):
                    inseq = encoder_output[batch_idx, :, :].unsqueeze(1)  # [T, 1, D]
                    logitlen = encoded_lengths[batch_idx]

                    partial_hypothesis = partial_hypotheses[batch_idx] if partial_hypotheses is not None else None
                    hypothesis = self._greedy_decode(inseq, logitlen, partial_hypotheses=partial_hypothesis)
                    hypotheses.append(hypothesis)

            # Pack results into Hypotheses
            packed_result = pack_hypotheses(hypotheses, encoded_lengths)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,)

    @torch.no_grad()
    def _greedy_decode(
        self, x: torch.Tensor, out_len: torch.Tensor, partial_hypotheses: Optional[rnnt_utils.Hypothesis] = None
    ):
        # x: [T, 1, D]
        # out_len: [seq_len]

        # Initialize blank state and empty label set in Hypothesis
        hypothesis = rnnt_utils.Hypothesis(score=0.0, y_sequence=[], dec_state=None, timestep=[], last_token=None)

        if partial_hypotheses is not None:
            hypothesis.last_token = partial_hypotheses.last_token
            hypothesis.y_sequence = partial_hypotheses.y_sequence.cpu().tolist()
            if partial_hypotheses.dec_state is not None:
                hypothesis.dec_state = self.decoder.batch_concat_states([partial_hypotheses.dec_state])
                hypothesis.dec_state = _states_to_device(hypothesis.dec_state, x.device)

        if self.preserve_alignments:
            # Alignments is a 2-dimensional dangling list representing T x U
            hypothesis.alignments = [[]]

        # For timestep t in X_t
        for time_idx in range(out_len):
            # Extract encoder embedding at timestep t
            # f = x[time_idx, :, :].unsqueeze(0)  # [1, 1, D]
            f = x.narrow(dim=0, start=time_idx, length=1)

            # Setup exit flags and counter
            not_blank = True
            symbols_added = 0

            # While blank is not predicted, or we dont run out of max symbols per timestep
            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                # In the first timestep, we initialize the network with RNNT Blank
                # In later timesteps, we provide previous predicted label as input.
                if hypothesis.last_token is None and hypothesis.dec_state is None:
                    last_label = self._SOS
                else:
                    last_label = label_collate([[hypothesis.last_token]])

                # Perform prediction network and joint network steps.
                g, hidden_prime = self._pred_step(last_label, hypothesis.dec_state)
                logp = self._joint_step(f, g, log_normalize=None)[0, 0, 0, :]

                del g

                # torch.max(0) op doesnt exist for FP 16.
                if logp.dtype != torch.float32:
                    logp = logp.float()

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.item()  # K is the label at timestep t_s in inner loop, s >= 0.

                if self.preserve_alignments:
                    # insert logprobs into last timestep
                    hypothesis.alignments[-1].append((logp.to('cpu'), torch.tensor(k, dtype=torch.int32)))

                del logp

                # If blank token is predicted, exit inner loop, move onto next timestep t
                if k == self._blank_index:
                    not_blank = False

                    if self.preserve_alignments:
                        # convert Ti-th logits into a torch array
                        hypothesis.alignments.append([])  # blank buffer for next timestep
                else:
                    # Append token to label set, update RNN state.
                    hypothesis.y_sequence.append(k)
                    hypothesis.score += float(v)
                    hypothesis.timestep.append(time_idx)
                    hypothesis.dec_state = hidden_prime
                    hypothesis.last_token = k

                # Increment token counter.
                symbols_added += 1

        # Remove trailing empty list of Alignments
        if self.preserve_alignments:
            if len(hypothesis.alignments[-1]) == 0:
                del hypothesis.alignments[-1]

        # Unpack the hidden states
        hypothesis.dec_state = self.decoder.batch_select_state(hypothesis.dec_state, 0)

        return hypothesis


class GreedyBatchedRNNTInfer(_GreedyRNNTInfer):
    """A batch level greedy transducer decoder.

    Batch level greedy decoding, performed auto-repressively.

    Args:
        decoder_model: rnnt_utils.AbstractRNNTDecoder implementation.
        joint_model: rnnt_utils.AbstractRNNTJoint implementation.
        blank_index: int index of the blank token. Can be 0 or len(vocabulary).
        max_symbols_per_step: Optional int. The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        preserve_alignments: Bool flag which preserves the history of alignments generated during
            greedy decoding (sample / batched). When set to true, the Hypothesis will contain
            the non-null value for `alignments` in it. Here, `alignments` is a List of List of
            Tuple(Tensor (of length V + 1), Tensor(scalar, label after argmax)).

            The length of the list corresponds to the Acoustic Length (T).
            Each value in the list (Ti) is a torch.Tensor (U), representing 1 or more targets from a vocabulary.
            U is the number of target tokens for the current timestep Ti.
    """

    def __init__(
        self,
        decoder_model: rnnt_abstract.AbstractRNNTDecoder,
        joint_model: rnnt_abstract.AbstractRNNTJoint,
        blank_index: int,
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments: bool = False,
    ):
        super().__init__(
            decoder_model=decoder_model,
            joint_model=joint_model,
            blank_index=blank_index,
            max_symbols_per_step=max_symbols_per_step,
            preserve_alignments=preserve_alignments,
        )

        # Depending on availability of `blank_as_pad` support
        # switch between more efficient batch decoding technique
        if self.decoder.blank_as_pad:
            self._greedy_decode = self._greedy_decode_blank_as_pad
        else:
            self._greedy_decode = self._greedy_decode_masked

    @typecheck()
    def forward(
        self,
        encoder_output: torch.Tensor,
        encoded_lengths: torch.Tensor,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        with torch.inference_mode():
            # Apply optional preprocessing
            encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
            logitlen = encoded_lengths

            self.decoder.eval()
            self.joint.eval()

            with self.decoder.as_frozen(), self.joint.as_frozen():
                inseq = encoder_output  # [B, T, D]
                hypotheses = self._greedy_decode(
                    inseq, logitlen, device=inseq.device, partial_hypotheses=partial_hypotheses
                )

            # Pack the hypotheses results
            packed_result = pack_hypotheses(hypotheses, logitlen)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        return (packed_result,)

    def _greedy_decode_blank_as_pad(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        with torch.inference_mode():
            # x: [B, T, D]
            # out_len: [B]
            # device: torch.device

            # Initialize list of Hypothesis
            batchsize = x.shape[0]
            hypotheses = [
                rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)
            ]

            # Initialize Hidden state matrix (shared by entire batch)
            hidden = None

            # If alignments need to be preserved, register a danling list to hold the values
            if self.preserve_alignments:
                # alignments is a 3-dimensional dangling list representing B x T x U
                for hyp in hypotheses:
                    hyp.alignments = [[]]

            # Last Label buffer + Last Label without blank buffer
            # batch level equivalent of the last_label
            last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)

            # Mask buffers
            blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

            # Get max sequence length
            max_out_len = out_len.max()
            for time_idx in range(max_out_len):
                f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

                # Prepare t timestamp batch variables
                not_blank = True
                symbols_added = 0

                # Reset blank mask
                blank_mask.mul_(False)

                # Update blank mask with time mask
                # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
                # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
                blank_mask = time_idx >= out_len

                # Start inner loop
                while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                    # Batch prediction and joint network steps
                    # If very first prediction step, submit SOS tag (blank) to pred_step.
                    # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                    if time_idx == 0 and symbols_added == 0 and hidden is None:
                        g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
                    else:
                        # Perform batch step prediction of decoder, getting new states and scores ("g")
                        g, hidden_prime = self._pred_step(last_label, hidden, batch_size=batchsize)

                    # Batched joint step - Output = [B, V + 1]
                    logp = self._joint_step(f, g, log_normalize=None)[:, 0, 0, :]

                    if logp.dtype != torch.float32:
                        logp = logp.float()

                    # Get index k, of max prob for batch
                    v, k = logp.max(1)
                    del g

                    # Update blank mask with current predicted blanks
                    # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                    k_is_blank = k == self._blank_index
                    blank_mask.bitwise_or_(k_is_blank)

                    del k_is_blank

                    # If preserving alignments, check if sequence length of sample has been reached
                    # before adding alignment
                    if self.preserve_alignments:
                        # Insert logprobs into last timestep per sample
                        logp_vals = logp.to('cpu')
                        logp_ids = logp_vals.max(1)[1]
                        for batch_idx in range(batchsize):
                            if time_idx < out_len[batch_idx]:
                                hypotheses[batch_idx].alignments[-1].append(
                                    (logp_vals[batch_idx], logp_ids[batch_idx])
                                )
                        del logp_vals
                    del logp

                    # If all samples predict / have predicted prior blanks, exit loop early
                    # This is equivalent to if single sample predicted k
                    if blank_mask.all():
                        not_blank = False

                        # If preserving alignments, convert the current Uj alignments into a torch.Tensor
                        # Then preserve U at current timestep Ti
                        # Finally, forward the timestep history to Ti+1 for that sample
                        # All of this should only be done iff the current time index <= sample-level AM length.
                        # Otherwise ignore and move to next sample / next timestep.
                        if self.preserve_alignments:

                            # convert Ti-th logits into a torch array
                            for batch_idx in range(batchsize):

                                # this checks if current timestep <= sample-level AM length
                                # If current timestep > sample-level AM length, no alignments will be added
                                # Therefore the list of Uj alignments is empty here.
                                if len(hypotheses[batch_idx].alignments[-1]) > 0:
                                    hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep
                    else:
                        # Collect batch indices where blanks occurred now/past
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                        # Recover prior state for all samples which predicted blank now/past
                        if hidden is not None:
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)

                        elif len(blank_indices) > 0 and hidden is None:
                            # Reset state if there were some blank and other non-blank predictions in batch
                            # Original state is filled with zeros so we just multiply
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)

                        # Recover prior predicted label for all samples which predicted blank now/past
                        k[blank_indices] = last_label[blank_indices, 0]

                        # Update new label and hidden state for next iteration
                        last_label = k.clone().view(-1, 1)
                        hidden = hidden_prime

                        # Update predicted labels, accounting for time mask
                        # If blank was predicted even once, now or in the past,
                        # Force the current predicted label to also be blank
                        # This ensures that blanks propogate across all timesteps
                        # once they have occured (normally stopping condition of sample level loop).
                        for kidx, ki in enumerate(k):
                            if blank_mask[kidx] == 0:
                                hypotheses[kidx].y_sequence.append(ki)
                                hypotheses[kidx].timestep.append(time_idx)
                                hypotheses[kidx].score += float(v[kidx])

                        symbols_added += 1

            # Remove trailing empty list of alignments at T_{am-len} x Uj
            if self.preserve_alignments:
                for batch_idx in range(batchsize):
                    if len(hypotheses[batch_idx].alignments[-1]) == 0:
                        del hypotheses[batch_idx].alignments[-1]

        # Preserve states
        for batch_idx in range(batchsize):
            hypotheses[batch_idx].dec_state = self.decoder.batch_select_state(hidden, batch_idx)

        return hypotheses

    def _greedy_decode_masked(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
        device: torch.device,
        partial_hypotheses: Optional[List[rnnt_utils.Hypothesis]] = None,
    ):
        if partial_hypotheses is not None:
            raise NotImplementedError("`partial_hypotheses` support is not supported")

        # x: [B, T, D]
        # out_len: [B]
        # device: torch.device

        # Initialize state
        batchsize = x.shape[0]
        hypotheses = [
            rnnt_utils.Hypothesis(score=0.0, y_sequence=[], timestep=[], dec_state=None) for _ in range(batchsize)
        ]

        # Initialize Hidden state matrix (shared by entire batch)
        hidden = None

        # If alignments need to be preserved, register a danling list to hold the values
        if self.preserve_alignments:
            # alignments is a 3-dimensional dangling list representing B x T x U
            for hyp in hypotheses:
                hyp.alignments = [[]]
        else:
            alignments = None

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long, device=device)
        last_label_without_blank = last_label.clone()

        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool, device=device)

        # Get max sequence length
        max_out_len = out_len.max()

        with torch.inference_mode():
            for time_idx in range(max_out_len):
                f = x.narrow(dim=1, start=time_idx, length=1)  # [B, 1, D]

                # Prepare t timestamp batch variables
                not_blank = True
                symbols_added = 0

                # Reset blank mask
                blank_mask.mul_(False)

                # Update blank mask with time mask
                # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
                # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
                blank_mask = time_idx >= out_len

                # Start inner loop
                while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                    # Batch prediction and joint network steps
                    # If very first prediction step, submit SOS tag (blank) to pred_step.
                    # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                    if time_idx == 0 and symbols_added == 0 and hidden is None:
                        g, hidden_prime = self._pred_step(self._SOS, hidden, batch_size=batchsize)
                    else:
                        # Set a dummy label for the blank value
                        # This value will be overwritten by "blank" again the last label update below
                        # This is done as vocabulary of prediction network does not contain "blank" token of RNNT
                        last_label_without_blank_mask = last_label == self._blank_index
                        last_label_without_blank[last_label_without_blank_mask] = 0  # temp change of label
                        last_label_without_blank[~last_label_without_blank_mask] = last_label[
                            ~last_label_without_blank_mask
                        ]

                        # Perform batch step prediction of decoder, getting new states and scores ("g")
                        g, hidden_prime = self._pred_step(last_label_without_blank, hidden, batch_size=batchsize)

                    # Batched joint step - Output = [B, V + 1]
                    logp = self._joint_step(f, g, log_normalize=None)[:, 0, 0, :]

                    if logp.dtype != torch.float32:
                        logp = logp.float()

                    # Get index k, of max prob for batch
                    v, k = logp.max(1)
                    del g

                    # Update blank mask with current predicted blanks
                    # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                    k_is_blank = k == self._blank_index
                    blank_mask.bitwise_or_(k_is_blank)

                    # If preserving alignments, check if sequence length of sample has been reached
                    # before adding alignment
                    if self.preserve_alignments:
                        # Insert logprobs into last timestep per sample
                        logp_vals = logp.to('cpu')
                        logp_ids = logp_vals.max(1)[1]
                        for batch_idx in range(batchsize):
                            if time_idx < out_len[batch_idx]:
                                hypotheses[batch_idx].alignments[-1].append(
                                    (logp_vals[batch_idx], logp_ids[batch_idx])
                                )
                        del logp_vals
                    del logp

                    # If all samples predict / have predicted prior blanks, exit loop early
                    # This is equivalent to if single sample predicted k
                    if blank_mask.all():
                        not_blank = False

                        # If preserving alignments, convert the current Uj alignments into a torch.Tensor
                        # Then preserve U at current timestep Ti
                        # Finally, forward the timestep history to Ti+1 for that sample
                        # All of this should only be done iff the current time index <= sample-level AM length.
                        # Otherwise ignore and move to next sample / next timestep.
                        if self.preserve_alignments:

                            # convert Ti-th logits into a torch array
                            for batch_idx in range(batchsize):

                                # this checks if current timestep <= sample-level AM length
                                # If current timestep > sample-level AM length, no alignments will be added
                                # Therefore the list of Uj alignments is empty here.
                                if len(hypotheses[batch_idx].alignments[-1]) > 0:
                                    hypotheses[batch_idx].alignments.append([])  # blank buffer for next timestep
                    else:
                        # Collect batch indices where blanks occurred now/past
                        blank_indices = (blank_mask == 1).nonzero(as_tuple=False)

                        # Recover prior state for all samples which predicted blank now/past
                        if hidden is not None:
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, hidden, blank_indices)

                        elif len(blank_indices) > 0 and hidden is None:
                            # Reset state if there were some blank and other non-blank predictions in batch
                            # Original state is filled with zeros so we just multiply
                            # LSTM has 2 states
                            hidden_prime = self.decoder.batch_copy_states(hidden_prime, None, blank_indices, value=0.0)

                        # Recover prior predicted label for all samples which predicted blank now/past
                        k[blank_indices] = last_label[blank_indices, 0]

                        # Update new label and hidden state for next iteration
                        last_label = k.view(-1, 1)
                        hidden = hidden_prime

                        # Update predicted labels, accounting for time mask
                        # If blank was predicted even once, now or in the past,
                        # Force the current predicted label to also be blank
                        # This ensures that blanks propogate across all timesteps
                        # once they have occured (normally stopping condition of sample level loop).
                        for kidx, ki in enumerate(k):
                            if blank_mask[kidx] == 0:
                                hypotheses[kidx].y_sequence.append(ki)
                                hypotheses[kidx].timestep.append(time_idx)
                                hypotheses[kidx].score += float(v[kidx])

                    symbols_added += 1

        # Remove trailing empty list of alignments at T_{am-len} x Uj
        if self.preserve_alignments:
            for batch_idx in range(batchsize):
                if len(hypotheses[batch_idx].alignments[-1]) == 0:
                    del hypotheses[batch_idx].alignments[-1]

        # Preserve states
        for batch_idx in range(batchsize):
            hypotheses[batch_idx].dec_state = self.decoder.batch_select_state(hidden, batch_idx)

        return hypotheses


class ONNXGreedyBatchedRNNTInfer:
    def __init__(
        self, encoder_model: str, decoder_joint_model: str, max_symbols_per_step: Optional[int] = None,
    ):
        try:
            import onnx
            import onnxruntime
        except (ModuleNotFoundError, ImportError):
            raise ImportError(f"`onnx` or `onnxruntime` could not be imported, please install the libraries.\n")

        onnx_model = onnx.load(encoder_model)
        onnx.checker.check_model(onnx_model, full_check=True)
        self.encoder_model = onnx_model
        self.encoder = onnxruntime.InferenceSession(onnx_model.SerializeToString())

        onnx_model = onnx.load(decoder_joint_model)
        onnx.checker.check_model(onnx_model, full_check=True)
        self.decoder_joint_model = onnx_model
        self.decoder_joint = onnxruntime.InferenceSession(onnx_model.SerializeToString())

        logging.info("Successfully loaded encoder, decoder and joint onnx models !")

        # Will be populated at runtime
        self._blank_index = None
        self.max_symbols_per_step = max_symbols_per_step

        self._setup_encoder_input_output_keys()
        self._setup_decoder_joint_input_output_keys()
        self._setup_blank_index()

    def _setup_encoder_input_output_keys(self):
        self.encoder_inputs = list(self.encoder_model.graph.input)
        self.encoder_outputs = list(self.encoder_model.graph.output)

    def _setup_decoder_joint_input_output_keys(self):
        self.decoder_joint_inputs = list(self.decoder_joint_model.graph.input)
        self.decoder_joint_outputs = list(self.decoder_joint_model.graph.output)

    def _setup_blank_index(self):
        # ASSUME: Single input with no time length information
        dynamic_dim = 257
        shapes = self.encoder_inputs[0].type.tensor_type.shape.dim
        ip_shape = []
        for shape in shapes:
            if hasattr(shape, 'dim_param') and 'dynamic' in shape.dim_param:
                ip_shape.append(dynamic_dim)  # replace dynamic axes with constant
            else:
                ip_shape.append(int(shape.dim_value))

        enc_logits, encoded_length = self.run_encoder(
            audio_signal=torch.randn(*ip_shape), length=torch.randint(0, 1, size=(dynamic_dim,))
        )

        # prepare states
        states = self._get_initial_states(batchsize=dynamic_dim)

        # run decoder 1 step
        joint_out, states = self.run_decoder_joint(enc_logits, None, None, *states)
        log_probs, lengths = joint_out

        self._blank_index = log_probs.shape[-1] - 1  # last token of vocab size is blank token
        logging.info(
            f"Enc-Dec-Joint step was evaluated, blank token id = {self._blank_index}; vocab size = {log_probs.shape[-1]}"
        )

    def __call__(self, audio_signal: torch.Tensor, length: torch.Tensor):
        """Returns a list of hypotheses given an input batch of the encoder hidden embedding.
        Output token is generated auto-repressively.

        Args:
            encoder_output: A tensor of size (batch, features, timesteps).
            encoded_lengths: list of int representing the length of each sequence
                output sequence.

        Returns:
            packed list containing batch number of sentences (Hypotheses).
        """
        with torch.no_grad():
            # Apply optional preprocessing
            encoder_output, encoded_lengths = self.run_encoder(audio_signal=audio_signal, length=length)
            encoder_output = encoder_output.transpose([0, 2, 1])  # (B, T, D)
            logitlen = encoded_lengths

            inseq = encoder_output  # [B, T, D]
            hypotheses, timestamps = self._greedy_decode(inseq, logitlen)

            # Pack the hypotheses results
            packed_result = [rnnt_utils.Hypothesis(score=-1.0, y_sequence=[]) for _ in range(len(hypotheses))]
            for i in range(len(packed_result)):
                packed_result[i].y_sequence = torch.tensor(hypotheses[i], dtype=torch.long)
                packed_result[i].length = timestamps[i]

            del hypotheses

        return packed_result

    def _greedy_decode(self, x, out_len):
        # x: [B, T, D]
        # out_len: [B]

        # Initialize state
        batchsize = x.shape[0]
        hidden = self._get_initial_states(batchsize)
        target_lengths = torch.ones(batchsize, dtype=torch.int32)

        # Output string buffer
        label = [[] for _ in range(batchsize)]
        timesteps = [[] for _ in range(batchsize)]

        # Last Label buffer + Last Label without blank buffer
        # batch level equivalent of the last_label
        last_label = torch.full([batchsize, 1], fill_value=self._blank_index, dtype=torch.long).numpy()

        # Mask buffers
        blank_mask = torch.full([batchsize], fill_value=0, dtype=torch.bool).numpy()

        # Get max sequence length
        max_out_len = out_len.max()
        for time_idx in range(max_out_len):
            f = x[:, time_idx : time_idx + 1, :]  # [B, 1, D]
            f = f.transpose([0, 2, 1])

            # Prepare t timestamp batch variables
            not_blank = True
            symbols_added = 0

            # Reset blank mask
            blank_mask *= False

            # Update blank mask with time mask
            # Batch: [B, T, D], but Bi may have seq len < max(seq_lens_in_batch)
            # Forcibly mask with "blank" tokens, for all sample where current time step T > seq_len
            blank_mask = time_idx >= out_len
            # Start inner loop
            while not_blank and (self.max_symbols_per_step is None or symbols_added < self.max_symbols_per_step):

                # Batch prediction and joint network steps
                # If very first prediction step, submit SOS tag (blank) to pred_step.
                # This feeds a zero tensor as input to AbstractRNNTDecoder to prime the state
                if time_idx == 0 and symbols_added == 0:
                    g = torch.tensor([self._blank_index] * batchsize, dtype=torch.int32).view(-1, 1)
                else:
                    g = last_label.astype(np.int32)

                # Batched joint step - Output = [B, V + 1]
                joint_out, hidden_prime = self.run_decoder_joint(f, g, target_lengths, *hidden)
                logp, pred_lengths = joint_out
                logp = logp[:, 0, 0, :]

                # Get index k, of max prob for batch
                k = np.argmax(logp, axis=1).astype(np.int32)

                # Update blank mask with current predicted blanks
                # This is accumulating blanks over all time steps T and all target steps min(max_symbols, U)
                k_is_blank = k == self._blank_index
                blank_mask |= k_is_blank

                del k_is_blank
                del logp

                # If all samples predict / have predicted prior blanks, exit loop early
                # This is equivalent to if single sample predicted k
                if blank_mask.all():
                    not_blank = False

                else:
                    # Collect batch indices where blanks occurred now/past
                    blank_indices = blank_mask.astype(np.int32).nonzero()
                    if type(blank_indices) in (list, tuple):
                        blank_indices = blank_indices[0]

                    # Recover prior state for all samples which predicted blank now/past
                    if hidden is not None:
                        # LSTM has 2 states
                        for state_id in range(len(hidden)):
                            hidden_prime[state_id][:, blank_indices, :] = hidden[state_id][:, blank_indices, :]

                    elif len(blank_indices) > 0 and hidden is None:
                        # Reset state if there were some blank and other non-blank predictions in batch
                        # Original state is filled with zeros so we just multiply
                        # LSTM has 2 states
                        for state_id in range(len(hidden_prime)):
                            hidden_prime[state_id][:, blank_indices, :] *= 0.0

                    # Recover prior predicted label for all samples which predicted blank now/past
                    k[blank_indices] = last_label[blank_indices, 0]

                    # Update new label and hidden state for next iteration
                    last_label = k.copy().reshape(-1, 1)
                    hidden = hidden_prime

                    # Update predicted labels, accounting for time mask
                    # If blank was predicted even once, now or in the past,
                    # Force the current predicted label to also be blank
                    # This ensures that blanks propogate across all timesteps
                    # once they have occured (normally stopping condition of sample level loop).
                    for kidx, ki in enumerate(k):
                        if blank_mask[kidx] == 0:
                            label[kidx].append(ki)
                            timesteps[kidx].append(time_idx)

                    symbols_added += 1

        return label, timesteps

    def run_encoder(self, audio_signal, length):
        if hasattr(audio_signal, 'cpu'):
            audio_signal = audio_signal.cpu().numpy()

        if hasattr(length, 'cpu'):
            length = length.cpu().numpy()

        ip = {
            self.encoder_inputs[0].name: audio_signal,
            self.encoder_inputs[1].name: length,
        }
        enc_out = self.encoder.run(None, ip)
        enc_out, encoded_length = enc_out  # ASSUME: single output
        return enc_out, encoded_length

    def run_decoder_joint(self, enc_logits, targets, target_length, *states):
        # ASSUME: Decoder is RNN Transducer
        if targets is None:
            targets = torch.zeros(enc_logits.shape[0], 1, dtype=torch.int32)
            target_length = torch.ones(enc_logits.shape[0], dtype=torch.int32)

        if hasattr(targets, 'cpu'):
            targets = targets.cpu().numpy()

        if hasattr(target_length, 'cpu'):
            target_length = target_length.cpu().numpy()

        ip = {
            self.decoder_joint_inputs[0].name: enc_logits,
            self.decoder_joint_inputs[1].name: targets,
            self.decoder_joint_inputs[2].name: target_length,
        }

        num_states = 0
        if states is not None and len(states) > 0:
            num_states = len(states)
            for idx, state in enumerate(states):
                if hasattr(state, 'cpu'):
                    state = state.cpu().numpy()

                ip[self.decoder_joint_inputs[len(ip)].name] = state

        dec_out = self.decoder_joint.run(None, ip)

        # unpack dec output
        if num_states > 0:
            new_states = dec_out[-num_states:]
            dec_out = dec_out[:-num_states]
        else:
            new_states = None

        return dec_out, new_states

    def _get_initial_states(self, batchsize):
        # ASSUME: LSTM STATES of shape (layers, batchsize, dim)
        input_state_nodes = [ip for ip in self.decoder_joint_inputs if 'state' in ip.name]
        num_states = len(input_state_nodes)
        if num_states == 0:
            return

        input_states = []
        for state_id in range(num_states):
            node = input_state_nodes[state_id]
            ip_shape = []
            for shape_idx, shape in enumerate(node.type.tensor_type.shape.dim):
                if hasattr(shape, 'dim_param') and 'dynamic' in shape.dim_param:
                    ip_shape.append(batchsize)  # replace dynamic axes with constant
                else:
                    ip_shape.append(int(shape.dim_value))

            input_states.append(torch.zeros(*ip_shape))

        return input_states


@dataclass
class GreedyRNNTInferConfig:
    max_symbols_per_step: Optional[int] = 10
    preserve_alignments: bool = False


@dataclass
class GreedyBatchedRNNTInferConfig:
    max_symbols_per_step: Optional[int] = 10
    preserve_alignments: bool = False
