# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import torch
from utils.constants import V_NEGATIVE_NUM


def viterbi_decoding(log_probs_batch, y_batch, T_batch, U_batch, viterbi_device):
    """
    Do Viterbi decoding with an efficient algorithm (the only for-loop in the 'forward pass' is over the time dimension). 
    Args:
        log_probs_batch: tensor of shape (B, T_max, V). The parts of log_probs_batch which are 'padding' are filled
            with 'V_NEGATIVE_NUM' - a large negative number which represents a very low probability.
        y_batch: tensor of shape (B, U_max) - contains token IDs including blanks in every other position. The parts of
            y_batch which are padding are filled with the number 'V'. V = the number of tokens in the vocabulary + 1 for
            the blank token.
        T_batch: tensor of shape (B, 1) - contains the durations of the log_probs_batch (so we can ignore the 
            parts of log_probs_batch which are padding)
        U_batch: tensor of shape (B, 1) - contains the lengths of y_batch (so we can ignore the parts of y_batch
            which are padding).
        viterbi_device: the torch device on which Viterbi decoding will be done.

    Returns:
        alignments_batch: list of lists containing locations for the tokens we align to at each timestep.
            Looks like: [[0, 0, 1, 2, 2, 3, 3, ...,  ], ..., [0, 1, 2, 2, 2, 3, 4, ....]].
            Each list inside alignments_batch is of length T_batch[location of utt in batch].
    """

    B, T_max, _ = log_probs_batch.shape
    U_max = y_batch.shape[1]

    # transfer all tensors to viterbi_device
    log_probs_batch = log_probs_batch.to(viterbi_device)
    y_batch = y_batch.to(viterbi_device)
    T_batch = T_batch.to(viterbi_device)
    U_batch = U_batch.to(viterbi_device)

    # make tensor that we will put at timesteps beyond the duration of the audio
    padding_for_log_probs = V_NEGATIVE_NUM * torch.ones((B, T_max, 1), device=viterbi_device)
    # make log_probs_padded tensor of shape (B, T_max, V +1 ) where all of
    # log_probs_padded[:,:,-1] is the 'V_NEGATIVE_NUM'
    log_probs_padded = torch.cat((log_probs_batch, padding_for_log_probs), dim=2)

    # initialize v_prev - tensor of previous timestep's viterbi probabilies, of shape (B, U_max)
    v_prev = V_NEGATIVE_NUM * torch.ones((B, U_max), device=viterbi_device)
    v_prev[:, :2] = torch.gather(input=log_probs_padded[:, 0, :], dim=1, index=y_batch[:, :2])

    # initialize backpointers_rel - which contains values like 0 to indicate the backpointer is to the same u index,
    # 1 to indicate the backpointer pointing to the u-1 index and 2 to indicate the backpointer is pointing to the u-2 index
    backpointers_rel = -99 * torch.ones((B, T_max, U_max), dtype=torch.int8, device=viterbi_device)

    # Make a letter_repetition_mask the same shape as y_batch
    # the letter_repetition_mask will have 'True' where the token (including blanks) is the same
    # as the token two places before it in the ground truth (and 'False everywhere else).
    # We will use letter_repetition_mask to determine whether the Viterbi algorithm needs to look two tokens back or
    # three tokens back
    y_shifted_left = torch.roll(y_batch, shifts=2, dims=1)
    letter_repetition_mask = y_batch - y_shifted_left
    letter_repetition_mask[:, :2] = 1  # make sure dont apply mask to first 2 tokens
    letter_repetition_mask = letter_repetition_mask == 0

    for t in range(1, T_max):

        # e_current is a tensor of shape (B, U_max) of the log probs of every possible token at the current timestep
        e_current = torch.gather(input=log_probs_padded[:, t, :], dim=1, index=y_batch)

        # apply a mask to e_current to cope with the fact that we do not keep the whole v_matrix and continue
        # calculating viterbi probabilities during some 'padding' timesteps
        t_exceeded_T_batch = t >= T_batch

        U_can_be_final = torch.logical_or(
            torch.arange(0, U_max, device=viterbi_device).unsqueeze(0) == (U_batch.unsqueeze(1) - 0),
            torch.arange(0, U_max, device=viterbi_device).unsqueeze(0) == (U_batch.unsqueeze(1) - 1),
        )

        mask = torch.logical_not(torch.logical_and(t_exceeded_T_batch.unsqueeze(1), U_can_be_final,)).long()

        e_current = e_current * mask

        # v_prev_shifted is a tensor of shape (B, U_max) of the viterbi probabilities 1 timestep back and 1 token position back
        v_prev_shifted = torch.roll(v_prev, shifts=1, dims=1)
        # by doing a roll shift of size 1, we have brought the viterbi probability in the final token position to the
        # first token position - let's overcome this by 'zeroing out' the probabilities in the firest token position
        v_prev_shifted[:, 0] = V_NEGATIVE_NUM

        # v_prev_shifted2 is a tensor of shape (B, U_max) of the viterbi probabilities 1 timestep back and 2 token position back
        v_prev_shifted2 = torch.roll(v_prev, shifts=2, dims=1)
        v_prev_shifted2[:, :2] = V_NEGATIVE_NUM  # zero out as we did for v_prev_shifted
        # use our letter_repetition_mask to remove the connections between 2 blanks (so we don't skip over a letter)
        # and to remove the connections between 2 consective letters (so we don't skip over a blank)
        v_prev_shifted2.masked_fill_(letter_repetition_mask, V_NEGATIVE_NUM)

        # we need this v_prev_dup tensor so we can calculated the viterbi probability of every possible
        # token position simultaneously
        v_prev_dup = torch.cat(
            (v_prev.unsqueeze(2), v_prev_shifted.unsqueeze(2), v_prev_shifted2.unsqueeze(2),), dim=2,
        )

        # candidates_v_current are our candidate viterbi probabilities for every token position, from which
        # we will pick the max and record the argmax
        candidates_v_current = v_prev_dup + e_current.unsqueeze(2)
        # we straight away save results in v_prev instead of v_current, so that the variable v_prev will be ready for the
        # next iteration of the for-loop
        v_prev, bp_relative = torch.max(candidates_v_current, dim=2)

        backpointers_rel[:, t, :] = bp_relative

    # trace backpointers
    alignments_batch = []
    for b in range(B):
        T_b = int(T_batch[b])
        U_b = int(U_batch[b])

        if U_b == 1:  # i.e. we put only a blank token in the reference text because the reference text is empty
            current_u = 0  # set initial u to 0 and let the rest of the code block run as usual
        else:
            current_u = int(torch.argmax(v_prev[b, U_b - 2 : U_b])) + U_b - 2
        alignment_b = [current_u]
        for t in range(T_max - 1, 0, -1):
            current_u = current_u - int(backpointers_rel[b, t, current_u])
            alignment_b.insert(0, current_u)
        alignment_b = alignment_b[:T_b]
        alignments_batch.append(alignment_b)

    return alignments_batch
