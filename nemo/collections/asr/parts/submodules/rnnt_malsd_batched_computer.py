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
from pathlib import Path
from typing import Optional, Any, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs
from nemo.core.utils.cuda_python_utils import (
    check_cuda_python_cuda_graphs_conditional_nodes_supported,
    cu_call,
    run_nvrtc,
    with_conditional_node,
)

from nemo.collections.asr.parts.submodules.ngram_lm import FastNGramLM
from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.asr.parts.utils.rnnt_batched_beam_utils import BlankLMScoreMode, BatchedBeamHyps, PruningMode

try:
    from cuda import cudart

    HAVE_CUDA_PYTHON = True
except ImportError:
    HAVE_CUDA_PYTHON = False

class MALSDState:
    """
    State for Loop Labels algorithm. Used only with CUDA graphs.
    In initialization phase it is possible to assign values (tensors) to the state.
    For algorithm code the storage should be reused (prefer copy data instead of assigning tensors).
    """

    INACTIVE_HYPOTHESIS_SCORE=float('-inf')

    max_time: int  # maximum length of internal storage for time dimension
    batch_size: int  # (maximum) length of internal storage for batch dimension
    device: torch.device  # device to store preallocated tensors
    beam_size: int
    blank_index: int
        
    encoder_output_projected: torch.Tensor  # projected output from the encoder for decoding algorithm
    encoder_output_length: torch.Tensor  # length of the (projected) output from the encoder

    # labels: torch.Tensor  # storage for current labels
    # scores: torch.Tensor  # storage for current scores

    batch_indices: torch.Tensor  # indices of elements in batch (constant, range [0, batch_size-1])

    time_indices: torch.Tensor  # current time indices for each element in batch
    safe_time_indices: torch.Tensor  # current time indices, but guaranteed to be < encoder_output_length
    time_indices_current_labels: torch.Tensor  # time indices for found labels (corresponding to `labels` field)
    last_timesteps: torch.Tensor  # indices of the last timesteps for each element (encoder_output_length - 1)
    last_labels_wb: torch.Tensor
    hyp_scores: torch.Tensor

    active_mask: torch.Tensor  # mask for active hypotheses (the decoding is finished for the utterance if it is False)
    # advance_mask: torch.Tensor  # mask for "advancing" hypotheses (blank is found for the element on the current step)
    blank_mask: torch.Tensor  # if the element is blank
    # if the element was active on the previous step: to identify the end of decoding and store final hidden state
    active_mask_prev: torch.Tensor
    became_inactive_mask: torch.Tensor  # mask for elements that became inactive (end of decoding)

    active_mask_any: torch.Tensor  # 0-dim bool tensor, condition for outer loop ('any element is still active')
    advance_mask_any: torch.Tensor  # 0-dim bool tensor, condition for inner loop ('should advance any index')

    last_decoder_state: Any  # last state from the decoder, needed for the output
    decoder_state: Any  # current decoder state
    decoder_output: torch.Tensor  # output from the decoder (projected)

    batched_hyps: BatchedBeamHyps  # batched hypotheses - decoding result
    alignments: Optional[rnnt_utils.BatchedAlignments] = None  # batched alignments

    batch_lm_states: Optional[torch.Tensor] = None
    lm_scores: Optional[torch.Tensor] = None
    batch_lm_states_candidates: Optional[torch.Tensor] = None
    ngram_lm_batch: Optional[FastNGramLM] = None
    
    def __init__(
        self,
        batch_size: int,
        beam_size: int,
        max_time: int,
        encoder_dim: int,
        max_symbols: int,
        device: torch.device,
        float_dtype: torch.dtype,
        blank_index: int,
    ):
        """

        Args:
            batch_size: batch size for encoder output storage
            max_time: maximum time for encoder output storage
            encoder_dim: last dimension for encoder output storage (projected encoder output)
            max_symbols: max symbols per step (to avoid infinite looping and pre-allocate storage)
            device: device to store tensors
            float_dtype: default float dtype for tensors (should match projected encoder output)
        """
        self.device = device
        self.float_dtype = float_dtype
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.max_time = max_time
        self.blank_index = blank_index

        self.encoder_output_projected = torch.zeros(
            (self.batch_size, self.max_time, encoder_dim),
            dtype=float_dtype,
            device=self.device,
        )
        self.encoder_output_length = torch.zeros([self.batch_size, self.beam_size], dtype=torch.long, device=self.device)

        # self.labels = torch.zeros([self.batch_size], dtype=torch.long, device=self.device)
        # self.scores = torch.zeros([self.batch_size], dtype=float_dtype, device=self.device)
        self.last_labels_wb = torch.zeros(
                [self.batch_size, self.beam_size],
                device=self.device,
                dtype=torch.long
            )
        self.hyp_scores = torch.full(
            [self.batch_size, self.beam_size],
            fill_value=self.INACTIVE_HYPOTHESIS_SCORE,
            device=self.device,
            dtype=torch.float
        )
        
        # indices of elements in batch (constant)
        self.batch_indices = (
            torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
            .expand(batch_size, self.beam_size)
            .clone()
        )

        self.time_indices = torch.zeros_like(self.batch_indices)
        self.safe_time_indices = torch.zeros_like(self.batch_indices)
        self.last_timesteps = torch.zeros_like(self.time_indices)

        self.active_mask = torch.zeros([self.batch_size, self.beam_size], dtype=torch.bool, device=self.device)
        # self.advance_mask = torch.zeros_like(self.active_mask)
        self.blank_mask = torch.zeros_like(self.active_mask)
        self.active_mask_prev = torch.zeros_like(self.active_mask)
        self.became_inactive_mask = torch.zeros_like(self.active_mask)

        self.active_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)
        self.advance_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)

        self.batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self.blank_index,
            init_length=max_time * (max_symbols + 1) if max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
        )
        
        self.alignments = None
        self.last_decoder_state = None

    def need_reinit(self, encoder_output_projected: torch.Tensor) -> bool:
        """Check if need to reinit state: larger batch_size/max_time, or new device"""
        return (
            self.batch_size < encoder_output_projected.shape[0]
            or self.max_time < encoder_output_projected.shape[1]
            or self.device.index != encoder_output_projected.device.index
        )
        
@dataclass
class SeparateGraphsMALSD:
    """Class to store Cuda graphs for decoding when separate graphs are used"""

    before_loop: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    loop_body: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)

class ModifiedALSDBatchedRNNTComputer(WithOptionalCudaGraphs, ConfidenceMethodMixin):
    """
    Label Looping algorithm implementation: optimized batched greedy decoding. Callable.
    Iterates over labels, on each step finding the next non-blank label
    (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
    to prediction network (with maximum possible batch size),
    which makes it especially useful for scaling the prediction network.
    During decoding all active hypotheses ("texts") have the same lengths.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to init state for Cuda graphs
    CUDA_PROGRAM_NAME = b"while_loop_labels_conditional_rnnt.cu"

    class CudaGraphsMode(PrettyStrEnum):
        FULL_GRAPH = "full_graph"  # Cuda graphs with conditional nodes, fastest implementation
        NO_WHILE_LOOPS = "no_while_loops"  # Decoding with PyTorch while loops + partial Cuda graphs
        NO_GRAPHS = "no_graphs"  # decoding without graphs, stateful implementation, only for testing purposes

    separate_graphs: Optional[SeparateGraphsMALSD]
    full_graph: Optional[torch.cuda.CUDAGraph]
    cuda_graphs_mode: Optional[CudaGraphsMode]
    state: Optional[MALSDState]
    ngram_lm_batch: Optional[FastNGramLM]

    def __init__(
        self,
        decoder,
        joint,
        blank_index: int,
        beam_size: int,
        max_symbols_per_step: Optional[int] = 10,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        confidence_method_cfg: Optional[DictConfig] = None,
        ngram_lm_model: Optional[str | Path] = None,
        ngram_lm_alpha: float = 0.0,
        blank_lm_score_mode: Optional[str | BlankLMScoreMode] = None,
        pruning_mode: Optional[str | PruningMode] = None,
        allow_recombine_hyps: bool = True,
        score_norm: bool = True,
        allow_cuda_graphs: bool = False
    ):
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        self._blank_index = blank_index
        self.beam_size = beam_size
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self.allow_recombine_hyps = allow_recombine_hyps
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        self.score_norm = score_norm
        self.allow_cuda_graphs = allow_cuda_graphs

        assert self._SOS == self._blank_index  # "blank as pad" algorithm only
        assert not self.preserve_alignments
        assert not self.preserve_frame_confidence
        
        self.state = None
        self.full_graph = None
        self.separate_graphs = None
        
        # prprpr
        self.cuda_graphs_mode = self.CudaGraphsMode("no_while_loops")
        # self.cuda_graphs_mode = self.CudaGraphsMode("no_graphs")
        # self.cuda_graphs_mode = None
        self.maybe_enable_cuda_graphs()
        
        if ngram_lm_model is not None:
            assert self._blank_index == self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
            self.ngram_lm_batch = FastNGramLM.from_arpa(lm_path=ngram_lm_model, vocab_size=self._blank_index)
            
            self.pruning_mode = (
                PruningMode.EARLY
                if pruning_mode is None
                else PruningMode(pruning_mode)
            )
            self.blank_lm_score_mode = (
                BlankLMScoreMode.LM_WEIGHTED_FULL
                if blank_lm_score_mode is None 
                else BlankLMScoreMode(blank_lm_score_mode)
            )
        else:
            self.ngram_lm_batch = None
            self.blank_lm_score_mode = None
        self.ngram_lm_alpha = ngram_lm_alpha

    def force_cuda_graphs_mode(self, mode: Optional[Union[str, CudaGraphsMode]]):
        """
        Method to set graphs mode. Use only for testing purposes.
        For debugging the algorithm use "no_graphs" mode, since it is impossible to debug CUDA graphs directly.
        """
        self.cuda_graphs_mode = self.CudaGraphsMode(mode) if mode is not None else None
        self.state = None

    def maybe_enable_cuda_graphs(self):
        """Enable CUDA graphs if conditions met"""
        if self.cuda_graphs_mode is not None:
            # CUDA graphs are already enabled
            return

        if not self.allow_cuda_graphs:
            self.cuda_graphs_mode = None
        else:
            # cuda graphs are allowed
            # check basic requirements for cuda graphs
            if self.max_symbols is None:
                logging.warning("Max symbols per step is None, which is not allowed with Cuda graphs. Setting to `10`")
                self.max_symbols = 10
            # basic requirements met, need to check while loops
            try:
                check_cuda_python_cuda_graphs_conditional_nodes_supported()
                self.cuda_graphs_mode = self.CudaGraphsMode.FULL_GRAPH
            except (ImportError, ModuleNotFoundError, EnvironmentError) as e:
                logging.warning(
                    "No conditional node support for Cuda.\n"
                    "Cuda graphs with while loops are disabled, decoding speed will be slower\n"
                    f"Reason: {e}"
                )
                self.cuda_graphs_mode = self.CudaGraphsMode.NO_WHILE_LOOPS
        self.reset_cuda_graphs_state()

    def disable_cuda_graphs(self):
        """Disable CUDA graphs, can be used to disable graphs temporary, e.g., in training process"""
        if self.cuda_graphs_mode is None:
            # nothing to disable
            return
        self.cuda_graphs_mode = None
        self.reset_cuda_graphs_state()

    def reset_cuda_graphs_state(self):
        """Reset state to release memory (for CUDA graphs implementations)"""
        self.state = None
        self.full_graph = None
        self.separate_graphs = None

    def modified_alsd_beam_torch(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> list[rnnt_utils.Hypothesis]:
        batch_size, max_time, vocab_size = encoder_output.shape
        device = encoder_output.device

        # do not recalculate joint projection, project only once
        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype

        # init output structures: BatchedHyps (for results), BatchedAlignments + last decoder state
        # init empty batched hypotheses
        batched_hyps = BatchedBeamHyps(
            batch_size=batch_size,
            beam_size=self.beam_size,
            blank_index=self._blank_index,
            init_length=max_time * (self.max_symbols + 1) if self.max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
        )

        last_labels_wb = torch.full(
            [batch_size, self.beam_size], fill_value=self._SOS, device=device, dtype=torch.long
        )

        batch_indices = (
            torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
            .expand(batch_size, self.beam_size)
            .clone()
        )
        time_indices = torch.zeros_like(batch_indices)
        safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
        last_timesteps = (encoder_output_length - 1)[:, None].expand_as(batch_indices)
        active_mask = time_indices <= last_timesteps

        if self.ngram_lm_batch is not None:
            self.ngram_lm_batch.to(device)
            
            batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(states=batch_lm_states)  # vocab_size_no_blank
            lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha

        decoder_output, decoder_state, *_ = self.decoder.predict(
            last_labels_wb.reshape(-1).unsqueeze(1), None, add_sos=False, batch_size=batch_size * self.beam_size
        )
        decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection
        # decoder_output: [(B x Beam), 1, Dim]

        while active_mask.any():
            # step 1: get joint output + fuse with LM (if present)
            logits = (
                self.joint.joint_after_projection(
                    encoder_output_projected[
                        batch_indices.view(-1),
                        safe_time_indices.view(-1)
                        ].unsqueeze(1),
                    decoder_output,
                )
                .squeeze(1)
                .squeeze(1)
            )
            log_probs = F.log_softmax(logits, dim=-1).view(batch_size, self.beam_size, -1)  # [(B x Beam), V]
            
            if self.ngram_lm_batch is not None:
                log_probs_top_k, labels_top_k = self.topk_lm(lm_scores, log_probs)
            else:
                log_probs_top_k, labels_top_k = torch.topk(
                    log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                )

            # step 2: Make hyps candidates. Add new scores to hyps, force blank if necessary, recombine hyps, prune
            # step 2.1: hyps candidates
            log_probs_blank = log_probs[..., self._blank_index]
            # size: batch_size x beam_size x beam_size (k)
            hyps_scores = batched_hyps.scores
            hyps_candidates_prob = hyps_scores.unsqueeze(-1) + log_probs_top_k  # hyps from top-k (top-k-prev x top_k)
            hyps_candidates_prob_forced_blank = (
                hyps_scores + log_probs_blank
            )  # hyps with forced blank (top-k-prev x blank)

            # step 2.2 force add final hyps with the same score to the beam
            # final hyps cannot be extended -> mask with minus inf, copy prev scores; label - set to -1
            hyps_candidates_prob = torch.where(
                active_mask.unsqueeze(-1),
                hyps_candidates_prob,
                batched_hyps.INACTIVE_SCORE,
            )
            hyps_candidates_prob[..., 0] = torch.where(
                active_mask,
                hyps_candidates_prob[..., 0],
                hyps_scores,
            )
            labels_top_k = torch.where(active_mask.unsqueeze(-1), labels_top_k, -1)

            # step 2.3: force blank extension with respect to self.max_symbols
            if self.max_symbols is not None:
                force_blank = (batched_hyps.last_timestep_lasts >= self.max_symbols) & active_mask
            else:
                force_blank = torch.full_like(active_mask, fill_value=False)
            # mask all extensions with -inf
            hyps_candidates_prob = torch.where(
                force_blank.unsqueeze(-1),
                batched_hyps.INACTIVE_SCORE,
                hyps_candidates_prob
            )
            # first element in beam - score for hyp with forced blank
            hyps_candidates_prob[..., 0] = torch.where(
                force_blank, hyps_candidates_prob_forced_blank, hyps_candidates_prob[..., 0]
            )
            labels_top_k = torch.where(
                force_blank.unsqueeze(-1), self._blank_index, labels_top_k
            )

            # step 2.4: final pruning - get top-k from (top-k x top-k) hyps
            hyps_indices = torch.arange(self.beam_size, dtype=torch.long, device=device)[None, :, None].expand(
                batch_size, -1, self.beam_size
            )
            next_hyps_prob, hyps_candidates_indices = torch.topk(
                hyps_candidates_prob.view(batch_size, -1), k=self.beam_size, largest=True, sorted=True
            )
            hyps_indices = torch.gather(hyps_indices.reshape(batch_size, -1), dim=-1, index=hyps_candidates_indices)
            next_labels = torch.gather(labels_top_k.reshape(batch_size, -1), dim=-1, index=hyps_candidates_indices)

            # step 3: store results
            if self.max_symbols is None:
                batched_hyps.add_results_(hyps_indices, next_labels, next_hyps_prob)
            else:
                batched_hyps.add_results_no_checks_(hyps_indices, next_labels, next_hyps_prob)
            if self.allow_recombine_hyps:
                batched_hyps.self_recombine_hyps_()

            # step 4: update decoder state + decoder output (+ lm state/scores)
            last_labels_wb = torch.where(
                next_labels >= 0, next_labels, self._blank_index
            )
            preserve_state = last_labels_wb == self._blank_index

            # update decoder + lm state
            # decoder_output: [(B x Beam), 1, Dim]
            prev_decoder_output = torch.gather(
                decoder_output.view(batch_size, self.beam_size, 1, -1),
                dim=1,
                index=hyps_indices[:, :, None, None].expand(batch_size, self.beam_size, 1, decoder_output.shape[-1]),
            ).view(batch_size * self.beam_size, 1, -1)
            
            # TODO: move state aggregation to decoder + support stateless decoder:
            #  self.decoder.batch_aggregate_states_beam(...)
            # state: tuple, each is of [Layers, (BxBeam), Dim]
            prev_decoder_state = (
                torch.gather(
                    decoder_state[0].view(decoder_state[0].shape[0], batch_size, self.beam_size, -1),
                    dim=2,
                    index=hyps_indices[None, :, :, None].expand(
                        decoder_state[0].shape[0], batch_size, self.beam_size, decoder_state[0].shape[-1]
                    ),
                ).view(decoder_state[0].shape[0], batch_size * self.beam_size, -1),
                torch.gather(
                    decoder_state[1].view(decoder_state[1].shape[0], batch_size, self.beam_size, -1),
                    dim=2,
                    index=hyps_indices[None, :, :, None].expand(
                        decoder_state[1].shape[0], batch_size, self.beam_size, decoder_state[1].shape[-1]
                    ),
                ).view(decoder_state[1].shape[0], batch_size * self.beam_size, -1),
            )

            decoder_output, decoder_state, *_ = self.decoder.predict(
                last_labels_wb.reshape(-1).unsqueeze(1),
                prev_decoder_state,
                add_sos=False,
                batch_size=batch_size * self.beam_size,
            )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            decoder_output = torch.where(preserve_state.view(-1)[:, None, None], prev_decoder_output, decoder_output)
            self.decoder.batch_replace_states_mask(
                src_states=prev_decoder_state, dst_states=decoder_state, mask=preserve_state.view(-1)
            ) 
            if self.ngram_lm_batch is not None:
                # batch_lm_states: [(BxBeam)]
                # batch_lm_states_candidates: [(BxBeam) x V (without blank)]
                batch_lm_states_candidates = torch.gather(
                    batch_lm_states_candidates.view(batch_size, self.beam_size, -1),
                    dim=1,
                    index=hyps_indices[:, :, None].expand(
                        batch_size, self.beam_size, batch_lm_states_candidates.shape[-1]
                    ),
                )
                batch_lm_states_prev = torch.gather(
                    batch_lm_states.view(batch_size, self.beam_size), dim=1, index=hyps_indices
                )
                last_labels_wb_blank_replaced = torch.where(
                    preserve_state, 0, last_labels_wb
                )

                batch_lm_states = torch.gather(
                    batch_lm_states_candidates, dim=-1, index=last_labels_wb_blank_replaced.unsqueeze(-1)
                ).squeeze(-1)
                batch_lm_states = torch.where(preserve_state, batch_lm_states_prev, batch_lm_states).view(-1)

                lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
                    states=batch_lm_states
                )  # vocab_size_no_blank
                lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha

            # step 5: update time indices + active mask
            time_indices = batched_hyps.next_timestep
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            active_mask = time_indices <= last_timesteps
            # torch.cuda.set_sync_debug_mode(0)

        return batched_hyps.to_hyps_list(score_norm=self.score_norm)
    
    def topk_lm(self, lm_scores, log_probs):
        if self.pruning_mode is PruningMode.LATE:
            if self.blank_lm_score_mode is BlankLMScoreMode.NO_SCORE:
                log_probs[..., :-1] += lm_scores
                log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
            elif self.blank_lm_score_mode is BlankLMScoreMode.PRESERVE_BLANK:
                _, labels_top_k_no_lm = torch.topk(log_probs, self.beam_size, dim=-1, largest=True, sorted=True)
                log_probs[..., :-1] += lm_scores
                _, labels_with_lm_nb_top_k = torch.topk(
                        log_probs[..., :-1], self.beam_size, dim=-1, largest=True, sorted=True
                    )
                    # [(BxBeam), beam]
                    # if blank was in labels_top_k -> add blank (last in beam)
                labels_top_k = labels_with_lm_nb_top_k
                labels_top_k[..., -1] = torch.where(
                        (labels_top_k_no_lm == self._blank_index).any(dim=-1),
                        self._blank_index,
                        labels_top_k[..., -1],
                    )
                log_probs_top_k = torch.gather(log_probs, dim=-1, index=labels_top_k)
            elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED:
                log_probs[..., :-1] += lm_scores
                log_probs[..., -1] *= 1 + self.ngram_lm_alpha
                log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
            elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL:
                blank_logprob = log_probs[..., -1]
                non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
                log_probs[..., :-1] += non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + lm_scores
                log_probs[..., -1] *= 1 + self.ngram_lm_alpha
                log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
            elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL_FIXED_BLANK:
                blank_logprob = log_probs[..., -1]
                non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
                log_probs[..., :-1] += non_blank_logprob.unsqueeze(-1) + lm_scores # blank prob - the same
                log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
            elif self.blank_lm_score_mode is BlankLMScoreMode.LM_MAX:
                log_probs[..., :-1] += lm_scores
                log_probs[..., -1] += lm_scores.max(dim=-1, keepdim=False).values
                log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
            elif self.blank_lm_score_mode is BlankLMScoreMode.LM_TOP_MAX:
                log_probs[..., :-1] += lm_scores
                _, labels_with_lm_nb_top_k = torch.topk(
                        log_probs[..., :-1], self.beam_size, dim=-1, largest=True, sorted=True
                    )
                    # [(BxBeam), beam]
                lm_only_scores = torch.gather(
                        lm_scores,
                        dim=-1,
                        index=labels_with_lm_nb_top_k,
                    )
                log_probs[..., -1] += lm_only_scores.max(dim=-1, keepdim=False).values
                log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
            else:
                raise NotImplementedError(
                        f"The combination of blank scoring mode '{self.blank_lm_score_mode}' "
                        f"and pruning mode '{self.pruning_mode}' is not implemented."
                )
        elif self.pruning_mode is PruningMode.EARLY:
            if self.blank_lm_score_mode is BlankLMScoreMode.NO_SCORE:
                # log_probs[..., :-1] += lm_scores
                log_probs_top_k, labels_top_k = torch.topk(
                        log_probs, self.beam_size, dim=-1, largest=True, sorted=True
                    )
                masked_labels = torch.where(labels_top_k==self._blank_index, 0, labels_top_k)
                log_probs_top_k = torch.where(
                    labels_top_k==self._blank_index,
                    log_probs_top_k,
                    log_probs_top_k + torch.gather(lm_scores, dim=-1,index=masked_labels))
            elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL:
                # choosing topk from acoustic model
                log_probs_top_k, labels_top_k = log_probs.topk(self.beam_size, dim=-1, largest=True, sorted=True)
                
                blank_logprob = log_probs[..., -1]
                non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
                
                masked_labels = torch.where(labels_top_k==self._blank_index, 0, labels_top_k)
                log_probs_top_k = torch.where(
                    labels_top_k==self._blank_index,
                    log_probs_top_k * (1 + self.ngram_lm_alpha),
                    log_probs_top_k + non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + torch.gather(lm_scores, dim=-1, index=masked_labels)
                )
            else:
                raise NotImplementedError(
                        f"The combination of blank scoring mode '{self.blank_lm_score_mode}' "
                        f"and pruning mode '{self.pruning_mode}' is not implemented."
                )
        else:
            raise NotImplementedError(f"Pruning mode {self.pruning_mode} is not implemented.")
            
        return log_probs_top_k, labels_top_k

    def modified_alsd_cuda_graphs(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        """
        Implementation with CUDA graphs.

        Args:
            encoder_output: output from the encoder
            encoder_output_length: lengths of the utterances in `encoder_output`
        """
        assert self.cuda_graphs_mode is not None

        # do not recalculate joint projection, project only once
        encoder_output = self.joint.project_encoder(encoder_output)
        current_batch_size = encoder_output.shape[0]
        current_max_time = encoder_output.shape[1]

        if torch.is_autocast_enabled():
            encoder_output = encoder_output.to(torch.get_autocast_gpu_dtype())

        # init or reinit graph
        if self.state is None or self.state.need_reinit(encoder_output):
            self._graph_reinitialize(encoder_output, encoder_output_length)


        # set length to zero for elements outside the current batch
        self.state.encoder_output_length.fill_(0)
        # copy (projected) encoder output and lenghts
        self.state.encoder_output_projected[:current_batch_size, :current_max_time, ...].copy_(encoder_output)
        self.state.encoder_output_length[:current_batch_size].copy_(encoder_output_length.unsqueeze(-1))
        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self.full_graph.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self.separate_graphs.before_loop.replay()
            while self.state.active_mask_any.item():
                self.separate_graphs.loop_body.replay()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # this mode is only for testing purposes
            # manual loop instead of using graphs
            self._before_loop()
            while self.state.active_mask_any.item():
                self._loop_body()
        else:
            raise NotImplementedError(f"Unknown graph mode: {self.cuda_graphs_mode}")

        return self.state.batched_hyps.to_hyps_list(score_norm=self.score_norm)[:current_batch_size]
        # return (
        #     self.state.batched_hyps,
        #     self.state.alignments,
        #     self.state.last_decoder_state,
        # )

    @classmethod
    def _create_outer_while_loop_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the outer loop body (not all hypotheses are decoded).
        Condition: while(active_mask_any).
        """
        kernel_string = r"""\
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
    
        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
    
        extern "C" __global__
        void outer_loop_labels_conditional(cudaGraphConditionalHandle handle, const bool *active_mask_any)
        {
         cudaGraphSetConditional(handle, *active_mask_any);
        }
        """
        return run_nvrtc(kernel_string, b"outer_loop_labels_conditional", cls.CUDA_PROGRAM_NAME)

    @classmethod
    def _create_inner_while_loop_kernel(cls):
        """
        Creates a kernel that evaluates whether to enter the inner loop body (not all non-blank labels found).
        Condition: while(advance_mask_any).
        """
        kernel_string = r"""\
        typedef __device_builtin__ unsigned long long cudaGraphConditionalHandle;
    
        extern "C" __device__ __cudart_builtin__ void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned int value);
    
        extern "C" __global__
        void inner_find_non_blank_conditional(cudaGraphConditionalHandle handle, const bool *advance_mask_any)
        {
         cudaGraphSetConditional(handle, *advance_mask_any);
        }
        """
        return run_nvrtc(kernel_string, b"inner_find_non_blank_conditional", cls.CUDA_PROGRAM_NAME)

    def _graph_reinitialize(
        self,
        encoder_output_projected: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ):
        batch_size, max_time, encoder_dim = encoder_output_projected.shape

        self.state = MALSDState(
            batch_size=batch_size,
            beam_size=self.beam_size,
            max_time=max(max_time, self.INITIAL_MAX_TIME),
            encoder_dim=encoder_dim,
            max_symbols=self.max_symbols,
            device=encoder_output_projected.device,
            float_dtype=encoder_output_projected.dtype,
            blank_index=self._blank_index
        )

        # self.state.last_decoder_state = self.decoder.initialize_state(encoder_output_projected)
        # self.state.decoder_state = self.decoder.initialize_state(encoder_output_projected)
        decoder_output, self.state.decoder_state, *_ = self.decoder.predict(
            self.state.last_labels_wb.reshape(-1).unsqueeze(1), None, add_sos=False, batch_size=self.state.batch_size * self.state.beam_size
        )
        # to avoid recalculation of joint projection, store decoder output in state
        self.state.decoder_output = self.joint.project_prednet(decoder_output)
        
        if self.ngram_lm_batch is not None:
            device = encoder_output_projected.device
            float_dtype = encoder_output_projected.dtype
            vocab_size = self.ngram_lm_batch.vocab_size
            self.ngram_lm_batch.to(device)
            self.state.batch_lm_states = self.ngram_lm_batch.get_init_states(
                batch_size=self.state.batch_size, bos=True
            )
            self.state.batch_lm_states_candidates = torch.zeros(
                [batch_size, vocab_size], dtype=torch.long, device=device
            )
            self.state.lm_scores = torch.zeros([batch_size, vocab_size], dtype=float_dtype, device=device)

        if self.cuda_graphs_mode is self.CudaGraphsMode.FULL_GRAPH:
            self._full_graph_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_WHILE_LOOPS:
            self._partial_graphs_compile()
        elif self.cuda_graphs_mode is self.CudaGraphsMode.NO_GRAPHS:
            # no graphs needed
            pass
        else:
            raise NotImplementedError

    def _partial_graphs_compile(self):
        """Compile decoding by parts"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        stream_for_graph.wait_stream(torch.cuda.default_stream(self.state.device))
        self.separate_graphs = SeparateGraphsMALSD()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.before_loop, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._before_loop()

        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(
                self.separate_graphs.loop_body, stream=stream_for_graph, capture_error_mode="thread_local"
            ),
        ):
            self._loop_body()

    def _full_graph_compile(self):
        """Compile full graph for decoding"""
        # Always create a new stream, because the per-thread default stream disallows stream capture to a graph.
        stream_for_graph = torch.cuda.Stream(self.state.device)
        self.full_graph = torch.cuda.CUDAGraph()
        with (
            torch.cuda.stream(stream_for_graph),
            torch.inference_mode(),
            torch.cuda.graph(self.full_graph, stream=stream_for_graph, capture_error_mode="thread_local"),
        ):
            self._before_loop()

            capture_status, _, graph, _, _ = cu_call(
                cudart.cudaStreamGetCaptureInfo(torch.cuda.current_stream(device=self.state.device).cuda_stream)
            )
            assert capture_status == cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive

            # capture: while self.active_mask_any:
            (outer_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
            outer_loop_kernel = self._create_outer_while_loop_kernel()
            active_mask_any_ptr = np.array([self.state.active_mask_any.data_ptr()], dtype=np.uint64)
            outer_loop_args = np.array(
                [outer_loop_conditional_handle.getPtr(), active_mask_any_ptr.ctypes.data],
                dtype=np.uint64,
            )
            # loop while there are active utterances
            with with_conditional_node(
                outer_loop_kernel, outer_loop_args, outer_loop_conditional_handle, device=self.state.device
            ):
                self._loop_body()
                # capture: while self.advance_mask_any.item():
                inner_while_loop_kernel = self._create_inner_while_loop_kernel()
                (inner_loop_conditional_handle,) = cu_call(cudart.cudaGraphConditionalHandleCreate(graph, 0, 0))
                advance_mask_any_ptr = np.array([self.state.advance_mask_any.data_ptr()], dtype=np.uint64)
                inner_loop_args = np.array(
                    [
                        inner_loop_conditional_handle.getPtr(),
                        advance_mask_any_ptr.ctypes.data,
                    ],
                    dtype=np.uint64,
                )

    def _before_loop(self):
        """Clear state and compute initial active mask"""
        self.state.batched_hyps.clear_()
        if self.state.alignments is not None:
            self.state.alignments.clear_()

        # initial state
        self.decoder.batch_replace_states_all(
            src_states=self.decoder.initialize_state_like(
                self.state.encoder_output_projected.shape[0]*self.beam_size,
                self.state.encoder_output_projected.dtype,
                self.state.encoder_output_projected.device
            ),
            dst_states=self.state.decoder_state,
        )
        
        # initial state - lm
        if self.ngram_lm_batch is not None:
            self.state.batch_lm_states.copy_(
                self.ngram_lm_batch.get_init_states(batch_size=self.state.batch_size, bos=True)
            )

        # last found labels - initially <SOS> (<blank>) symbol
        self.state.last_labels_wb.fill_(self._SOS)
        # self.state.scores.fill_(0.0)

        # time indices
        self.state.time_indices.fill_(0)
        self.state.safe_time_indices.fill_(0)  # safe time indices: guaranteed to be < encoder_output_length
        # self.state.time_indices_current_labels.fill_(0).
        
        torch.sub(
            self.state.encoder_output_length,
            1,
            out=self.state.last_timesteps
        )

        # masks for utterances in batch
        # same as: active_mask = self.encoder_output_length > 0
        torch.greater(
            self.state.encoder_output_length,
            0,
            out=self.state.active_mask
        )

        # for storing the last state we need to know what elements became "inactive" on this step
        # same as: self.active_mask_any = active_mask.any()
        torch.any(self.state.active_mask, out=self.state.active_mask_any)
        
        decoder_output, decoder_state, *_ = self.decoder.predict(
            self.state.last_labels_wb.reshape(-1).unsqueeze(1), None, add_sos=False, batch_size=self.state.encoder_output_projected.shape[0] * self.beam_size
        )
        self.decoder.batch_replace_states_all(src_states=decoder_state, dst_states=self.state.decoder_state)
        self.state.decoder_output.copy_(self.joint.project_prednet(decoder_output))  # do not recalculate joint projection

    def _loop_body(self):
        """Get Joint output after decoder output, prepare inner loop to search for all next non-blank labels"""
        # stage 2: get joint output, iteratively seeking for non-blank labels
        # blank label in `labels` tensor means "end of hypothesis" (for this index)
        logits = (
            self.joint.joint_after_projection(
                self.state.encoder_output_projected[
                    self.state.batch_indices.view(-1), 
                    self.state.safe_time_indices.view(-1)
                    ].unsqueeze(1),
                self.state.decoder_output,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # same as: scores, labels = logits.max(-1)
        log_probs = F.log_softmax(logits, dim=-1).view(self.state.batch_size, self.beam_size, -1)  # [(B x Beam), V]
        if self.ngram_lm_batch is not None:
            log_probs_top_k, labels_top_k = self.topk_lm(self.state.lm_scores, log_probs)
        else:
            log_probs_top_k, labels_top_k = torch.topk(
                log_probs, self.beam_size, dim=-1, largest=True, sorted=True
            )

        # step 2: Make hyps candidates. Add new scores to hyps, force blank if necessary, recombine hyps, prune
        # step 2.1: hyps candidates
        log_probs_blank = log_probs[..., self._blank_index]
        # size: batch_size x beam_size x beam_size (k)
        hyps_scores = self.state.batched_hyps.scores
        hyps_candidates_prob = hyps_scores.unsqueeze(-1) + log_probs_top_k  # hyps from top-k (top-k-prev x top_k)
        hyps_candidates_prob_forced_blank = (
            hyps_scores + log_probs_blank
        )  # hyps with forced blank (top-k-prev x blank)


        # step 2.2 force add final hyps with the same score to the beam
        # final hyps cannot be extended -> mask with minus inf, copy prev scores; label - set to -1
        hyps_candidates_prob = torch.where(
            self.state.active_mask.unsqueeze(-1),
            hyps_candidates_prob,
            self.state.INACTIVE_HYPOTHESIS_SCORE,
        )
        hyps_candidates_prob[..., 0] = torch.where(
            self.state.active_mask,
            hyps_candidates_prob[..., 0],
            hyps_scores,
        )
        labels_top_k = torch.where(self.state.active_mask.unsqueeze(-1), labels_top_k, -1)
            
        # step 2.3: force blank extension with respect to self.max_symbols
        if self.max_symbols is not None:
            force_blank = (self.state.batched_hyps.last_timestep_lasts >= self.max_symbols) & self.state.active_mask
        else:
            force_blank = torch.full_like(self.state.active_mask, fill_value=False)
        # mask all extensions with -inf
        hyps_candidates_prob = torch.where(
            force_blank.unsqueeze(-1),
            self.state.INACTIVE_HYPOTHESIS_SCORE,
            hyps_candidates_prob
        )
        # first element in beam - score for hyp with forced blank
        hyps_candidates_prob[..., 0] = torch.where(
            force_blank, hyps_candidates_prob_forced_blank, hyps_candidates_prob[..., 0]
        )
        labels_top_k = torch.where(
            force_blank.unsqueeze(-1), self._blank_index, labels_top_k
        )

        # step 2.4: final pruning - get top-k from (top-k x top-k) hyps
        hyps_indices = torch.arange(self.beam_size, dtype=torch.long, device=self.state.device)[None, :, None].expand(
            self.state.batch_size, -1, self.beam_size
        )
        next_hyps_prob, hyps_candidates_indices = torch.topk(
            hyps_candidates_prob.view(self.state.batch_size, -1), k=self.beam_size, largest=True, sorted=True
        )
        hyps_indices = torch.gather(hyps_indices.reshape(self.state.batch_size, -1), dim=-1, index=hyps_candidates_indices)
        next_labels = torch.gather(labels_top_k.reshape(self.state.batch_size, -1), dim=-1, index=hyps_candidates_indices)

        # step 3: store results
        if self.max_symbols is None:
            self.state.batched_hyps.add_results_(hyps_indices, next_labels, next_hyps_prob)
        else:
            self.state.batched_hyps.add_results_no_checks_(hyps_indices, next_labels, next_hyps_prob)
        if self.allow_recombine_hyps:
            self.state.batched_hyps.self_recombine_hyps_()

        # step 4: update decoder state + decoder output (+ lm state/scores)
        last_labels_wb = torch.where(
            next_labels >= 0, next_labels, self._blank_index
        )
        preserve_state = last_labels_wb == self._blank_index

        # update decoder + lm state
        # decoder_output: [(B x Beam), 1, Dim]
        prev_decoder_output = torch.gather(
            self.state.decoder_output.view(self.state.batch_size, self.beam_size, 1, -1),
            dim=1,
            index=hyps_indices[:, :, None, None].expand(self.state.batch_size, self.beam_size, 1, self.state.decoder_output.shape[-1]),
        ).view(self.state.batch_size * self.beam_size, 1, -1)
        
        # TODO: move state aggregation to decoder + support stateless decoder:
        #  self.decoder.batch_aggregate_states_beam(...)
        # state: tuple, each is of [Layers, (BxBeam), Dim]
        prev_decoder_state = (
                torch.gather(
                    self.state.decoder_state[0].view(self.state.decoder_state[0].shape[0], self.state.batch_size, self.state.beam_size, -1),
                    dim=2,
                    index=hyps_indices[None, :, :, None].expand(
                        self.state.decoder_state[0].shape[0],
                        self.state.batch_size,
                        self.state.beam_size,
                        self.state.decoder_state[0].shape[-1]
                    ),
                ).view(self.state.decoder_state[0].shape[0], self.state.batch_size * self.beam_size, -1),
                torch.gather(
                    self.state.decoder_state[1].view(self.state.decoder_state[1].shape[0], self.state.batch_size, self.beam_size, -1),
                    dim=2,
                    index=hyps_indices[None, :, :, None].expand(
                        self.state.decoder_state[1].shape[0], self.state.batch_size, self.beam_size, self.state.decoder_state[1].shape[-1]
                    ),
                ).view(self.state.decoder_state[1].shape[0], self.state.batch_size * self.beam_size, -1),
            )

        decoder_output, decoder_state, *_ = self.decoder.predict(
            last_labels_wb.reshape(-1).unsqueeze(1),
            prev_decoder_state,
            add_sos=False,
            batch_size=self.state.batch_size * self.beam_size,
        )
        self.state.decoder_output.copy_(self.joint.project_prednet(decoder_output))  # do not recalculate joint projection
        self.state.decoder_output = torch.where(preserve_state.view(-1)[:, None, None], prev_decoder_output, self.state.decoder_output)
        self.decoder.batch_replace_states_all(src_states=decoder_state, dst_states=self.state.decoder_state)
        self.decoder.batch_replace_states_mask(
            src_states=prev_decoder_state, 
            dst_states=self.state.decoder_state,
            mask=preserve_state.view(-1)
        )
        
        if self.ngram_lm_batch is not None:
            # batch_lm_states: [(BxBeam)]
            # batch_lm_states_candidates: [(BxBeam) x V (without blank)]
            batch_lm_states_candidates = torch.gather(
                batch_lm_states_candidates.view(self.state.batch_size, self.beam_size, -1),
                dim=1,
                index=hyps_indices[:, :, None].expand(
                    self.state.batch_size, self.beam_size, batch_lm_states_candidates.shape[-1]
                ),
            )
            batch_lm_states_prev = torch.gather(
                batch_lm_states.view(self.state.batch_size, self.beam_size), dim=1, index=hyps_indices
            )
            last_labels_wb_blank_replaced = torch.where(
                preserve_state, 0, last_labels_wb
            )

            batch_lm_states = torch.gather(
                batch_lm_states_candidates, dim=-1, index=last_labels_wb_blank_replaced.unsqueeze(-1)
            ).squeeze(-1)
            batch_lm_states = torch.where(preserve_state, batch_lm_states_prev, batch_lm_states).view(-1)

            lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
                states=batch_lm_states
            )  # vocab_size_no_blank
            lm_scores = lm_scores.to(dtype=self.state.float_dtype).view(self.state.batch_size, self.beam_size, -1) * self.ngram_lm_alpha
            
        # step 5: update time indices + active mask
        self.state.time_indices.copy_(self.state.batched_hyps.next_timestep)
        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)
        torch.less_equal(self.state.time_indices, self.state.last_timesteps, out=self.state.active_mask)
        torch.any(self.state.active_mask, out=self.state.active_mask_any)
            
    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        if self.cuda_graphs_mode is not None and x.device.type == "cuda":
            return self.modified_alsd_cuda_graphs(encoder_output=x, encoder_output_length=out_len)

        return self.modified_alsd_beam_torch(encoder_output=x, encoder_output_length=out_len)

# class ModifiedALSDBatchedRNNTComputer(ConfidenceMethodMixin):
#     """
#     mALSD decoding: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053040
#     """

#     def __init__(
#         self,
#         decoder,
#         joint,
#         blank_index: int,
#         beam_size: int,
#         max_symbols_per_step: Optional[int] = 10,
#         preserve_alignments=False,
#         preserve_frame_confidence=False,
#         confidence_method_cfg: Optional[DictConfig] = None,
#         ngram_lm_model: Optional[str | Path] = None,
#         ngram_lm_alpha: float = 0.0,
#         blank_lm_score_mode: Optional[str | BlankLMScoreMode] = None,
#         pruning_mode: Optional[str | PruningMode] = None,
#         allow_recombine_hyps: bool = True,
#         score_norm: bool = True,
#     ):
#         super().__init__()
#         self.decoder = decoder
#         self.joint = joint
#         self._blank_index = blank_index
#         self.beam_size = beam_size
#         self.max_symbols = max_symbols_per_step
#         self.preserve_alignments = preserve_alignments
#         self.preserve_frame_confidence = preserve_frame_confidence
#         self.allow_recombine_hyps = allow_recombine_hyps
#         self._SOS = self._blank_index
#         self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
#         self.score_norm = score_norm
        
#         assert self._SOS == self._blank_index  # "blank as pad" algorithm only
#         assert not self.preserve_alignments
#         assert not self.preserve_frame_confidence
        
#         if ngram_lm_model is not None:
#             assert self._blank_index == self.joint.num_classes_with_blank - self.joint.num_extra_outputs - 1
#             self.ngram_lm_batch = FastNGramLM.from_arpa(lm_path=ngram_lm_model, vocab_size=self._blank_index)
            
#             self.pruning_mode = (
#                 PruningMode.EARLY
#                 if pruning_mode is None
#                 else PruningMode(pruning_mode)
#             )
#             self.blank_lm_score_mode = (
#                 BlankLMScoreMode.LM_WEIGHTED_FULL
#                 if blank_lm_score_mode is None 
#                 else BlankLMScoreMode(blank_lm_score_mode)
#             )
#         else:
#             self.ngram_lm_batch = None
#             self.blank_lm_score_mode = None
#         self.ngram_lm_alpha = ngram_lm_alpha

#     def modified_alsd_beam_torch(
#         self,
#         encoder_output: torch.Tensor,
#         encoder_output_length: torch.Tensor,
#     ) -> list[rnnt_utils.Hypothesis]:
#         batch_size, max_time, vocab_size = encoder_output.shape
#         device = encoder_output.device

#         # do not recalculate joint projection, project only once
#         encoder_output_projected = self.joint.project_encoder(encoder_output)
#         float_dtype = encoder_output_projected.dtype

#         # init output structures: BatchedHyps (for results), BatchedAlignments + last decoder state
#         # init empty batched hypotheses
#         batched_hyps = BatchedBeamHyps(
#             batch_size=batch_size,
#             beam_size=self.beam_size,
#             blank_index=self._blank_index,
#             init_length=max_time * (self.max_symbols + 1) if self.max_symbols is not None else max_time,
#             device=device,
#             float_dtype=float_dtype,
#         )

#         last_labels_wb = torch.full(
#             [batch_size, self.beam_size], fill_value=self._SOS, device=device, dtype=torch.long
#         )

#         batch_indices = (
#             torch.arange(batch_size, dtype=torch.long, device=device)[:, None]
#             .expand(batch_size, self.beam_size)
#             .clone()
#         )
#         time_indices = torch.zeros_like(batch_indices)
#         safe_time_indices = torch.zeros_like(time_indices)  # time indices, guaranteed to be < out_len
#         last_timesteps = (encoder_output_length - 1)[:, None].expand_as(batch_indices)
#         active_mask = time_indices <= last_timesteps

#         if self.ngram_lm_batch is not None:
#             self.ngram_lm_batch.to(device)
            
#             batch_lm_states = self.ngram_lm_batch.get_init_states(batch_size=batch_size * self.beam_size, bos=True)
#             lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(states=batch_lm_states)  # vocab_size_no_blank
#             lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha

#         decoder_output, decoder_state, *_ = self.decoder.predict(
#             last_labels_wb.reshape(-1).unsqueeze(1), None, add_sos=False, batch_size=batch_size * self.beam_size
#         )
#         decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection
#         # decoder_output: [(B x Beam), 1, Dim]

#         while active_mask.any():
#             # step 1: get joint output + fuse with LM (if present)
#             logits = (
#                 self.joint.joint_after_projection(
#                     encoder_output_projected[batch_indices.view(-1), safe_time_indices.view(-1)].unsqueeze(1),
#                     decoder_output,
#                 )
#                 .squeeze(1)
#                 .squeeze(1)
#             )
#             log_probs = F.log_softmax(logits, dim=-1).view(batch_size, self.beam_size, -1)  # [(B x Beam), V]
            
#             if self.ngram_lm_batch is not None:
#                 log_probs_top_k, labels_top_k = self.topk_lm(lm_scores, log_probs)
#             else:
#                 log_probs_top_k, labels_top_k = torch.topk(
#                     log_probs, self.beam_size, dim=-1, largest=True, sorted=True
#                 )

#             # step 2: Make hyps candidates. Add new scores to hyps, force blank if necessary, recombine hyps, prune
#             # step 2.1: hyps candidates
#             log_probs_blank = log_probs[..., self._blank_index]
#             # size: batch_size x beam_size x beam_size (k)
#             hyps_scores = batched_hyps.scores
#             hyps_candidates_prob = hyps_scores.unsqueeze(-1) + log_probs_top_k  # hyps from top-k (top-k-prev x top_k)
#             hyps_candidates_prob_forced_blank = (
#                 hyps_scores + log_probs_blank
#             )  # hyps with forced blank (top-k-prev x blank)

#             # step 2.2 force add final hyps with the same score to the beam
#             # final hyps cannot be extended -> mask with minus inf, copy prev scores; label - set to -1
#             hyps_candidates_prob = torch.where(
#                 active_mask.unsqueeze(-1),
#                 hyps_candidates_prob,
#                 batched_hyps.INACTIVE_SCORE,
#             )
#             hyps_candidates_prob[..., 0] = torch.where(
#                 active_mask,
#                 hyps_candidates_prob[..., 0],
#                 hyps_scores,
#             )
#             labels_top_k = torch.where(active_mask.unsqueeze(-1), labels_top_k, -1)

#             # step 2.3: force blank extension with respect to self.max_symbols
#             if self.max_symbols is not None:
#                 force_blank = (batched_hyps.last_timestep_lasts >= self.max_symbols) & active_mask
#             else:
#                 force_blank = torch.full_like(active_mask, fill_value=False)
#             # mask all extensions with -inf
#             hyps_candidates_prob = torch.where(
#                 force_blank.unsqueeze(-1),
#                 batched_hyps.INACTIVE_SCORE,
#                 hyps_candidates_prob
#             )
#             # first element in beam - score for hyp with forced blank
#             hyps_candidates_prob[..., 0] = torch.where(
#                 force_blank, hyps_candidates_prob_forced_blank, hyps_candidates_prob[..., 0]
#             )
#             labels_top_k = torch.where(
#                 force_blank.unsqueeze(-1), self._blank_index, labels_top_k
#             )

#             # step 2.4: final pruning - get top-k from (top-k x top-k) hyps
#             hyps_indices = torch.arange(self.beam_size, dtype=torch.long, device=device)[None, :, None].expand(
#                 batch_size, -1, self.beam_size
#             )
#             next_hyps_prob, hyps_candidates_indices = torch.topk(
#                 hyps_candidates_prob.view(batch_size, -1), k=self.beam_size, largest=True, sorted=True
#             )
#             hyps_indices = torch.gather(hyps_indices.reshape(batch_size, -1), dim=-1, index=hyps_candidates_indices)
#             next_labels = torch.gather(labels_top_k.reshape(batch_size, -1), dim=-1, index=hyps_candidates_indices)

#             # step 3: store results
#             if self.max_symbols is None:
#                 batched_hyps.add_results_(hyps_indices, next_labels, next_hyps_prob)
#             else:
#                 batched_hyps.add_results_no_checks_(hyps_indices, next_labels, next_hyps_prob)
#             if self.allow_recombine_hyps:
#                 batched_hyps.self_recombine_hyps_()

#             # step 4: update decoder state + decoder output (+ lm state/scores)
#             last_labels_wb = torch.where(
#                 next_labels >= 0, next_labels, self._blank_index
#             )
#             preserve_state = last_labels_wb == self._blank_index

#             # update decoder + lm state
#             # decoder_output: [(B x Beam), 1, Dim]
#             prev_decoder_output = torch.gather(
#                 decoder_output.view(batch_size, self.beam_size, 1, -1),
#                 dim=1,
#                 index=hyps_indices[:, :, None, None].expand(batch_size, self.beam_size, 1, decoder_output.shape[-1]),
#             ).view(batch_size * self.beam_size, 1, -1)
            
#             # TODO: move state aggregation to decoder + support stateless decoder:
#             #  self.decoder.batch_aggregate_states_beam(...)
#             # state: tuple, each is of [Layers, (BxBeam), Dim]
#             prev_decoder_state = (
#                 torch.gather(
#                     decoder_state[0].view(decoder_state[0].shape[0], batch_size, self.beam_size, -1),
#                     dim=2,
#                     index=hyps_indices[None, :, :, None].expand(
#                         decoder_state[0].shape[0], batch_size, self.beam_size, decoder_state[0].shape[-1]
#                     ),
#                 ).view(decoder_state[0].shape[0], batch_size * self.beam_size, -1),
#                 torch.gather(
#                     decoder_state[1].view(decoder_state[1].shape[0], batch_size, self.beam_size, -1),
#                     dim=2,
#                     index=hyps_indices[None, :, :, None].expand(
#                         decoder_state[1].shape[0], batch_size, self.beam_size, decoder_state[1].shape[-1]
#                     ),
#                 ).view(decoder_state[1].shape[0], batch_size * self.beam_size, -1),
#             )

#             decoder_output, decoder_state, *_ = self.decoder.predict(
#                 last_labels_wb.reshape(-1).unsqueeze(1),
#                 prev_decoder_state,
#                 add_sos=False,
#                 batch_size=batch_size * self.beam_size,
#             )
#             decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

#             decoder_output = torch.where(preserve_state.view(-1)[:, None, None], prev_decoder_output, decoder_output)
#             self.decoder.batch_replace_states_mask(
#                 src_states=prev_decoder_state, dst_states=decoder_state, mask=preserve_state.view(-1)
#             ) 
#             if self.ngram_lm_batch is not None:
#                 # batch_lm_states: [(BxBeam)]
#                 # batch_lm_states_candidates: [(BxBeam) x V (without blank)]
#                 batch_lm_states_candidates = torch.gather(
#                     batch_lm_states_candidates.view(batch_size, self.beam_size, -1),
#                     dim=1,
#                     index=hyps_indices[:, :, None].expand(
#                         batch_size, self.beam_size, batch_lm_states_candidates.shape[-1]
#                     ),
#                 )
#                 batch_lm_states_prev = torch.gather(
#                     batch_lm_states.view(batch_size, self.beam_size), dim=1, index=hyps_indices
#                 )
#                 last_labels_wb_blank_replaced = torch.where(
#                     preserve_state, 0, last_labels_wb
#                 )

#                 batch_lm_states = torch.gather(
#                     batch_lm_states_candidates, dim=-1, index=last_labels_wb_blank_replaced.unsqueeze(-1)
#                 ).squeeze(-1)
#                 batch_lm_states = torch.where(preserve_state, batch_lm_states_prev, batch_lm_states).view(-1)

#                 lm_scores, batch_lm_states_candidates = self.ngram_lm_batch.advance(
#                     states=batch_lm_states
#                 )  # vocab_size_no_blank
#                 lm_scores = lm_scores.to(dtype=float_dtype).view(batch_size, self.beam_size, -1) * self.ngram_lm_alpha

#             # step 5: update time indices + active mask
#             time_indices = batched_hyps.next_timestep
#             torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
#             active_mask = time_indices <= last_timesteps
#             # torch.cuda.set_sync_debug_mode(0)

#         return batched_hyps.to_hyps_list(score_norm=self.score_norm)
    
#     def topk_lm(self, lm_scores, log_probs):
#         if self.pruning_mode is PruningMode.LATE:
#             if self.blank_lm_score_mode is BlankLMScoreMode.NO_SCORE:
#                 log_probs[..., :-1] += lm_scores
#                 log_probs_top_k, labels_top_k = torch.topk(
#                         log_probs, self.beam_size, dim=-1, largest=True, sorted=True
#                     )
#             elif self.blank_lm_score_mode is BlankLMScoreMode.PRESERVE_BLANK:
#                 _, labels_top_k_no_lm = torch.topk(log_probs, self.beam_size, dim=-1, largest=True, sorted=True)
#                 log_probs[..., :-1] += lm_scores
#                 _, labels_with_lm_nb_top_k = torch.topk(
#                         log_probs[..., :-1], self.beam_size, dim=-1, largest=True, sorted=True
#                     )
#                     # [(BxBeam), beam]
#                     # if blank was in labels_top_k -> add blank (last in beam)
#                 labels_top_k = labels_with_lm_nb_top_k
#                 labels_top_k[..., -1] = torch.where(
#                         (labels_top_k_no_lm == self._blank_index).any(dim=-1),
#                         self._blank_index,
#                         labels_top_k[..., -1],
#                     )
#                 log_probs_top_k = torch.gather(log_probs, dim=-1, index=labels_top_k)
#             elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED:
#                 log_probs[..., :-1] += lm_scores
#                 log_probs[..., -1] *= 1 + self.ngram_lm_alpha
#                 log_probs_top_k, labels_top_k = torch.topk(
#                         log_probs, self.beam_size, dim=-1, largest=True, sorted=True
#                     )
#             elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL:
#                 blank_logprob = log_probs[..., -1]
#                 non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
#                 log_probs[..., :-1] += non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + lm_scores
#                 log_probs[..., -1] *= 1 + self.ngram_lm_alpha
#                 log_probs_top_k, labels_top_k = torch.topk(
#                         log_probs, self.beam_size, dim=-1, largest=True, sorted=True
#                     )
#             elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL_FIXED_BLANK:
#                 blank_logprob = log_probs[..., -1]
#                 non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
#                 log_probs[..., :-1] += non_blank_logprob.unsqueeze(-1) + lm_scores # blank prob - the same
#                 log_probs_top_k, labels_top_k = torch.topk(
#                         log_probs, self.beam_size, dim=-1, largest=True, sorted=True
#                     )
#             elif self.blank_lm_score_mode is BlankLMScoreMode.LM_MAX:
#                 log_probs[..., :-1] += lm_scores
#                 log_probs[..., -1] += lm_scores.max(dim=-1, keepdim=False).values
#                 log_probs_top_k, labels_top_k = torch.topk(
#                         log_probs, self.beam_size, dim=-1, largest=True, sorted=True
#                     )
#             elif self.blank_lm_score_mode is BlankLMScoreMode.LM_TOP_MAX:
#                 log_probs[..., :-1] += lm_scores
#                 _, labels_with_lm_nb_top_k = torch.topk(
#                         log_probs[..., :-1], self.beam_size, dim=-1, largest=True, sorted=True
#                     )
#                     # [(BxBeam), beam]
#                 lm_only_scores = torch.gather(
#                         lm_scores,
#                         dim=-1,
#                         index=labels_with_lm_nb_top_k,
#                     )
#                 log_probs[..., -1] += lm_only_scores.max(dim=-1, keepdim=False).values
#                 log_probs_top_k, labels_top_k = torch.topk(
#                         log_probs, self.beam_size, dim=-1, largest=True, sorted=True
#                     )
#             else:
#                 raise NotImplementedError(
#                         f"The combination of blank scoring mode '{self.blank_lm_score_mode}' "
#                         f"and pruning mode '{self.pruning_mode}' is not implemented."
#                 )
#         elif self.pruning_mode is PruningMode.EARLY:
#             if self.blank_lm_score_mode is BlankLMScoreMode.NO_SCORE:
#                 # log_probs[..., :-1] += lm_scores
#                 log_probs_top_k, labels_top_k = torch.topk(
#                         log_probs, self.beam_size, dim=-1, largest=True, sorted=True
#                     )
#                 masked_labels = torch.where(labels_top_k==self._blank_index, 0, labels_top_k)
#                 log_probs_top_k = torch.where(
#                     labels_top_k==self._blank_index,
#                     log_probs_top_k,
#                     log_probs_top_k + torch.gather(lm_scores, dim=-1,index=masked_labels))
#             elif self.blank_lm_score_mode is BlankLMScoreMode.LM_WEIGHTED_FULL:
#                 # choosing topk from acoustic model
#                 log_probs_top_k, labels_top_k = log_probs.topk(self.beam_size, dim=-1, largest=True, sorted=True)
                
#                 blank_logprob = log_probs[..., -1]
#                 non_blank_logprob = torch.log1p(-torch.clamp(torch.exp(blank_logprob), max=1.0 - 1e-6))
                
#                 masked_labels = torch.where(labels_top_k==self._blank_index, 0, labels_top_k)
#                 log_probs_top_k = torch.where(
#                     labels_top_k==self._blank_index,
#                     log_probs_top_k * (1 + self.ngram_lm_alpha),
#                     log_probs_top_k + non_blank_logprob.unsqueeze(-1) * self.ngram_lm_alpha + torch.gather(lm_scores, dim=-1, index=masked_labels)
#                 )
#             else:
#                 raise NotImplementedError(
#                         f"The combination of blank scoring mode '{self.blank_lm_score_mode}' "
#                         f"and pruning mode '{self.pruning_mode}' is not implemented."
#                 )
#         else:
#             raise NotImplementedError(f"Pruning mode {self.pruning_mode} is not implemented.")
            
#         return log_probs_top_k, labels_top_k

#     def __call__(
#         self,
#         x: torch.Tensor,
#         out_len: torch.Tensor,
#     ) -> list[rnnt_utils.Hypothesis]:
#         return self.modified_alsd_beam_torch(encoder_output=x, encoder_output_length=out_len)
