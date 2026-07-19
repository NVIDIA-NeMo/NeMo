# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import numpy as np
import pytest
import torch
from lightning.pytorch import Trainer
from torch.nn.utils.rnn import pad_sequence

from nemo.collections.asr.models import EncDecCTCModelBPE
from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import (
    BoostingTreeModelConfig,
    GPUBoostingTreeModel,
    PhraseItem,
)
from nemo.collections.asr.parts.context_biasing.context_graph_universal import ContextGraph

DEVICES = [torch.device("cpu")]

if torch.cuda.is_available():
    DEVICES.append(torch.device("cuda"))


@pytest.fixture(scope="module")
def test_context_graph():
    phrases = ["abc", "abd", "c"]
    phrases_ids = [[1, 2, 3], [1, 2, 4], [3]]
    context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
    context_graph.build(token_ids=phrases_ids, phrases=phrases, scores=None, uniform_weights=False)
    return context_graph


@pytest.fixture(scope="module")
def test_boosting_tree(test_context_graph):
    boosting_tree = GPUBoostingTreeModel.from_context_graph(
        context_graph=test_context_graph,
        vocab_size=5,
        unk_score=0.0,
        final_eos_score=0.0,
        use_triton=True,
        uniform_weights=False,
    )
    return boosting_tree


@pytest.fixture(scope="module")
def conformer_ctc_bpe_model():
    model = EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
    model.set_trainer(Trainer(devices=1, accelerator="cpu"))
    model = model.eval()
    return model


class TestGPUBoostingTreeModel:
    @pytest.mark.unit
    def test_building_context_graph(self, test_context_graph):
        """Test initial python-based context graph"""
        context_graph = test_context_graph
        assert context_graph.num_nodes == 5
        # end nodes
        assert context_graph.root.next[1].next[2].next[3].is_end
        assert context_graph.root.next[1].next[2].next[4].is_end
        assert context_graph.root.next[3].is_end
        # words in the end nodes
        assert context_graph.root.next[1].next[2].next[3].phrase == "abc"
        assert context_graph.root.next[1].next[2].next[4].phrase == "abd"
        assert context_graph.root.next[3].phrase == "c"
        # fail links
        assert context_graph.root.next[1].next[2].next[3].fail.token == 3
        assert context_graph.root.next[1].next[2].next[4].fail.token == -1  # root
        assert context_graph.root.next[3].fail.token == -1  # root
        # node scores
        assert round(context_graph.root.next[1].next[2].next[3].node_score, 2) == 4.79
        assert round(context_graph.root.next[1].next[2].next[4].node_score, 2) == 4.79
        assert round(context_graph.root.next[3].node_score, 2) == 1.0

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    @pytest.mark.parametrize("batch_size", [1, 3, 8])
    def test_advance_method(self, test_boosting_tree, device, batch_size):
        """Test advance method with different batch sizes"""
        test_boosting_tree.to(device)
        # Test with initial states
        init_states = test_boosting_tree.get_init_states(batch_size=batch_size, bos=True)
        scores, next_states = test_boosting_tree.advance(init_states)

        assert scores.shape == (batch_size, 5)  # vocab_size=5
        assert next_states.shape == (batch_size, 5)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_get_final_method(self, test_boosting_tree, device):
        """Test get_final method for EOS scoring"""
        test_boosting_tree.to(device)
        # Test with various states
        states = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
        final_scores = test_boosting_tree.get_final(states)

        assert final_scores.shape == (3,)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_boosting_tree_inference(self, test_boosting_tree, device):
        """Test boosting tree inference with predefined sentences"""
        test_boosting_tree.to(device)

        sentences_ids = [[1, 2, 3, 2, 1], [2, 2, 1, 2, 4], [3, 1, 2, 1], []]  # ['abcba', 'bbabd', 'caba', '']
        boosting_scores = test_boosting_tree(
            labels=pad_sequence([torch.LongTensor(sentence) for sentence in sentences_ids], batch_first=True).to(
                device
            ),
            labels_lengths=torch.LongTensor([len(sentence) for sentence in sentences_ids]).to(device),
            bos=False,
            eos=False,
        )
        correct_answer = torch.tensor(
            [
                [1.0000, 1.6931, 2.0986, 0.0000, 1.0000],
                [0.0000, 0.0000, 1.0000, 1.6931, 2.0986],
                [1.0000, 1.0000, 1.6931, -1.6931, 0.0000],
                [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            ],
            device=device,
        )
        assert torch.allclose(boosting_scores, correct_answer, atol=1e-4)

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_triton_vs_pytorch_consistency(self, test_context_graph):
        """Compare Triton vs PyTorch implementations"""
        device = torch.device("cuda")

        # Create two identical models with different implementations
        boosting_tree_triton = GPUBoostingTreeModel.from_context_graph(
            context_graph=test_context_graph, vocab_size=5, use_triton=True
        ).to(device)

        boosting_tree_pytorch = GPUBoostingTreeModel.from_context_graph(
            context_graph=test_context_graph, vocab_size=5, use_triton=False
        ).to(device)

        # Test with same input
        sentences_ids = [[1, 2, 3, 2, 1], [2, 2, 1, 2, 4]]
        labels = pad_sequence([torch.LongTensor(s) for s in sentences_ids], batch_first=True).to(device)
        lengths = torch.LongTensor([len(s) for s in sentences_ids]).to(device)

        scores_triton = boosting_tree_triton(labels=labels, labels_lengths=lengths, bos=False, eos=False)
        scores_pytorch = boosting_tree_pytorch(labels=labels, labels_lengths=lengths, bos=False, eos=False)

        assert torch.allclose(scores_triton, scores_pytorch, atol=1e-5)

    @pytest.mark.unit
    def test_eos_handling(self, test_context_graph):
        """Test EOS token handling (important for AED models)"""
        boosting_tree = GPUBoostingTreeModel.from_context_graph(
            context_graph=test_context_graph, vocab_size=5, unk_score=0.0, final_eos_score=1.0
        )

        # Test advance with EOS
        init_states = torch.tensor([1, 2], dtype=torch.long)
        scores, next_states = boosting_tree.advance(init_states, eos_id=0)

        # state 2 in the 1st batch should have final_eos_score value
        assert (
            round(scores[0, 0].item(), 2) == 1.69
        )  # (1.69+0): 1.69 as max score for state 1 and 0 because it is not final state
        assert scores[1, 0] == 2.0  # (1+1): 1 as max score for state 2 and 1 because it is final state

    @pytest.mark.unit
    # I need to test that the boosting tree model is built correctly from the config using model_path, key_phrases_file, key_phrases_list
    def test_boosting_tree_model_from_config(self, conformer_ctc_bpe_model, tmp_path):
        """Test that the boosting tree model is built correctly from the config using model_path, key_phrases_file, key_phrases_list"""

        # 1. build boosting tree model from model path
        boosting_tree_cfg = BoostingTreeModelConfig()
        phrases = ["abc", "abd", "c"]
        phrases_ids = [conformer_ctc_bpe_model.tokenizer.text_to_ids(phrase) for phrase in phrases]
        context_graph = ContextGraph(
            context_score=boosting_tree_cfg.context_score, depth_scaling=boosting_tree_cfg.depth_scaling
        )
        context_graph.build(
            token_ids=phrases_ids, phrases=phrases, scores=None, uniform_weights=boosting_tree_cfg.uniform_weights
        )
        test_boosting_tree = GPUBoostingTreeModel.from_context_graph(
            context_graph=context_graph,
            vocab_size=conformer_ctc_bpe_model.tokenizer.vocab_size,
            unk_score=boosting_tree_cfg.unk_score,
            final_eos_score=boosting_tree_cfg.final_eos_score,
            use_triton=boosting_tree_cfg.use_triton,
            uniform_weights=boosting_tree_cfg.uniform_weights,
        )

        test_boosting_tree.save_to(tmp_path / "test_boosting_tree.nemo")
        boosting_tree_cfg = BoostingTreeModelConfig(model_path=tmp_path / "test_boosting_tree.nemo")
        boosting_tree_from_model_path = GPUBoostingTreeModel.from_config(
            boosting_tree_cfg, tokenizer=conformer_ctc_bpe_model.tokenizer
        )

        # 2. build boosting tree model from key phrases file
        with open(tmp_path / "test_boosting_tree.txt", "w") as f:
            f.write("abc\nabd\nc")
        boosting_tree_cfg = BoostingTreeModelConfig(key_phrases_file=tmp_path / "test_boosting_tree.txt")
        boosting_tree_from_key_phrases_file = GPUBoostingTreeModel.from_config(
            boosting_tree_cfg, tokenizer=conformer_ctc_bpe_model.tokenizer
        )

        # 3. build boosting tree model from key phrases list
        boosting_tree_cfg = BoostingTreeModelConfig(key_phrases_list=["abc", "abd", "c"])
        boosting_tree_from_key_phrases_list = GPUBoostingTreeModel.from_config(
            boosting_tree_cfg, tokenizer=conformer_ctc_bpe_model.tokenizer
        )

        # check that the boosting tree models are the same
        assert torch.allclose(
            boosting_tree_from_model_path.arcs_weights, boosting_tree_from_key_phrases_file.arcs_weights
        )
        assert torch.allclose(
            boosting_tree_from_model_path.arcs_weights, boosting_tree_from_key_phrases_list.arcs_weights
        )


class TestPerPhraseBoostingParams:
    @pytest.mark.unit
    def test_per_phrase_context_score_affects_only_that_phrase(self):
        """Per-phrase context_score changes arc scores of that phrase only (shared prefixes take the max)"""
        phrases = ["abc", "abd", "c"]
        phrases_ids = [[1, 2, 3], [1, 2, 4], [3]]
        context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
        context_graph.build(token_ids=phrases_ids, phrases=phrases, scores=[None, 2.0, None], uniform_weights=False)

        root = context_graph.root
        # shared prefix "ab" is raised to the max of the two phrase scores
        assert root.next[1].token_score == pytest.approx(2.0)
        assert root.next[1].next[2].token_score == pytest.approx(2.0)
        # "abd" leaf uses its custom score, "abc" leaf keeps the global one
        assert root.next[1].next[2].next[4].token_score == pytest.approx(2.0 + np.log(3))
        assert root.next[1].next[2].next[3].token_score == pytest.approx(1.0 + np.log(3))
        # "c" is unaffected
        assert root.next[3].token_score == pytest.approx(1.0)

    @pytest.mark.unit
    @pytest.mark.parametrize("device", DEVICES)
    def test_per_phrase_alpha_scales_whole_phrase(self, device):
        """Uniform per-phrase alpha scales all graph scores (arcs and backoffs) by the same factor"""
        phrases = ["abc", "abd", "c"]
        phrases_ids = [[1, 2, 3], [1, 2, 4], [3]]
        alpha = 2.0

        trees = []
        for alphas in (None, [alpha] * len(phrases)):
            context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
            context_graph.build(
                token_ids=phrases_ids, phrases=phrases, scores=None, alphas=alphas, uniform_weights=False
            )
            trees.append(
                GPUBoostingTreeModel.from_context_graph(
                    context_graph=context_graph,
                    vocab_size=5,
                    unk_score=0.0,
                    final_eos_score=0.0,
                    use_triton=True,
                    uniform_weights=False,
                ).to(device)
            )
        baseline_tree, scaled_tree = trees

        # node-level check: every token/node score is scaled by alpha
        assert scaled_tree.num_states == baseline_tree.num_states
        assert torch.allclose(scaled_tree.arcs_weights, alpha * baseline_tree.arcs_weights)
        assert torch.allclose(scaled_tree.backoff_weights, alpha * baseline_tree.backoff_weights)

        # end-to-end check (includes backoff transitions): scores are exactly alpha * baseline
        sentences_ids = [[1, 2, 3, 2, 1], [2, 2, 1, 2, 4], [3, 1, 2, 1]]
        labels = pad_sequence([torch.LongTensor(s) for s in sentences_ids], batch_first=True).to(device)
        lengths = torch.LongTensor([len(s) for s in sentences_ids]).to(device)
        baseline_scores = baseline_tree(labels=labels, labels_lengths=lengths, bos=False, eos=False)
        scaled_scores = scaled_tree(labels=labels, labels_lengths=lengths, bos=False, eos=False)
        assert torch.allclose(scaled_scores, alpha * baseline_scores, atol=1e-4)

    @pytest.mark.unit
    def test_per_phrase_alpha_shared_suffix_backoff(self):
        """Backoff transitions between phrases with different alphas keep alpha-consistent cumulative scores"""
        phrases = ["ab", "c"]
        phrases_ids = [[1, 2], [3]]
        context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
        context_graph.build(token_ids=phrases_ids, phrases=phrases, scores=None, alphas=[2.0, 3.0])
        boosting_tree = GPUBoostingTreeModel.from_context_graph(
            context_graph=context_graph, vocab_size=5, unk_score=0.0, final_eos_score=0.0, use_triton=True
        )

        # [1, 3]: "a" arc (2.0), then backoff from non-final "a" state (-2.0) + "c" arc (3.0) = 1.0
        # [1, 2, 3]: "a" (2.0), "b" (2 * (1 + log(2))), then "c" from final state without penalty (3.0)
        sentences_ids = [[1, 3, 0], [1, 2, 3]]
        boosting_scores = boosting_tree(
            labels=torch.LongTensor(sentences_ids),
            labels_lengths=torch.LongTensor([2, 3]),
            bos=False,
            eos=False,
        )
        expected = torch.tensor(
            [
                [2.0, 1.0, 0.0],
                [2.0, 2.0 * (1.0 + np.log(2)), 3.0],
            ],
            dtype=boosting_scores.dtype,
        )
        assert torch.allclose(boosting_scores, expected, atol=1e-4)

    @pytest.mark.unit
    def test_per_phrase_alpha_scales_final_eos_score(self):
        """final_eos_score of an end state is scaled by the alpha of the phrase ending there"""
        phrases = ["abc", "abd", "c"]
        phrases_ids = [[1, 2, 3], [1, 2, 4], [3]]
        context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
        context_graph.build(token_ids=phrases_ids, phrases=phrases, scores=None, alphas=[2.0, None, 3.0])
        boosting_tree = GPUBoostingTreeModel.from_context_graph(
            context_graph=context_graph, vocab_size=5, unk_score=0.0, final_eos_score=1.0, use_triton=True
        )

        # three end states scaled by their phrase alphas (2.0, 1.0, 3.0), all other states are not final
        final_weights = boosting_tree.final_weights[: boosting_tree.num_states]
        assert sorted(w for w in final_weights.tolist() if w != 0.0) == pytest.approx([1.0, 2.0, 3.0])

    @pytest.mark.unit
    def test_explicit_zero_context_score_is_honored(self):
        """An explicit per-phrase context_score of 0.0 is not treated as 'use the default' anymore"""
        context_graph = ContextGraph(context_score=1.0, depth_scaling=1.0)
        context_graph.build(token_ids=[[1, 2]], phrases=["ab"], scores=[0.0])

        assert context_graph.root.next[1].token_score == pytest.approx(0.0)
        # only the depth term remains for tokens after the first one
        assert context_graph.root.next[1].next[2].token_score == pytest.approx(np.log(2))

    @pytest.mark.unit
    def test_negative_alpha_raises(self):
        """Negative per-phrase boosting_tree_alpha is rejected"""
        with pytest.raises(ValueError, match="Negative boosting_tree_alpha"):
            GPUBoostingTreeModel._validate_phrase_items([PhraseItem("abc", boosting_tree_alpha=-1.0)])

    @pytest.mark.unit
    def test_per_phrase_params_none_matches_baseline(self, conformer_ctc_bpe_model):
        """key_phrase_items_list with all-default params builds the same graph as key_phrases_list"""
        phrases = ["abc", "abd", "c"]
        baseline_tree = GPUBoostingTreeModel.from_config(
            BoostingTreeModelConfig(key_phrases_list=phrases), tokenizer=conformer_ctc_bpe_model.tokenizer
        )
        items_tree = GPUBoostingTreeModel.from_config(
            BoostingTreeModelConfig(key_phrase_items_list=[PhraseItem(phrase) for phrase in phrases]),
            tokenizer=conformer_ctc_bpe_model.tokenizer,
        )

        assert torch.allclose(baseline_tree.arcs_weights, items_tree.arcs_weights)
        assert torch.allclose(baseline_tree.backoff_weights, items_tree.backoff_weights)
        assert torch.allclose(baseline_tree.final_weights, items_tree.final_weights)

    @pytest.mark.unit
    def test_per_phrase_context_score_precedence_over_score_per_phrase(self, conformer_ctc_bpe_model):
        """Per-phrase context_score wins over score_per_phrase for that phrase only"""
        tokenizer = conformer_ctc_bpe_model.tokenizer
        cfg = BoostingTreeModelConfig(
            key_phrase_items_list=[PhraseItem("abc", context_score=2.0), PhraseItem("abd")],
            score_per_phrase=4.0,
        )
        items_tree = GPUBoostingTreeModel.from_config(cfg, tokenizer=tokenizer)

        # reference graph: "abc" uses the per-phrase score, "abd" uses score_per_phrase / len(phrase)
        reference_graph = ContextGraph(context_score=cfg.context_score, depth_scaling=cfg.depth_scaling)
        reference_graph.build(
            token_ids=[tokenizer.text_to_ids(phrase) for phrase in ("abc", "abd")],
            phrases=["abc", "abd"],
            scores=[2.0, round(4.0 / len("abd"), 2)],
        )
        reference_tree = GPUBoostingTreeModel.from_context_graph(
            context_graph=reference_graph,
            vocab_size=tokenizer.vocab_size,
            unk_score=cfg.unk_score,
            final_eos_score=cfg.final_eos_score,
        )

        assert torch.allclose(items_tree.arcs_weights, reference_tree.arcs_weights)
        assert torch.allclose(items_tree.backoff_weights, reference_tree.backoff_weights)

    @pytest.mark.unit
    def test_per_phrase_params_with_bpe_dropout(self, conformer_ctc_bpe_model):
        """All alternative BPE tokenizations of a phrase get that phrase's alpha.

        Note: two consecutive builds with BPE dropout sample different alternative tokenizations
        (sentencepiece seeds its random generator only on first use), so the check is done
        within a single build via the final weights of the end states.
        """
        cfg = BoostingTreeModelConfig(
            key_phrase_items_list=[PhraseItem("nvidia", boosting_tree_alpha=2.0), PhraseItem("omniverse")],
            use_bpe_dropout=True,
            final_eos_score=1.0,
        )
        boosting_tree = GPUBoostingTreeModel.from_config(cfg, tokenizer=conformer_ctc_bpe_model.tokenizer)

        # every end state of a "nvidia" tokenization carries final_eos_score * 2.0,
        # every end state of an "omniverse" tokenization keeps final_eos_score * 1.0
        final_weights = boosting_tree.final_weights[: boosting_tree.num_states]
        assert set(w for w in final_weights.tolist() if w != 0.0) == {1.0, 2.0}
