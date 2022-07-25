# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import collections

import pytest
import torch

from nemo.collections.nlp.data.question_answering.dataset.qa_dataset import QADataset
from nemo.collections.nlp.data.question_answering.dataset.qa_gpt_dataset import GPTQADataset
from nemo.collections.nlp.metrics.qa_metrics import QAMetrics


@pytest.mark.unit
def test_remove_articles():
    sentences = [
        "this is an apple",
        "this is the apple",
        "this is a fruit",
    ]

    expected_article_removed_sents = ["this is   apple", "this is   apple", "this is   fruit"]

    article_removed_sents = [QAMetrics.remove_articles(sent) for sent in sentences]

    assert article_removed_sents == expected_article_removed_sents


@pytest.mark.unit
def test_white_space_fix():
    sentences = [
        "sentence with a space",
        "sentence with multiple   spaces",
    ]

    expected_white_space_fixed_sents = [
        "sentence with a space",
        "sentence with multiple spaces",
    ]

    white_space_fixed_sents = [QAMetrics.white_space_fix(sent) for sent in sentences]

    assert white_space_fixed_sents == expected_white_space_fixed_sents


@pytest.mark.unit
def test_remove_punc():
    sentence = "this, is. a! sentence: with; punctuations?"
    expected_punc_removed_sent = "this is a sentence with punctuations"

    punc_removed_sent = QAMetrics.remove_punc(sentence)

    assert punc_removed_sent == expected_punc_removed_sent


@pytest.mark.unit
def test_get_normalized_tokens():
    sentence = 'I am happy'
    tokens = ['i', 'am', 'happy']
    assert tokens == QAMetrics._get_normalized_tokens(sentence)

    sentence = 'I am a person'
    tokens = ['i', 'am', 'person']
    assert tokens == QAMetrics._get_normalized_tokens(sentence)

    sentence = 'I am a person.'
    tokens = ['i', 'am', 'person']
    assert tokens == QAMetrics._get_normalized_tokens(sentence)


@pytest.mark.unit
def test_get_one_f1():
    generated_field = 'That is so good'
    ground_truth_field = 'That is so awesome'

    f1 = QAMetrics.get_one_f1(generated_field, ground_truth_field)
    assert f1 == 0.75

    generated_field = ''
    ground_truth_field = 'That'

    f1 = QAMetrics.get_one_f1(generated_field, ground_truth_field)
    assert f1 == 0


@pytest.mark.unit
def test_get_one_exact_match():
    generated_field = 'That is so good'
    ground_truth_field = 'That is so awesome'

    em = QAMetrics.get_one_exact_match(generated_field, ground_truth_field)
    assert em == 0

    generated_field = 'That is so good!'
    ground_truth_field = 'That is so good.'

    em = QAMetrics.get_one_exact_match(generated_field, ground_truth_field)
    assert em == 1

    generated_field = 'That is so good'
    ground_truth_field = 'that is so good'

    em = QAMetrics.get_one_exact_match(generated_field, ground_truth_field)
    assert em == 1


@pytest.mark.unit
def test_split_into_words():
    text = 'hi yo'
    char_to_word_offset = [0, 0, 0, 1, 1]
    doc_tokens = ["hi", "yo"]
    output = QADataset.split_into_words(text)
    assert output[0] == doc_tokens
    assert output[1] == char_to_word_offset

    text = 'i am good'
    char_to_word_offset = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    doc_tokens = ["i", "am", 'good']
    output = QADataset.split_into_words(text)
    assert output[0] == doc_tokens
    assert output[1] == char_to_word_offset


@pytest.mark.unit
def test_get_doc_spans():
    all_doc_tokens = ['a'] * 15
    max_tokens_for_doc = 10
    doc_stride = 5
    doc_spans = QADataset.get_docspans(all_doc_tokens, max_tokens_for_doc, doc_stride)

    assert len(doc_spans) == 2
    assert doc_spans[0].start == 0
    assert doc_spans[0].length == 10
    assert doc_spans[1].start == 5
    assert doc_spans[1].length == 10


@pytest.mark.unit
def test_get_average_dist_to_tok_start_and_end():
    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

    doc_span = _DocSpan(start=0, length=5)

    tok_start_position = 1
    tok_end_position = 3

    assert 2 == QADataset.get_average_dist_to_tok_start_and_end(doc_span, tok_start_position, tok_end_position)

    doc_span = _DocSpan(start=5, length=5)

    tok_start_position = 1
    tok_end_position = 2

    assert 6 == QADataset.get_average_dist_to_tok_start_and_end(doc_span, tok_start_position, tok_end_position)

    doc_span = _DocSpan(start=5, length=4)

    tok_start_position = 1
    tok_end_position = 2

    assert 5 == QADataset.get_average_dist_to_tok_start_and_end(doc_span, tok_start_position, tok_end_position)


@pytest.mark.unit
def test_keep_relevant_docspans():

    _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

    doc_spans = [_DocSpan(start=start, length=5) for start in range(15)]

    tok_start_position = 1
    tok_end_position = 2

    mode = 'all'
    assert doc_spans == QADataset.keep_relevant_docspans(doc_spans, tok_start_position, tok_end_position, mode)

    doc_spans = [_DocSpan(start=start, length=5) for start in range(15)]

    tok_start_position = -1
    tok_end_position = -1

    mode = 'only_positive'

    expected_doc_spans = []
    assert expected_doc_spans == QADataset.keep_relevant_docspans(
        doc_spans, tok_start_position, tok_end_position, mode
    )

    doc_spans = [_DocSpan(start=start, length=5) for start in range(15)]

    tok_start_position = 1
    tok_end_position = 2

    mode = 'only_positive'

    expected_doc_spans = [_DocSpan(start=0, length=5), _DocSpan(start=1, length=5)]
    assert expected_doc_spans == QADataset.keep_relevant_docspans(
        doc_spans, tok_start_position, tok_end_position, mode
    )

    doc_spans = [_DocSpan(start=start, length=5) for start in range(15)]

    tok_start_position = 1
    tok_end_position = 2

    mode = 'limited_negative'

    expected_doc_spans = [_DocSpan(start=start, length=5) for start in range(10)]
    assert expected_doc_spans == QADataset.keep_relevant_docspans(
        doc_spans, tok_start_position, tok_end_position, mode
    )


@pytest.mark.unit
def test_gpt_no_pad_loss_masking():
    input_ids = [1] * 15 + [50257] * 15
    input_ids = torch.tensor(input_ids)

    input_attn_mask = [1] * 16 + [0] * 14
    input_attn_mask = torch.Tensor(input_attn_mask)

    training_mask_end = 10

    expected_labels = [-100] * 10 + [1] * 5 + [50257] + [-100] * 14
    expected_labels = torch.tensor(expected_labels)

    labels = GPTQADataset.update_labels_for_no_pad_loss(input_ids, training_mask_end, input_attn_mask)

    assert torch.all(labels.eq(expected_labels))
