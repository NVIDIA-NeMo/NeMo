# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# Copyright 2018 The Google AI Team Authors.
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

# Most of the code here has been copied from:
#   https://github.com/google-research/albert/blob/master/create_pretraining_data.py
# with some modifications.

import collections
import math
import os
import subprocess
import time

import numpy as np
import torch

from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.collections.nlp.data.language_modeling.megatron.indexed_dataset import make_dataset as make_indexed_dataset
from nemo.collections.nlp.data.language_modeling.megatron.lm_adapted_t5_dataset import T5LMAdaptedDataset
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero

try:
    from apex.transformer import parallel_state

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False


DSET_TYPE_BERT = 'standard_bert'
DSET_TYPE_ICT = 'ict'
DSET_TYPE_T5 = 't5'
DSET_TYPE_T5_LM = 't5_prefix_lm'

DSET_TYPES = [DSET_TYPE_BERT, DSET_TYPE_ICT, DSET_TYPE_T5, DSET_TYPE_T5_LM]


def compile_helper():
    """Compile helper function ar runtime. Make sure this
    is invoked on a single process."""

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(['make', '-C', path])
    if ret.returncode != 0:
        logging.error("Making C++ dataset helpers module failed, exiting.")
        import sys

        sys.exit(1)


def get_a_and_b_segments(sample, np_rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, 'make sure each sample has at least two sentences.'

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in numpy is exclusive.
        a_end = np_rng.randint(1, n_sentences)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if np_rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random


def truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, np_rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    # print(len_a, len_b, max_num_tokens)
    assert len_a > 0
    if len_a + len_b <= max_num_tokens:
        return False
    while len_a + len_b > max_num_tokens:
        if len_a > len_b:
            len_a -= 1
            tokens = tokens_a
        else:
            len_b -= 1
            tokens = tokens_b
        if np_rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()
    return True


def create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id):
    """Merge segments A and B, add [CLS] and [SEP] and build tokentypes."""

    tokens = []
    tokentypes = []
    # [CLS].
    tokens.append(cls_id)
    tokentypes.append(0)
    # Segment A.
    for token in tokens_a:
        tokens.append(token)
        tokentypes.append(0)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(0)
    # Segment B.
    for token in tokens_b:
        tokens.append(token)
        tokentypes.append(1)
    if tokens_b:
        # [SEP].
        tokens.append(sep_id)
        tokentypes.append(1)

    return tokens, tokentypes


MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def is_start_piece(piece):
    """Check if the current word piece is the starting piece. (BERT)"""
    # When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    return not piece.startswith("##")


def create_masked_lm_predictions(
    tokens,
    vocab_id_list,
    vocab_id_to_token_dict,
    masked_lm_prob,
    cls_id,
    sep_id,
    mask_id,
    max_predictions_per_seq,
    np_rng,
    max_ngram_size=3,
    mean_ngram_size=None,
    whole_word_masking=True,
    favor_long_ngrams=False,
    permutation=False,
    geometric_dist=False,
    masking_style="bert",
    tokenizer_type="wordpiece",
):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""

    if not geometric_dist and mean_ngram_size is not None:
        raise ValueError(f"Mean ngram size is only supported for geometric distribution.")

    cand_indexes = []
    # Note(mingdachen): We create a list for recording if the piece is
    # the starting piece of current token, where 1 means true, so that
    # on-the-fly whole word masking is possible.
    token_boundary = [0] * len(tokens)

    for (i, token) in enumerate(tokens):
        if token == cls_id or token == sep_id:
            token_boundary[i] = 1
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if whole_word_masking and len(cand_indexes) >= 1 and not is_start_piece(vocab_id_to_token_dict[token]):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
            if is_start_piece(vocab_id_to_token_dict[token]):
                token_boundary[i] = 1

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []

    if masked_lm_prob == 0:
        return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    ngrams = np.arange(1, max_ngram_size + 1, dtype=np.int64)
    if not geometric_dist:
        # Note(mingdachen):
        # By default, we set the probilities to favor shorter ngram sequences.
        pvals = 1.0 / np.arange(1, max_ngram_size + 1)
        pvals /= pvals.sum(keepdims=True)
        if favor_long_ngrams:
            pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx : idx + n])
        ngram_indexes.append(ngram_index)

    np_rng.shuffle(ngram_indexes)

    (masked_lms, masked_spans) = ([], [])
    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        if not geometric_dist:
            n = np_rng.choice(
                ngrams[: len(cand_index_set)],
                p=pvals[: len(cand_index_set)] / pvals[: len(cand_index_set)].sum(keepdims=True),
            )
        else:
            # Sampling "n" from the geometric distribution and clipping it to
            # the max_ngrams. Using p=0.2 default from the SpanBERT paper
            # https://arxiv.org/pdf/1907.10529.pdf (Sec 3.1)

            # The expectation of a geometric distribution is E[X] = 1 / p
            p = 1 / mean_ngram_size if mean_ngram_size is not None else 0.2
            n = min(np_rng.geometric(p), max_ngram_size)

        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if masking_style == "bert":
                # 80% of the time, replace with [MASK]
                if np_rng.random() < 0.8:
                    masked_token = mask_id
                else:
                    # 10% of the time, keep original
                    if np_rng.random() < 0.5:
                        masked_token = tokens[index]
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_id_list[np_rng.randint(0, len(vocab_id_list))]
            elif masking_style == "t5":
                masked_token = mask_id
            else:
                raise ValueError("invalid value of masking style")

            output_tokens[index] = masked_token
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_spans.append(MaskedLmInstance(index=index_set, label=[tokens[index] for index in index_set]))

    assert len(masked_lms) <= num_to_predict
    np_rng.shuffle(ngram_indexes)

    select_indexes = set()
    if permutation:
        for cand_index_set in ngram_indexes:
            if len(select_indexes) >= num_to_predict:
                break
            if not cand_index_set:
                continue
            # Note(mingdachen):
            # Skip current piece if they are covered in lm masking or previous ngrams.
            for index_set in cand_index_set[0]:
                for index in index_set:
                    if index in covered_indexes or index in select_indexes:
                        continue

            n = np.random.choice(
                ngrams[: len(cand_index_set)],
                p=pvals[: len(cand_index_set)] / pvals[: len(cand_index_set)].sum(keepdims=True),
            )
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1

            while len(select_indexes) + len(index_set) > num_to_predict:
                if n == 0:
                    break
                index_set = sum(cand_index_set[n - 1], [])
                n -= 1
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(select_indexes) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or index in select_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                select_indexes.add(index)
        assert len(select_indexes) <= num_to_predict

        select_indexes = sorted(select_indexes)
        permute_indexes = list(select_indexes)
        np_rng.shuffle(permute_indexes)
        orig_token = list(output_tokens)

        for src_i, tgt_i in zip(select_indexes, permute_indexes):
            output_tokens[src_i] = orig_token[tgt_i]
            masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    # Sort the spans by the index of the first span
    masked_spans = sorted(masked_spans, key=lambda x: x.index[0])

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary, masked_spans)


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions, masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length, dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np


def build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    max_seq_length,
    masked_lm_prob,
    short_seq_prob,
    seed,
    skip_warmup,
    binary_head=False,
    max_seq_length_dec=None,
    dataset_type='standard_bert',
    tokenizer=None,
    max_ngram_size=3,
    mean_ngram_size=None,
    geometric_dist=True,
    permutation=False,
    whole_word_masking=True,
    favor_long_ngrams=False,
):

    if len(data_prefix) == 1:
        return _build_train_valid_test_datasets(
            cfg,
            trainer,
            data_prefix[0],
            data_impl,
            splits_string,
            train_valid_test_num_samples,
            max_seq_length,
            masked_lm_prob,
            short_seq_prob,
            seed,
            skip_warmup,
            binary_head,
            max_seq_length_dec,
            dataset_type=dataset_type,
            tokenizer=tokenizer,
            max_ngram_size=max_ngram_size,
            mean_ngram_size=mean_ngram_size,
            geometric_dist=geometric_dist,
            permutation=permutation,
            whole_word_masking=whole_word_masking,
            favor_long_ngrams=favor_long_ngrams,
        )
    # Blending dataset.
    # Parse the values.
    output = get_datasets_weights_and_num_samples(data_prefix, train_valid_test_num_samples)
    prefixes, weights, datasets_train_valid_test_num_samples = output

    # Build individual datasets.
    train_datasets = []
    valid_datasets = []
    test_datasets = []
    for i in range(len(prefixes)):
        train_ds, valid_ds, test_ds = _build_train_valid_test_datasets(
            cfg,
            trainer,
            prefixes[i],
            data_impl,
            splits_string,
            datasets_train_valid_test_num_samples[i],
            max_seq_length,
            masked_lm_prob,
            short_seq_prob,
            seed,
            skip_warmup,
            binary_head,
            max_seq_length_dec,
            dataset_type=dataset_type,
            tokenizer=tokenizer,
            max_ngram_size=max_ngram_size,
            mean_ngram_size=mean_ngram_size,
            geometric_dist=geometric_dist,
            permutation=permutation,
            whole_word_masking=whole_word_masking,
            favor_long_ngrams=favor_long_ngrams,
        )
        if train_ds:
            train_datasets.append(train_ds)
        if valid_ds:
            valid_datasets.append(valid_ds)
        if test_ds:
            test_datasets.append(test_ds)

        # Blend.
    blending_train_dataset = None
    if train_datasets:
        blending_train_dataset = BlendableDataset(train_datasets, weights)
    blending_valid_dataset = None
    if valid_datasets:
        blending_valid_dataset = BlendableDataset(valid_datasets, weights)
    blending_test_dataset = None
    if test_datasets:
        blending_test_dataset = BlendableDataset(test_datasets, weights)

    return (blending_train_dataset, blending_valid_dataset, blending_test_dataset)


def _build_train_valid_test_datasets(
    cfg,
    trainer,
    data_prefix,
    data_impl,
    splits_string,
    train_valid_test_num_samples,
    max_seq_length,
    masked_lm_prob,
    short_seq_prob,
    seed,
    skip_warmup,
    binary_head,
    max_seq_length_dec,
    dataset_type='standard_bert',
    tokenizer=None,
    max_ngram_size=3,
    mean_ngram_size=None,
    geometric_dist=True,
    permutation=False,
    whole_word_masking=True,
    favor_long_ngrams=False,
):

    if dataset_type not in DSET_TYPES:
        raise ValueError("Invalid dataset_type: ", dataset_type)

    # Indexed dataset.
    indexed_dataset = get_indexed_dataset_(data_prefix, data_impl, skip_warmup)

    if dataset_type == DSET_TYPE_ICT:
        title_dataset = get_indexed_dataset_(args.titles_data_path, data_impl, skip_warmup)

    # Get start and end indices of train/valid/train into doc-idx
    # Note that doc-idx is desinged to be num-docs + 1 so we can
    # easily iterate over it.
    total_num_of_documents = indexed_dataset.doc_idx.shape[0] - 1
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)

    # Print stats about the splits.
    logging.info(' > dataset split:')

    def print_split_stats(name, index):
        logging.info('    {}:'.format(name))
        logging.info(
            '     document indices in [{}, {}) total of {} '
            'documents'.format(splits[index], splits[index + 1], splits[index + 1] - splits[index])
        )
        start_index = indexed_dataset.doc_idx[splits[index]]
        end_index = indexed_dataset.doc_idx[splits[index + 1]]
        logging.info(
            '     sentence indices in [{}, {}) total of {} '
            'sentences'.format(start_index, end_index, end_index - start_index)
        )

    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        # from nemo.collections.nlp.data.language_modeling.megatron.ict_dataset import ICTDataset
        from nemo.collections.nlp.data.language_modeling.megatron.bert_dataset import BertDataset
        from nemo.collections.nlp.data.language_modeling.megatron.t5_dataset import T5Dataset

        dataset = None
        if splits[index + 1] > splits[index]:
            # Get the pointer to the original doc-idx so we can set it later.
            doc_idx_ptr = indexed_dataset.get_doc_idx()
            # Slice the doc-idx
            start_index = splits[index]
            # Add +1 so we can index into the dataset to get the upper bound.
            end_index = splits[index + 1] + 1
            # New doc_idx view.
            indexed_dataset.set_doc_idx(doc_idx_ptr[start_index:end_index])
            # Build the dataset accordingly.
            kwargs = dict(
                name=name,
                data_prefix=data_prefix,
                num_epochs=None,
                max_num_samples=int(train_valid_test_num_samples[index]),
                max_seq_length=max_seq_length,
                seed=seed,
            )

            if dataset_type == DSET_TYPE_ICT:
                raise NotImplementedError("ICT dataset is not implemented yet.")
                '''
                dataset = ICTDataset(
                    block_dataset=indexed_dataset,
                    title_dataset=title_dataset,
                    query_in_block_prob=args.query_in_block_prob,
                    use_one_sent_docs=args.use_one_sent_docs,
                    binary_head=binary_head,
                    **kwargs,
                )
                '''
            elif dataset_type == DSET_TYPE_T5:
                assert tokenizer is not None, "Tokenizer is required for T5 dataset"
                logging.info("Instatiating T5 Dataset ...")
                dataset = T5Dataset(
                    cfg=cfg,
                    trainer=trainer,
                    tokenizer=tokenizer,
                    indexed_dataset=indexed_dataset,
                    masked_lm_prob=masked_lm_prob,
                    max_seq_length_dec=max_seq_length_dec,
                    short_seq_prob=short_seq_prob,
                    max_ngram_size=max_ngram_size,
                    mean_ngram_size=mean_ngram_size,
                    geometric_dist=geometric_dist,
                    permutation=permutation,
                    whole_word_masking=whole_word_masking,
                    favor_long_ngrams=favor_long_ngrams,
                    **kwargs,
                )
            elif dataset_type == DSET_TYPE_BERT:
                logging.info("Instatiating BERT Dataset ...")
                dataset = BertDataset(
                    indexed_dataset=indexed_dataset,
                    masked_lm_prob=masked_lm_prob,
                    short_seq_prob=short_seq_prob,
                    binary_head=binary_head,
                    tokenizer=tokenizer,
                    **kwargs,
                )
            elif dataset_type == DSET_TYPE_T5_LM:
                documents = np.arange(start=splits[index], stop=splits[index + 1], step=1, dtype=np.int32)
                logging.info("Instatiating T5 Prefix-LM Dataset ...")
                dataset = T5LMAdaptedDataset(
                    cfg=cfg,
                    trainer=trainer,
                    tokenizer=tokenizer,
                    documents=documents,
                    indexed_dataset=indexed_dataset,
                    num_samples=int(train_valid_test_num_samples[index]),
                    **kwargs,
                )
            else:
                raise NotImplementedError("Dataset type not fully implemented.")

            # Set the original pointer so dataset remains the main dataset.
            indexed_dataset.set_doc_idx(doc_idx_ptr)
            # Checks.
            assert indexed_dataset.doc_idx[0] == 0
            assert indexed_dataset.doc_idx.shape[0] == (total_num_of_documents + 1)
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


def get_indexed_dataset_(data_prefix, data_impl, skip_warmup):

    logging.info(' > building dataset index ...')

    start_time = time.time()
    indexed_dataset = make_indexed_dataset(data_prefix, data_impl, skip_warmup)
    assert indexed_dataset.sizes.shape[0] == indexed_dataset.doc_idx[-1]
    logging.info(' > finished creating indexed dataset in {:4f} ' 'seconds'.format(time.time() - start_time))

    logging.info(' > indexed dataset stats:')
    logging.info('    number of documents: {}'.format(indexed_dataset.doc_idx.shape[0] - 1))
    logging.info('    number of sentences: {}'.format(indexed_dataset.sizes.shape[0]))

    return indexed_dataset


def get_samples_mapping(
    indexed_dataset, data_prefix, num_epochs, max_num_samples, max_seq_length, short_seq_prob, seed, name, binary_head
):
    """Get a list that maps a sample index to a starting sentence index, end sentence index, and length"""

    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples " "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() == 0 and not os.path.isfile(indexmap_filename):
        print(
            ' > WARNING: could not find index map file {}, building '
            'the indices on rank 0 ...'.format(indexmap_filename)
        )

        # Make sure the types match the helpers input types.
        assert indexed_dataset.doc_idx.dtype == np.int64
        assert indexed_dataset.sizes.dtype == np.int32

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        logging.info(' > building samples index mapping for {} ...'.format(name))
        # First compile and then import.
        try:
            if is_global_rank_zero():
                compile_helper()
            from nemo.collections.nlp.data.language_modeling.megatron import helpers
        except ImportError:
            raise ImportError(
                f'Could not compile megatron dataset C++ helper functions and therefore cannot import helpers python file.'
            )

        samples_mapping = helpers.build_mapping(
            indexed_dataset.doc_idx,
            indexed_dataset.sizes,
            num_epochs,
            max_num_samples,
            max_seq_length,
            short_seq_prob,
            seed,
            verbose,
            2 if binary_head else 1,
        )
        logging.info(' > done building samples index maping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        logging.info(' > saved the index mapping in {}'.format(indexmap_filename))
        # Make sure all the ranks have built the mapping
        logging.info(
            ' > elasped time to build and save samples mapping ' '(seconds): {:4f}'.format(time.time() - start_time)
        )

    torch.distributed.barrier()
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
    torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
    assert counts[0].item() == (
        torch.distributed.get_world_size()
        // torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group())
    )

    # Load indexed dataset.
    logging.info(' > loading indexed mapping from {}'.format(indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True, mmap_mode='r')
    logging.info('    loaded indexed file in {:3.3f} seconds'.format(time.time() - start_time))
    logging.info('    total number of samples: {}'.format(samples_mapping.shape[0]))

    return samples_mapping
