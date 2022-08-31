# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import argparse
import multiprocessing as mp
from pathlib import Path

from nemo.collections.nlp.data.token_classification.punctuation_capitalization_tarred_dataset import (
    DEFAULT_CAPIT_LABEL_VOCAB_FILE_NAME,
    DEFAULT_PUNCT_LABEL_VOCAB_FILE_NAME,
    METADATA_CAPIT_LABEL_VOCAB_KEY,
    METADATA_PUNCT_LABEL_VOCAB_KEY,
    build_label_ids_from_list_of_labels,
    check_labels_for_being_unique_before_building_label_ids,
    check_tar_file_prefix,
    create_tarred_dataset,
)


"""
A tarred dataset allows to train on large amounts without storing it all into memory simultaneously. In case of
punctuation and capitalization model, tarred dataset is a directory which contains metadata file, tar files with
batches, punct_label_vocab.csv and capit_label_vocab.csv files.

A metadata file is a JSON file with 4 fields: 'num_batches', 'tar_files', 'punct_label_vocab_file',
'capit_label_vocab_file'. 'num_batches' (int) is a total number of batches in tarred dataset. 'tar_files' is a list of
paths to tar files relative to directory containing the metadata file. 'punct_label_vocab_file' and
'capit_label_vocab_file' are paths to .csv files containing all unique punctuation and capitalization labels. Each
label in these files is written in a separate line. The first labels in both files are equal and serve for padding and
as neutral labels.

Every tar file contains objects written using `webdataset.TarWriter`. Each object is a dictionary with two items:
'__key__' and 'batch.pyd'. '__key__' is a name of a batch and 'batch.pyd' is a pickled dictionary which contains
'input_ids', 'subtokens_mask', 'punct_labels', 'capit_labels'. 'input_ids' is an array containing ids of source tokens,
'subtokens_mask' is a boolean array showing first tokens in words, 'punct_labels' and 'capit_labels' are arrays with
ids of labels. Metadata file should be passed to constructor of
`nemo.collections.nlp.data.token_classification.PunctuationCapitalizationTarredDataset` and the instance of 
the class will handle iteration and constructing masks and token types for BERT model.

Example of usage:

python create_punctuation_capitalization_tarred_dataset.py \
  --text <PATH/TO/TEXT/FILE> \
  --labels <PATH/TO/LABELS/FILE> \
  --output_dir <PATH/TO/OUTPUT/DIR> \
  --lines_per_dataset_fragment 10000 \
  --tokens_in_batch 8000 \
  --num_batches_per_tarfile 5 \
  --tokenizer_name char \
  --vocab_file <PATH_TO_CHAR_TOKENIZER_VOCABULARY>
"""


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=f"A tarred dataset allows to train on large amounts without storing it all into memory "
        f"simultaneously. In case of punctuation and capitalization model, tarred dataset is a directory which "
        f"contains metadata file, tar files with batches, {DEFAULT_PUNCT_LABEL_VOCAB_FILE_NAME} and "
        f"{DEFAULT_CAPIT_LABEL_VOCAB_FILE_NAME} files. A metadata file is a JSON file with 4 fields: 'num_batches', "
        f"'tar_files', '{METADATA_PUNCT_LABEL_VOCAB_KEY}', '{METADATA_CAPIT_LABEL_VOCAB_KEY}'. 'num_batches' (int) is "
        f"a total number of batches in tarred dataset. 'tar_files' is a list of paths to tar files relative "
        f"to directory containing the metadata file. '{METADATA_PUNCT_LABEL_VOCAB_KEY}' and "
        f"'{METADATA_CAPIT_LABEL_VOCAB_KEY}' are paths to .csv files containing all unique punctuation and "
        f"capitalization labels. Each label in these files is written in a separate line. The first labels in both "
        f"files are equal and serve for padding and as neutral labels. Every tar file contains objects written "
        f"using `webdataset.TarWriter`. Each object is a dictionary with two items: '__key__' and 'batch.pyd'. "
        f"'__key__' is a name of a batch and 'batch.pyd' is a pickled dictionary which contains 'input_ids', "
        f"'subtokens_mask', 'punct_labels', 'capit_labels'. 'input_ids' is an array containing ids of source tokens, "
        f"'subtokens_mask' is a boolean array showing first tokens in words, 'punct_labels' and 'capit_labels' are "
        f"arrays with ids of labels. Metadata file should be passed to constructor of "
        "`nemo.collections.nlp.data.token_classification.PunctuationCapitalizationTarredDataset` and the instance of "
        "the class will handle iteration and constructing masks and token types for BERT model.",
    )
    parser.add_argument(
        "--text",
        "-t",
        help="Path to source lowercased text without punctuation. Number of lines in `--text` file has to be equal "
        "to number of lines in `--labels` file.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--audio_file",
        type=Path,
        required=False,
        help="Path to source file which contains paths to audio one path per line. "
        "Number of lines in `--audio_file` has to be equal to number of lines in `--labels` file",
    )
    parser.add_argument(
        "--use_audio",
        required=False,
        action="store_true",
        help="If set to `True` script creates lexical audio dataset which can be used with `PunctuationCapitalizationLexicalAudioModel`.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        required=False,
        help="Target sample rate of audios. Can be used for downsampling or upsampling.",
    )
    parser.add_argument(
        "--labels",
        "-L",
        type=Path,
        required=True,
        help="Path to file with labels in the format described here "
        "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/punctuation_and_capitalization.html#"
        "nemo-data-format . Number of lines in `--labels` file has to be equal to the number of lines in `--text` "
        "file.",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        required=True,
        help="Path to directory where .tar files, metadata file, label id files are stored.",
    )
    parser.add_argument(
        "--max_seq_length",
        "-s",
        type=int,
        default=512,
        help="Maximum number of subtokens in an input sequence. A source sequence which contain too many subtokens are "
        "clipped to `--max_seq_length - 2` subtokens and then [CLS] token is prepended to the clipped sequence and "
        "[SEP] token is appended to the clipped sequence. The clipping is performed via removal of subtokens in the "
        "end of a source sequence.",
    )
    parser.add_argument(
        "--tokens_in_batch",
        "-b",
        type=int,
        default=15000,
        help="Maximum number of tokens in a batch including [CLS], [SEP], [UNK], and [PAD] tokens. Before packing into "
        "batches source sequences are sorted by number of tokens in order to reduce number of pad tokens. So the "
        "number of sequences in a batch may be different.",
    )
    parser.add_argument(
        "--lines_per_dataset_fragment",
        type=int,
        default=10 ** 6,
        help="A number of lines processed by one worker during creation of tarred dataset. A worker tokenizes "
        "`--lines_per_dataset_fragment` lines and keeps in RAM tokenized text labels before packing them into "
        "batches. Reducing `--lines_per_dataset_fragment` leads to reducing of the amount of memory required by this "
        "script.",
    )
    parser.add_argument(
        "--num_batches_per_tarfile",
        type=int,
        default=1000,
        help="A number of batches saved in a tar file. If you increase `--num_batches_per_tarfile`, then there will "
        "be less tar files in the dataset. There cannot be less then `--num_batches_per_tarfile` batches in a tar "
        "file, and all excess batches are removed. Maximum number of discarded batches is "
        "`--num_batches_per_tarfile - 1`.",
    )
    parser.add_argument(
        "--tokenizer_name",
        "-T",
        default="bert-base-uncased",
        help="Name of the tokenizer used for tokenization of source sequences. Possible options are 'sentencepiece', "
        "'word', 'char', HuggingFace tokenizers. For more options see function "
        "`nemo.collections.nlp.modules.common.get_tokenizer`. The tokenizer has to have properties `cls_id`, "
        "`pad_id`, `sep_id`, `unk_id`.",
    )
    parser.add_argument(
        "--tokenizer_model", "-m", type=Path, help="Path to tokenizer model required for 'sentencepiece' tokenizer."
    )
    parser.add_argument(
        "--vocab_file",
        "-v",
        type=Path,
        help="Path to vocabulary file which can be used in 'word', 'char', and HuggingFace tokenizers.",
    )
    parser.add_argument(
        "--merges_file", "-M", type=Path, help="Path to merges file which can be used in HuggingFace tokenizers."
    )
    parser.add_argument(
        "--special_token_names",
        "-n",
        nargs="+",
        help="Names of special tokens which may be passed to constructors of 'char', 'word', 'sentencepiece', and "
        "HuggingFace tokenizers.",
    )
    parser.add_argument(
        "--special_token_values",
        "-V",
        nargs="+",
        help="Values of special tokens which may be passed to constructors of 'char', 'word', 'sentencepiece', and "
        "HuggingFace tokenizers.",
    )
    parser.add_argument(
        "--use_fast_tokenizer", "-f", action="store_true", help="Whether to use fast HuggingFace tokenizer."
    )
    parser.add_argument(
        "--pad_label",
        "-P",
        default='O',
        help="Pad label both for punctuation and capitalization. This label is also is used for marking words which "
        "do not need punctuation and capitalization. It is also a neutral label used for marking words which do "
        "not require punctuation and capitalization.",
    )
    punct = parser.add_mutually_exclusive_group(required=False)
    punct.add_argument(
        "--punct_labels",
        "-p",
        nargs="+",
        help="All punctuation labels EXCEPT PAD LABEL. Punctuation labels are strings separated by spaces. "
        "Alternatively you can use parameter `--punct_label_vocab_file`. If none of parameters `--punct_labels` "
        "and `--punct_label_vocab_file` are provided, then punctuation label ids will be inferred from `--labels` "
        "file.",
    )
    punct.add_argument(
        "--punct_label_vocab_file",
        type=Path,
        help="A path to file with punctuation labels. These labels include pad label. Pad label has to be the first "
        "label in the file. Each label is written on separate line. Alternatively you can use `--punct_labels` "
        "parameter. If none of parameters `--punct_labels` and `--punct_label_vocab_file` are provided, then "
        "punctuation label ids will be inferred from `--labels` file.",
    )
    capit = parser.add_mutually_exclusive_group(required=False)
    capit.add_argument(
        "--capit_labels",
        "-c",
        nargs="+",
        help="All capitalization labels EXCEPT PAD LABEL. Capitalization labels are strings separated by spaces. "
        "Alternatively you can use parameter `--capit_label_vocab_file`. If none of parameters `--capit_labels` "
        "and `--capit_label_vocab_file` are provided, then capitalization label ids will be inferred from `--labels` "
        "file.",
    )
    capit.add_argument(
        "--capit_label_vocab_file",
        type=Path,
        help="A path to file with capitalization labels. These labels include pad label. Pad label has to be the "
        "first label in the file. Each label is written on separate line. Alternatively you can use `--capit_labels` "
        "parameter. If none of parameters `--capit_labels` and `--capit_label_vocab_file` are provided, then "
        "capitalization label ids will be inferred from `--labels` file.",
    )
    parser.add_argument(
        "--tar_file_prefix",
        "-x",
        default="punctuation_capitalization",
        help="A string from which tar file names start. It can contain only characters 'A-Z', 'a-z', '0-9', '_', '-', "
        "'.'.",
    )
    parser.add_argument(
        "--n_jobs",
        "-j",
        type=int,
        default=mp.cpu_count(),
        help="Number of workers for creating tarred dataset. By default it is equal to the number of CPU cores.",
    )
    args = parser.parse_args()
    for name in [
        "text",
        "labels",
        "output_dir",
        "tokenizer_model",
        "vocab_file",
        "merges_file",
        "punct_label_vocab_file",
        "capit_label_vocab_file",
    ]:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser())
    if args.special_token_names is not None or args.special_token_values is not None:
        if args.special_token_names is None:
            parser.error(
                "If you provide parameter `--special_token_values` you have to provide parameter "
                "`--special_token_names`."
            )
        if args.special_token_values is None:
            parser.error(
                "If you provide parameter `--special_token_names` you have to provide parameter "
                "`--special_token_values`."
            )
        if len(args.special_token_names) != len(args.special_token_values):
            parser.error(
                f"Parameters `--special_token_names` and `--special_token_values` have to have equal number of values "
                f"whereas parameter `--special_token_names` has {len(args.special_token_names)} values and parameter "
                f"`--special_token_values` has {len(args.special_token_values)} values."
            )
        if len(set(args.special_token_names)) != len(args.special_token_names):
            for i in range(len(args.special_token_names) - 1):
                if args.special_token_names[i] in args.special_token_names[i + 1 :]:
                    parser.error(
                        f"Values of parameter `--special_token_names` has to be unique. Found duplicate value "
                        f"'{args.special_token_names[i]}'."
                    )
    if args.punct_labels is not None:
        check_labels_for_being_unique_before_building_label_ids(
            args.pad_label, args.punct_labels, '--pad_label', '--punct_labels', parser.error
        )
        check_labels_for_being_unique_before_building_label_ids(
            args.pad_label, args.capit_labels, '--pad_label', '--capit_labels', parser.error
        )
    check_tar_file_prefix(args.tar_file_prefix, parser.error, '--tar_file_prefix')
    return args


def main() -> None:
    args = get_args()
    if args.special_token_names is None:
        special_tokens = None
    else:
        special_tokens = dict(zip(args.special_token_names, args.special_token_values))

    if args.punct_labels is not None:
        punct_label_ids = build_label_ids_from_list_of_labels(args.pad_label, args.punct_labels)
    else:
        punct_label_ids = None

    if args.capit_labels is not None:
        capit_label_ids = build_label_ids_from_list_of_labels(args.pad_label, args.capit_labels)
    else:
        capit_label_ids = None

    create_tarred_dataset(
        args.text,
        args.labels,
        args.output_dir,
        args.max_seq_length,
        args.tokens_in_batch,
        args.lines_per_dataset_fragment,
        args.num_batches_per_tarfile,
        args.tokenizer_name,
        tokenizer_model=args.tokenizer_model,
        vocab_file=args.vocab_file,
        merges_file=args.merges_file,
        special_tokens=special_tokens,
        use_fast_tokenizer=args.use_fast_tokenizer,
        pad_label=args.pad_label,
        punct_label_ids=punct_label_ids,
        capit_label_ids=capit_label_ids,
        punct_label_vocab_file=args.punct_label_vocab_file,
        capit_label_vocab_file=args.capit_label_vocab_file,
        tar_file_prefix=args.tar_file_prefix,
        n_jobs=args.n_jobs,
        audio_file=args.audio_file,
        sample_rate=args.sample_rate,
        use_audio=args.use_audio,
    )


if __name__ == "__main__":
    main()
