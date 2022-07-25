# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import json
from ast import literal_eval

from tqdm import tqdm


def load_json(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


def dump_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f)


def get_context_from_passages(passages, keep_only_relevant_passages):
    contexts = []
    if keep_only_relevant_passages:
        for passage in passages:
            if passage["is_selected"] == 1:
                contexts.append(passage["passage_text"])
    else:
        contexts = [passage["passage_text"] for passage in passages]

    return " ".join(contexts)


def format_answers_into_squad_format(answers):
    is_impossible = True if "No Answer Present." in answers else False
    if is_impossible:
        answers = []
    else:
        answers = [{"text": ans, "answer_start": -1} for ans in answers]

    return answers


def convert_msmarco_to_squad_format(msmarco_data, args):
    ids = list(msmarco_data["query"])
    squad_data = {"data": [{"title": "MSMARCO", "paragraphs": []}], "version": "v2.1"}
    for index, _id in enumerate(tqdm(ids)):

        context = get_context_from_passages(msmarco_data["passages"][_id], args.keep_only_relevant_passages)
        if not context:
            continue

        query = msmarco_data["query"][_id]

        # use well formed answers if present, else use the 'answers' field
        well_formed_answers = msmarco_data['wellFormedAnswers'][_id]
        well_formed_answers = (
            well_formed_answers if isinstance(well_formed_answers, list) else literal_eval(well_formed_answers)
        )
        answers = well_formed_answers if well_formed_answers else msmarco_data["answers"][_id]
        answers = format_answers_into_squad_format(answers)
        if args.exclude_negative_samples and (not answers):
            continue

        squad_data["data"][0]["paragraphs"].append(
            {
                "context": context,
                "qas": [
                    {"id": index, "question": query, "answers": answers, "is_impossible": False if answers else True,}
                ],
            }
        )

    return squad_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--msmarco_train_input_filepath", default=None, type=str, required=True)
    parser.add_argument("--msmarco_dev_input_filepath", default=None, type=str, required=True)
    parser.add_argument("--converted_train_save_path", default=None, type=str, required=True)
    parser.add_argument("--converted_dev_save_path", default=None, type=str, required=True)
    parser.add_argument(
        "--exclude_negative_samples",
        default=False,
        type=bool,
        help="whether to keep No Answer samples in the dataset",
        required=False,
    )
    parser.add_argument(
        "--keep_only_relevant_passages",
        default=False,
        type=bool,
        help="if True, will only use passages with is_selected=True for context",
        required=False,
    )
    args = parser.parse_args()

    print("converting MS-MARCO train dataset...")
    msmarco_train_data = load_json(args.msmarco_train_input_filepath)
    squad_train_data = convert_msmarco_to_squad_format(msmarco_train_data, args)
    dump_json(args.converted_train_save_path, squad_train_data)

    print("converting MS-MARCO dev dataset...")
    msmarco_dev_data = load_json(args.msmarco_dev_input_filepath)
    squad_dev_data = convert_msmarco_to_squad_format(msmarco_dev_data, args)
    dump_json(args.converted_dev_save_path, squad_dev_data)


if __name__ == "__main__":
    """
    Please agree to the Terms of Use at:
        https://microsoft.github.io/msmarco/ 
    Download data at:
        https://msmarco.blob.core.windows.net/msmarco/train_v2.1.json.gz
        https://msmarco.blob.core.windows.net/msmarco/dev_v2.1.json.gz

    Example usage:
        python convert_msmarco_to_squad_format.py \
            --msmarco_train_input_filepath=/path/to/msmarco_train_v2.1.json \
            --msmarco_dev_input_filepath=/path/to/msmarco_dev_v2.1.json \
            --converted_train_save_path=/path/to/msmarco_squad_format_train.json \
            --converted_dev_save_path=/path/to/msmarco_squad_format_dev.json \
            --exclude_negative_samples=False \
            --keep_only_relevant_passages=False
    """
    main()
