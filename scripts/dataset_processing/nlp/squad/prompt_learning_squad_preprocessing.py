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
import random

from tqdm import tqdm


"""
Dataset preprocessing script for the SQuAD dataset: https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
Converts the dataset into a jsonl format that can be used for p-tuning/prompt tuning in NeMo. 

Inputs:
    data-dir: (str) The directory where the squad dataset was downloaded, files will be saved here
    file-name: (str) Name of the input file you want to process
    save-name-base: (str) The base name for each of the train, val, and test files. If save-name-base were 'squad' for
                    example, the files would be saved as squad_train.jsonl, squad_val.jsonl, and squad_test.jsonl
    make-ground-truth: (bool) If true, test files will include answers, if false, test files will not include answers. 
    include-topic-name: Whether to include the topic name for the paragraph in the data json. See the squad explaination
                        below for more context on what is ment by 'topic name'.
    random-seed: (int) Random seed for repeatable shuffling of train/val/test splits. 
    train-percent: (float) Precentage of data that should be used for the train split. The val and test splits will be made
                    by splitting the remaining data evenly. 

Saves train, val, and test files for the SQuAD dataset.

The SQuAD dataset consists of various topics like Beyoncé, IPod, and Symbiosis. Each topic has several paragraphs 
associated with it, and each paragraph has several questions and answers related to it. When we separated the 
train/validation/test splits, we separated them on the topic level. For example, if the training set contains paragraphs 
and questions about the topic Beyoncé, neither the validation nor test sets will contain any questions on this topic. 
All questions about a certain topic are isolated to one split of the data.

An example of the processed output written to file:
    
    {
        "taskname": "squad", 
        "context": "Red is the traditional color of warning and danger. In the Middle Ages, a red flag announced that the defenders of a town or castle would fight to defend it, and a red flag hoisted by a warship meant they would show no mercy to their enemy. In Britain, in the early days of motoring, motor cars had to follow a man with a red flag who would warn horse-drawn vehicles, before the Locomotives on Highways Act 1896 abolished this law. In automobile races, the red flag is raised if there is danger to the drivers. In international football, a player who has made a serious violation of the rules is shown a red penalty card and ejected from the game.", 
        "question": "What did a red flag signal in the Middle Ages?", 
        "answer": " defenders of a town or castle would fight to defend it"
    },


"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/SQuAD")
    parser.add_argument("--file-name", type=str, default="train-v2.0.json")
    parser.add_argument("--save-name-base", type=str, default="squad")
    parser.add_argument("--make-ground-truth", action='store_true')
    parser.add_argument("--include-topic-name", action='store_true')
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--train-percent", type=float, default=0.8)
    args = parser.parse_args()

    data_dict = json.load(open(f"{args.data_dir}/{args.file_name}"))
    data = data_dict['data']
    save_name_base = f"{args.data_dir}/{args.save_name_base}"

    process_data(
        data, save_name_base, args.train_percent, args.random_seed, args.include_topic_name, args.make_ground_truth
    )


def process_data(data, save_name_base, train_percent, random_seed, include_topic, make_ground_truth=False):
    data = extract_questions(data, include_topic)

    # Data examples are currently grouped by topic, shuffle topic groups
    random.seed(random_seed)
    random.shuffle(data)

    # Decide train/val/test splits on the topic level
    data_total = len(data)
    train_total = int(data_total * train_percent)
    val_total = (data_total - train_total) // 2

    train_set = data[0:train_total]
    val_set = data[train_total : train_total + val_total]
    test_set = data[train_total + val_total :]

    # Flatten data for each split now that topics have been confined to one split
    train_set = [question for topic in train_set for question in topic]
    val_set = [question for topic in val_set for question in topic]
    test_set = [question for topic in test_set for question in topic]

    # Shuffle train set questions
    random.shuffle(train_set)

    gen_file(train_set, save_name_base, 'train')
    gen_file(val_set, save_name_base, 'val')
    gen_file(test_set, save_name_base, 'test', make_ground_truth)


def extract_questions(data, include_topic):
    processed_data = []

    # Iterate over topics, want to keep them seprate in train/val/test splits
    for question_group in data:
        processed_topic_data = []
        topic = question_group['title']
        questions = question_group['paragraphs']

        # Iterate over paragraphs related to topics
        for qa_group in questions:
            context = qa_group['context']
            qas = qa_group['qas']

            # Iterate over questions about paragraph
            for qa in qas:
                question = qa['question']

                try:
                    answer = qa['answers'][0]['text']
                except IndexError:
                    continue

                example_json = {"taskname": "squad", "context": context, "question": question, "answer": " " + answer}

                if include_topic:
                    example_json["topic"] = topic

                processed_topic_data.append(example_json)
        processed_data.append(processed_topic_data)

    return processed_data


def gen_file(data, save_name_base, split_type, make_ground_truth=False):
    save_path = f"{save_name_base}_{split_type}.jsonl"
    print(f"Saving {split_type} split to {save_path}")

    with open(save_path, 'w') as save_file:
        for example_json in tqdm(data):

            # Dont want labels in the test set
            if split_type == "test" and not make_ground_truth:
                del example_json["answer"]

            save_file.write(json.dumps(example_json) + '\n')


if __name__ == "__main__":
    main()
