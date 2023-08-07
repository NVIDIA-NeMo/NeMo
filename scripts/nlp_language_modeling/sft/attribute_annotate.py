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

"""script to annotate the the datasets with using trained attribute prediciton model.
First, we need to launch the NeMo Megatron inference server
Example:
```bash
 python examples/nlp/language_modeling/megatron_gpt_eval.py \
        gpt_model_file=/models/TRAINED_ATTR_PREDICTION_MODEL.nemo \
        pipeline_model_parallel_split_rank=0 \
        server=True \
        tensor_model_parallel_size=TP_SIZE \
        pipeline_model_parallel_size=PP_SIZE \
        trainer.precision=bf16 \
        trainer.devices=TP_SIZE*PP_SIZE \
        trainer.num_nodes=1 \
        web_server=False \
        port=1424
```

Then, we can run this script to annotate the dataset.
Example usage:

python scripts/nlp_language_modeling/sft/attribute_annotate.py  --batch_size=1 --host=localhost --input_file_name=input.jsonl --output_file_name=output.jsonl --port_num=1424
"""

import json
import os

import fire
import tqdm
from langchain.prompts.few_shot import PromptTemplate

from nemo.collections.nlp.modules.common.megatron.retrieval_services.util import text_generation

langs = [
    'ar',
    'bg',
    'bn',
    'ca',
    'cs',
    'da',
    'de',
    'el',
    'en',
    'eo',
    'es',
    'eu',
    'fa',
    'fi',
    'fr',
    'gl',
    'he',
    'hu',
    'id',
    'it',
    'ja',
    'ko',
    'nb',
    'nl',
    'pl',
    'pt',
    'ro',
    'ru',
    'sk',
    'sv',
    'th',
    'tr',
    'uk',
    'vi',
    'zh',
]

SFT_PREFIX = """<extra_id_0>System
{system_message}"""

ONE_TRUN_WITH_VAL = """<extra_id_1>{user_name}
{user_message}
<extra_id_2>{label}
"""

ONE_TRUN_WITHOUT_VAL = """<extra_id_1>{user_name}
{user_message}
"""
SYSTEM = PromptTemplate(input_variables=["system_message"], template=SFT_PREFIX)
EXAMPLE_PROMPT_WITH_VAL = PromptTemplate(
    input_variables=["user_name", "user_message", "label"], template=ONE_TRUN_WITH_VAL
)
EXAMPLE_PROMPT_WITHOUT_VAL = PromptTemplate(
    input_variables=["user_name", "user_message"], template=ONE_TRUN_WITHOUT_VAL
)

selected_keys = [
    'quality',
    'toxicity',
    'humor',
    'creativity',
    'violence',
    'helpfulness',
    'not_appropriate',
    'hate_speech',
    'sexual_content',
    'fails_task',
    'political_content',
    'moral_judgement',
    'lang',
]


def calculate_key(obj):
    return ":".join([item['value'] for item in obj['conversations']])


def load_data(path):
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            yield json.loads(line)


def get_prompt(data_obj, turn, current_label="", label_id=0):
    if len(data_obj['conversations']) < turn + 1:
        return None

    examples = []
    for i in range(0, turn):
        d = data_obj['conversations'][i]
        if 'label' in d:
            examples.append(
                EXAMPLE_PROMPT_WITH_VAL.format(
                    **{'user_name': d['from'], 'user_message': d['value'], 'label': d['label']}
                )
            )
        else:
            examples.append(EXAMPLE_PROMPT_WITHOUT_VAL.format(**{'user_name': d['from'], 'user_message': d['value']}))

    example_text = "".join(examples)
    d = data_obj['conversations'][turn]
    predict_message = EXAMPLE_PROMPT_WITHOUT_VAL.format(**{'user_name': d['from'], 'user_message': d['value']})

    if label_id != 0:
        current_label = current_label + ',' + selected_keys[label_id] + ':'
    else:
        current_label = '<extra_id_2>' + selected_keys[label_id] + ':'
    return SYSTEM.format(**{'system_message': data_obj['system']}) + example_text + predict_message + current_label


def create_gen_function(host='localhost', port=5555):
    def request(prompts, greedy, add_BOS, token_to_gen, min_tokens, temp, top_p, top_k, repetition, end_strings):
        data = {
            "sentences": prompts,
            "tokens_to_generate": int(token_to_gen),
            "temperature": temp,
            "add_BOS": add_BOS,
            "top_k": top_k,
            "top_p": top_p,
            "greedy": greedy,
            "all_probs": False,
            "repetition_penalty": repetition,
            "min_tokens_to_generate": int(min_tokens),
            "end_strings": end_strings,
        }
        response = text_generation(data, ip=host, port=port)
        sentences = response['sentences']
        return sentences

    return request


class Worker(object):
    def __init__(self, host='localhost', port=5555, progress_bar=None, output_file=None, process_lang=False):
        self.req = create_gen_function(host=host, port=port)
        self.fout = open(output_file, "a", encoding='utf-8')
        self.progress_bar = progress_bar
        self.process_lang = process_lang

    def process_result(self, batch):
        while True:
            try:
                items = [i['item'] for i in batch]
                turns = [i['turn'] for i in batch]
                prompts = [i['prompt'] for i in batch]

                for label_id in range(1, len(selected_keys)):
                    results = self.req(
                        prompts,
                        greedy=True,
                        add_BOS=False,
                        token_to_gen=1,
                        min_tokens=1,
                        temp=0.1,
                        top_p=1.0,
                        top_k=1,
                        repetition=1.0,
                        end_strings=["<extra_id_1>", "<|endoftext|>"],
                    )
                    # get current value from result
                    current_values = []
                    nums = []
                    for result in results:
                        # promblem result[-1] is '\n'
                        current_val = result.split('quality')[-1]
                        current_val = 'quality' + current_val
                        # remove whatever after new line
                        current_val = current_val.split('\n')[0].strip()
                        # remove everything that is >= selected_keys[label_id]
                        splits = current_val.split(',')
                        filtered = []
                        for item in splits:
                            filtered.append(item)
                            if item.split(':')[0] == selected_keys[label_id - 1]:
                                nums.append(item.split(':')[1])
                                break
                        current_val = '<extra_id_2>' + ','.join(filtered)
                        current_values.append(current_val)

                    filtered_items = []
                    filtered_turns = []
                    filtered_prompts = []
                    filtered_current_values = []

                    for result, item, turn, num, current_value in zip(results, items, turns, nums, current_values):
                        try:
                            value = int(num)
                        except Exception as e:
                            print(f'error {e} when convert {num} to int')
                            continue
                        filtered_current_values.append(current_value)
                        filtered_items.append(item)
                        filtered_turns.append(turn)
                        if label_id < len(selected_keys):
                            prompt = get_prompt(item, turn, current_label=current_value, label_id=label_id)
                            filtered_prompts.append(prompt)
                    items = filtered_items
                    turns = filtered_turns
                    prompts = filtered_prompts
                    current_values = filtered_current_values

                if self.process_lang:
                    results = self.req(
                        prompts,
                        greedy=True,
                        add_BOS=False,
                        token_to_gen=1,
                        min_tokens=1,
                        temp=0.1,
                        top_p=1.0,
                        top_k=1,
                        repetition=1.0,
                        end_strings=["<extra_id_1>", "<|endoftext|>"],
                    )
                    # get current value from result
                    current_values = []
                    for result in results:
                        # promblem result[-1] is '\n'
                        if result.endswith('\n'):
                            result = result[:-1] + '@'
                        current_values.append(result.split('\n')[-1])

                    nums = []
                    for result in results:
                        # promblem result[-1] is '\n'
                        current_val = result.split('quality')[-1]
                        current_val = 'quality' + current_val
                        # remove whatever after new line
                        current_val = current_val.split('\n')[0].strip()
                        # remove everything that is >= selected_keys[label_id]
                        splits = current_val.split(',')
                        filtered = []
                        for item in splits:
                            filtered.append(item)
                            if item.split(':')[0] == selected_keys[label_id]:
                                nums.append(item.split(':')[1])
                                break
                        current_val = '<extra_id_2>' + ','.join(filtered)
                        current_values.append(current_val)

                    filtered_items = []
                    filtered_turns = []
                    filtered_prompts = []
                    filtered_current_values = []

                    for result, item, turn, num, current_value in zip(results, items, turns, nums, current_values):
                        if num not in langs:
                            print(f'error {num} not in langs')
                            continue
                        filtered_current_values.append(current_value)
                        filtered_items.append(item)
                        filtered_turns.append(turn)
                    items = filtered_items
                    turns = filtered_turns
                    current_values = filtered_current_values

                batch = []
                for item, turn, current_value in zip(items, turns, current_values):
                    response_text = current_value[12:]
                    if 'label' in item['conversations'][turn]:
                        item['conversations'][turn]['gt_label'] = item['conversations'][turn]['label']
                    item['conversations'][turn]['label'] = response_text
                    prompt = get_prompt(item, turn + 1, current_label='', label_id=0)
                    if prompt is not None:
                        batch.append({'prompt': prompt, 'item': item, 'turn': turn + 1})
                    else:
                        self.progress_bar.update(1)
                        self.fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                        self.fout.flush()
                        if self.progress_bar.n >= self.progress_bar.total:
                            break
                if len(batch) == 0:
                    break
            except Exception as e:
                print(f'error {e} when processing {batch}')
                # ignore the error and continue
                self.progress_bar.update(1)
                if self.progress_bar.n >= self.progress_bar.total:
                    break


def main(
    batch_size=1,
    host='localhost',
    input_file_name='input.jsonl',
    output_file_name='output.jsonl',
    port_num=1424,
    process_lang=True,
):
    input_data = load_data(f'{input_file_name}')
    output_path = f'{output_file_name}'
    existing_requests = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line = json.loads(line)
                existing_requests.add(calculate_key(line))
        print(f"Loaded {len(existing_requests)} existing requests")

    filter_data = [d for d in input_data if calculate_key(d) not in existing_requests]

    progress_bar = tqdm.tqdm(total=len(filter_data))

    worker = Worker(
        host=host, port=port_num, progress_bar=progress_bar, output_file=output_path, process_lang=process_lang
    )
    for batch_idx in range(0, len(filter_data), batch_size):
        batch = [line for line in filter_data[batch_idx : batch_idx + batch_size]]
        turns = [
            0 if 'mask' not in d['conversations'][0]['from'] or d['conversations'][0]['from'] == d['mask'] else 1
            for d in batch
        ]
        task = [{'prompt': get_prompt(d, turn, "", 0), 'item': d, 'turn': turn} for d, turn in zip(batch, turns)]
        worker.process_result(task)
    worker.fout.close()


if __name__ == '__main__':
    fire.Fire(main)
