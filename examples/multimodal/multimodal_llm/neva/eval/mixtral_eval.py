"""Script to query Mixtral-8x7B as a judge via NGC API for evaluation"""
import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import requests
import shortuuid
from tqdm import tqdm

"""Usage: (for image inference)
API_TOKEN=xxx python3 --model-name-list name-of-model-1 name-of-model-2
                      --media-type image
                      --question-file path/to/prompts.jsonl
                      --responses-list path/to/responses-1.jsonl path/to/responses-2.jsonl
                      --answers-dir path/to/desired/preprocessed/answers/dir
                      --context-file path/to/context.jsonl
                      --rule-file path/to/rule.json
                      --output path/to/desired/output.json
"""

invoke_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/8f4118ba-60a8-4e6b-8574-e38a4067a4a3"

API_TOKEN = os.getenv("API_TOKEN", "")  # ADD NGC API TOKEN HERE

headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "accept": "text/event-stream",
    "content-type": "application/json",
}


def summarize(review_files):
    for review_file in sorted(review_files):
        scores = defaultdict(list)
        with open(review_file) as f:
            for review_str in f:
                review = json.loads(review_str)
                if 'category' in review:
                    scores[review['category']].append(review['tuple'])
                    scores['all'].append(review['tuple'])
                else:
                    if 'tuple' in review:
                        scores['all'].append(review['tuple'])
                    else:
                        scores['all'].append(review['score'])
        for k, v in sorted(scores.items()):
            stats = np.asarray(v).mean(0).tolist()
            stats = [round(x, 3) for x in stats]
            # print(k, round(stats[1] / stats[0] * 100, 1), round(stats[0] * 10, 1), round(stats[1] * 10, 1))
            print(k, round(stats[0] * 10, 1), round(stats[1] * 10, 1))
        print('=================================')


def get_eval(content: str, max_tokens: int):
    payload = {
        "messages": [
            {
                'role': 'system',
                'content': 'You are a helpful and precise assistant for checking the quality of the answer.',
            },
            {'role': 'user', 'content': content,},
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": max_tokens,
        "seed": 42,
        "stream": True,
    }
    response = requests.post(invoke_url, headers=headers, json=payload, stream=True)
    output = ""
    for line in response.iter_lines():
        if line:
            try:
                res = json.loads(line.decode("utf-8").split("data: ")[1])
            except:
                continue
            output += res['choices'][0]['delta']['content']
    print(output)
    return output


def parse_score(review):
    try:
        score_pair = review.split('\n')[0]
        score_pair = score_pair.replace(',', ' ')
        sp = score_pair.split()
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print('error', review)
            return [-1, -1]
    except Exception as e:
        print(e.messsage)
        print('error', review)
        return [-1, -1]


def generate_prompt(args, answer_list):
    f_q = open(os.path.expanduser(args.question_file))
    f_ans1 = open(os.path.expanduser(answer_list[0]))
    f_ans2 = open(os.path.expanduser(answer_list[1]))
    rule_dict = json.load(open(os.path.expanduser(args.rule_file), 'r'))

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [json.loads(line) for line in open(os.path.expanduser(args.output))]
    else:
        cur_reviews = []

    review_file = open(f'{args.output}', 'a')

    context_list = [json.loads(line) for line in open(os.path.expanduser(args.context_file))]
    image_to_context = {context['image']: context for context in context_list}

    idx = 0
    for ques_js, ans1_js, ans2_js in zip(f_q, f_ans1, f_ans2):
        ques = json.loads(ques_js)
        ans1 = json.loads(ans1_js)
        ans2 = json.loads(ans2_js)

        inst = image_to_context[ques['image']]

        if isinstance(inst['caption'], list):
            cap_str = '\n'.join(inst['caption'])
        else:
            cap_str = inst['caption']

        category = 'llava_bench_' + json.loads(ques_js)['category']
        if category in rule_dict:
            rule = rule_dict[category]
        else:
            assert False, f"Visual QA category not found in rule file: {category}."
        prompt = rule['prompt']
        role = rule['role']
        content = (
            f'[Context]\n{cap_str}\n\n'
            f'[Question]\n{ques["text"]}\n\n'
            f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
            f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
            f'[System]\n{prompt}\n\n'
        )
        cur_js = {
            'id': idx + 1,
            'question_id': ques['question_id'],
            'answer1_id': ans1.get('answer_id', ans1['question_id']),
            'answer2_id': ans2.get('answer_id', ans2['answer_id']),
            'category': category,
        }
        if idx >= len(cur_reviews):
            print(content)
            review = get_eval(content, args.max_tokens)
            scores = parse_score(review)
            cur_js['content'] = review
            cur_js['tuple'] = scores
            review_file.write(json.dumps(cur_js) + '\n')
            review_file.flush()
        else:
            print(f'Skipping {idx} as we already have it.')
        idx += 1
        print(idx)
    review_file.close()

    return args.output


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def preprocess(args, response_file, model_name):
    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    responses = [json.loads(r) for r in open(os.path.expanduser(response_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    responses = get_chunk(responses, args.num_chunks, args.chunk_idx)
    base, ext = os.path.splitext(os.path.basename(response_file))
    answer_file = os.path.join(args.answers_dir, f'{base}_answer{ext}')
    answers_file = os.path.expanduser(answer_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for line, resp in tqdm(zip(questions, responses), total=len(questions)):
        idx = line["question_id"]
        resp_key = "response_id"
        resp_text_key = "response"
        if resp_key not in resp:
            resp_key = "question_id"
            resp_text_key = "text"
        resp_idx = resp[resp_key]

        if int(idx) == int(resp_idx):
            # image_file = line[args.media_type]
            qs = line["text"]
            cur_prompt = qs
            outputs = resp[resp_text_key]
            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {},
                    }
                )
                + "\n"
            )
            ans_file.flush()
    ans_file.close()

    return answer_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-list", nargs='+', default=[])
    parser.add_argument("--media-type", type=str, default="image")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument('--responses-list', nargs='+', default=[])
    parser.add_argument("--answers-dir", type=str, default="answers")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument('--context-file', type=str, default="context.jsonl")
    parser.add_argument('--rule-file', type=str, default="rule.json")
    parser.add_argument('--output', type=str, default="output.json")
    parser.add_argument('--max-tokens', type=int, default=1024, help='maximum number of tokens produced in the output')
    args = parser.parse_args()

    answer_list = []
    for response, model_name in zip(args.responses_list, args.model_name_list):
        answer = preprocess(args, response, model_name)
        answer_list.append(answer)

    review = generate_prompt(args, answer_list)

    summarize([review])
