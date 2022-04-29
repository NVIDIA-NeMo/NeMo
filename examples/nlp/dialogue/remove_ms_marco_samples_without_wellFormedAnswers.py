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


import argparse
import json
from ast import literal_eval

import ijson


def main(filename):
    with open(filename, 'r') as file:
        objects = ijson.kvitems(file, 'wellFormedAnswers')
        valid_old_key_to_new_key = {}
        new_key = 0
        for key, well_formed_answer in objects:
            value = well_formed_answer if isinstance(well_formed_answer, list) else literal_eval(well_formed_answer)
            if len(value) > 0:
                valid_old_key_to_new_key[key] = str(new_key)
                new_key += 1
        filtered_data = {}
        fieldnames = ['query', 'query_type', 'answers', 'wellFormedAnswers', 'passages']
        for fieldname in fieldnames:
            add_data(filename, filtered_data, fieldname, valid_old_key_to_new_key)

    with open(filename, 'w') as fw:
        json.dump(filtered_data, fw)


def add_data(filename, filtered_data, fieldname, valid_old_key_to_new_key):
    with open(filename, 'r') as f:
        objects = ijson.kvitems(f, fieldname)
        filtered_data[fieldname] = {
            valid_old_key_to_new_key[key]: query for key, query in objects if key in valid_old_key_to_new_key
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename")
    args = parser.parse_args()
    main(args.filename)
