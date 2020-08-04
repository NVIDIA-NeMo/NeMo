# -*- coding: utf-8 -*-
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

""" Script responsible for generation of a JSON file with list of NeMo collections.

Exemplary call:
    nemo-get-collections

Returns:
    Format of the output JSON file (indicated  as -o, --output_filename):

[
    {
        "name": "nlp",
        "id": "nemo.collections.nlp",
        "description": "Natural Language Processing collection",
        "version": "0.11.0",
        "author": "NVIDIA Corporation"
    },
...
]
"""

import argparse
import importlib
import json
import os

import nemo
from nemo.utils import logging


def process_collection(id, col):
    """ Helper function processing the collection.
    
    Args:
        id: (short) name of the collection.
        col: a collection (python module).
    """
    return {
        "name": id,
        "id": col.__name__,  # Full specification.
        "description": col.__description__,
        "version": col.__version__,
        "author": col.__author__,
    }


def get_collections():
    """ Main function generating a JSON file with list of NeMo collections. """
    # Parse arguments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output_filename',
        '-o',
        help='Name of the output JSON file (DEFAULT: collections.json)',
        type=str,
        default="collections.json",
    )
    args = parser.parse_args()

    # Get collections directory.
    colletions_dir = os.path.dirname(nemo.collections.__file__)
    logging.info('Analysing collections in `{}`'.format(colletions_dir))
    logging.info("=" * 80)

    # Generate list of NeMo collections - from the list of collection subfolders.
    collections = {}
    for sub_dir in os.listdir(colletions_dir):
        # Skip cache.
        if sub_dir == "__pycache__":
            continue
        # Check if it is a directory.
        if os.path.isdir(os.path.join(colletions_dir, sub_dir)):
            collections[sub_dir] = "nemo.collections." + sub_dir

    output_list = []
    # Iterate over all collections.
    for key, val in collections.items():
        # Try to get module specification.
        module_spec = importlib.util.find_spec(val)
        if module_spec is None:
            logging.warning("  * Failed to process `{}`".format(val))
        else:
            try:
                # Import the module from the module specification.
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)
                # Add to list.
                output_list.append(process_collection(key, module))
                logging.info(" * Processed `{}`".format(val))
            except AttributeError:
                logging.warning(" ! Failed to process `{}`".format(val))

    # Export to JSON.
    with open(args.output_filename, 'w') as outfile:
        json.dump(output_list, outfile)

    logging.info("=" * 80)
    outputs = "".join("* {} \n".format(o) for o in output_list)
    logging.info("Finshed the analysis, found {} collections:\n{}".format(len(output_list), outputs))
    logging.info("Results exported to `{}`.".format(args.output_filename))


if __name__ == '__main__':
    get_collections()
