# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#!/bin/bash

concatenate_chunk() {
    local data_folder=$1
    local chunk_number=$2
    local chunk_folder="$data_folder/chunk$chunk_number"
    local output_file="$data_folder/concatenated_chunk$chunk_number.jsonl"

    echo "Combining files for $data_folder/chunk$chunk_number to $output_file."

    if [ ! -d "$chunk_folder" ]; then
        echo "Chunk folder $chunk_folder does not exist"
        return 1
    fi

    # Check if the concatenated file already exists
    if [ -f "$output_file" ]; then
        echo "Concatenated file for chunk$chunk_number already exists. Skipping."
        return 0
    fi

    # Use find to get all files in the chunk folder and sort them
    files=$(find $chunk_folder -maxdepth 1 -type f -name "*.jsonl" | sort)

    # Concatenate all files in the chunk folder
    cat $files > "$output_file"

    if [ $? -eq 0 ]; then
        echo "Successfully concatenated files for chunk$chunk_number"
    else
        echo "Failed to concatenate files for chunk$chunk_number"
    fi
}

# Check if enough arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <data_folder> <chunk_number1> [<chunk_number2> ...]"
    exit 1
fi

# Get the train folder from the first argument
data_folder=$1
shift

# Check if the train folder exists
if [ ! -d "$data_folder" ]; then
    echo "Error: Data folder '$data_folder' does not exist"
    exit 1
fi

# Process each provided chunk number
for chunk_number in "$@"; do
    if [[ -n "$chunk_number" ]]; then
        concatenate_chunk "$data_folder" "$chunk_number"
    fi
done
