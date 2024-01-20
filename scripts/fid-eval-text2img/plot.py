# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

"""
python plot_fid_vs_clip.py \
  --fid_scores_csv path/to/fid_scores.csv \
  --clip_scores_csv path/to/clip_scores.csv
Replace path/to/fid_scores.csv and path/to/clip_scores.csv with the paths
to the respective CSV files. The script will display the plot with FID
scores against CLIP scores, with cfg values annotated on each point.
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_fid_vs_clip(fid_scores_csv, clip_scores_csv, ax, label):
    fid_scores = pd.read_csv(fid_scores_csv)
    clip_scores = pd.read_csv(clip_scores_csv)
    merged_data = pd.merge(fid_scores, clip_scores, on='cfg').sort_values('cfg')
    merged_data.index = range(len(merged_data))

    ax.plot(
        merged_data['clip_score'], merged_data['fid'], marker='o', linestyle='-', label=label
    )  # Connect points with a line

    for i, txt in enumerate(merged_data['cfg']):
        ax.annotate(txt, (merged_data['clip_score'][i], merged_data['fid'][i]))

    ax.set_xlabel('CLIP Score')
    ax.set_ylabel('FID')
    ax.set_title('FID vs CLIP Score')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fid_scores_csv', nargs='+', required=True, type=str, help='Paths to the FID scores CSV files'
    )
    parser.add_argument(
        '--clip_scores_csv', nargs='+', required=True, type=str, help='Paths to the CLIP scores CSV files'
    )
    parser.add_argument(
        '--labels', nargs='+', required=False, type=str, help='If provided, curves will be named with these names'
    )
    parser.add_argument(
        '--save_plot_path', required=False, type=str, help='If provided, the plot will be stored at this path'
    )
    args = parser.parse_args()

    if not args.labels:
        args.labels = [None] * len(args.fid_scores_csv)

    assert len(args.fid_scores_csv) == len(args.clip_scores_csv) == len(args.labels), (
        len(args.fid_scores_csv),
        len(args.clip_scores_csv),
        len(args.labels),
    )

    fig, ax = plt.subplots()

    for fid, clip, label in zip(args.fid_scores_csv, args.clip_scores_csv, args.labels):
        plot_fid_vs_clip(fid, clip, ax, label)

    plt.show()
    if args.save_plot_path:
        plt.savefig(args.save_plot_path)
