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


"""
This script can be used to after google_data_preprocessing_before_alignment.py
to obtain separate "parallel" corpora for each semiotic class.

USAGE Example:
1. Download the Google TN dataset from https://www.kaggle.com/google-nlu/text-normalization
2. Unzip the English subset (e.g., by running `tar zxvf  en_with_types.tgz`).
   Then there will a folder named `en_with_types`.
3. Run python google_data_preprocessing_before_alignment.py
     which will produce a file data.tsv in its --output-dir
4. [Optional]. sort -u and rewrite data.tsv
5. Clone https://github.com/moses-smt/giza-pp.git, run "make" from its root folder.
6. Run this script
     python ${NEMO}/examples/nlp/text_normalization_as_tagging/dataset_preparation/prepare_corpora_for_alignment.py \
        --data_dir=<--output-dir from the previous step> \
        --out_dir=<destination directory for giza alignment folders> \
        --giza_dir=/.../giza-pp/GIZA++-v2 \
        --mckls_binary=/.../giza-pp/mkcls-v2/mkcls \
        --lang={en,ru}


Each corpus will be stored within <--data-dir> in the subdirectory with the name of the semiotic class,
 containing files ready to be fed to Giza++:
    src - written form, tokenized as characters
    dst - spoken form, tokenized as words
    run.sh - script for running Giza++

"""
from argparse import ArgumentParser
from collections import Counter
from os import listdir, mkdir
from os.path import isdir, join
from shutil import rmtree

from nemo.collections.nlp.data.text_normalization_as_tagging.utils import get_src_and_dst_for_alignment

parser = ArgumentParser(description='Split corpus to subcorpora for giza alignment')
parser.add_argument('--data_dir', type=str, required=True, help='Path to folder with data')
parser.add_argument('--out_dir', type=str, required=True, help='Path to output folder')
parser.add_argument('--giza_dir', type=str, required=True, help='Path to folder with GIZA++ binaries')
parser.add_argument('--mckls_binary', type=str, required=True, help='Path to mckls binary')
parser.add_argument('--lang', type=str, required=True, help='Language')
args = parser.parse_args()


def prepare_subcorpora_from_data() -> None:
    """Preprocess a corpus in Google TN Dataset format, extract TN-ITN phrase pairs, prepare input for GIZA++ alignment.
    """
    semiotic_vcb = Counter()
    cache_vcb = {}
    filenames = []
    for fn in listdir(args.data_dir + "/train"):
        filenames.append(args.data_dir + "/train/" + fn)
    for fn in listdir(args.data_dir + "/dev"):
        filenames.append(args.data_dir + "/dev/" + fn)
    for fn in filenames:
        with open(fn, "r", encoding="utf-8") as f:
            # Loop through each line of the file
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                if len(parts) != 3:
                    raise ValueError("Expect 3 parts, got " + str(len(parts)))
                semiotic_class, written, spoken = parts[0], parts[1].strip(), parts[2].strip()
                if spoken == "<self>":
                    continue
                semiotic_class = semiotic_class.casefold()
                semiotic_vcb[semiotic_class] += 1
                classdir = join(args.out_dir, semiotic_class)
                if not isdir(classdir):
                    mkdir(classdir)
                src, dst, _, _ = get_src_and_dst_for_alignment(semiotic_class, written, spoken, args.lang)
                if src == "" or dst == "":
                    continue
                if len(src.split(" ")) >= 100:
                    continue
                if semiotic_class not in cache_vcb:
                    cache_vcb[semiotic_class] = Counter()
                cache_vcb[semiotic_class][(src, dst)] += 1
    for sem in semiotic_vcb:
        classdir = join(args.out_dir, sem)
        if not isdir(classdir):
            raise ValueError("No such directory: " + classdir)
        print(classdir, " has ", semiotic_vcb[sem], " instances")
        with open(join(classdir, "run.sh"), "w") as out:
            out.write("GIZA_PATH=\"" + args.giza_dir + "\"\n")
            out.write("MKCLS=\"" + args.mckls_binary + "\"\n")
            out.write("\n")
            out.write("${GIZA_PATH}/plain2snt.out src dst\n")
            out.write("${MKCLS} -m2 -psrc -c15 -Vsrc.classes opt >& mkcls1.log\n")
            out.write("${MKCLS} -m2 -pdst -c15 -Vdst.classes opt >& mkcls2.log\n")
            out.write("${GIZA_PATH}/snt2cooc.out src.vcb dst.vcb src_dst.snt > src_dst.cooc\n")
            out.write(
                "${GIZA_PATH}/GIZA++ -S src.vcb -T dst.vcb -C src_dst.snt -coocurrencefile src_dst.cooc -p0 0.98 -o GIZA++ >& GIZA++.log\n"
            )
            out.write("##reverse direction\n")
            out.write("${GIZA_PATH}/snt2cooc.out dst.vcb src.vcb dst_src.snt > dst_src.cooc\n")
            out.write(
                "${GIZA_PATH}/GIZA++ -S dst.vcb -T src.vcb -C dst_src.snt -coocurrencefile dst_src.cooc -p0 0.98 -o GIZA++reverse >& GIZA++reverse.log\n"
            )
        out_src = open(join(classdir, "src"), 'w', encoding="utf-8")
        out_dst = open(join(classdir, "dst"), 'w', encoding="utf-8")
        out_freq = open(join(classdir, "freq"), 'w', encoding="utf-8")
        for src, dst in cache_vcb[sem]:
            freq = cache_vcb[sem][(src, dst)]
            out_src.write(src + "\n")
            out_dst.write(dst + "\n")
            out_freq.write(str(freq) + "\n")
        out_freq.close()
        out_dst.close()
        out_src.close()


# Main code
if __name__ == '__main__':
    for name in listdir(args.out_dir):
        path = join(args.out_dir, name)
        if isdir(path):
            rmtree(path)

    # Processing
    prepare_subcorpora_from_data()
