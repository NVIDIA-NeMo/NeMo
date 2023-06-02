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
This script can be used after GIZA++ alignment to extract final alignments for each semiotic class.
"""

import re
from argparse import ArgumentParser

from nemo.collections.nlp.data.text_normalization_as_tagging.utils import (
    check_monotonicity,
    fill_alignment_matrix,
    get_targets,
    get_targets_from_back,
)


parser = ArgumentParser(description='Extract final alignments from GIZA++ alignments')
parser.add_argument('--mode', type=str, required=True, help='tn or itn')
parser.add_argument('--giza_dir', type=str, required=True, help='Path to folder with GIZA++ alignment')
parser.add_argument(
    '--giza_suffix', type=str, required=True, help='suffix of alignment files, e.g. \"Ahmm.5\", \"A3.final\"'
)
parser.add_argument('--out_filename', type=str, required=True, help='Output file')
parser.add_argument('--lang', type=str, required=True, help="Language")
args = parser.parse_args()


def main() -> None:
    g = open(args.giza_dir + "/GIZA++." + args.giza_suffix, "r", encoding="utf-8")
    f = open(args.giza_dir + "/GIZA++reverse." + args.giza_suffix, "r", encoding="utf-8")
    target_inner_delimiter = ""
    if args.mode == "tn":
        g, f = f, g
        target_inner_delimiter = "_"
    out = open(args.giza_dir + "/" + args.out_filename, "w", encoding="utf-8")
    cache = {}
    good_count, not_mono_count, not_covered_count, exception_count = 0, 0, 0, 0
    n = 0
    while True:
        n += 3
        if n % 10000 == 0:
            print(n, "lines processed")
        fline1 = f.readline().strip()
        fline2 = f.readline().strip()
        fline3 = f.readline().strip()
        gline1 = g.readline().strip()
        gline2 = g.readline().strip()
        gline3 = g.readline().strip()
        if fline1 == "" and gline1 == "":
            break
        cache_key = fline1 + "\t" + fline2 + "\t" + gline1 + "\t" + gline2
        if cache_key in cache:
            out.write(cache[cache_key] + "\n")
            continue
        if fline1 == "" or gline1 == "" or fline2 == "" or gline2 == "" or fline3 == "" or gline3 == "":
            raise ValueError("Empty line: " + str(n))
        try:
            matrix, srctokens, dsttokens = fill_alignment_matrix(fline2, fline3, gline2, gline3)
        except Exception:
            print(fline1)
            print(fline2)
            print(fline3)
            print(gline1)
            print(gline2)
            print(gline3)
            exception_count += 1
            out_str = "-exception:\t" + fline2 + "\t" + gline2
            out.write(out_str + "\n")
            continue
        else:
            matrix[matrix <= 2] = 0  # leave only 1-to-1 alignment points
            if check_monotonicity(matrix):
                targets = get_targets(matrix, dsttokens, delimiter=target_inner_delimiter)
                targets_from_back = get_targets_from_back(matrix, dsttokens, delimiter=target_inner_delimiter)
                if len(targets) != len(srctokens):
                    raise ValueError(
                        "targets length doesn't match srctokens length: len(targets)="
                        + str(len(targets))
                        + "; len(srctokens)="
                        + str(len(srctokens))
                    )
                leftside_align = " ".join(targets)
                rightside_align = " ".join(targets_from_back)

                rightside_align = rightside_align.replace("<DELETE> <DELETE> _11100_", "_11 <DELETE> 100_")
                leftside_align = leftside_align.replace("<DELETE> <DELETE> _11100_", "_11 <DELETE> 100_")

                # _1 4000_ => _14 000_
                # 1 5,000 => 15 ,000
                rightside_align = re.sub(r"^_1 ([\d])(,?000)", r"_1\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"^_1 ([\d])(,?000)", r"_1\g<1> \g<2>", leftside_align)

                # "_2 10 0_" => "_2 <DELETE> 100_"
                rightside_align = re.sub(r"([\d]) 10 0_", r"\g<1> <DELETE> 100_", rightside_align)
                leftside_align = re.sub(r"([\d]) 10 0_", r"\g<1> <DELETE> 100_", leftside_align)

                if srctokens[0] in [
                    "ten",
                    "twenty",
                    "thirty",
                    "forty",
                    "fifty",
                    "sixty",
                    "seventy",
                    "eighty",
                    "ninety",
                ]:
                    #  ten thousand sixty  _1 00 60_  =>  _10 0 60_
                    rightside_align = re.sub(r"^(_\d) 00 (\d)", r"\g<1>0 0 \g<2>", rightside_align)
                    leftside_align = re.sub(r"^(_\d) 00 (\d)", r"\g<1>0 0 \g<2>", leftside_align)

                #  ten thousand sixty three    _1 0, 06 3_ => _10 ,0 6 3_
                rightside_align = re.sub(r"([ _]\d) 0, 0(\d)", r"\g<1>0 ,0 \g<2>", rightside_align)
                leftside_align = re.sub(r"([ _]\d) 0, 0(\d)", r"\g<1>0 ,0 \g<2>", leftside_align)

                #  _3 0, 7 7 4=> _30 , 7 7 4_
                rightside_align = re.sub(r"(\d) 0, ", r"\g<1>0 , ", rightside_align)
                leftside_align = re.sub(r"(\d) 0, ", r"\g<1>0 , ", leftside_align)

                #   _1 1, 1 <DELETE> 40_  =>  _11 , 1 <DELETE> 40_
                rightside_align = re.sub(r"1 1, (\d)", r"11 , \g<1>", rightside_align)
                leftside_align = re.sub(r"1 1, (\d)", r"11 , \g<1>", leftside_align)

                if re.match(r".+надцат", srctokens[0]) or srctokens[0] in [
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                ]:
                    # "_1 <DELETE> 12 14_" -> "_11 <DELETE> 2 14_"
                    rightside_align = re.sub(
                        r"^(_1) (<DELETE>) ([\d])([\d])", r"\g<1>\g<3> \g<2> \g<4>", rightside_align
                    )
                    leftside_align = re.sub(
                        r"^(_1) (<DELETE>) ([\d])([\d])", r"\g<1>\g<3> \g<2> \g<4>", leftside_align
                    )

                    # "_1 10 10_" -> "_11 0 10_"
                    rightside_align = re.sub(r"^_1 ([\d])0 ([\d] ?[\d])", r"_1\g<1> 0 \g<2>", rightside_align)
                    leftside_align = re.sub(r"^_1 ([\d])0 ([\d] ?[\d])", r"_1\g<1> 0 \g<2>", leftside_align)

                if args.giza_dir.endswith("decimal") and args.lang == "ru":
                    # "_1 <DELETE> 0, 5_" => "_10 <DELETE> , 5_"      #десять целых и пять десятых
                    rightside_align = re.sub(
                        r"(\d) (<DELETE>) ([0123456789])(,) ([\d])", r"\g<1>\g<3> \g<2> \g<4> \g<5>", rightside_align
                    )
                    leftside_align = re.sub(
                        r"(\d) (<DELETE>) ([0123456789])(,) ([\d])", r"\g<1>\g<3> \g<2> \g<4> \g<5>", leftside_align
                    )

                if args.giza_dir.endswith("decimal") and args.lang == "en":
                    # "_7 0. 7_" => _70 . 7_
                    rightside_align = re.sub(r"^(_\d) 0\. (\d)", r"\g<1>0 . \g<2>", rightside_align)
                    leftside_align = re.sub(r"^(_\d) 0\. (\d)", r"\g<1>0 . \g<2>", leftside_align)

                if args.giza_dir.endswith("money") and args.lang == "en":
                    # "_1 , 000__£<<" => "_1 ,000_ _£<<"
                    rightside_align = re.sub(r"(\d) , 000_(_[£$€])", r"\g<1> ,000_ \g<2>", rightside_align)
                    leftside_align = re.sub(r"(\d) , 000_(_[£$€])", r"\g<1> ,000_ \g<2>", leftside_align)

                if args.giza_dir.endswith("money"):
                    # "_5 <DELETE> 000000__иен_" => "_5 000000_ _иен_"
                    rightside_align = re.sub(
                        r"([\d]) <DELETE> 000000_(_[^\d])", r"\g<1> 000000_ \g<2>", rightside_align
                    )
                    leftside_align = re.sub(r"([\d]) <DELETE> 000000_(_[^\d])", r"\g<1> 000000_ \g<2>", leftside_align)

                    # _5_ <DELETE> _m__£<< => "_5_ _m_ _£<<"
                    rightside_align = re.sub(
                        r"([\d]_) <DELETE> (_[mk]_)(_[^\d])", r"\g<1> \g<2> \g<3>", rightside_align
                    )
                    leftside_align = re.sub(r"([\d]_) <DELETE> (_[mk]_)(_[^\d])", r"\g<1> \g<2> \g<3>", leftside_align)

                    # "_3 <DELETE> 0__m__£<<" => "_30 _m_ _£<<"
                    rightside_align = re.sub(
                        r"([\d]) <DELETE> 0_(_[mk]_)(_[^\d])", r"\g<1>0 \g<2> \g<3>", rightside_align
                    )
                    leftside_align = re.sub(
                        r"([\d]) <DELETE> 0_(_[mk]_)(_[^\d])", r"\g<1>0 \g<2> \g<3>", leftside_align
                    )

                # "_15 <DELETE> 000__руб._" => "_15 000_ _руб._"
                rightside_align = re.sub(r"([\d]) <DELETE> (000_)(_[^\d])", r"\g<1> \g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) <DELETE> (000_)(_[^\d])", r"\g<1> \g<2> \g<3>", leftside_align)

                # "_2 5 0 000__$<<" => "_2 50 000_ _$<<"
                rightside_align = re.sub(r"([\d]) 0 000_(_[^\d])", r"\g<1>0 000_ \g<2>", rightside_align)
                leftside_align = re.sub(r"([\d]) 0 000_(_[^\d])", r"\g<1>0 000_ \g<2>", leftside_align)

                # "_5 0 0000__$_" => "_500 000_ _$_"
                rightside_align = re.sub(r"([\d]) 0 0000_(_[^\d])", r"\g<1>00 000_ \g<2>", rightside_align)
                leftside_align = re.sub(r"([\d]) 0 0000_(_[^\d])", r"\g<1>00 000_ \g<2>", leftside_align)

                # "_1 000__руб._" => "_1000_ _руб._"
                rightside_align = re.sub(r"_1 000_(_[^\d])", r"_1000_ \g<1>", rightside_align)
                leftside_align = re.sub(r"_1 000_(_[^\d])", r"_1000_ \g<1>", leftside_align)

                # replace cases like "2 0__января" with "20_ _января"
                leftside_align = re.sub(r"([\d]) (00?_)(_[^\d])", r"\g<1>\g<2> \g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) (00?_)(_[^\d])", r"\g<1>\g<2> \g<3>", rightside_align)

                #  "_3 <DELETE> 0__september_ _2 014_" => "_30_ <DELETE> _september_ _2 014_"
                #  "_3 <DELETE> 00__тыс.__руб._" => "_300_ <DELETE> _тыс.__руб._"
                leftside_align = re.sub(
                    r"([\d]) <DELETE> (00?_)(_[^\d])", r"\g<1>\g<2> <DELETE> \g<3>", leftside_align
                )
                rightside_align = re.sub(
                    r"([\d]) <DELETE> (00?_)(_[^\d])", r"\g<1>\g<2> <DELETE> \g<3>", rightside_align
                )

                # "_october_ _2 0,2 015_" => "_october_ _20 ,2 015_"
                leftside_align = re.sub(r"([\d]) (0),(\d)", r"\g<1>\g<2> ,\g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) (0),(\d)", r"\g<1>\g<2> ,\g<3>", rightside_align)

                # "_3 0_.10. _1 9 4 3_" =>  "_30_ .10. _1 9 4 3_"
                leftside_align = re.sub(r"([\d]) (0_)(\.[\d])", r"\g<1>\g<2> \g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) (0_)(\.[\d])", r"\g<1>\g<2> \g<3>", rightside_align)

                # replace cases like "_1 0000_" with "_10 000_"
                # replace cases like "_5 00000_" with "_500 000_"
                rightside_align = re.sub(r"([\d]) ([0][0]?)(000000000_)", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) ([0][0]?)(000000000_)", r"\g<1>\g<2> \g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) ([0][0]?)(000000_)", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) ([0][0]?)(000000_)", r"\g<1>\g<2> \g<3>", leftside_align)
                rightside_align = re.sub(r"([\d]) ([0][0]?)(000_)", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) ([0][0]?)(000_)", r"\g<1>\g<2> \g<3>", leftside_align)

                # "_4 00,000_" -> "_400 ,000_"
                rightside_align = re.sub(r"([\d]) ([0][0]?),(000_)", r"\g<1>\g<2> ,\g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) ([0][0]?),(000_)", r"\g<1>\g<2> ,\g<3>", leftside_align)

                # "_9 3 ,0__²_> _км_" => "_9 3 ,0__²_> _км_"
                rightside_align = re.sub(r"([\d]) (,00?_)(_[^\d])", r"\g<1>\g<2> \g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) (,00?_)(_[^\d])", r"\g<1>\g<2> \g<3>", leftside_align)

                # "_0 <DELETE> , <DELETE> <DELETE> 01__г_" => "_0 <DELETE> , 01 <DELETE> _г_"
                rightside_align = re.sub(
                    r"(,) <DELETE> <DELETE> 01_(_[^\d])", r"\g<1> 01_ <DELETE> \g<2>", rightside_align
                )
                leftside_align = re.sub(
                    r"(,) <DELETE> <DELETE> 01_(_[^\d])", r"\g<1> 01_ <DELETE> \g<2>", leftside_align
                )

                # "_0 <DELETE> , 7 6 <DELETE> <DELETE> 1__км_" => "_0 <DELETE> , 7 6 1_ <DELETE> _км_"
                rightside_align = re.sub(
                    r"(,) (\d) (\d) <DELETE> <DELETE> 1_(_[^\d])",
                    r"\g<1> \g<2> \g<3> 1_ <DELETE> \g<4>",
                    rightside_align,
                )
                leftside_align = re.sub(
                    r"(,) (\d) (\d) <DELETE> <DELETE> 1_(_[^\d])",
                    r"\g<1> \g<2> \g<3> 1_ <DELETE> \g<4>",
                    leftside_align,
                )

                # "_5 <DELETE> 0000__рублей_" => "_50 000_ рублей"
                rightside_align = re.sub(
                    r"([\d]) <DELETE> ([0][0]?)(000_)(_)", r"\g<1>\g<2> \g<3> \g<4>", rightside_align
                )
                leftside_align = re.sub(
                    r"([\d]) <DELETE> ([0][0]?)(000_)(_)", r"\g<1>\g<2> \g<3> \g<4>", leftside_align
                )

                # "_1 <DELETE> 115_" -> "_1 1 15_"
                rightside_align = re.sub(r"<DELETE> ([1])([1][\d])", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> ([1])([1][\d])", r"\g<1> \g<2>", leftside_align)

                # "_1 <DELETE> 990-х_" -> "_1 9 90-х_"
                rightside_align = re.sub(r"<DELETE> (9)(90)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (9)(90)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (8)(80)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (8)(80)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (7)(70)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (7)(70)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (6)(60)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (6)(60)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (5)(50)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (5)(50)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (4)(40)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (4)(40)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (3)(30)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (3)(30)", r"\g<1> \g<2>", leftside_align)
                rightside_align = re.sub(r"<DELETE> (2)(20)", r"\g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> (2)(20)", r"\g<1> \g<2>", leftside_align)

                # восемь ноль ноль ноль ноль ноль ноль ноль _8 0 0 0 0 0 0 0_
                # _8 <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 0000000_
                rightside_align = re.sub(
                    r"<DELETE> <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 0000000_",
                    r"0 0 0 0 0 0 0_",
                    rightside_align,
                )
                leftside_align = re.sub(
                    r"<DELETE> <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 0000000_",
                    r"0 0 0 0 0 0 0_",
                    leftside_align,
                )

                # _8 <DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 000000_
                rightside_align = re.sub(
                    r"<DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 000000_", r"0 0 0 0 0 0_", rightside_align
                )
                leftside_align = re.sub(
                    r"<DELETE> <DELETE> <DELETE> <DELETE> <DELETE> 000000_", r"0 0 0 0 0 0_", leftside_align
                )

                # _8 <DELETE> <DELETE> <DELETE> <DELETE> 00000_
                rightside_align = re.sub(r"<DELETE> <DELETE> <DELETE> <DELETE> 00000_", r"0 0 0 0 0_", rightside_align)
                leftside_align = re.sub(r"<DELETE> <DELETE> <DELETE> <DELETE> 00000_", r"0 0 0 0 0_", leftside_align)

                # _8 <DELETE> <DELETE> <DELETE> 0000_
                rightside_align = re.sub(r"<DELETE> <DELETE> <DELETE> 0000_", r"0 0 0 0_", rightside_align)
                leftside_align = re.sub(r"<DELETE> <DELETE> <DELETE> 0000_", r"0 0 0 0_", leftside_align)

                # _8 <DELETE> <DELETE> 000_
                rightside_align = re.sub(r"<DELETE> <DELETE> 000_", r"0 0 0_", rightside_align)
                leftside_align = re.sub(r"<DELETE> <DELETE> 000_", r"0 0 0_", leftside_align)

                # "_2 <DELETE> <DELETE> 010/11" => "_2 0 10 /11"
                rightside_align = re.sub(
                    r"<DELETE> <DELETE> (0)([1][\d])/([\d])", r"\g<1> \g<2> /\g<3>", rightside_align
                )
                leftside_align = re.sub(
                    r"<DELETE> <DELETE> (0)([1][\d])/([\d])", r"\g<1> \g<2> /\g<3>", leftside_align
                )

                # "_2 0 <DELETE> 11/12_" => "_2 0 11 /12_"
                rightside_align = re.sub(r"<DELETE> ([\d]+)/([\d])", r"\g<1> /\g<2>", rightside_align)
                leftside_align = re.sub(r"<DELETE> ([\d]+)/([\d])", r"\g<1> /\g<2>", leftside_align)

                # "_2 0 1 0/2 0 11_" => "_2 0 10 /2 0 11_"
                rightside_align = re.sub(r"([\d]) ([\d]+)/([\d])", r"\g<1>\g<2> /\g<3>", rightside_align)
                leftside_align = re.sub(r"([\d]) ([\d]+)/([\d])", r"\g<1>\g<2> /\g<3>", leftside_align)

                # "_5 0%_" => "_50 %_"
                # "_1 00%_" => "_100 %_"
                # "_1 00,00%_" => "_100,00 %_"
                rightside_align = re.sub(r"([\d]) ([0,]+)%", r"\g<1>\g<2> %", rightside_align)
                leftside_align = re.sub(r"([\d]) ([0,]+)%", r"\g<1>\g<2> %", leftside_align)

                # ATTENTION: keep the order of next two rules
                # "_2 0½_" => "_20 ½_"
                rightside_align = re.sub(r"([\d]) ([\d]+)½", r"\g<1>\g<2> ½", rightside_align)
                leftside_align = re.sub(r"([\d]) ([\d]+)½", r"\g<1>\g<2> ½", leftside_align)
                # "_1 ½_ <DELETE> <DELETE> <DELETE>" => "_1 <DELETE> <DELETE> <DELETE> ½_" #одна целая и одна вторая
                rightside_align = re.sub(
                    r"([\d]) (_?½_)? <DELETE> <DELETE> <DELETE>",
                    r"\g<1> <DELETE> <DELETE> <DELETE> \g<2>",
                    rightside_align,
                )
                leftside_align = re.sub(
                    r"([\d]) (_?½_)? <DELETE> <DELETE> <DELETE>",
                    r"\g<1> <DELETE> <DELETE> <DELETE> \g<2>",
                    leftside_align,
                )

                if args.lang == "en" and srctokens[-1] == "half":
                    #  _2 <DELETE> 1/ 2_ => _2 <DELETE> <DELETE> ½_
                    rightside_align = re.sub(r"(\d) <DELETE> 1/ 2_$", r"\g<1> <DELETE> <DELETE> ½_", rightside_align)
                    leftside_align = re.sub(r"(\d) <DELETE> 1/ 2_$", r"\g<1> <DELETE> <DELETE> ½_", leftside_align)

                # "_1 50_ <DELETE> _тыс.__руб._"  => "_1 50_ _тыс._ _руб._"
                rightside_align = re.sub(r"_ <DELETE> (_[^\d]+_)(_[^\d]+_)", r"_ \g<1> \g<2>", rightside_align)
                leftside_align = re.sub(r"_ <DELETE> (_[^\d]+_)(_[^\d]+_)", r"_ \g<1> \g<2>", leftside_align)

                # _1000 000__$_ => "_1000000_ _$_"
                rightside_align = re.sub(r"_1000 000_(_[^\d])", r"_1000000_ \g<1>", rightside_align)
                leftside_align = re.sub(r"_1000 000_(_[^\d])", r"_1000000_ \g<1>", leftside_align)

                if args.giza_dir.endswith("date") and args.lang == "en":
                    #  "_1 2_ <DELETE> _november_ _2 014_" => " <DELETE> _12_ <DELETE> _november_ _2 014_"
                    if srctokens[0] == "the":
                        leftside_align = re.sub(r"^_1 (\d_)", r"<DELETE> _1\g<1>", leftside_align)
                        rightside_align = re.sub(r"^_1 (\d_)", r"<DELETE> _1\g<1>", rightside_align)

                    # "<DELETE> <DELETE> _12,2012_" => "_12_ ,20 12_"
                    leftside_align = re.sub(r"^<DELETE> <DELETE> _12,2012_", r"_12_ ,20 12_", leftside_align)
                    rightside_align = re.sub(r"^<DELETE> <DELETE> _12,2012_", r"_12_ ,20 12_", rightside_align)

                    # "<DELETE> _1,20 14_" => "_1 ,20 14_"
                    leftside_align = re.sub(r"^<DELETE> _1,(\d)", r"_1 ,\g<1>", leftside_align)
                    rightside_align = re.sub(r"^<DELETE> _1,(\d)", r"_1 ,\g<1>", rightside_align)

                    # "_2 <DELETE> 1,20 14_" => "_2 1 ,20 14_"
                    leftside_align = re.sub(r"<DELETE> 1,(\d)", r"1 ,\g<1>", leftside_align)
                    rightside_align = re.sub(r"<DELETE> 1,(\d)", r"1 ,\g<1>", rightside_align)

                    #  <DELETE> _11,19 9 7_  =>   _11 ,19 9 7_
                    leftside_align = re.sub(r"<DELETE> _11,(\d)", r"_11 ,\g<1>", leftside_align)
                    rightside_align = re.sub(r"<DELETE> _11,(\d)", r"_11 ,\g<1>", rightside_align)

                    if len(srctokens) >= 2 and srctokens[-2] == "twenty":
                        # "<DELETE> <DELETE> _12,200 9_" => "_12 ,20 09_"
                        leftside_align = re.sub(
                            r"^<DELETE> <DELETE> _12,200 (\d_)", r"_12_ ,20 0\g<1>", leftside_align
                        )
                        rightside_align = re.sub(
                            r"^<DELETE> <DELETE> _12,200 (\d_)", r"_12_ ,20 0\g<1>", rightside_align
                        )

                        # "_april_ _2 015_" => "_april_ _20 15_"
                        leftside_align = re.sub(r"2 0(\d\d_)$", r"20 \g<1>", leftside_align)
                        rightside_align = re.sub(r"2 0(\d\d_)$", r"20 \g<1>", rightside_align)
                    elif len(srctokens) >= 2 and srctokens[-2] == "thousand":
                        # "<DELETE> <DELETE> _12,200 9_" => "_12 ,2 00 9_"
                        leftside_align = re.sub(
                            r"^<DELETE> <DELETE> _12,200 (\d_)", r"_12_ ,2 00 \g<1>", leftside_align
                        )
                        rightside_align = re.sub(
                            r"^<DELETE> <DELETE> _12,200 (\d_)", r"_12_ ,2 00 \g<1>", rightside_align
                        )

                    # thirtieth twenty fifteen   _3 0th__,20 15_ => _30th_ _,20 15_
                    leftside_align = re.sub(r"(\d) 0th_(_,\d)", r"\g<1>0th_ \g<2>", leftside_align)
                    rightside_align = re.sub(r"(\d) 0th_(_,\d)", r"\g<1>0th_ \g<2>", rightside_align)

                if args.giza_dir.endswith("date") and args.lang == "ru":
                    # тысяча девятьсот шестидесятого года  _1 9 6 0_  => _1 9 60_ <DELETE>
                    if srctokens[-1] == "года":
                        leftside_align = re.sub(r"(\d) 0_", r"\g<1>0_ <DELETE>", leftside_align)
                        rightside_align = re.sub(r"(\d) 0_", r"\g<1>0_ <DELETE>", rightside_align)

                if args.giza_dir.endswith("time"):
                    if srctokens[-1] == "hundred":
                        # fifteen hundred     <DELETE> _15:00_
                        rightside_align = re.sub(r"<DELETE> (_\d\d:)00_", r"\g<1> 00_", rightside_align)
                        leftside_align = re.sub(r"<DELETE> (_\d\d:)00_", r"\g<1> 00_", leftside_align)

                        #  !! Do not change the order of next two rules
                        # twenty one hundred      _2 1:00_ <DELETE>
                        rightside_align = re.sub(r"(_\d) (\d:)00_ <DELETE>", r"\g<1> \g<2> 00_", rightside_align)
                        leftside_align = re.sub(r"(_\d) (\d:)00_ <DELETE>", r"\g<1> \g<2> 00_", leftside_align)
                        # twenty hundred      _2 0:00_
                        rightside_align = re.sub(r"(_\d) (\d:)00_", r"\g<1>\g<2> 00_", rightside_align)
                        leftside_align = re.sub(r"(_\d) (\d:)00_", r"\g<1>\g<2> 00_", leftside_align)

                    if srctokens[-1] == "o'clock":
                        #  nine o'clock    <DELETE> _09:00_   => "_09:00_ <DELETE>"
                        rightside_align = re.sub(r"^<DELETE> ([^ ])$", r"\g<1> <DELETE>", rightside_align)
                        leftside_align = re.sub(r"^<DELETE> ([^ ])$", r"\g<1> <DELETE>", leftside_align)

                    # "_1 1:3 3_" => "_11: 3 3_"
                    rightside_align = re.sub(r"_(\d) (\d:)(\d)", r"\g<1>\g<2> \g<3>", rightside_align)
                    leftside_align = re.sub(r"_(\d) (\d:)(\d)", r"\g<1>\g<2> \g<3>", leftside_align)

                ban = False
                if args.giza_dir.endswith("ordinal"):
                    if dsttokens[0] == "_—":  # тысяча девятьсот сорок пятом    _— 1 9 4 5_
                        ban = True

                # ban roman numbers with at least two symbols, because we do not split them to parts
                for t in rightside_align.split():
                    if re.match(r"^_?[ivxl][ivxl]+_?$", t):
                        ban = True

                # ban cases like "_11/05/2013_", "_2005-11-25_", because they are source of incorrect alignments
                if args.giza_dir.endswith("date") and args.lang == "en":
                    if "/" in rightside_align or "-" in rightside_align:
                        ban = True

                # ban brackets
                if "(" in rightside_align or ")" in rightside_align:
                    ban = True

                if ban:
                    out_str = (
                        "ban:\t"
                        + " ".join(srctokens)
                        + "\t"
                        + " ".join(dsttokens)
                        + "\t"
                        + leftside_align
                        + "\t"
                        + rightside_align
                    )
                else:
                    out_str = (
                        "good:\t"
                        + " ".join(srctokens)
                        + "\t"
                        + " ".join(dsttokens)
                        + "\t"
                        + leftside_align
                        + "\t"
                        + rightside_align
                    )
                out.write(out_str + "\n")
                cache[cache_key] = out_str
            else:
                out_str = "-mon:\t" + " ".join(srctokens) + "\t" + " ".join(dsttokens)
                out.write(out_str + "\n")
                cache[cache_key] = out_str
                not_mono_count += 1

    f.close()
    g.close()
    out.close()


# Main code
if __name__ == '__main__':
    main()
