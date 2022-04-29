#!/bin/bash

## This bash-script reproduces the pipeline of data preparation for the Thutmose Tagger model (tagger-based ITN model)

## In order to use it, you need:
## 1. install and compile GIZA++
##     git clone https://github.com/moses-smt/giza-pp.git giza-pp
##     cd giza-pp
##     make
## 2. Download Google TN Dataset
##    https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish
## 3. install NeMo
##     git clone https://github.com/NVIDIA/NeMo
## 4. Specify the following paths

## path to NeMo repository, e.g. /home/user/nemo
NEMO_PATH=
## path to GIZA++, e.g. /home/user/giza-pp/GIZA++-v2
GIZA_BIN_DIR=
## path to MCKLS_BINARY, e.g. /home/user/giza-pp/mkcls-v2/mkcls
MCKLS_BINARY=
## initial unzipped Google Text Normalization Dataset, e.g. /home/user/data/ru_with_types
GOOGLE_CORPUS_DIR=


## corpus language
CORPUS_LANG=ru

WORK_DIR=`pwd`   # directory from which this bash-script is run
echo "Working directory:" ${WORK_DIR}

## names of working subfolders
CORPUS_DIR=${WORK_DIR}/corpus
ALIGNMENT_DIR=${WORK_DIR}/alignment

## read the data and split it into train, dev, test
## files in test folder is truncated to match the default test dataset from the paper on Google TN Dataset
## option --add_test_full=true creates additional test_full folder, which is not truncated
python ${NEMO_PATH}/examples/nlp/duplex_text_normalization/data/data_split.py \
  --data_dir=${GOOGLE_CORPUS_DIR} \
  --output_dir=${CORPUS_DIR}_tmp \
  --lang=${CORPUS_LANG} \
  --add_test_full

## we need only output-00099-of-00100.tsv as the final test data
rm ${CORPUS_DIR}_tmp/test/output-00095-of-00100.tsv ${CORPUS_DIR}_tmp/test/output-00096-of-00100.tsv ${CORPUS_DIR}_tmp/test/output-00097-of-00100.tsv ${CORPUS_DIR}_tmp/test/output-00098-of-00100.tsv

## apply a blacklist of erroneous conversions to remove the sentences containing them from the corpus
mkdir ${CORPUS_DIR}
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/filter_sentences_with_errors.py \
  --data_dir=${CORPUS_DIR}_tmp \
  --out_dir=${CORPUS_DIR} \
  --lang=${CORPUS_LANG} \
  --errors_vocab_filename=${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/corpus_errors.${CORPUS_LANG}

rm -r ${CORPUS_DIR}_tmp

## This script extracts all unique ITN phrase-pairs from the Google TN dataset, tokenizes them and stores in separate
## folders for each semiotic class. In each folder we generate a bash script for running the alignment.
mkdir ${ALIGNMENT_DIR}
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/prepare_corpora_for_alignment.py \
  --data_dir=${CORPUS_DIR} \
  --out_dir=${ALIGNMENT_DIR} \
  --giza_dir=${GIZA_BIN_DIR} \
  --mckls_binary=${MCKLS_BINARY} \
  --lang=${CORPUS_LANG}

##exclude punct class
rm -r ${ALIGNMENT_DIR}/punct

## for better GIZA++ alignments mix in examples from other classes
## they will append to the tail of "src" and "dst" files and they will not have corresponding freqs in "freq" file
## all these appended lines will be skipped in the get_replacement_vocab step
for fn in "src" "dst"
do
    cat ${ALIGNMENT_DIR}/money/${fn} \
        ${ALIGNMENT_DIR}/cardinal/${fn} \
        ${ALIGNMENT_DIR}/decimal/${fn} \
        ${ALIGNMENT_DIR}/fraction/${fn} \
        ${ALIGNMENT_DIR}/measure/${fn} > ${ALIGNMENT_DIR}/money/${fn}.new

    cat ${ALIGNMENT_DIR}/time/${fn} \
        ${ALIGNMENT_DIR}/cardinal/${fn} > ${ALIGNMENT_DIR}/time/${fn}.new

    cat ${ALIGNMENT_DIR}/measure/${fn} \
        ${ALIGNMENT_DIR}/cardinal/${fn} \
        ${ALIGNMENT_DIR}/decimal/${fn} \
        ${ALIGNMENT_DIR}/fraction/${fn} \
        ${ALIGNMENT_DIR}/money/${fn} > ${ALIGNMENT_DIR}/measure/${fn}.new

    cat ${ALIGNMENT_DIR}/fraction/${fn} \
        ${ALIGNMENT_DIR}/cardinal/${fn} \
        ${ALIGNMENT_DIR}/measure/${fn} \
        ${ALIGNMENT_DIR}/money/${fn} > ${ALIGNMENT_DIR}/fraction/${fn}.new

    cat ${ALIGNMENT_DIR}/decimal/${fn} \
        ${ALIGNMENT_DIR}/cardinal/${fn} \
        ${ALIGNMENT_DIR}/measure/${fn} \
        ${ALIGNMENT_DIR}/money/${fn} > ${ALIGNMENT_DIR}/decimal/${fn}.new
done

for c in "decimal" "fraction" "measure" "time" "money"
do
    mv ${ALIGNMENT_DIR}/${c}/src.new ${ALIGNMENT_DIR}/${c}/src
    mv ${ALIGNMENT_DIR}/${c}/dst.new ${ALIGNMENT_DIR}/${c}/dst
done

for subfolder in ${ALIGNMENT_DIR}/*
do
    echo ${subfolder}
    chmod +x ${subfolder}/run.sh
done

## Run alignment using multiple processes
for subfolder in ${ALIGNMENT_DIR}/*
do
    cd ${subfolder}
    ./run.sh &
done
wait

## Extract final alignments for each semiotic class
for subfolder in ${ALIGNMENT_DIR}/*
do
    python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/extract_giza_alignments.py \
      --mode=itn \
      --giza_dir=${subfolder} \
      --out_filename=itn.out \
      --giza_suffix=A3.final \
      --lang=ru &
done
wait

## add column with frequencies of phrase pairs in the corpus
for subfolder in ${ALIGNMENT_DIR}/*
do
    paste -d"\t" ${subfolder}/freq ${subfolder}/itn.out > ${subfolder}/itn.out2
    awk 'BEGIN {FS="\t"} match($3, " "){print $0}' < ${subfolder}/itn.out2 | sort -rn > ${subfolder}/itn.debug
done

## loop through the obtained alignments and collect vocabularies (for each semiotic class)
## of all possible replacement fragments (aka tags)
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/prepare_corpora_after_alignment.py \
  --mode=get_replacement_vocab \
  --giza_dir=${ALIGNMENT_DIR} \
  --alignment_filename=itn.out2 \
  --data_dir="" \
  --vocab_filename=${WORK_DIR}/replacement_vocab_full.txt \
  --out_filename="" \
  --lang=${CORPUS_LANG}

## Here we put some voluntary thresholds on how many tags we take.
## Tags with low frequencies are likely to be derived from sporadic alignment mistakes
head -n 57 ${WORK_DIR}/replacement_vocab_full.txt.verbatim > ${WORK_DIR}/replacement_vocab_verbatim.txt
grep -v "0__" ${WORK_DIR}/replacement_vocab_full.txt.time | head -n 106 > ${WORK_DIR}/replacement_vocab_time.txt
grep -v "0__" ${WORK_DIR}/replacement_vocab_full.txt.telephone | head -n 134 > ${WORK_DIR}/replacement_vocab_telephone.txt
head -n 763 ${WORK_DIR}/replacement_vocab_full.txt.plain > ${WORK_DIR}/replacement_vocab_plain.txt
head -n 1397 ${WORK_DIR}/replacement_vocab_full.txt.ordinal > ${WORK_DIR}/replacement_vocab_ordinal.txt
grep -v "0__" ${WORK_DIR}/replacement_vocab_full.txt.money | grep -v '0\$' | head -n 294 > ${WORK_DIR}/replacement_vocab_money.txt
grep -v "0__" ${WORK_DIR}/replacement_vocab_full.txt.measure | head -n 508 > ${WORK_DIR}/replacement_vocab_measure.txt
cp ${WORK_DIR}/replacement_vocab_full.txt.letters ${WORK_DIR}/replacement_vocab_letters.txt
head -n 163 ${WORK_DIR}/replacement_vocab_full.txt.fraction > ${WORK_DIR}/replacement_vocab_fraction.txt
head -n 262 ${WORK_DIR}/replacement_vocab_full.txt.electronic > ${WORK_DIR}/replacement_vocab_electronic.txt
cp ${WORK_DIR}/replacement_vocab_full.txt.digit ${WORK_DIR}/replacement_vocab_digit.txt
head -n 270 ${WORK_DIR}/replacement_vocab_full.txt.decimal > ${WORK_DIR}/replacement_vocab_decimal.txt
head -n 271 ${WORK_DIR}/replacement_vocab_full.txt.date > ${WORK_DIR}/replacement_vocab_date.txt
head -n 455 ${WORK_DIR}/replacement_vocab_full.txt.cardinal > ${WORK_DIR}/replacement_vocab_cardinal.txt

## concatenate all tags in a single vocabulary (repetitions don't matter)
cat ${WORK_DIR}/replacement_vocab_cardinal.txt \
  ${WORK_DIR}/replacement_vocab_date.txt \
  ${WORK_DIR}/replacement_vocab_decimal.txt \
  ${WORK_DIR}/replacement_vocab_digit.txt \
  ${WORK_DIR}/replacement_vocab_electronic.txt \
  ${WORK_DIR}/replacement_vocab_fraction.txt \
  ${WORK_DIR}/replacement_vocab_letters.txt \
  ${WORK_DIR}/replacement_vocab_measure.txt \
  ${WORK_DIR}/replacement_vocab_money.txt \
  ${WORK_DIR}/replacement_vocab_ordinal.txt \
  ${WORK_DIR}/replacement_vocab_plain.txt \
  ${WORK_DIR}/replacement_vocab_telephone.txt \
  ${WORK_DIR}/replacement_vocab_time.txt \
  ${WORK_DIR}/replacement_vocab_verbatim.txt > ${WORK_DIR}/replacement_vocab.select.txt

## Here we loop once again through the alignments and discard those examples that are not fully covered
## by our restricted tag vocabulary
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/prepare_corpora_after_alignment.py \
  --mode=filter_by_vocab \
  --giza_dir=${ALIGNMENT_DIR} \
  --alignment_filename=itn.out2 \
  --data_dir="" \
  --vocab_filename=${WORK_DIR}/replacement_vocab.select.txt \
  --out_filename=itn.select.out \
  --lang=${CORPUS_LANG}

## We now have a large collection of ITN phrase conversions that we know how to tag.
## Once again we loop through the Google TN dataset and create tag-labeled datasets, containing full sentences.
## If a sentence contains something that we do not know how to tag, we discard the whole sentence.
for subset in "train" "dev"
do
    python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/prepare_corpora_after_alignment.py \
      --mode=get_labeled_corpus \
      --giza_dir=${ALIGNMENT_DIR} \
      --alignment_filename=itn.select.out \
      --data_dir=${CORPUS_DIR}/${subset} \
      --vocab_filename="" \
      --out_filename=${CORPUS_DIR}/${subset}.labeled \
      --lang=${CORPUS_LANG}
done

## Loop through the obtained datasets and get final tag vocabulary
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/get_label_vocab.py \
  --train_filename=${CORPUS_DIR}/train.labeled \
  --dev_filename=${CORPUS_DIR}/dev.labeled \
  --out_filename=${CORPUS_DIR}/label_map.txt

## The full dataset is very large, while some tags occur rarely. So we can try some sampling.
## Here we try to sample sentences that contain at least one tag that we have not yet seen at least for N times
## This script will split the input dataset into two parts: sampled sentences and the rest sentences.
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/sample_each_label.py \
  --filename=${CORPUS_DIR}/dev.labeled \
  --max_count=10

python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/dataset_preparation/sample_each_label.py \
  --filename=${CORPUS_DIR}/train.labeled \
  --max_count=500

## Here we create final train and dev datasets, mixing sampled sentences and some quantity of the rest sentences.
mkdir ${WORK_DIR}/datasets
DATASET=${WORK_DIR}/datasets/itn_sample500k_rest1500k_select_vocab
mkdir $DATASET
cat ${CORPUS_DIR}/train.labeled.sample_500 > ${DATASET}/train.tsv
head -n 1500000 ${CORPUS_DIR}/train.labeled.rest_500 >> ${DATASET}/train.tsv
cat ${CORPUS_DIR}/dev.labeled.sample_10 > ${DATASET}/valid.tsv
head -n 12000 ${CORPUS_DIR}/dev.labeled.rest_10 >> ${DATASET}/valid.tsv
cp ${DATASET}/valid.tsv ${DATASET}/test.tsv

echo "CARDINAL" > ${CORPUS_DIR}/semiotic_classes.txt
echo "DATE" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "DECIMAL" >> ${CORPUS_DIR}/semiotic_classes.txt 
echo "DIGIT" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "ELECTRONIC" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "FRACTION" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "LETTERS" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "MEASURE" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "MONEY" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "ORDINAL" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "PLAIN" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "PUNCT" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "TELEPHONE" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "TIME" >> ${CORPUS_DIR}/semiotic_classes.txt
echo "VERBATIM" >> ${CORPUS_DIR}/semiotic_classes.txt

cp ${CORPUS_DIR}/label_map.txt ${WORK_DIR}/datasets/label_map.txt
cp ${CORPUS_DIR}/semiotic_classes.txt ${WORK_DIR}/datasets/semiotic_classes.txt

## Now all data is ready to train the model.

## We also prepare the test data to test the model after the training.
## The test data knows nothing about alignment, it only contains input sentences and references.

## The original test data from Google TN Dataset contains a single reference for each ITN span.
## In order to take into account more than one acceptable variant,
## we prepare a dictionary of multiple possible references.
## The following script maps the whole input text of ITN span to the list of different conversions that occurred
## with this input anywhere in the Google TN Dataset.
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/get_multi_reference_vocab.py \
  --data_dir=${CORPUS_DIR} \
  --out_filename=${CORPUS_DIR}/reference_vocab.txt

## Generate the "default" test data for Google TN Dataset (same as usually used in papers)
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/prepare_corpora_for_testing.py \
  --data_dir=${CORPUS_DIR}/test \
  --reference_vocab=${CORPUS_DIR}/reference_vocab.txt \
  --output_file=${WORK_DIR}/datasets/test.labeled \
  --sampling_count=-1
awk 'BEGIN {FS="\t"}{print $1}' < ${WORK_DIR}/datasets/test.labeled > ${WORK_DIR}/datasets/test.input
awk 'BEGIN {FS="\t"}{print $1 "\t" $3}' < ${WORK_DIR}/datasets/test.labeled > ${WORK_DIR}/datasets/test.input_ref

## Generate the "hard" test data for Google TN Dataset.
## We try to sample at least 1000 examples per semiotic class.
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/prepare_corpora_for_testing.py \
  --data_dir=${CORPUS_DIR}/test_full \
  --reference_vocab=${CORPUS_DIR}/reference_vocab.txt \
  --output_file=${WORK_DIR}/datasets/test1000.labeled \
  --sampling_count=1000
awk 'BEGIN {FS="\t"}{print $1}' < ${WORK_DIR}/datasets/test1000.labeled > ${WORK_DIR}/datasets/test1000.input
awk 'BEGIN {FS="\t"}{print $1 "\t" $3}' < ${WORK_DIR}/datasets/test1000.labeled > ${WORK_DIR}/datasets/test1000.input_ref

## After we have train a model, we can run inference and evaluation like below

##export TOKENIZERS_PARALLELISM=false
##PRETRAINED_MODEL=./nemo_experiments/training.nemo
### run inference on default Google Dataset test
#python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
#  pretrained_model=${PRETRAINED_MODEL} \
#  inference.from_file=${DATA_PATH}/test.input \
#  inference.out_file=./final_test.output \
#  model.max_sequence_len=1024 #\
#  inference.batch_size=128
#
### run inference on "hard" test (sample of at least 1000 examples of each semiotic class)
#python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
#  pretrained_model=${PRETRAINED_MODEL} \
#  inference.from_file=${DATA_PATH}/test1000.input \
#  inference.out_file=./final_test1000.output \
#  model.max_sequence_len=1024 \
#  inference.batch_size=128
#
### compare inference results to the reference
#python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval.py \
#  --reference_file=${DATA_PATH}/test.labeled \
#  --inference_file=final_test.output \
#  > final_test.report
#
#python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval.py \
#  --reference_file=${DATA_PATH}/test1000.labeled \
#  --inference_file=final_test1000.output \
#  --print_other_errors \
#  > final_test1000.report
#
### compare inference results to the reference, get separate report per semiotic class
#python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval_per_class.py \
#  --reference_file=${DATA_PATH}/test.labeled \
#  --inference_file=final_test.output \
#  --output_file=per_class.report
#
#python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval_per_class.py \
#  --reference_file=${DATA_PATH}/test1000.labeled \
#  --inference_file=final_test1000.output \
#  --output_file=per_class1000.report
