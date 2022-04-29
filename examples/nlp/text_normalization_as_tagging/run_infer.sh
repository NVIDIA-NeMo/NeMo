#!/bin/bash

## This bash-script demonstrates how to run inference and evaluation for the Thutmose Tagger model (tagger-based ITN model)

## In order to use it, you need:
## 1. install NeMo
##     git clone https://github.com/NVIDIA/NeMo
## 2. Specify the following paths

## path to NeMo repository, e.g. /home/user/nemo
NEMO_PATH=

## name or local path to pretrained model, e.g. ./nemo_experiments/training.nemo
PRETRAINED_MODEL=   

## path to input and reference files 
# (see the last steps in examples/nlp/text_normalization_as_tagging/prepare_dataset_en.sh,
#   starting from "python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/get_multi_reference_vocab.py"
#)
INPUT_FILE=
REFERENCE_FILE=


export TOKENIZERS_PARALLELISM=false

### run inference on default Google Dataset test
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/normalization_as_tagging_infer.py \
  pretrained_model=${PRETRAINED_MODEL} \
  inference.from_file=${INPUT_FILE} \
  inference.out_file=./final_test.output \
  model.max_sequence_len=1024 #\
  inference.batch_size=128

### compare inference results to the reference
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval.py \
  --reference_file=${REFERENCE_FILE} \
  --inference_file=final_test.output \
  > final_test.report

### compare inference results to the reference, get separate report per semiotic class
python ${NEMO_PATH}/examples/nlp/text_normalization_as_tagging/evaluation/eval_per_class.py \
  --reference_file=${REFERENCE_FILE} \
  --inference_file=final_test.output \
  --output_file=per_class.report
