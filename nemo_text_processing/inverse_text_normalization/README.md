# Inverse Text Normalization

Inverse text normalization (ITN) is a part of the Automatic Speech Recognition (ASR) post-processing pipeline.
ITN is the task of converting the raw spoken output of the ASR model into its written form to improve text readability.

For example, `one hundred twenty three kilograms` -> `123 kg` 

# Documentation

[ITN documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/text_normalization/wfst/wfst_inverse_text_normalization.html).

[TN/INT tutorials NeMo/tutorials/text_processing](https://github.com/NVIDIA/NeMo/tree/stable/tutorials/text_processing).

# Installation

``` bash setup.sh ```

# Integrate ITN to a text processing pipeline

```
# import WFST-based ITN module
from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer

# initialize inverse normalizer
inverse_normalizer = InverseNormalizer(lang="en", cache_dir="CACHE_DIR")

# try normalizer on a few examples
print(inverse_normalizer.normalize("it costs one hundred and twenty three dollars"))
# >>>"it costs $123"

print(inverse_normalizer.normalize("in nineteen seventy"))
# >>> "in 1970"
```

# Prediction

```
# run prediction on <INPUT_TEXT_FILE>
python run_predict.py  --input=<INPUT_TEXT_FILE> --output=<OUTPUT_PATH> --lang=<LANGUAGE> \
    [--verbose]

# single input prediction
python inverse_normalize.py --lang=<LANGUAGE> <INPUT_TEXT> \
    [--verbose] [--overwrite_cache] [--cache_dir=<CACHE_DIR>]
```

The input is expected to be lower-cased. Punctuation are outputted with separating spaces after semiotic tokens, e.g. `"i see, it is ten o'clock..."` -> `"I see, it is 10:00  .  .  ."`.
Inner-sentence white-space characters in the input are not maintained.
See the above scripts for more details.

# Supported Languages

ITN supports: English, Spanish, German, French, Vietnamese, and Russian languages.

# Evaluation
Evaluation on text normalization data, e.g. [Google text normalization dataset](https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish).

```
python run_evaluate.py  --input=./en_with_types/output-00001-of-00100 \
    [--cat CATEGORY] [--filter]
```