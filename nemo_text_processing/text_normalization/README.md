# Text Normalization

NeMo Text Normalization converts text from written form into its verbalized form. It is used as a preprocessing step before Text to Speech (TTS). It could also be used for preprocessing Automatic Speech Recognition (ASR) training transcripts.

For example, `123 kg` -> `one hundred twenty three kilograms`

# Documentation

[TN documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/text_normalization/wfst/wfst_text_normalization.html).

[TN/INT tutorials: NeMo/tutorials/text_processing](https://github.com/NVIDIA/NeMo/tree/stable/tutorials/text_processing).

# Installation
`bash ../setup.sh`

# Integrate TN to a text processing pipeline

```
# import WFST-based TN module
from nemo_text_processing.text_normalization.normalize import Normalizer

# initialize normalizer
normalizer = Normalizer(input_case="cased", lang="en")

# try normalizer on a few examples
print(normalizer.normalize("123"))
# >>> one hundred twenty three
print(normalizer.normalize_list(["at 10:00", "it weights 10kg."], punct_post_process=True))
# >>> ["at ten o'clock", 'it weights ten kilograms.']
```

# Prediction

```
# run prediction on <INPUT_TEXT_FILE>
python run_predict.py  --input=<INPUT_TEXT_FILE> --output=<OUTPUT_PATH> --lang=<LANGUAGE> \
    [--input_case=<INPUT_CASE>]

# single input prediction
python normalize.py --lang=<LANGUAGE> <INPUT_TEXT> \
    [--verbose] [--overwrite_cache] [--cache_dir=<CACHE_DIR>] [--input_case=<INPUT_CASE>]
``` 

# Evaluation

Evaluation on text normalization data, e.g. [Google text normalization dataset](https://www.kaggle.com/richardwilliamsproat/text-normalization-for-english-russian-and-polish).


``` python run_evaluate.py  --input=./en_with_types/output-00001-of-00100 [--cat CATEGORY] ```

# Audio-based normalization

See [documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/text_normalization/wfst/wfst_text_normalization.html#audio-based-text-normalization) for more details.