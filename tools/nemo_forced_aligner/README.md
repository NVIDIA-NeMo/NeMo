# NeMo Forced Aligner (NFA)

<p align="center">
Try it out: <a href="https://huggingface.co/spaces/erastorgueva-nv/NeMo-Forced-Aligner">HuggingFace Space 🎤</a> | Tutorial: <a href="https://colab.research.google.com/github/NVIDIA-NeMo/Speech/blob/main/tutorials/tools/NeMo_Forced_Aligner_Tutorial.ipynb">"How to use NFA?" 🚀</a> | Blog post: <a href="https://nvidia.github.io/NeMo/blogs/2023/2023-08-forced-alignment/">"How does forced alignment work?" 📚</a>
</p>

<p align="center">
<img width="80%" src="https://github.com/NVIDIA-NeMo/Speech/releases/download/v1.20.0/nfa_forced_alignment_pipeline.png">
</p>

NFA is a tool for generating token-, word- and segment-level timestamps of speech in audio using NeMo's CTC-based Automatic Speech Recognition models. You can provide your own reference text, or use ASR-generated transcription. You can use NeMo's ASR Model checkpoints out of the box in [14+ languages](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/results.html#speech-recognition-languages), or train your own model. NFA can be used on long audio files of 1+ hours duration (subject to your hardware and the ASR model used).


## Quickstart
1. Install [NeMo](https://docs.nvidia.com/nemo/speech/nightly/starthere/install.html) with the ASR collection.
2. Prepare a NeMo-style manifest containing the paths of audio files you would like to process, and (optionally) their text.
3. Run NFA's `align.py` script with the desired config, e.g.:
    ``` bash
    python <path_to_NeMo>/tools/nemo_forced_aligner/align.py \
	    pretrained_name="stt_en_fastconformer_hybrid_large_pc" \
	    manifest_filepath=<path to manifest of utterances you want to align> \
	    output_dir=<path to where your output files will be saved>
    ```

<p align="center">
	<img src="https://github.com/NVIDIA-NeMo/Speech/releases/download/v1.20.0/nfa_run.png">
</p>

## Documentation
More documentation is available [here](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/tools/nemo_forced_aligner.html).

## Multispeaker t-SOT timestamps

`align_multispeaker_sot.py` is a specialized PEE + TransformerCTC tool for
force-aligning a Nemotron-Transcribe t-SOT transcript with Sortformer speaker
probabilities. It reads each WAV directly from the JSONL manifest's
`audio_filepath`, and emits timestamps relative to that file,
one CTM per recording (numeric t-SOT speaker ID in the second column), RTTM,
and a `.seglst` file.

Run `python tools/nemo_forced_aligner/align_multispeaker_sot.py --help` for
its required model, tokenizer, input, and output arguments. The project-level
`extract_timestamps.sh` wrapper provides the normal, shorter interface:

```bash
./extract_timestamps.sh \
  --input manifest.jsonl \
  --output-dir results
```

The wrapper uses the project-owned
`conf/pee_transformer_ctc_timestamps.yaml` for model paths and alignment
defaults. Each record's `audio_filepath` is the only audio location used;
source-audio fields and recording offsets are ignored.
