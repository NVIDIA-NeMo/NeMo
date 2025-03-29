coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/asr/asr_chunked_inference/aed/speech_to_text_aed_chunked_infer.py \
    model_path=/home/TestData/asr/canary/models/canary-1b-flash_HF_20250318.nemo \
    audio_dir=/home/TestData/asr/longform/earnings22 \
    output_filename=preds.json \
    timestamps=True
