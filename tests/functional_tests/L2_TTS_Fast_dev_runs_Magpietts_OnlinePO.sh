# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
coverage run --branch -a --data-file=/workspace/.coverage --source=/workspace/nemo examples/tts/magpietts.py \
    --config-name magpietts_multilingual_v1 \
    +mode="onlinepo_train" \
    ~model.text_tokenizers.multilingual_sentencepiece \
    +model.text_tokenizers.english_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.english_chartokenizer.pretrained_model="google/byt5-small" \
    +model.text_tokenizers.spanish_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.spanish_chartokenizer.pretrained_model="google/byt5-small" \
    +model.text_tokenizers.french_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.french_chartokenizer.pretrained_model="google/byt5-small" \
    +model.text_tokenizers.dutch_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.dutch_chartokenizer.pretrained_model="google/byt5-small" \
    +model.text_tokenizers.italian_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.italian_chartokenizer.pretrained_model="google/byt5-small" \
    +model.text_tokenizers.german_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.german_chartokenizer.pretrained_model="google/byt5-small" \
    +model.text_tokenizers.portugese_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.portugese_chartokenizer.pretrained_model="google/byt5-small" \
    +model.text_tokenizers.polish_chartokenizer._target_=AutoTokenizer \
    +model.text_tokenizers.polish_chartokenizer.pretrained_model="google/byt5-small" \
    +train_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/an4_train.json" \
    +train_ds_meta.an4.audio_dir="/" \
    +train_ds_meta.an4.tokenizer_names="[english_phoneme]" \
    +train_ds_meta.an4.feature_dir=null \
    +val_ds_meta.an4.manifest_path="/home/TestData/an4_dataset/an4_val.json" \
    +val_ds_meta.an4.audio_dir="/" \
    +val_ds_meta.an4.tokenizer_names="[english_phoneme]" \
    +val_ds_meta.an4.feature_dir=null \
    +init_from_ptl_ckpt="/home/TestData/tts/2506_SeenSpeaker/T5TTS--val_loss=0.3125-epoch=8.ckpt" \
    max_epochs=1 \
    batch_size=2 \
    +model.grpo_beta=0.0 \
    +model.num_generations_per_item=6 \
    +model.reference_free=true \
    +model.inference_cfg_prob=0.0 \
    +model.inference_cfg_scale=2.5 \
    +model.cer_reward_weight=0.33 \
    +model.ssim_reward_weight=0.33 \
    +model.pesq_reward_weight=0.33 \
    +model.use_pesq=true \
    model.local_transformer_type="none" \
    model.cfg_unconditional_prob=0.0 \
    model.model_type="multi_encoder_context_tts" \
    model.transcript_decoder_layers="[0,2,4,6,8,10]" \
    model.context_decoder_layers="[1,3,5,7,9,11]" \
    model.context_duration_min=3.0 \
    model.context_duration_max=8.0 \
    model.decoder.p_dropout=0.0 \
    model.context_encoder.p_dropout=0.0 \
    model.encoder.p_dropout=0.0 \
    model.decoder.kernel_size=1 \
    model.decoder.xa_n_heads=1 \
    model.context_encoder.n_layers=6 \
    model.encoder.is_causal=false \
    model.use_text_conditioning_encoder=true \
    +model.forced_num_all_tokens_per_codebook=2048 \
    +model.forced_audio_eos_id=2047 \
    +model.forced_audio_bos_id=2046 \
    +model.forced_context_audio_eos_id=2045 \
    +model.forced_context_audio_bos_id=2044 \
    model.codecmodel_path="/home/TestData/tts/AudioCodec_21Hz_no_eliz_without_wavlm_disc.nemo" \
    model.alignment_loss_scale=0.0 \
    model.prior_scaling_factor=null \
    trainer.log_every_n_steps=10 \
    +model.inference_topk=2016 \
    model.optim.lr=2e-7 \
    ~model.optim.sched \
    +model.use_kv_cache_during_online_po=true \
    exp_manager.checkpoint_callback_params.monitor="val_cer_gt" \
    exp_manager.checkpoint_callback_params.mode="min" \
    trainer.precision=32 \
    trainer.devices="[0]" \
    +trainer.limit_train_batches=1 \
    +trainer.limit_val_batches=1 \
    trainer.strategy=auto \
    model.train_ds.dataloader_params.num_workers=0 \
    model.validation_ds.dataloader_params.num_workers=0 \
    ~trainer.check_val_every_n_epoch
