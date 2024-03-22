
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

data_dir="/media/data3/datasets/librispeech_origin"
# train_manifests="[${data_dir}/train_clean_360_cleaned.json,${data_dir}/train_clean_100_cleaned.json,${data_dir}/train_other_500_cleaned.json]"
# train_manifests="[${data_dir}/test_clean.json]"
dev_manifests="${data_dir}/dev_clean_cleaned.json"
batch_size=16
num_workers=0

TRAIN_IS_TARRED=True
TRAIN_MANIFEST='/media/data3/librispeech_tarred/tarred_audio_manifest.json'
TRAIN_FILEPATHS="/media/data3/librispeech_tarred/audio__OP_0..511_CL_.tar"
noise_manifest="[/media/data3/datasets/noise_data/musan/musan_nonspeech_manifest.json,/media/data3/datasets/noise_data/freesound/freesound_noise_manifest_filtered.json]"

exp_name=ssl_fastconformer_large_rq_ls_dns_debug_r2

CUDA_VISIBLE_DEVICES="0" python speech_pretrain_denoise.py \
    --config-path="configs" \
    --config-name="fastconformer_large_ssl_rq_dns" \
    model.train_ds.manifest_filepath=${TRAIN_MANIFEST} \
    model.train_ds.is_tarred=${TRAIN_IS_TARRED} \
    model.train_ds.tarred_audio_filepaths=${TRAIN_FILEPATHS} \
    model.train_ds.noise_manifest=$noise_manifest \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.validation_ds.noise_manifest=$noise_manifest \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    ++trainer.gradient_clip_val=1.0 \
    exp_manager.name=$exp_name \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name=$exp_name \
    exp_manager.wandb_logger_kwargs.project="ssl_WavLM"

