
NEMO_BASEPATH="/home/heh/codes/nemo-ssl"
export PYTHONPATH=$NEMO_BASEPATH:$PYTHONPATH

vox1_dir="/media/data2/datasets/speaker_datasets/voxceleb1"
vox2_dir="/media/data2/datasets/voxceleb2"
train_manifests="[${vox1_dir}/vox1_train_manifest_train_chunk3s.json,${vox2_dir}/vox2_all_manifest_train_chunk3s.json]"
dev_manifests="[${vox1_dir}/vox1_train_manifest_val_chunk30s.json,${vox2_dir}/vox2_all_manifest_val_chunk30s.json]"

noise_manifest="/media/data2/simulated_data/rir_noise_data/white_noise_1ch_37h.json"
rir_manifest="/media/data2/simulated_data/rir_noise_data/real_rirs_isotropic_noises_1ch.json"

EXP_NAME="ecapa_tdnn_ssl_sid_ssl_WavLM"
POSTFIX=epoch49

SSL_CKPT="/home/heh/codes/nemo-ssl/workspace/nemo_experiments/pretrained_checkpoints/oci_ll_unlab-60k_bs2048_adamwlr0.004_wd1e-3_warmup25000_epoch1000_mask0.01x40pre_conv_wavLM0.2x0.1_n16_r5--val_loss5.0808-epoch43-last.ckpt"

batch_size=4
num_workers=4

CUDA_VISIBLE_DEVICES="0" python speaker_id_train.py \
    --config-path="configs" \
    --config-name="ecapa_tdnn_small_ssl" \
    ++init_from_ptl_ckpt=$SSL_CKPT \
    trainer.log_every_n_steps=10 \
    model.train_ds.manifest_filepath=$train_manifests \
    model.train_ds.augmentor.noise.manifest_path=$noise_manifest \
    model.train_ds.augmentor.impulse.manifest_path=$rir_manifest \
    ++model.train_ds.min_duration=0.3 \
    ++model.validation_ds.min_duration=0.3 \
    model.validation_ds.manifest_filepath=$dev_manifests \
    model.train_ds.batch_size=$batch_size \
    model.validation_ds.batch_size=$batch_size \
    model.train_ds.num_workers=$num_workers \
    model.validation_ds.num_workers=$num_workers \
    exp_manager.checkpoint_callback_params.save_top_k=1 \
    exp_manager.name="$EXP_NAME-${POSTFIX}" \
    exp_manager.create_wandb_logger=False \
    exp_manager.wandb_logger_kwargs.name="$EXP_NAME-${POSTFIX}" \
    exp_manager.wandb_logger_kwargs.project="ssl_sid"
