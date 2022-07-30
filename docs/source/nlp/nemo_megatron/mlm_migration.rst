Migrating from Megatron-LM
--------------------------

NeMo Megatron and Megatron-LM share many underlying technology. You should be able to convert your GPT model checkpoints trained with Megatron-LM into NeMo Megatron.
Example conversion script:

.. code-block:: bash

   <NeMo_ROOT_FOLDER>/examples/nlp/language_modeling/megatron_lm_ckpt_to_nemo.py \
     --checkpoint_folder <path_to_PTL_checkpoints_folder> \
     --checkpoint_name megatron_gpt--val_loss=99.99-step={steps}-consumed_samples={consumed}.0 \
     --nemo_file_path <path_to_output_nemo_file> \
     --model_type <megatron model type> \
     --tensor_model_parallel_size <tensor_model_parallel_size> \
     --pipeline_model_parallel_size <pipeline_model_parallel_size>  \
     --gpus_per_node  <gpus per node>



To resume the training from converted MegatronLM checkpoint, make sure to set the 
`trainer.max_steps=round(lr-warmup-fraction * lr-decay-iters + lr-decay-iters)`
where  `lr-warmup-fraction` and `lr-decay-iters` are arguments from MegatronLM training
so the learning rate scheduler will follow the same curve.

