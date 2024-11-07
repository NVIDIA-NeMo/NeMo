Preparing Image / Video Megatron Energon WebDataset with Cosmos Tokenizer
===========================

This script is an example on preparing a WebDataset for an image / video + text dataset using distributed processing with the Cosmos Tokenizer. It processes each sample by generating a **continuous** image / video latent using the Cosmos video tokenizer and a T5 embedding from the text caption. Then, the processed data is stored in a WebDataset-compatible format.

Requirements
------------
- **Dependencies**:
  - Please use the latest NeMo dev container: ``nvcr.io/nvidia/nemo:dev``
  - You may also need to install ``jammy`` and ``mediapy`` depending on your dev container version.

- **Data**:
  - The script uses an example dataset that comes in parquet format. To use a custom, you will need to write a custom ``process_func`` and create a new factory recipe that uses your new ``process_func``.

Usage
-----
1. **Set up your environment**:
   Pull and launch the NeMo dev container to run your script.

2. **Customize Cache Path**:
   Set the T5 cache directory path in the script by specifying the `t5_cache_dir` variable.

3. **Running the Script**:
   To run the script on 8 GPUs, use the following command:
   
   ``bash torchrun --nproc_per_node=8 nemo/collections/diffusion/data/prepare_energon_dataset.py``
