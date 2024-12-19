**Introduction**

Welcome to the NeMo SlimPajama Data Pipeline and Pretraining tutorial! This tutorial provides a step-by-step guide to preprocessing the SlimPajama dataset and pretraining a Llama based model using the NeMo 2.0 library.

The tutorial includes two Jupyter notebooks: `data_pipeline.ipynb` and `pretraining.ipynb`. The `data_pipeline.ipynb` notebook provides a data pipeline to preprocess the SlimPajama dataset, including downloading, extracting, concatenating and tokenizing the data. The `pretraining.ipynb` notebook provides a pretraining recipe to train a language model using the preprocessed data.

This repository is designed to be used with the NeMo 2.0 and NeMo-Run.

**Pre-requisites / Requirements**

- System Configuration
  - For Preprocessing: access to any CPU node should be sufficient. Please reach out to us if you run into errors.
  - For Pretraining: access to at least 1 NVIDIA GPUs with a cumulative memory of at least 48GB.
  - A Docker-enabled environment, with NVIDIA Container Runtime installed, which will make the container GPU-aware.
- Software Requirements
  - Use the latest [NeMo Framework Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags) . Note that you must be logged in to the container registry to view this page.
  - This notebook uses the container: nvcr.io/nvidia/nemo:dev.
  - Get your Hugging Face [access token](https://huggingface.co/docs/hub/en/security-tokens), which will be used to download assets from Hugging Face.
  - Download Jupyter Lab or Jupyter Notebook in your environment if not already installed.
- NeMo 2.0 and NeMo-Run
  - We will use NeMo 2.0 and NeMo-Run for this tutorial. Both are already available in the NeMo Framework Container.

**Getting started**

Assuming you have all the pre-requisites installed, you can get started by following these steps:
1. Start and enter the dev container by running:
   ```bash
    docker run \
    --gpus device=all \
    --shm-size=2g \
    --net=host \
    --ulimit memlock=-1 \
    --rm -it \
    -v ${PWD}:/workspace \
    -w /workspace \
    nvcr.io/nvidia/nemo:dev bash
    ```
2. Log in through huggingface-cli using your Hugging Face token.
    ```huggingface-cli login```
3. From within the container, start the Jupyter lab:
    ```jupyter lab --ip 0.0.0.0 --port=8888 --allow-root```
4. Follow the directions in data_pipeline.ipynb and pretraining.ipynb notebooks to preprocess the SlimPajama dataset and pretrain a model.

**Note**

* Make sure to replace placeholder paths with the actual paths on your machine. Make sure to update the docker volume mounts to persist data.
* The `data_pipeline.ipynb` notebook assumes that the SlimPajama dataset is stored in the `/data/slimpajama` directory.
* The `pretraining.ipynb` notebook assumes that the preprocessed data is stored in the `/data/slimpajama_megatron` directory.