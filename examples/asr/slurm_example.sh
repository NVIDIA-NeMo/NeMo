#!/bin/bash
#SBATCH -A <ACCOUNT>		# <CHANGE !>
#SBATCH -J "<JOB_NAME>"            # job name (<< CHANGE ! >>)
#SBATCH -p <PARTITION>             # partition (queue) (<< CHANGE ! >>)
#SBATCH -N 1                       # number of nodes
#SBATCH -t 4:00:00              # wall time
#SBATCH --time-min 04:00:00  
#SBATCH --ntasks-per-node=8    # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --overcommit
#SBATCH --mem=0


# """""""""""""""NOTE"""
# This is a template script to run NeMo on Slurm. Only provided as a reference to get you started on multi-node multi-gpu training.
# Please adjust the paths, account, partition, and other parameters as per your cluster setup.
# Make sure to have the correct paths to the data, code, and results directory.
# Make sure to have the correct WandB key.
# Make sure to have the correct NGC key in ~/.config/enroot/.credentials
# Make sure to have the correct container image. (use sqsh to improve loading times)

# This script is not tested and may require adjustments based on your cluster setup.

# Once you have adjusted the script, you can run it using the following command:
# sbatch slurm_example.sh
# """


set -x

WANDB="<WANDB_KEY>" # replace with your own WandB API key

CONTAINER=nvcr.io/nvidia/nemo:25.02.rc4 # Adjust to your needs. and make sure you have ngc key in ~/.config/enroot/.credentials

CODE_DIR=<PATH_TO_NEMO> # Adjust to your needs.
DATA_DIR=<PATH_TO_DATA> # Adjust to your needs on how you want to mount the data.

MOUNTS="--container-mounts=${CODE_DIR},${DATA_DIR}:/data"

CONFIG_PATH="${CODE_DIR}/examples/asr/conf/speech_multitask"  # Adjust if launching from outside this directory.
CONFIG_NAME="fast-conformer_aed.yaml"
EXP_NAME="canary-1b-repro"
RESULTS_DIR="<PATH_TO_RESULTS>" # Adjust to your needs.
mkdir -p ${RESULTS_DIR}

# && export AIS_ENDPOINT="<AIS_ENDPOINT>" \ # Update and uncomment if you want to use AIS for data storage

read -r -d '' cmd <<EOF
cd ${CODE_DIR} \
&& export WANDB_API_KEY="${WANDB}" \
&& export PYTHONPATH="${CODE_DIR}:${PYTHONPATH}" \
&& export TOKENIZERS_PARALLELISM=false \
&& export LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE=0.3 \
&& HYDRA_FULL_ERROR=1 TORCH_CUDNN_V8_API_ENABLED=1 python -u -B ${CODE_DIR}/examples/asr/speech_multitask/speech_to_text_aed.py  \
--config-path=$CONFIG_PATH \
--config-name=$CONFIG_NAME \
trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
trainer.devices=${SLURM_NTASKS_PER_NODE} \
name=${EXP_NAME} \
exp_manager.explicit_log_dir=${RESULTS_DIR}
EOF


OUTFILE=${RESULTS_DIR}/slurm-%j-%n.out
ERRFILE=${RESULTS_DIR}/error-%j-%n.out

srun -o $OUTFILE -e $ERRFILE --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"