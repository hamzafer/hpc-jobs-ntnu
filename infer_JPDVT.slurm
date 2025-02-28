#!/bin/bash
#SBATCH --partition=GPUQ                 # Use the GPU queue
#SBATCH --gres=gpu:h100:4                # Request 4 H100 GPUs
#SBATCH --nodes=1                        # 1 node
#SBATCH --ntasks=1                       # 1 task, but we'll spawn multiple processes
#SBATCH --cpus-per-task=16               # Enough CPU cores
#SBATCH --mem=200G                       # Memory
#SBATCH --time=14-00:00:00               # Set the max runtime (14 days)
#SBATCH --job-name=JPDVT_inference       # Name of the job
#SBATCH --output=JPDVT_infer_%j.out      # Stdout log
#SBATCH --error=JPDVT_infer_%j.err       # Stderr log

# Load modules
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.4.0

# Activate your conda env
source activate /cluster/home/muhamhz/jpdvt_env

echo "============================================="
echo "Starting JPDVT distributed inference at $(date)"
echo "Node: $(hostname)"
nvidia-smi
echo "============================================="

cd /cluster/home/muhamhz/JPDVT/image_model

# We'll launch 1 node with 4 processes (one per GPU):
srun torchrun --nnodes=1 --nproc_per_node=4 inference_ddp.py

echo "============================================="
echo "Inference completed at $(date)"
echo "============================================="
