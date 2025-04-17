#!/bin/bash
#SBATCH --partition=GPUQ                  # Use the GPU queue
#SBATCH --gres=gpu:h100:1                 # Request 1 H100 GPU
#SBATCH --nodes=1                         # Use 1 node
#SBATCH --ntasks=1                        # One task only
#SBATCH --cpus-per-task=8                 # Enough CPU per GPU
#SBATCH --mem=200G                        # Max available memory per node
#SBATCH --time=14-00:00:00                # Run for up to 14 days
#SBATCH --job-name=FCViT_H100             # Job name
#SBATCH --output=fcvit_h100_%j.out        # Stdout log
#SBATCH --error=fcvit_h100_%j.err         # Stderr log

# Load modules
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.4.0

# Activate FCViT Conda environment
source activate /cluster/home/muhamhz/fcvit_env

# System info
echo "=============================================="
echo "Starting FCViT job on $(hostname) at $(date)"
nvidia-smi
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
echo "=============================================="

# Go to FCViT repo
cd /cluster/home/muhamhz/fcvit-mt-ntnu

# Run training (adjust epochs if needed)
python main_train.py \
  --backbone vit_base_patch16_224 \
  --size_puzzle 225 \
  --size_fragment 75 \
  --num_fragment 9 \
  --lr 3e-05 \
  --epochs 100 \
  --weight_decay 0.05 \
  --batch_size 64 \
  --data_path /cluster/home/muhamhz/data/imagenet \
  --output_dir /cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit

# Done
echo "=============================================="
echo "FCViT Training completed at $(date)"
echo "=============================================="
 