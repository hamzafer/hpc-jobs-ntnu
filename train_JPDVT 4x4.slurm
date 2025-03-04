#!/bin/bash
#SBATCH --partition=GPUQ                 # Use the GPU queue
#SBATCH --gres=gpu:h100:4                # Request the 4 H100 GPUs (adjust as needed)
#SBATCH --nodes=1                        # Use 1 node (adjust if more GPUs free up)
#SBATCH --ntasks=1                       # 4 tasks (1 per GPU)
#SBATCH --cpus-per-task=16               # Allocate more CPU cores per GPU for better performance
#SBATCH --mem=200G                       # Max available memory per node
#SBATCH --time=14-00:00:00               # Set the max runtime (14 days)
#SBATCH --job-name=JPDVT_H100            # Name of the job
#SBATCH --output=JPDVT_H100_%j.out       # Save stdout log
#SBATCH --error=JPDVT_H100_%j.err        # Save stderr log

# Load required modules
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.4.0

# Activate the existing Conda environment
source activate /cluster/home/muhamhz/jpdvt_env

# Generate a dynamic port for rendezvous (avoids conflicts with other jobs)
# export MASTER_PORT=$((RANDOM % 1000 + 29000))  # Pick a random port between 29000-29999

# Print system info
echo "=============================================="
echo "Starting job on $(hostname) at $(date)"
echo "CUDA and GPU Information:"
nvidia-smi  # Show GPU usage
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
echo "=============================================="

# Install missing dependencies if needed (optional)
# pip install --upgrade timm einops diffusers --no-cache-dir

# Navigate to the project directory
cd /cluster/home/muhamhz/JPDVT/image_model

# Run training using all available H100 GPUs
# srun torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py --dataset imagenet --data-path /cluster/home/muhamhz/data/imagenet --image-size 192 --crop
srun torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT.py \
    --dataset imagenet \
    --data-path /cluster/home/muhamhz/data/imagenet \
    --image-size 192 \
    --crop \
    --ckpt /cluster/home/muhamhz/JPDVT/image_model/results/004-imagenet-JPDVT-crop/checkpoints/4070000.pt \
    --epochs 207 \
    --ckpt-every 50000

# --rdzv_endpoint=localhost:$MASTER_PORT

# Print completion message
echo "=============================================="
echo "Training completed at $(date)"
echo "=============================================="
