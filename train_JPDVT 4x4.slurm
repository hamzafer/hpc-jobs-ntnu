#!/bin/bash
#SBATCH --partition=GPUQ                 # Use the GPU queue
#SBATCH --gres=gpu:h100:4                # Request 4 H100 GPUs
#SBATCH --nodes=1                        # 1 node
#SBATCH --ntasks=1                       # 1 task
#SBATCH --cpus-per-task=16               # 16 CPU cores
#SBATCH --mem=200G                       # Memory per node
#SBATCH --time=14-00:00:00               # Max runtime (14 days)
#SBATCH --job-name=JPDVT_H100            # Name of the job
#SBATCH --output=JPDVT_H100_4_%j.out     # Stdout log
#SBATCH --error=JPDVT_H100_4_%j.err      # Stderr log

# 1) Load modules
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.4.0

# 2) Activate the existing conda env
source activate /cluster/home/muhamhz/jpdvt_env

# 3) Print system info
echo "=============================================="
echo "Starting job on $(hostname) at $(date)"
echo "CUDA and GPU Information:"
nvidia-smi
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
echo "=============================================="

# 4) Optionally install any missing pip packages
# pip install --upgrade timm einops diffusers --no-cache-dir

# 5) Navigate to project directory
cd /cluster/home/muhamhz/JPDVT/image_model

# 6) Run training using all available H100 GPUs
#    --model     => sets the model to "JPDVT-T" (tiny variant for 4×4)
#    --image-size 256 => 4×64 puzzle pieces
#    --epochs 100 => fewer epochs recommended
#    --ckpt       => you can comment this out if you want to start from scratch
#    (You can also adjust --ckpt-every to control how often checkpoints are saved.)
srun torchrun --nnodes=1 --nproc_per_node=4 train_JPDVT_4x4.py \
    --dataset imagenet \
    --data-path /cluster/home/muhamhz/data/imagenet \
    --image-size 256 \
    --model JPDVT-T \
    --crop \
    --epochs 100 \
    --ckpt-every 100000

# Locally:
# torchrun --nnodes=1 --nproc_per_node=1 train_JPDVT_4x4.py --dataset imagenet --data-path /cluster/home/muhamhz/data/imagenet --image-size 256 --model JPDVT-T --crop

echo "=============================================="
echo "Training completed at $(date)"
echo "=============================================="
