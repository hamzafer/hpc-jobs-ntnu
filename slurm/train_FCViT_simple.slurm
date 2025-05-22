#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:h100:4              # Request 4 H100s
#SBATCH --nodes=1
#SBATCH --ntasks=1                     # Single task with multiple processes
#SBATCH --cpus-per-task=32             # 8 CPUs per GPU × 4 GPUs
#SBATCH --mem=128G
#SBATCH --time=14-00:00:00
#SBATCH --job-name=FCViT_simple
#SBATCH --output=logs/out/fcvit_simple_%j.out
#SBATCH --error=logs/err/fcvit_simple_%j.err

# ── modules & conda env ───────────────────────────────────────────────
module purge
module load Anaconda3/2024.02-1
module load CUDA/12.4.0

source activate /cluster/home/muhamhz/fcvit_env

# ── diagnostics ───────────────────────────────────────────────────────
echo "=============================================="
echo "Starting FCViT job on $(hostname) at $(date)"
nvidia-smi
python - <<'PY'
import torch, os
print("CUDA visible:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("GPUs:", torch.cuda.device_count())
PY
echo "=============================================="

# ── move to repo ──────────────────────────────────────────────────────
cd /cluster/home/muhamhz/fcvit-mt-ntnu

# Launch with srun + torchrun for better integration with SLURM
srun torchrun --nnodes=1 --nproc_per_node=4 main_train.py \
  --backbone vit_base_patch16_224 \
  --size_puzzle 225 \
  --size_fragment 75 \
  --num_fragment 9 \
  --lr 3e-05 \
  --epochs 100 \
  --weight_decay 0.05 \
  --batch_size 16 \
  --num_workers 8 \
  --data_path /cluster/home/muhamhz/data/imagenet \
  --output_dir /cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_simple

echo "=============================================="
echo "FCViT Training completed at $(date)"
echo "=============================================="
