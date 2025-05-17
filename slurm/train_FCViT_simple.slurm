#!/bin/bash
#SBATCH --partition=GPUQ                       # GPU queue
#SBATCH --gres=gpu:h100:1                      # 1× H100 GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                    # 1 rank per GPU
#SBATCH --cpus-per-task=8                      # 8 CPU threads per rank
#SBATCH --mem=200G
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

# ── launch standard training with standard parameters ─────────────────
python main_train.py \
  --backbone vit_base_patch16_224 \
  --size_puzzle 225 \
  --size_fragment 75 \
  --num_fragment 9 \
  --lr 3e-05 \
  --epochs 100 \
  --weight_decay 0.05 \
  --batch_size 64 \
  --num_workers 12 \
  --data_path /cluster/home/muhamhz/data/imagenet \
  --output_dir /cluster/home/muhamhz/fcvit-mt-ntnu/output_fcvit_simple

echo "=============================================="
echo "FCViT Training completed at $(date)"
echo "=============================================="
