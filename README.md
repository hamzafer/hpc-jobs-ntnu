# hpc-jobs-ntnu
hpc-jobs-ntnu

python sample.py \
    --model JPDVT \
    --dataset imagenet \
    --data-path /cluster/home/muhamhz/data/imagenet/val \
    --image-size 192 \
    --ckpt /cluster/home/muhamhz/JPDVT/image_model/results/009-imagenet-JPDVT-crop/checkpoints/2850000.pt \
    --num-sampling-steps 250
