#!/bin/bash
# check_storage.sh
# This script displays disk usage information, project directory sizes, and Lustre quota.

echo "--------------------------------------------------"
echo "Disk usage information (df -h):"
df -h
echo "--------------------------------------------------"
echo ""

echo "Project directory usage:"
echo "Inference directory:"
du -sh /cluster/home/muhamhz/JPDVT/image_model/inference
echo "Entire project directory:"
du -sh /cluster/home/muhamhz/JPDVT/image_model
echo "--------------------------------------------------"
echo ""

echo "Lustre quota for /cluster/home/muhamhz:"
lfs quota -h /cluster/home/muhamhz
echo "--------------------------------------------------"
echo ""

echo "Personal quota (if available):"
quota -s
echo "--------------------------------------------------"
