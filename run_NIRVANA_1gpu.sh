#!/bin/bash
# =============================================
# Run MoDM throughput experiments on two datasets
# =============================================


echo "=========================================="
echo "Running on diffusionDB dataset..."
echo "=========================================="

python3 ./serving/throughput/serving_system_N_1gpu.py \
  --large_model sd3.5 \
  --num_req 1000 \
  --cache_directory "./MoDM_cache/DiffusionDB/nohit" \
  --dataset diffusiondb \
  --image_directory ./images/NIRVANA_throughput_diffusionDB \
  2>&1 | tee NIRVANA_throughput_diffusionDB.txt


echo "=========================================="
echo "Running on MJHQ dataset..."
echo "=========================================="

python3 ./serving/throughput/serving_system_N_1gpu.py \
  --large_model sd3.5 \
  --num_req 1000 \
  --cache_directory "./MoDM_cache/MJHQ/cache_images" \
  --dataset MJHQ \
  --image_directory ./images/NIRVANA_throughput_MJHQ \
  2>&1 | tee NIRVANA_throughput_MJHQ.txt



echo "=========================================="
echo "All experiments completed."
echo "=========================================="
