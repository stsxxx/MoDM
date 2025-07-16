#!/bin/bash
# Run MoDM throughput experiments on MJHQ and diffusionDB datasets

echo "Running on diffusionDB..."
python3 ./serving/throughput/serving_system_1gpu.py \
  --large_model sd3.5 \
  --small_model sdxl \
  --num_req 1000 \
  --warm_up_size 0 \
  --dataset "diffusiondb" \
  --cache_directory "./MoDM_cache/DiffusionDB/nohit" \
  --image_directory ./images/MoDM_throughput_diffusionDB_sdxl \
  2>&1 | tee MoDM_throughput_diffusionDB_sdxl.txt

python3 ./serving/throughput/serving_system_1gpu.py \
  --large_model sd3.5 \
  --small_model sana \
  --num_req 1000 \
  --warm_up_size 0 \
  --dataset "diffusiondb" \
  --cache_directory "./MoDM_cache/DiffusionDB/nohit" \
  --image_directory ./images/MoDM_throughput_diffusionDB_sana \
  2>&1 | tee MoDM_throughput_diffusionDB_sana.txt

echo "Running on MJHQ..."
python3 ./serving/throughput/serving_system_1gpu.py \
  --large_model sd3.5 \
  --small_model sdxl \
  --num_req 1000 \
  --warm_up_size 0 \
  --dataset "MJHQ" \
  --cache_directory "./MoDM_cache/MJHQ/cache_images" \
  --image_directory ./images/MoDM_throughput_MJHQ_sdxl \
  2>&1 | tee MoDM_throughput_MJHQ_sdxl.txt

python3 ./serving/throughput/serving_system_1gpu.py \
  --large_model sd3.5 \
  --small_model sana \
  --num_req 1000 \
  --warm_up_size 0 \
  --dataset "MJHQ" \
  --cache_directory "./MoDM_cache/MJHQ/cache_images" \
  --image_directory ./images/MoDM_throughput_MJHQ_sana \
  2>&1 | tee MoDM_throughput_MJHQ_sana.txt

echo "Done."
