#!/bin/bash
# Run MoDM throughput experiments on MJHQ and diffusionDB datasets

echo "Running on diffusionDB..."
python3 ./serving/throughput/serving_system.py \
  --large_model sd3.5 \
  --small_model sdxl \
  --num_req 1000 \
  --warm_up_size 0 \
  --dataset "diffusiondb" \
  --cache_directory "./MoDM_cache/DiffusionDB/nohit" \
  --image_directory ./images/MoDM_throughput_diffusionDB \
  2>&1 | tee MoDM_throughput_diffusionDB.txt

echo "Running on MJHQ..."
python3 ./serving/throughput/serving_system.py \
  --large_model sd3.5 \
  --small_model sdxl \
  --num_req 1000 \
  --warm_up_size 0 \
  --dataset "MJHQ" \
  --cache_directory "./MoDM_cache/MJHQ/cache_images" \
  --image_directory ./images/MoDM_throughput_MJHQ \
  2>&1 | tee MoDM_throughput_MJHQ.txt


echo "Done."
