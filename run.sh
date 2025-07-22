python3 ./serving/throughput/serving_system_1gpu_docker.py \
  --large_model flux \
  --small_model sdxl \
  --num_req 1000 \
  --warm_up_size 0 \
  --dataset "diffusiondb" \
  --cache_directory "../Cached_images" \
  --image_directory ../Generated_images\
  > ../Logs/MoDM_throughput_diffusionDB_FLUX_SDXL.txt