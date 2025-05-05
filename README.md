# MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models


## Hardware Requirement

Experiments are conducted using NVIDIA A40 GPU with 48GB memory and AMD MI210 GPU with 64GB.

## Getting Started

First, install all required libraries

```bash
# For Nvidia GPU install PyTorch 2.1.0 compatible with CUDA 11.8
pip install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# For AMD GPU install PyTorch 2.3.0 compatible with ROCM 5.6
pip install torch==2.3.0.dev20240111+rocm5.6 torchvision==0.18.0.dev20240116+rocm5.6
# install other dependencies
pip install -r requirements.txt
```

then update model pipelines in the diffusers library

```bash
chmod +x replace_pipelines.sh
./replace_pipelines.sh 
```

## Experiments Workflow

Download DiffusionDB metadata first
```bash
python3 DiffusionDB_parquet.py
```

## Throughput 
```bash
# --large_model can be selected from flux and sd3.5
# --small_model can be selected from sdxl and sana
python3 ./serving/throughput/serving_system.py \
  --large_model flux \
  --small_model sdxl \
  --num_req 1000 \
  --warm_up_size 10 \
  --cache_directory "the directory where the pre-cached images are stored" \
  --image_directory ./MoDM_throughput \
  2>&1 | tee MoDM_throughput.txt
```

## SLO
```bash
# Run SLO experiment
# --req_rate can be set for specific request rate (#/min)
python3 ./serving/SLO/serving_system_MoDM_SLO.py \
  --large_model flux \
  --small_model sdxl \
  --num_req 1000 \
  --cache_directory "the directory where the pre-cached images are stored" \
  --image_directory ./MoDM_SLO \
  2>&1 | tee MoDM_SLO.txt

# Generate statistics
python3 ./serving/SLO/stats.py --log_file MoDM_SLO.txt
```

## Increasing Request Rate
```bash
python3 ./serving/serving_system_MoDM.py \
  --large_model flux \
  --small_model sdxl \
  --num_req 1000 \
  --cache_directory "the directory where the pre-cached images are stored" \
  --image_directory ./MoDM_increasing_rate \
  2>&1 | tee MoDM_increasing_rate.txt
```

## Citation

If this work is helpful, please cite as:

```bibtex
@misc{xia2025modmefficientservingimage,
      title={MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models}, 
      author={Yuchen Xia and Divyam Sharma and Yichao Yuan and Souvik Kundu and Nishil Talati},
      year={2025},
      eprint={2503.11972},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2503.11972}, 
}
```

## Disclaimer

This “research quality code” is for Non-Commercial purposes and provided by the contributors “As Is” without any express or implied warranty of any kind. The organizations (University of Michigan or Intel) involved do not own the rights to the data sets used or generated and do not confer any rights to it. The organizations (University of Michigan or Intel) do not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security or ethical review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.