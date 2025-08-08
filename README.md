# MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models


## Hardware Requirement

Experiments are conducted using NVIDIA A40 GPU with 48GB memory and AMD MI210 GPU with 64GB.

## Getting Started
First, Create and activate a conda environment

```bash
conda create -n MoDM python=3.10 -y
conda activate MoDM
```

Then install all required libraries

```bash
# For Nvidia GPU install PyTorch 2.1.0 compatible with CUDA 11.8
pip install https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu118/torchvision-0.16.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=033712f65d45afe806676c4129dfe601ad1321d9e092df62b15847c02d4061dc
pip install https://download.pytorch.org/whl/cu118/torchaudio-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=cdfd0a129406155eee595f408cafbb92589652da4090d1d2040f5453d4cae71f
# For AMD GPU install PyTorch 2.3.0 compatible with ROCM 5.6
pip install torch==2.3.0.dev20240111+rocm5.6 torchvision==0.18.0.dev20240116+rocm5.6
```
```bash
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

Download pre-computed cache images and latents
```bash
gdown --folder --remaining-ok https://drive.google.com/drive/folders/1OFfbd_BgwTVY38bq_s0R0zyDaUytKB-Y
unzip ./MoDM_cache/MJHQ/cache_images.zip -d ./MoDM_cache/MJHQ/
unzip ./MoDM_cache/DiffusionDB/nohit.zip -d ./MoDM_cache/DiffusionDB/
```

## Throughput Experiments

For one GPU experiments
```bash
# MoDM
./run_MoDM_1gpu.sh

# Vanilla
./run_Vanilla_1gpu.sh

# NIRVANA
./run_NIRVANA_1gpu.sh
```

For multiple GPUs experiments
```bash
# MoDM
./run_MoDM.sh

# Vanilla
./run_Vanilla.sh

# NIRVANA
./run_NIRVANA.sh
```

## Image Quality
```bash
# Clip Score
python3 calculate_clip_dir.py > clip_score.txt

# IS
python3 IS_score.py > IS_score.txt

# Pick Score
# Create a separate Conda environment and install all dependencies as described in the PickScore repository:
# https://github.com/yuvalkirstain/PickScore
# After setting up, return to the MoDM directory and run
python3 pick_score.py > pick_score.txt

```

## Citation

If this work is helpful, please cite as:

```bibtex
@inproceedings{xia2026modm,
  author    = {Yuchen Xia and Divyam Sharma and Yichao Yuan and Souvik Kundu and Nishil Talati},
  title     = {MoDM: Efficient Serving for Image Generation via Mixture-of-Diffusion Models},
  booktitle = {Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1 (ASPLOS '26)},
  year      = {2026},
  month     = mar,
  address   = {Pittsburgh, PA, USA},
  publisher = {Association for Computing Machinery},
}
```

## Disclaimer

This “research quality code” is for Non-Commercial purposes and provided by the contributors “As Is” without any express or implied warranty of any kind. The organizations (University of Michigan or Intel) involved do not own the rights to the data sets used or generated and do not confer any rights to it. The organizations (University of Michigan or Intel) do not warrant or assume responsibility for the accuracy or completeness of any information, text, graphics, links or other items within the code. A thorough security or ethical review has not been performed on this code. Additionally, this repository may contain components that are out of date or contain known security vulnerabilities.