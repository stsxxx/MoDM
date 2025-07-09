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
pip install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
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
```

## Throughput Experiments
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