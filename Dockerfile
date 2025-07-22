FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python 3.10 and tools
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-distutils python3-pip git curl && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    python -m pip install --upgrade pip

# Set workdir
WORKDIR /workspace

# Copy your MoDM project files
COPY . .

# Install Python dependencies
RUN pip install https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu118/torchvision-0.16.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=033712f65d45afe806676c4129dfe601ad1321d9e092df62b15847c02d4061dc
RUN pip install https://download.pytorch.org/whl/cu118/torchaudio-2.1.0%2Bcu118-cp310-cp310-linux_x86_64.whl#sha256=cdfd0a129406155eee595f408cafbb92589652da4090d1d2040f5453d4cae71f
RUN pip install -r requirements.txt
RUN ./replace_pipelines.sh 
RUN python3 DiffusionDB_parquet.py
RUN chmod +x run.sh


# Optional: set default command
# CMD []