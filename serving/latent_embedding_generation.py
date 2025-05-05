import torch
import time
import queue
import multiprocessing as mp
import pandas as pd
from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
)
from tqdm import tqdm
import os
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import re
import argparse

parser = argparse.ArgumentParser(description="cache directory")
parser.add_argument("--cache_directory", type=str, required=False, help="directory of cached images")
args = parser.parse_args()

image_directory = args.cache_directory
image_paths = [os.path.join(image_directory, img_file) for img_file in os.listdir(image_directory) if img_file.endswith(('.png', '.jpg', '.jpeg'))]

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

strengths = [ 0.9, 0.8, 0.7, 0.6, 0.5]
def extract_prompt(filename):
    """Extract prompt from filename by replacing underscores with spaces."""
    name = os.path.splitext(os.path.basename(filename))[0]  # Remove extension
    return re.sub(r'[_]+', ' ', name).strip()  # Convert underscores to spaces

cached_latents = None

num_chunks = 3
chunk_size = len(image_paths) // num_chunks

for j in range(num_chunks):
    start_idx = j * chunk_size
    end_idx = (j + 1) * chunk_size if j < num_chunks - 1 else len(image_paths)
    subset_paths = image_paths[start_idx:end_idx]

    cached_latents = None

    for path in tqdm(subset_paths, desc=f"Chunk {j}"):
        prompt = extract_prompt(path)
        init_image = Image.open(path).convert("RGB")
        for strength in strengths:
            latent = pipe.get_latents(
                prompt=prompt,
                image=init_image,
                strength=strength,
                guidance_scale=7.5,
                negative_prompt=None
            )
            if cached_latents is None:
                cached_latents = latent.cpu().clone()
            else:
                cached_latents = torch.cat((cached_latents, latent.cpu().clone()), dim=0)

    # Print and save
    print(f"Chunk {j}: final cached_latents size: {cached_latents.shape}")
    save_path = f"cached_latents_{j}.pt"
    torch.save(cached_latents, save_path)
    print(f"Saved cached latents to {save_path}")
