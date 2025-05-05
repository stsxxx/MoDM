import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionXLPipeline

from huggingface_hub import login


import pandas as pd
import math
from typing import List, Tuple, Any
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
# from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
# from scheduling_euler_discrete import EulerDiscreteScheduler
from datasets import load_dataset
import argparse
from PIL import Image
from torchvision import models, transforms
from torch.nn.functional import adaptive_avg_pool2d
import numpy as np
from scipy import linalg
import skimage.metrics
import os
from transformers import CLIPProcessor, CLIPModel
import queue
from queue import Queue
import faiss
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import heapq

if torch.cuda.is_available():
    print(f"GPU is available. Number of GPUs: {torch.cuda.device_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    print("GPU is not available.")

cache_size = 1500000

class KMinHeapCache:
    def __init__(self, max_size):
        self.heap = []  # Min-heap for (score, index)
        self.item_map = {}  # Maps index to (score, index) for fast updates
        self.max_size = max_size

    def compute_lcbfu_score(self, f_i, k_i):
        return f_i * k_i

    def insert(self, index, f_i, k_i):
        score = self.compute_lcbfu_score(f_i, k_i)
        # Check if the index already exists
        if index in self.item_map:
            self.update_score(index, f_i, k_i)
        else:
            # Insert into heap and map
            entry = (score, index)
            heapq.heappush(self.heap, entry)
            self.item_map[index] = entry

        # Evict if necessary
        if len(self.item_map) > self.max_size:
            self.evict()

    def update_score(self, index, f_i, k_i):
        # Mark the old entry as invalid by removing it from the dictionary
        if index in self.item_map:
            del self.item_map[index]

        # Insert the updated entry
        self.insert(index, f_i, k_i)

    def evict(self):
        while self.heap:
            score, index = heapq.heappop(self.heap)
            # Check if the index is still valid
            if index in self.item_map and self.item_map[index] == (score, index):
                del self.item_map[index]  # Remove from the map
                return  # Evict only one item

    def retrieve(self, k_optimal):
        # Example: Find the largest K ≤ K_optimal
        for score, index in sorted(self.heap, reverse=True):
            # For simplicity, assume index points to an external list with (f_i, k_i)
            _, k_i = self.item_map[index]
            if k_i <= k_optimal:
                return index
        return None  # Fallback to vanilla model

# Example Usage
cache = KMinHeapCache(max_size=cache_size)

for i in range(cache_size):
    k = i % 5
    cache.insert(i, f_i=0, k_i= (k + 1) * 5)
# Insert items using indices
  # Representing "item_1" with index 0
cache.insert(1, f_i=30, k_i=20)  # Representing "item_2" with index 1

# Update item at index 0 after a hit
cache.update_score(0, f_i=60, k_i=25)  # New frequency increases the score

# Evict items when the cache is full
cache.insert(2, f_i=40, k_i=15)  # Representing "item_3" with index 2


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_clip_score(images, text):

    inputs = processor(text=text, images=images, return_tensors="pt", truncation=True, padding=True,max_length=77)
    # print(inputs)

    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # print(outputs)
 
    logits_per_image = outputs.logits_per_image.max(dim=1)[0]
    # print(logits_per_image, logits_per_image.shape)  # 1,4
    # probs = logits_per_image.softmax(dim=1)
    # mean_score = torch.mean(logits_per_image,dim=0)
    # print(f"average CLIP:{mean_score}")
    return logits_per_image



k = 10
num_inference_steps = 50
strength = 1 - (k / num_inference_steps)

metadata_df = pd.read_parquet('/home/stilex/metadata.parquet')
filtered_df = metadata_df[(metadata_df['part_id'] >= 1) & (metadata_df['part_id'] <= 1020)].sort_values(by='timestamp')
timestamps = filtered_df['timestamp']
# for timestamp in timestamps:
#     print(timestamp)
# for i in range(min(10000, len(filtered_df))):
#     prompt = [filtered_df['prompt'].iloc[i]]
#     image_name = filtered_df['image_name'].iloc[i]  # No need to wrap in a list here
#     image_path = f"/data6/stilex/diffusionDB/{image_name}"  # f-string for interpolation
#     image = Image.open(image_path).convert("RGB")
#     texts = processor(text=prompt, return_tensors="pt",truncation=True, padding=True,max_length=77).to(device)
#     with torch.no_grad():
#         embedding = model.get_text_features(**texts).cpu()
#     if i == 0:
#         text_embeddings = embedding.clone()
#     else:
#         text_embeddings = torch.cat((text_embeddings,embedding), dim=0)
        
# Batch size
batch_size = 128  # Adjust based on memory constraints
num_batches = (20000 + batch_size - 1) // batch_size  # Calculate number of batches

# Initialize tensor for embeddings

text_embeddings = []
image_embeddings = []


for batch_idx in tqdm(range(num_batches)):
    # Select batch slice
    batch_start = 100000 + batch_idx * batch_size
    batch_end = min(100000 + (batch_idx + 1) * batch_size, 120000)
    batch = filtered_df.iloc[batch_start:batch_end]

    # Process prompts (text)
    prompts = batch['prompt'].tolist()
    texts = processor(text=prompts, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)

    # Process images
    # image_paths = [f"/data6/stilex/diffusionDB/{row['image_name']}" for _, row in batch.iterrows()]
    # images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    # # Process both text and image embeddings in the same batch
    # image_tensors = processor(images=images, return_tensors="pt").to(device)

    # Initialize lists to hold batch embeddings for both text and images
    batch_text_embeddings = []
    batch_image_embeddings = []

    with torch.no_grad():
        # Get image embeddings
        # image_embeddings_batch = model.get_image_features(**image_tensors).cpu()  # Image model inference
        # batch_image_embeddings.append(image_embeddings_batch)
        # print(image_embeddings_batch.shape)

        # Get text embeddings
        text_embeddings_batch = model.get_text_features(**texts).cpu()  # Text model inference
        batch_text_embeddings.append(text_embeddings_batch)

    # Concatenate embeddings for the batch (both text and image)
    batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)
    # batch_image_embeddings = torch.cat(batch_image_embeddings, dim=0)

    # Append the concatenated embeddings to the respective lists
    text_embeddings.append(batch_text_embeddings)
    # image_embeddings.append(batch_image_embeddings)

# Final concatenation of all text and image embeddings
final_text_embeddings = torch.cat(text_embeddings, dim=0)
# final_image_embeddings = torch.cat(image_embeddings, dim=0)
# print(final_text_embeddings.shape)
final_text_embeddings_np = final_text_embeddings.cpu().numpy()
# final_image_embeddings_np = final_image_embeddings.cpu().numpy()

save_path_npy = "final_text_embeddings_20000.npy"
np.save(save_path_npy, final_text_embeddings_np)
print(f"Final text embeddings saved to {save_path_npy}")
# save_path_npy = "final_image_embeddings_100000.npy"
# np.save(save_path_npy, final_image_embeddings_np)
# print(f"Final image embeddings saved to {save_path_npy}")
# print(final_text_embeddings[0])
# Path to save the embeddings
# save_path = "final_text_embeddings.pt"

# # Save the tensor
# torch.save(final_text_embeddings.cpu(), save_path)
# print(f"Final text embeddings saved to {save_path}")
# l2_norm = torch.linalg.norm(final_text_embeddings[0])

# # Check if it's normalized
# is_normalized = torch.isclose(l2_norm, torch.tensor(1.0), atol=1e-6)

# print(f"L2 Norm: {l2_norm}")
# print(f"Is normalized: {is_normalized}")



























# embedding_dim = 768  # CLIP model output dimension
# index = faiss.IndexFlatL2(embedding_dim)  # Using FAISS for ANN search
# index.add(final_text_embeddings.numpy())  # Add text embeddings to FAISS index
# final_image_embeddings = final_image_embeddings.to(device)
# # Define k-values for top-k closest prompts
# top_k_values = [1, 5, 10, 15, 20, 25, 30, 100]

# # Calculate memory usage in bytes
# memory_usage = final_image_embeddings.element_size() * final_image_embeddings.numel()

# # Convert bytes to GB for easier reading
# memory_usage_gb = memory_usage / (1024 ** 3)

# print(f"Memory usage of final_image_embeddings: {memory_usage_gb:.4f} GB")

# # Initialize distribution tracker
# distribution = {k: 0 for k in top_k_values}
# distribution['out'] = 0
# non_hit = 0
# top_k_max_similarities_diff = {k: 0 for k in top_k_values}


# seed = 42 #any
# generator = torch.Generator(device).manual_seed(seed)

# pipe_xl = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0",
#     torch_dtype=torch.float16  # Use fp16 for efficiency if on GPU
# )
# pipe_xl = pipe_xl.to("cuda")  # Move to GPU if available

# img_quality_og = {}
# img_quality_img = {}
# img_quality_text = {}

# # Iterate over the new requests
# for i in tqdm(range(1000000,  1020000)):  # Adjust range if necessary
# # for i in tqdm(range(100)):  # Adjust range if necessary

#     prompt = [filtered_df['prompt'].iloc[i]]
#     # image_name = filtered_df['image_name'].iloc[i]
    
#     # Get the text embedding for the current prompt
#     texts = processor(text=prompt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)
    
#     with torch.no_grad():
#         text_embedding = model.get_text_features(**texts).cpu()
#     # Step 1: Use FAISS to find top-k closest prompts
#     query_embedding = text_embedding.numpy().reshape(1, -1)
#     distances, indices = index.search(query_embedding, k=max(top_k_values))
    
#     # Get the top-k closest prompt images (indices of top-k closest prompts)
#     closest_images = [filtered_df.iloc[indices[0][j]]['image_name'] for j in range(max(top_k_values))]
#     closest_prompt = filtered_df.iloc[indices[0][0]]['prompt']
    
#     image_path = f"/data6/stilex/diffusionDB/{filtered_df['image_name'].iloc[indices[0][0]]}"
#     image = Image.open(image_path).convert("RGB")
#     height = filtered_df['height'].iloc[indices[0][0]]
#     width = filtered_df['width'].iloc[indices[0][0]]
    
#     closest_texts = processor(text=closest_prompt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)
#     with torch.no_grad():
#         closest_text_embedding = model.get_text_features(**closest_texts)
    
#     text_embedding = text_embedding.to(device)
#     # Step 2: Compute CLIP scores between new prompt and all image embeddings
#     with torch.no_grad():
#         # Normalize the embeddings for cosine similarity (CLIP model does this internally)
#         text_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # Normalize text embedding
#         image_norm = final_image_embeddings / final_image_embeddings.norm(dim=-1, keepdim=True)  # Normalize image embeddings
#         closest_text_norm = closest_text_embedding / closest_text_embedding.norm(dim=-1, keepdim=True) 
#         # Calculate cosine similarity (dot product of normalized embeddings)
#         similarity_scores = torch.matmul(text_norm, image_norm.T)  # This gives a tensor of cosine similarities
#         text_similarity_scores = torch.matmul(text_norm, closest_text_norm.T)
#     similarity_scores = torch.clamp(similarity_scores, min=0)
#     text_similarity_scores = torch.clamp(text_similarity_scores, min=0)
    
    
#     # Convert to numpy for easier handling
#     similarity_scores = similarity_scores.cpu().numpy().flatten()
#     text_similarity_scores = text_similarity_scores.cpu().numpy().flatten()
#     if text_similarity_scores.item() < 0.65:
#         non_hit += 1
#         continue
#     if text_similarity_scores.item() > 0.95:
#         strength = 0.5
#         k = 25
#     elif text_similarity_scores.item() > 0.9:
#         strength = 0.6
#         k = 20
#     elif text_similarity_scores.item() > 0.85:
#         strength = 0.7
#         k = 15
        
#     elif text_similarity_scores.item() > 0.75:
#         strength = 0.8
#         k = 10
        
#     elif text_similarity_scores.item() > 0.65:
#         strength = 0.9
#         k = 5
        
#     ranked_indices = np.argsort(-similarity_scores)  # Sort by highest cosine similarity
#     # Dictionary to store results
    
#     clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt[0])[:300]

#     # Step 3: Check if the image with highest CLIP score is in top-k closest prompts
#     highest_clip_score_index = ranked_indices[0]  # Image with the highest CLIP score
#     highest_clip_score = similarity_scores[highest_clip_score_index]
#     print("hit")
#     print("k:", k)
#     print("image similarity ann:", similarity_scores[indices[0][0]])
#     print("highest image similarity:", highest_clip_score)
#     image_path = f"/data6/stilex/diffusionDB/{filtered_df['image_name'].iloc[highest_clip_score_index]}"
#     image1 = Image.open(image_path).convert("RGB")
#     height1 = filtered_df['height'].iloc[highest_clip_score_index]
#     width1 = filtered_df['width'].iloc[highest_clip_score_index]
    
#     if k not in img_quality_img:
#         img_quality_img[k] = []
#         img_quality_og[k] = []
#         img_quality_text[k] = []
        
#     model_ouputs_xl = pipe_xl(prompt=prompt,
#                 image=image1,
#                 strength=1,
#                 guidance_scale=7.5,
#                 negative_prompt=None,
#                 original_size = (height1, width1),
#                 target_size = (height1, width1),
#                 generator=generator)
#     clip = get_clip_score(model_ouputs_xl.images,prompt)
#     clip = clip.item()
#     img_quality_og[k].append(clip)
#     try:
#         filename = f"/data6/stilex/img_similarity/og/{clean_prompt}.png"    
#         model_ouputs_xl.images[0].save(filename)        
#     except Exception as e:
#         print(f"Failed to save image: {e}")
#         print("image name:", prompt[0])      
    
#     model_ouputs_xl = pipe_xl(prompt=prompt,
#                 image=image,
#                 strength=strength,
#                 guidance_scale=7.5,
#                 negative_prompt=None,
#                 original_size = (height, width),
#                 target_size = (height, width),
#                 generator=generator)
#     clip = get_clip_score(model_ouputs_xl.images,prompt)
#     clip = clip.item()
#     img_quality_text[k].append(clip)
#     try:
#         filename = f"/data6/stilex/img_similarity/text/{clean_prompt}_{k}.png"    
#         model_ouputs_xl.images[0].save(filename)        
#     except Exception as e:
#         print(f"Failed to save image: {e}")
#         print("image name:", prompt[0])       
        
        
#     model_ouputs_xl = pipe_xl(prompt=prompt,
#                 image=image1,
#                 strength=strength,
#                 guidance_scale=7.5,
#                 negative_prompt=None,
#                 original_size = (height1, width1),
#                 target_size = (height1, width1),
#                 generator=generator)
#     clip = get_clip_score(model_ouputs_xl.images,prompt)
#     clip = clip.item()
#     img_quality_img[k].append(clip)
#     try:
#         filename = f"/data6/stilex/img_similarity/img/{clean_prompt}_{k}.png"    
#         model_ouputs_xl.images[0].save(filename)        
#     except Exception as e:
#         print(f"Failed to save image: {e}")
#         print("image name:", prompt[0])       
#     # print("highest:", highest_clip_score)
#         # Loop through the top-k values
#     # print(highest_clip_score)

    
        
# # Output the distribution of image CLIP scores in top-k closest prompts
# print("number of nonhit:", non_hit)

# print("img")
# for k, values in img_quality_img.items():
#     print(f"k: {k}, number: {len(values)}, Clip score: {values}")
#     print("avg score:", sum(values)/len(values) )
    
# print("text")
# for k, values in img_quality_text.items():
#     print(f"k: {k}, number: {len(values)}, Clip score: {values}")
#     print("avg score:", sum(values)/len(values) )

# print("og")
# for k, values in img_quality_og.items():
#     print(f"k: {k}, Clip score: {values}")
#     print("avg score:", sum(values)/len(values) )
