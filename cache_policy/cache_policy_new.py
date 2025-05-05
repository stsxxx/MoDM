import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionXLPipeline

from huggingface_hub import login


import pandas as pd
import math
from typing import List, Tuple, Any
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from scheduling_euler_discrete import EulerDiscreteScheduler
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

cache_size = 300000

# class KMinHeapCache:
#     def __init__(self, max_size):
#         self.heap = []  # Min-heap for (score, index)
#         self.item_map = {}  # Maps index to (score, index) for fast updates
#         self.max_size = max_size

#     def compute_lcbfu_score(self, f_i, k_i):
#         return f_i * k_i

#     def insert(self, index, f_i, k_i):
#         score = self.compute_lcbfu_score(f_i, k_i)
#         # Check if the index already exists
#         if index in self.item_map:
#             self.update_score(index, f_i, k_i)
#         else:
#             # Insert into heap and map
#             entry = (score, index)
#             heapq.heappush(self.heap, entry)
#             self.item_map[index] = entry

#         # Evict if necessary
#         if len(self.item_map) > self.max_size:
#             self.evict()

#     def update_score(self, index, f_i, k_i):
#         # Mark the old entry as invalid by removing it from the dictionary
#         if index in self.item_map:
#             del self.item_map[index]

#         # Insert the updated entry
#         self.insert(index, f_i, k_i)

#     def evict(self):
#         while self.heap:
#             score, index = heapq.heappop(self.heap)
#             # Check if the index is still valid
#             if index in self.item_map and self.item_map[index] == (score, index):
#                 del self.item_map[index]  # Remove from the map
#                 return  # Evict only one item

#     def retrieve(self, k_optimal):
#         # Example: Find the largest K ≤ K_optimal
#         for score, index in sorted(self.heap, reverse=True):
#             # For simplicity, assume index points to an external list with (f_i, k_i)
#             _, k_i = self.item_map[index]
#             if k_i <= k_optimal:
#                 return index
#         return None  # Fallback to vanilla model

# # Example Usage
# cache = KMinHeapCache(max_size=cache_size)

# for i in range(cache_size):
#     k = i % 5
#     cache.insert(i, f_i=0, k_i= (k + 1) * 5)
# # Insert items using indices
#   # Representing "item_1" with index 0
# cache.insert(1, f_i=30, k_i=20)  # Representing "item_2" with index 1

# # Update item at index 0 after a hit
# cache.update_score(0, f_i=60, k_i=25)  # New frequency increases the score

# # Evict items when the cache is full
# cache.insert(2, f_i=40, k_i=15)  # Representing "item_3" with index 2


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
filtered_df = metadata_df[(metadata_df['part_id'] >= 1) & (metadata_df['part_id'] <= 1520)].sort_values(by='timestamp')

    
if 'timestamp' in filtered_df.columns:
    # print("First few entries in the 'timestamp' column:")
    # print(metadata_df['timestamp'].head())
    
    # Optionally sort by date
    # metadata_df['timestamp'] = pd.to_datetime(metadata_df['timestamp'])  # Ensure it's in datetime format

    start_time = filtered_df['timestamp'].iloc[0]
    
    # Add a new column for seconds relative to the start time
    filtered_df['seconds_from_start'] = (filtered_df['timestamp'] - start_time).dt.total_seconds()

    # Display the modified DataFrame
    print(filtered_df[['timestamp', 'seconds_from_start']].head())
    # print("Sorted DataFrame by timestamp:")
    # print(sorted_df.head())
else:
    print("No timestamp column found in the DataFrame.")
    
    
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
num_batches = (100000 + batch_size - 1) // batch_size  # Calculate number of batches

# Initialize tensor for embeddings

text_embeddings = []
image_embeddings = []
embedding_map = [i for i in range(100000)]
retrieve_time = [0 for _ in range(100000)]
for batch_idx in tqdm(range(num_batches)):
    # Select batch slice
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, 100000)
    batch = filtered_df.iloc[batch_start:batch_end]

    # Process prompts (text)
    prompts = batch['prompt'].tolist()
    texts = processor(text=prompts, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)

    # Process images
    image_paths = [f"/data6/stilex/diffusionDB/{row['image_name']}" for _, row in batch.iterrows()]
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    # Process both text and image embeddings in the same batch
    image_tensors = processor(images=images, return_tensors="pt").to(device)

    # Initialize lists to hold batch embeddings for both text and images

    batch_image_embeddings = []

    with torch.no_grad():
        # Get image embeddings
        image_embeddings_batch = model.get_image_features(**image_tensors).cpu()  # Image model inference
        batch_image_embeddings.append(image_embeddings_batch)
        # print(image_embeddings_batch.shape)


    # Concatenate embeddings for the batch (both text and image)

    batch_image_embeddings = torch.cat(batch_image_embeddings, dim=0)

    # Append the concatenated embeddings to the respective lists

    image_embeddings.append(batch_image_embeddings)

# Final concatenation of all text and image embeddings

final_image_embeddings = torch.cat(image_embeddings, dim=0)

# Path to save the embeddings
save_path = "final_image_embeddings.pt"

# Save the tensor
torch.save(final_image_embeddings.cpu(), save_path)
print(f"Final image embeddings saved to {save_path}")

# print(final_text_embeddings.shape)
# print(final_text_embeddings[0])

# l2_norm = torch.linalg.norm(final_text_embeddings[0])

# # Check if it's normalized
# is_normalized = torch.isclose(l2_norm, torch.tensor(1.0), atol=1e-6)

# print(f"L2 Norm: {l2_norm}")
# print(f"Is normalized: {is_normalized}")

# embedding_dim = 768  # CLIP model output dimension
# index = faiss.IndexFlatL2(embedding_dim)  # Using FAISS for ANN search
# index.add(final_text_embeddings.numpy())  # Add text embeddings to FAISS index
final_image_embeddings = final_image_embeddings.to(device)
# Define k-values for top-k closest prompts
top_k_values = [10, 15, 20, 25]

# Calculate memory usage in bytes
memory_usage = final_image_embeddings.element_size() * final_image_embeddings.numel()

# Convert bytes to GB for easier reading
memory_usage_gb = memory_usage / (1024 ** 3)

print(f"Memory usage of final_image_embeddings: {memory_usage_gb:.4f} GB")

# Initialize distribution tracker
distribution = {k: 0 for k in top_k_values}

non_hit = 0

time_diffs = {}

# Iterate over the new requests
for i in tqdm(range(100000,  500000)):  # Adjust range if necessary
# for i in tqdm(range(100)):  # Adjust range if necessary

    prompt = [filtered_df['prompt'].iloc[i]]
    start_time = filtered_df['seconds_from_start'].iloc[i]
    # image_name = filtered_df['image_name'].iloc[i]
    image_path = f"/data6/stilex/diffusionDB/{filtered_df['image_name'].iloc[i]}"
    images = Image.open(image_path).convert("RGB")
    image_tensor = processor(images=images, return_tensors="pt").to(device)

    # Get the text embedding for the current prompt
    texts = processor(text=prompt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)
    
    with torch.no_grad():
        text_embedding = model.get_text_features(**texts)
        image_embedding = model.get_image_features(**image_tensor)  # Image model inference
        

    with torch.no_grad():
        # Normalize the embeddings for cosine similarity (CLIP model does this internally)
        text_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # Normalize text embedding
        image_norm = final_image_embeddings / final_image_embeddings.norm(dim=-1, keepdim=True)  # Normalize image embeddings

        similarity_scores = torch.matmul(text_norm, image_norm.T)  # This gives a tensor of cosine similarities

    similarity_scores = torch.clamp(similarity_scores, min=0)

    
    # Convert to numpy for easier handling
    similarity_scores = similarity_scores.cpu().numpy().flatten()
    ranked_indices = np.argsort(-similarity_scores)  # Sort by highest cosine similarity
    highest_clip_score_index = ranked_indices[0]  # Image with the highest CLIP score
    highest_clip_score = similarity_scores[highest_clip_score_index]
    real_index = embedding_map[highest_clip_score_index]
    
    if highest_clip_score.item() < 0.15:
        non_hit += 1
        final_image_embeddings = torch.cat([final_image_embeddings, image_embedding], dim=0)  
        embedding_map.append(i)  
        retrieve_time.append(0)    
        continue
    if highest_clip_score.item() >= 0.25:
        k = 25
        distribution[k] += 1
    elif highest_clip_score.item() >= 0.23:
        k = 20
        distribution[k] += 1
    elif highest_clip_score.item() >= 0.21:
        k = 15
        distribution[k] += 1
    elif highest_clip_score.item() >= 0.15:
        k = 10
        distribution[k] += 1

    retrieve_time[highest_clip_score_index] += 1
    
    start_time_cached = filtered_df['seconds_from_start'].iloc[real_index]
    
    time_diff = start_time - start_time_cached
    
    if time_diff in time_diffs:
        time_diffs[time_diff] += 1
    else:
        time_diffs[time_diff] = 1

retrieve_time = [count for count in retrieve_time if count != 0]
        
# Output the distribution of image CLIP scores in top-k closest prompts
print("number of nonhit:", non_hit)


# Convert time_diffs to a list of time differences (flattened data)
time_diffs_list = [time for time, count in time_diffs.items() for _ in range(count)]

# Create a histogram
plt.hist(time_diffs_list, bins=60, color='lightgreen', edgecolor='black')

# Add labels and title
plt.xlabel('Time Difference (seconds)')
plt.ylabel('Frequency')
plt.title('Histogram of Time Differences')

# Show the plot
plt.show()
plt.savefig('time_differences_histogram.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the plot to avoid overlap with the next plot

# Create a histogram for retrieve_time
plt.hist(retrieve_time, bins=60, color='skyblue', edgecolor='black')

# Add labels and title
plt.xlabel('Retrieve Count')
plt.ylabel('Frequency')
plt.title('Histogram of Retrieve Times')
# Show the plot
plt.show()
plt.savefig('retrieve_times_histogram.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the plot

print(distribution)
print(time_diffs)
print(retrieve_time)