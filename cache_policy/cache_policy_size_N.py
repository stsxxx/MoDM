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

cache_size = 10000
init_size = 10000
maximum_hit = 10000
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

class KMinHeapCache:
    def __init__(self, max_size, initial_embeddings):
        self.heap = []  # Min-heap for (LCBFU score, (index, k_i))
        self.item_map = {}  # Maps (index, k_i) to (score, index, k_i) for fast lookups
        self.index_map = {}  # Maps index to a set of remaining k_i values for eviction check
        self.max_size = max_size
        
        # Define the k_i values in order
        k_values = [5, 10, 15, 20, 25]

        # Initialize cache with existing embeddings
        for index in range(len(initial_embeddings)):
            self.index_map[index] = set(k_values)  # All k_i values are initially present for each index
            
            f_i = 0  # Default initial frequency
            for k_i in k_values:
                score = self.compute_lcbfu_score(f_i, k_i)
                entry = (score, (index, k_i))  # Store both index and k_i
                heapq.heappush(self.heap, entry)
                self.item_map[(index, k_i)] = entry  # Use tuple key (index, k_i)

    def compute_lcbfu_score(self, f_i, k_i):
        return f_i * k_i

    def insert(self, index, f_i, k_i):
        k_values = [5, 10, 15, 20, 25]
        score = self.compute_lcbfu_score(f_i, k_i)
        if (index, k_i) in self.item_map:
            self.update_score(index,  k_i)
        else:
            entry = (score, (index, k_i))
            heapq.heappush(self.heap, entry)
            self.item_map[(index, k_i)] = entry
        if index not in self.index_map:
            self.index_map[index] = set(k_values)
        if len(self.item_map) > self.max_size:
            self.evict()

    def update_score(self, index, k_i):
        if (index, k_i) in self.item_map:
            old_score, _ = self.item_map[(index, k_i)]
            new_score = old_score + k_i  # Increment score directly

            # Remove old entry
            del self.item_map[(index, k_i)]

            # Reinsert with updated score
            entry = (new_score, (index, k_i))
            heapq.heappush(self.heap, entry)
            self.item_map[(index, k_i)] = entry

    def evict(self):
        # Initialize variable to store evicted index
        evicted_index = None

        while self.heap:
            score, (index, k_i) = heapq.heappop(self.heap)
            if (index, k_i) in self.item_map and self.item_map[(index, k_i)] == (score, (index, k_i)):
                # Remove the (index, k_i) from the cache
                del self.item_map[(index, k_i)]

                # Update the set of remaining k_i values for the corresponding index
                self.index_map[index].remove(k_i)

                # If all k_i values for this index are evicted, remove the entire embedding
                if not self.index_map[index]:
                    del self.index_map[index]  # Evict the entire index
                    evicted_index = index  # Store the evicted index
                    print(evicted_index)
                    break  # Stop eviction after removing one index (adjust this based on your policy)

        return evicted_index

    def retrieve(self, k_optimal, index_to_search):
        # Find the best match for a specific index where k_i ≤ k_optimal and closest to k_optimal
        candidates = [
            (score, (index, k_i)) for score, (index, k_i) in self.heap
            if index == index_to_search and k_i <= k_optimal
        ]
        
        # Find the candidate with k_i closest to k_optimal
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[1][1])
            score, (index, k_i) = best_candidate
        
            # Update the score of the best candidate
            self.update_score(index, k_i)
            return best_candidate
        return None

    

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
filtered_df = metadata_df[(metadata_df['part_id'] >= 1) & (metadata_df['part_id'] <= 2000)].sort_values(by='timestamp')
cached_df=filtered_df.iloc[100000-init_size:100000]
core_requests = cached_df['prompt'].tolist() 
    
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
num_batches = (init_size + batch_size - 1) // batch_size  # Calculate number of batches

# Initialize tensor for embeddings

text_embeddings = []
image_embeddings = []
embedding_map = [i for i in range(init_size)]
retrieve_time = [0 for _ in range(init_size)]
# for batch_idx in tqdm(range(num_batches)):
#     # Select batch slice
#     batch_start = batch_idx * batch_size + 99000
#     batch_end = min((batch_idx + 1) * batch_size + 99000, 100000)
#     batch = filtered_df.iloc[batch_start:batch_end]

#     # Process prompts (text)
#     prompts = batch['prompt'].tolist()
#     texts = processor(text=prompts, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)

#     # Process images
#     image_paths = [f"/data6/stilex/diffusionDB/{row['image_name']}" for _, row in batch.iterrows()]
#     images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

#     # Process both text and image embeddings in the same batch
#     image_tensors = processor(images=images, return_tensors="pt").to(device)

#     # Initialize lists to hold batch embeddings for both text and images

#     batch_image_embeddings = []

#     with torch.no_grad():
#         # Get image embeddings
#         image_embeddings_batch = model.get_image_features(**image_tensors).cpu()  # Image model inference
#         batch_image_embeddings.append(image_embeddings_batch)
#         # print(image_embeddings_batch.shape)
batch_size = 128  # Adjust based on memory constraints
num_batches = (init_size + batch_size - 1) // batch_size  # Calculate number of batches

# Initialize tensor for embeddings

text_embeddings = []
image_embeddings = []


for batch_idx in tqdm(range(num_batches)):
    # Select batch slice
    batch_start = (100000 - init_size) + batch_idx * batch_size
    batch_end = min((100000 - init_size) + (batch_idx + 1) * batch_size, 100000)
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
final_text_embeddings = final_text_embeddings.cpu()
# final_image_embeddings = torch.cat(image_embeddings, dim=0)
# print(final_text_embeddings.shape)


#     # Concatenate embeddings for the batch (both text and image)

#     batch_image_embeddings = torch.cat(batch_image_embeddings, dim=0)

#     # Append the concatenated embeddings to the respective lists

#     image_embeddings.append(batch_image_embeddings)

# # Final concatenation of all text and image embeddings

# final_image_embeddings = torch.cat(image_embeddings, dim=0)

# Path to save the embeddings
save_path = f"final_text_embeddings_{init_size}.pt"
torch.save(final_text_embeddings, save_path)


cache = KMinHeapCache(max_size=cache_size * 5, initial_embeddings=final_text_embeddings)

# Save the tensor
# torch.save(final_image_embeddings.cpu(), save_path)
# print(f"Final image embeddings saved to {save_path}")

# print(final_text_embeddings.shape)
# print(final_text_embeddings[0])

# l2_norm = torch.linalg.norm(final_text_embeddings[0])

# # Check if it's normalized
# is_normalized = torch.isclose(l2_norm, torch.tensor(1.0), atol=1e-6)
embedding_dim = 768  # CLIP model output dimension
index = faiss.IndexFlatL2(embedding_dim)  # Using FAISS for ANN search
index.add(final_text_embeddings.numpy())  # Add text embeddings to FAISS index

cache_map = [i for i in range(cache_size)]
# print(f"L2 Norm: {l2_norm}")
# print(f"Is normalized: {is_normalized}")

# embedding_dim = 768  # CLIP model output dimension
# index = faiss.IndexFlatL2(embedding_dim)  # Using FAISS for ANN search
# index.add(final_text_embeddings.numpy())  # Add text embeddings to FAISS index

# Define k-values for top-k closest prompts
top_k_values = [5, 10, 15, 20, 25]

# Calculate memory usage in bytes
memory_usage = final_text_embeddings.element_size() * final_text_embeddings.numel()

# Convert bytes to GB for easier reading
memory_usage_gb = memory_usage / (1024 ** 3)

print(f"Memory usage of final_image_embeddings: {memory_usage_gb:.4f} GB")
final_text_embeddings = final_text_embeddings.numpy()
# Initialize distribution tracker
distribution = {k: 0 for k in top_k_values}

def evict_from_faiss(index, embeddings, remove_index):
    """
    Removes the embedding at remove_index from the FAISS index.
    
    Args:
        index (faiss.Index): The FAISS index.
        embeddings (np.ndarray): The array of embeddings.
        remove_index (int): The index of the embedding to remove.
        
    Returns:
        faiss.Index: A new FAISS index without the removed embedding.
    """
    # Create a mask to exclude the embedding to be removed
    mask = np.ones(len(embeddings), dtype=bool)
    mask[remove_index] = False
    
    # Filter the embeddings to keep only the ones that are not removed
    filtered_embeddings = embeddings[mask]
    
    # Create a new FAISS index and add the remaining embeddings
    new_index = faiss.IndexFlatL2(embeddings.shape[1])
    new_index.add(filtered_embeddings)
    
    return new_index, filtered_embeddings


non_hit = 0
num_5hit =0
time_diffs = {}
similarity_count = {}
# Iterate over the new requests
for i in tqdm(range(100000,  120000)):  # Adjust range if necessary
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
        embeddings = model.get_text_features(**texts).cpu()
        
    query_embedding = embeddings.numpy().reshape(1, -1)
    distances, indices = index.search(query_embedding, k=1)
    
    closest_prompt = core_requests[indices[0][0]]
    closest_texts = processor(text=closest_prompt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)
    
    with torch.no_grad():
        closest_text_embedding = model.get_text_features(**closest_texts)
    embeddings = embeddings.to(device)
    with torch.no_grad():
        text_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        closest_text_norm = closest_text_embedding / closest_text_embedding.norm(dim=-1, keepdim=True) 
        text_similarity_scores = torch.matmul(text_norm, closest_text_norm.T)
    text_similarity_scores = torch.clamp(text_similarity_scores, min=0)
    embeddings = embeddings.cpu()
    
    if text_similarity_scores > 0.65:
        # print("hit")
        if text_similarity_scores > 0.95:
            k = 4
            closest_index = 25
        elif text_similarity_scores > 0.9:
            k = 3
            closest_index = 20
        elif text_similarity_scores > 0.85:
            k = 2
            closest_index = 15
        elif text_similarity_scores > 0.75:
            k = 1
            closest_index = 10
        elif text_similarity_scores > 0.65:
            k = 0
            closest_index = 5

        best_candidate = cache.retrieve(closest_index, indices[0][0])
        if best_candidate is not None:
            distribution[best_candidate[1][1]] += 1 
            # core_embeddings = embeddings.clone()
            # core_embeddings = core_embeddings.numpy()
            # while len(cache.item_map) + 5  > cache.max_size:
            #     evicted_index = cache.evict()
            #     if evicted_index is not None:
            #         del core_requests[evicted_index]
            #         index , final_text_embeddings = evict_from_faiss(index, final_text_embeddings, evicted_index)
            # num_embeddings = index.ntotal
            # core_requests.append(prompt[0]) 
            # index.add(core_embeddings)  # Add text embeddings to FAISS index
            # final_text_embeddings = np.concatenate((final_text_embeddings, core_embeddings), axis=0)
            
            # for k in  top_k_values:
            #     cache.insert( num_embeddings, 0, k)
        else:
            non_hit += 1
            core_embeddings = embeddings.clone()
            core_embeddings = core_embeddings.numpy()
            while len(cache.item_map) + 5  > cache.max_size:
                evicted_index = cache.evict()
                if evicted_index is not None:
                    del core_requests[evicted_index]
                    index , final_text_embeddings = evict_from_faiss(index, final_text_embeddings, evicted_index)
            num_embeddings = index.ntotal
            core_requests.append(prompt[0]) 
            index.add(core_embeddings)  # Add text embeddings to FAISS index
            final_text_embeddings = np.concatenate((final_text_embeddings, core_embeddings), axis=0)
            
            for k in  top_k_values:
                cache.insert( num_embeddings, 0, k)
            
    else:
        non_hit += 1
        core_embeddings = embeddings.clone()
        core_embeddings = core_embeddings.numpy()
        while len(cache.item_map) + 5  > cache.max_size:
            evicted_index = cache.evict()
            if evicted_index is not None:
                del core_requests[evicted_index]
                index , final_text_embeddings = evict_from_faiss(index, final_text_embeddings, evicted_index)
        num_embeddings = index.ntotal
        core_requests.append(prompt[0]) 
        index.add(core_embeddings)  # Add text embeddings to FAISS index
        final_text_embeddings = np.concatenate((final_text_embeddings, core_embeddings), axis=0)
        
        for k in  top_k_values:
            cache.insert( num_embeddings, 0, k)
            
    # with torch.no_grad():
    #     text_embedding = model.get_text_features(**texts)
    #     image_embedding = model.get_image_features(**image_tensor)  # Image model inference
        

    # with torch.no_grad():
    #     # Normalize the embeddings for cosine similarity (CLIP model does this internally)
    #     text_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)  # Normalize text embedding
    #     image_norm = final_image_embeddings / final_image_embeddings.norm(dim=-1, keepdim=True)  # Normalize image embeddings

    #     similarity_scores = torch.matmul(text_norm, image_norm.T)  # This gives a tensor of cosine similarities

    # similarity_scores = torch.clamp(similarity_scores, min=0)

    
    # # Convert to numpy for easier handling
    # similarity_scores = similarity_scores.cpu().numpy().flatten()
    # ranked_indices = np.argsort(-similarity_scores)  # Sort by highest cosine similarity
    # highest_clip_score_index = ranked_indices[0]  # Image with the highest CLIP score
    # highest_clip_score = similarity_scores[highest_clip_score_index]
    # real_index = embedding_map[highest_clip_score_index]
    # highest_score = round(highest_clip_score.item(), 2)
    
    
# Output the distribution of image CLIP scores in top-k closest prompts
print("number of nonhit:", non_hit)
# print("number of 5 hit:", num_5hit)



print(distribution)


