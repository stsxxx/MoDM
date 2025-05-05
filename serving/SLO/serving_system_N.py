import torch
import time
import queue
import torch.multiprocessing as mp
import pandas as pd
from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
)
import faiss
import os
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import re
import heapq
from tqdm import tqdm
import argparse
# Cache Data Structure

parser = argparse.ArgumentParser(description="model selection")
parser.add_argument("--large_model", type=str, default='sd3.5',required=False, help="which large model you wanna use")
parser.add_argument("--num_req", type=int, default=1000, required=True, help="number of requests")
parser.add_argument("--cache_size", type=int, default=10000, help="cache size")
parser.add_argument("--cache_directory", type=str, required=False, help="directory of cached images")
parser.add_argument("--image_directory", type=str, required=False, help="directory of generated images")
parser.add_argument("--req_rate", type=str, required=True, help="request rate / minute")
args = parser.parse_args()

directory = args.image_directory
os.makedirs(directory, exist_ok=True)
print(f"Directory '{directory}' created successfully.")


def extract_prompt(filename):
    """Extract prompt from filename by replacing underscores with spaces."""
    name = os.path.splitext(os.path.basename(filename))[0]  # Remove extension
    return re.sub(r'[_]+', ' ', name).strip()  # Convert underscores to spaces

def generate_rapidly_increasing_seconds_from_start(num_requests, min_rate=2, max_rate=9, duration=100*60):
    """
    Generate request timestamps with a smoothly increasing request rate using a sigmoid function.

    Args:
        num_requests (int): Total number of requests.
        min_rate (float): Minimum request rate (req/min).
        max_rate (float): Maximum request rate (req/min).
        duration (int): Total duration in seconds.

    Returns:
        np.array: Array of `seconds_from_start` values.
    """
    # Convert rates to requests per second
    min_rate_per_sec = min_rate / 60
    max_rate_per_sec = max_rate / 60

    # Adjust sigmoid input range to ensure full scaling from min_rate to max_rate
    x = np.linspace(-2, 6, num_requests)  # Increase range to get a full sigmoid transition

    # Apply sigmoid transformation
    sigmoid_growth = 1 / (1 + np.exp(-1.5 * x))

    # Normalize sigmoid values to scale precisely between min_rate_per_sec and max_rate_per_sec
    request_rates = min_rate_per_sec + (max_rate_per_sec - min_rate_per_sec) * sigmoid_growth

    # Compute interarrival times inversely proportional to the rate
    interarrival_times = 1 / np.maximum(request_rates, 1e-3)  # Avoid division by zero

    # Cumulatively sum interarrival times to get timestamps
    seconds_from_start = np.cumsum(interarrival_times)

    return seconds_from_start

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

def precompute_timesteps_for_labels_35(scheduler, labels, device, index):
    timesteps = []
    for label in labels:
        if label == 0:
            # For label 0, use full 50 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps.tolist())
        elif label == 1:
            # For label 1, use last 40 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device)
            timesteps.append(scheduler.timesteps[index:].tolist())         
        else:
            timesteps.append([])  

    return timesteps

class KMinHeapCache:
    def __init__(self, max_size, initial_embeddings, latents):
        self.heap = []  # Min-heap for (LCBFU score, (index, k_i))
        self.item_map = {}  # Maps (index, k_i) to (score, index, k_i, latent) for fast lookups
        self.index_map = {}  # Maps index to a set of remaining k_i values for eviction check
        self.max_size = max_size
        
        # Define the k_i values in order
        k_values = [5, 10, 15, 20, 25]

        # Initialize cache with existing embeddings
        for index, _ in enumerate(initial_embeddings):
            self.index_map[index] = set(k_values)  # All k_i values are initially present for each index
            
            f_i = 0  # Default initial frequency
            for idx, k_i in enumerate(k_values):
                score = self.compute_lcbfu_score(f_i, k_i)
                entry = (score, (index, k_i, latents[index * 5 + idx].share_memory_()))  # Store both index, k_i, and latent tensor
                heapq.heappush(self.heap, entry)
                self.item_map[(index, k_i)] = entry  # Use tuple key (index, k_i)

    def compute_lcbfu_score(self, f_i, k_i):
        return f_i * k_i

    def insert(self, index, f_i, k_i, latent):
        k_values = [5, 10, 15, 20, 25]
        score = self.compute_lcbfu_score(f_i, k_i)

        if (index, k_i) in self.item_map:
            self.update_score(index, k_i)
        else:
            entry = (score, (index, k_i, latent.share_memory_()))
            heapq.heappush(self.heap, entry)
            self.item_map[(index, k_i)] = entry

        if index not in self.index_map:
            self.index_map[index] = set(k_values)

        if len(self.item_map) > self.max_size:
            self.evict()

    def update_score(self, index, k_i):
        if (index, k_i) in self.item_map:
            old_score, (_, _, latent) = self.item_map[(index, k_i)]
            new_score = old_score + k_i  # Increment score directly

            # Remove old entry
            del self.item_map[(index, k_i)]

            # Reinsert with updated score
            entry = (new_score, (index, k_i, latent))
            heapq.heappush(self.heap, entry)
            self.item_map[(index, k_i)] = entry

    def evict(self):
        evicted_index = None

        while self.heap:
            score, (index, k_i, latent) = heapq.heappop(self.heap)
            if (index, k_i) in self.item_map and self.item_map[(index, k_i)] == (score, (index, k_i, latent)):
                del self.item_map[(index, k_i)]
                self.index_map[index].remove(k_i)

                if not self.index_map[index]:
                    del self.index_map[index]
                    evicted_index = index
                    print(evicted_index)
                    break  # Stop eviction after removing one index

        return evicted_index

    def retrieve(self, k_optimal, index_to_search):
        candidates = [
            (score, (index, k_i, latent)) for score, (index, k_i, latent) in self.heap
            if index == index_to_search and k_i <= k_optimal
        ]
        
        if candidates:
            best_candidate = max(candidates, key=lambda x: x[1][1])
            score, (index, k_i, latent) = best_candidate

            self.update_score(index, k_i)
            return best_candidate  # Includes latent tensor
        return None


# Function to load the Stable Diffusion 3.5 model
def load_model(device):
    return StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
    ).to(device)

# Request scheduler
def request_scheduler(req_queue,  selected_requests, start_time, index, cache, new_cache_queue, cached_requests, final_text_embeddings, k_values, processor, clip_model, log_file, worker_status):
    device = clip_model.device
    agg_k_distribution = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0} 
    minute = 0
    with open(log_file, "w") as f:
        f.write("timestamp,request_rate,throughput\n")
    last_check_time = time.time()
    last_check_time_queue = last_check_time
    request_count_per_min = 0
    size_of_queues = 0
    throughput = 0        
    for _, row in selected_requests.iterrows():
        while time.time() - start_time < row['seconds_from_start']:
            time.sleep(0.1)
            
            
        request_arrival_time = time.time()  # Start time of request processing

        row['start_time'] = request_arrival_time  # Add start_time to request
        
        while not new_cache_queue.empty():
            cache_data = new_cache_queue.get()  # Retrieves the dictionary from the queue
            new_cached_latents = cache_data['cached_latents']
            new_cached_prompt = cache_data['prompt']
            new_query_embedding = cache_data['query_embedding']
            while len(cache.item_map) + 5  > cache.max_size:
                evicted_index = cache.evict()
                if evicted_index is not None:
                    del cached_requests[evicted_index]
                    index , final_text_embeddings = evict_from_faiss(index, final_text_embeddings, evicted_index)

            
            num_embeddings = index.ntotal
            cached_requests.append(new_cached_prompt) 
            index.add(new_query_embedding)


            final_text_embeddings = np.concatenate((final_text_embeddings, new_query_embedding), axis=0)
            
            for idx, k in enumerate(k_values):
                cache.insert(num_embeddings , 0, k, new_cached_latents[idx])

        prompt = row['prompt']

        # Process the prompt embedding
        texts = processor(text=[prompt], return_tensors="pt", truncation=True, padding=True, max_length=77).to(clip_model.device)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**texts).cpu()
        query_embedding = text_embedding.numpy().reshape(1, -1)
  

        distances, indices = index.search(query_embedding, k=1)
        closest_prompt = cached_requests[indices[0][0]]
        closest_texts = processor(text=closest_prompt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)
        with torch.no_grad():
            closest_text_embedding = clip_model.get_text_features(**closest_texts)
        text_embedding = text_embedding.to(device)
        with torch.no_grad():
            text_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            closest_text_norm = closest_text_embedding / closest_text_embedding.norm(dim=-1, keepdim=True) 
            text_similarity_scores = torch.matmul(text_norm, closest_text_norm.T)
        text_similarity_scores = torch.clamp(text_similarity_scores, min=0)
        text_embedding = text_embedding.cpu()
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
            
            if best_candidate:
                score, (idex, k_i, latent) = best_candidate
                row['cached'] = True
                row['k'] = k_i
                row['latent'] = latent.clone().to(dtype=torch.float32).cpu()
                row['query_embedding'] = text_embedding.clone()
                req_queue.put(row.to_dict())
                agg_k_distribution[k_i] += 1
            else:
                row['cached'] = None
                row['k'] = None
                row['latent'] = None
                row['query_embedding'] = text_embedding.clone()
                req_queue.put(row.to_dict())
            
            
        else:
            row['cached'] = None
            row['k'] = None
            row['latent'] = None
            row['query_embedding'] = text_embedding.clone()
            req_queue.put(row.to_dict())

        print(agg_k_distribution)

        request_count_per_min += 1
        current_time = time.time()
        
        if current_time - last_check_time_queue >= 60:
            elapsed_time = current_time - last_check_time_queue
            minute += 1
            new_size_of_queues =  req_queue.qsize() 
            throughput = size_of_queues + request_count_per_min - new_size_of_queues
            size_of_queues = new_size_of_queues
            with open(log_file, "a") as f:
                f.write(f"{minute},{request_count_per_min / elapsed_time * 60},{throughput / elapsed_time * 60}\n")
            request_count_per_min = 0
            last_check_time_queue = current_time

    while not req_queue.empty():
        while not new_cache_queue.empty():
            cache_data = new_cache_queue.get()  # Retrieves the dictionary from the queue
            new_cached_latents = cache_data['cached_latents']
            new_cached_prompt = cache_data['prompt']
            new_query_embedding = cache_data['query_embedding']
            while len(cache.item_map) + 5  > cache.max_size:
                evicted_index = cache.evict()
                if evicted_index is not None:
                    del cached_requests[evicted_index]
                    index , final_text_embeddings = evict_from_faiss(index, final_text_embeddings, evicted_index)

            
            num_embeddings = index.ntotal
            cached_requests.append(new_cached_prompt) 
            index.add(new_query_embedding)


            final_text_embeddings = np.concatenate((final_text_embeddings, new_query_embedding), axis=0)
            
            for idx, k in enumerate(k_values):
                cache.insert(num_embeddings , 0, k, new_cached_latents[idx])
                
    while True:
        if req_queue.empty():
            # Check if all workers have finished or dropped
            all_done = all(status in ["finished", "dropped"] for status in worker_status.values())
            if all_done:
                print("[Scheduler] All workers have finished. Terminating.")
                break  # Exit scheduler loop

        time.sleep(1)  # Prevent busy waiting


# Worker Process
def worker(gpu_id, req_queue, new_cache_queue,latency_queue, worker_status):
    device = f"cuda:{gpu_id}"
    seed = 42 #any
    generator = torch.Generator(device).manual_seed(seed)
    model = load_model(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(model.scheduler.config)
    idle_counter = 0
    max_idle_iterations = 100 # Set a threshold for termination
    while True:
        try:
            
            request = req_queue.get(timeout=10)
            idle_counter = 0
            prompt = request['prompt']
            clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:220]
            # no hit
            if request['cached'] is None:
                timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [0], "cpu",0)[0]
                prompt_embeds, pooled_prompt_embeds, latents = model.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["latents"])
                model_outputs = model(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
                cached_timestep=None ,labels_batch = 0, current_latents=latents, height=1024, width=1024)
                
                generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                
                cached_latent = model_outputs[0].cpu()
                cached_latents = cached_latent.clone().share_memory_()
                query_embedding = request['query_embedding']
                new_cache_queue.put({'cached_latents': cached_latents.cpu().share_memory_(), 'prompt':prompt, 'query_embedding': query_embedding.cpu()})
                try:
                    model_outputs[1][0].save(generated_image_path)     
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    print("image name:", prompt)
                    
            else:
                # print(request['latent'].size())
                timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [1], "cpu",request['k'])[0]
                prompt_embeds, pooled_prompt_embeds, latents = model.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["latents"])
                model_outputs = model(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
                cached_timestep=None ,labels_batch = 1, current_latents=request['latent'].unsqueeze(0).to(dtype=torch.bfloat16).to(device), height=1024, width=1024)
                
                generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                try:
                    model_outputs[0].save(generated_image_path)
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    print("image name:", prompt)
                    
            finish_time = time.time() - request['start_time']
            print(f"[Worker {gpu_id}] Processed request latency: {finish_time}")
            latency_queue.put(finish_time)

        except queue.Empty:
            idle_counter += 1
            if idle_counter >= max_idle_iterations:
                print(f"[Worker {gpu_id}] No requests received for a while. Exiting...")
                worker_status[gpu_id] = "dropped"
                break
            continue

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = "cuda:0"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    metadata_df = pd.read_parquet('./metadata.parquet')

    np.random.seed(42)
    if 'timestamp' in metadata_df.columns:
        # Sort by timestamp
        sorted_df = metadata_df.sort_values(by='timestamp')
        start_time = sorted_df['timestamp'].iloc[0]

        # Convert timestamp to seconds from start
        sorted_df['seconds_from_start'] = (sorted_df['timestamp'] - start_time).dt.total_seconds()

        # Select 1000 requests from the dataset
        selected_requests = sorted_df.iloc[50000:50000+args.num_req].copy()
        num_requests = len(selected_requests)

        # Define Poisson process parameters
        rate_per_minute = args.req_rate   # Requests per minute
        rate_per_second = rate_per_minute / 60  # Poisson process rate (lambda)

        # Generate Poisson-distributed inter-arrival times
        timestamps = []
        current_time = 0

        for _ in range(num_requests):
            inter_arrival_time = np.random.exponential(scale=1/rate_per_second)  # Sample inter-arrival time
            current_time += inter_arrival_time  # Update timestamp
            timestamps.append(current_time)  # Store timestamp

        # Assign new Poisson-distributed timestamps
        selected_requests["seconds_from_start"] = timestamps

        print(f"First request: {timestamps[0]:.4f} sec")
        print(f"Last request: {timestamps[-1]:.4f} sec")
        print(f"Total duration: {timestamps[-1] - timestamps[0]:.4f} sec")

    else:
        print("No timestamp column found in the DataFrame.")
        
    # if 'timestamp' in metadata_df.columns:
    #     # print("First few entries in the 'timestamp' column:")
    #     # print(metadata_df['timestamp'].head())
        
    #     # Optionally sort by date
    #     # metadata_df['timestamp'] = pd.to_datetime(metadata_df['timestamp'])  # Ensure it's in datetime format
    #     sorted_df = metadata_df.sort_values(by='timestamp')
    #     start_time = sorted_df['timestamp'].iloc[0]
        
    #     # Add a new column for seconds relative to the start time
    #     sorted_df['seconds_from_start'] = (sorted_df['timestamp'] - start_time).dt.total_seconds()

    #     # Display the modified DataFrame
    #     print(sorted_df[['timestamp', 'seconds_from_start']].head())
    #     # print("Sorted DataFrame by timestamp:")
    #     # print(sorted_df.head())
    # else:
    #     print("No timestamp column found in the DataFrame.")
        
    image_directory = args.cache_directory

    # Get a list of all image file paths in the directory
    image_paths = [os.path.join(image_directory, img_file) for img_file in os.listdir(image_directory) if img_file.endswith(('.png', '.jpg', '.jpeg'))]
    cached_requests = [extract_prompt(image_path) for image_path in image_paths]
    print("number of cached:", len(cached_requests))
    # sorted_df = sorted_df.sort_values(by='seconds_from_start')
    # selected_requests = sorted_df.iloc[50000:51000].copy()
    num_requests = len(selected_requests)
    # selected_requests['seconds_from_start'] = generate_rapidly_increasing_seconds_from_start(
    # num_requests, min_rate=1, max_rate=9
    # )

 

    batch_size = 128
    num_batches = (len(cached_requests) + batch_size - 1) // batch_size

    # Placeholder to store all image embeddings
    text_embeddings = []

    # Process each batch
    for batch_idx in tqdm(range(num_batches)):
        # Select batch slice
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(cached_requests))
        batch_image_paths = cached_requests[batch_start:batch_end]
        
        texts = processor(text=batch_image_paths, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)




        batch_text_embeddings = []


        with torch.no_grad():
            # Get image embeddings
            # image_embeddings_batch = model.get_image_features(**image_tensors).cpu()  # Image model inference
            # batch_image_embeddings.append(image_embeddings_batch)
            # print(image_embeddings_batch.shape)

            # Get text embeddings
            text_embeddings_batch = clip_model.get_text_features(**texts).cpu()  # Text model inference
            batch_text_embeddings.append(text_embeddings_batch)
        batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)
        # batch_image_embeddings = torch.cat(batch_image_embeddings, dim=0)

        # Append the concatenated embeddings to the respective lists
        text_embeddings.append(batch_text_embeddings)

    # Concatenate all image embeddings into a single tensor
    final_text_embeddings = torch.cat(text_embeddings, dim=0)
    final_text_embeddings = final_text_embeddings.cpu()
    torch.save(final_text_embeddings, "final_text_embeddings.pt")
    print(f"Generated embeddings for {len(cached_requests)} images.")
    
    # final_text_embeddings = final_text_embeddings[0:3333]
    # cached_requests = cached_requests[0:3333]
    
    final_latents = torch.load("../cached_latents_1.pt", map_location="cpu")
    final_latents_1 = torch.load("../cached_latents_2.pt", map_location="cpu")
    final_latents_2 = torch.load("../cached_latents_3.pt", map_location="cpu")

    final_latents = torch.cat((final_latents,final_latents_1), dim=0)
    final_latents = torch.cat((final_latents,final_latents_2), dim=0)


    assert final_latents.shape[0] == 5 * final_text_embeddings.shape[0], \
    f"Assertion failed: {final_latents.shape[0]} != 5 * {final_text_embeddings.shape[0]}"
    
    embedding_dim = 768  # CLIP model output dimension
    index = faiss.IndexFlatL2(embedding_dim)  # Using FAISS for ANN search
    index.add(final_text_embeddings.numpy())  # Add text embeddings to FAISS index
    
    final_text_embeddings.share_memory_()
    final_latents.share_memory_()

    num_gpus = torch.cuda.device_count()
    req_queue = mp.Queue()
    new_cache_queue = mp.Queue()
    latency_queue = mp.Queue()
    manager = mp.Manager()
    worker_status = manager.dict()
    log_file=f"request_throughput_{num_requests}_rr{args.rate}_N.csv"
    
    cache = KMinHeapCache(max_size=args.cache_size * 5, initial_embeddings=final_text_embeddings, latents=final_latents)

    k_values = [5, 10, 15, 20, 25]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    start_time = time.time()

    scheduler = mp.Process(target=request_scheduler, args=(req_queue,  selected_requests, start_time, index, cache,new_cache_queue,cached_requests, final_text_embeddings, k_values, processor, clip_model,log_file, worker_status))

    scheduler.start()

    workers = []
    for gpu_id in range(num_gpus):
        worker_status[gpu_id] = "starting"
        p = mp.Process(target=worker, args=(gpu_id, req_queue, new_cache_queue,latency_queue, worker_status))
        p.start()
        workers.append(p)


    # Wait for all workers to finish
    for p in workers:
        p.join()
        
    scheduler.join()

    print("[Main] All requests processed.")
    
    all_latencies = []
    while not latency_queue.empty():
        all_latencies.append(latency_queue.get())

    print("\n🚀 **Final Latency Report**")
    for i, latency in enumerate(all_latencies):
        print(f"{latency:.4f}")

    print(f"\n📊 **Latency Summary**: Min = {min(all_latencies):.4f}s, Max = {max(all_latencies):.4f}s, Avg = {sum(all_latencies)/len(all_latencies):.4f}s")

    print("[Main] All requests processed.")