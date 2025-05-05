import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.rpc import RRef
import queue
import os
import time
import argparse
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
import argparse
from tqdm import tqdm

import torch.distributed.rpc as rpc

cache_size = 100000
# === Argument Parsing for SLURM ===
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers per node")
parser.add_argument("--rate", type=float, default=4)
parser.add_argument("--metadata_path", type=str, required=True, help="path of metadata")
parser.add_argument("--image_directory", type=str, required=True, help="directory of generated images")

# os.environ["TP_SOCKET_IFNAME"] = "ibp148s0f0"
# os.environ["GLOO_SOCKET_IFNAME"] = "ibp148s0f0"
args = parser.parse_args()
directory = args.image_directory
os.makedirs(directory, exist_ok=True)
print(f"Directory '{directory}' created successfully.")

def extract_prompt(filename):
    """Extract prompt from filename by replacing underscores with spaces."""
    name = os.path.splitext(os.path.basename(filename))[0]  # Remove extension
    return re.sub(r'[_]+', ' ', name).strip()  # Convert underscores to spaces

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
    
def get_current_time():
    return time.time()

def get_worker_rref():
    global worker_rref
    return worker_rref

def load_model(device):
    return StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
    ).to(device)

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

class WorkerStatusTracker:
    """Tracks the status of all worker nodes"""
    def __init__(self, worker_ranks):
        self.worker_status = {worker: "active" for worker in worker_ranks}
        self.total_workers = len(worker_ranks)
        self.active_workers = self.total_workers
        print(f"[StatusTracker] Initialized tracking {self.total_workers} workers")
    
    def mark_worker_dropped(self, worker_rank):
        """Mark a worker as dropped and return the number of active workers"""
        if worker_rank in self.worker_status and self.worker_status[worker_rank] == "active":
            self.worker_status[worker_rank] = "dropped"
            self.active_workers -= 1
            print(f"[StatusTracker] Worker {worker_rank} marked as dropped. {self.active_workers} workers still active.")
        return self.active_workers
    
    def all_workers_dropped(self):
        """Check if all workers have dropped"""
        return self.active_workers == 0
    
    def get_status(self):
        """Get the current status of all workers"""
        return self.worker_status

# Function to access the global worker_status_tracker
def get_worker_status_tracker():
    global worker_status_tracker
    return worker_status_tracker

# Function for workers to notify they're dropping
def notify_worker_dropped(worker_rank):
    global worker_status_tracker
    remaining = worker_status_tracker.mark_worker_dropped(worker_rank)
    return remaining

def get_scheduler_ref():
    global scheduler_rref
    return scheduler_rref

class NodeWorker:
    """Worker that fetches and processes requests from a shared queue."""
    def __init__(self, worker_id, gpu_id):
        self.worker_id = worker_id
        self.request_queue = queue.Queue()
        self.running = True  # Used for graceful shutdown
        self.device = f"cuda:{gpu_id}"

    def fetch_and_process_requests(self):
        """Continuously fetches requests from the shared queue."""
        seed = 42 #any
        generator = torch.Generator(self.device).manual_seed(seed)
        model = load_model(self.device)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(model.scheduler.config)
        idle_counter = 0
        max_idle_iterations = 10 # Set a threshold for termination
        log_file = f"SLO_LOG/NIRVANA_slo_{args.rate}.log"
        while self.running:
            try:
                request = self.request_queue.get(timeout=60)
                prompt = request['prompt']

                print(f"[Worker {self.worker_id}] Processing request: {prompt}")
                idle_counter = 0
                clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:220]
                if request['cached'] is None:
                    timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [0], "cpu",0)[0]
                    prompt_embeds, pooled_prompt_embeds, latents = model.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
                    callback_on_step_end_tensor_inputs=["latents"])
                    model_outputs = model(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
                cached_timestep=None ,labels_batch = 0, current_latents=latents, height=1024, width=1024)
                
                    generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                    try:
                        model_outputs[1][0].save(generated_image_path)     
                    except Exception as e:
                        print(f"Failed to save image: {e}")
                        print("image name:", prompt)
                    end_time = rpc.rpc_sync("scheduler", get_current_time)
                    finish_time = end_time - request['start_time']
                    with open(log_file,"a") as f:
                        f.write(f"[Worker {self.worker_id}] Processed request latency: {finish_time}\n")
                    cached_latent = model_outputs[0].cpu()
                    cached_latents = cached_latent.clone()
                    query_embedding = request['query_embedding']
                    new_cache = {
                    'cached_latents': cached_latents,
                    'prompt': prompt,
                    'query_embedding': query_embedding
                    }   
                    scheduler_rref = rpc.rpc_sync(f"scheduler", get_scheduler_ref)  # ✅ Correct
                    scheduler_rref.rpc_sync().add_new_cache(new_cache)  # ✅ Call method remotely
                else:
                    timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [1], "cpu",request['k'])[0]
                    prompt_embeds, pooled_prompt_embeds, latents = model.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
                    callback_on_step_end_tensor_inputs=["latents"])
                    model_outputs = model(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
                    callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
                    cached_timestep=None ,labels_batch = 1, current_latents=request['latent'].unsqueeze(0).to(dtype=torch.bfloat16).to(self.device), height=1024, width=1024)
                
                    generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                    try:
                        model_outputs[0].save(generated_image_path) 
                    except Exception as e:
                        print(f"Failed to save image: {e}")
                        print("image name:", prompt)
                    end_time = rpc.rpc_sync("scheduler", get_current_time)
                    finish_time = end_time - request['start_time']
                    with open(log_file,"a") as f:
                        f.write(f"[Worker {self.worker_id}] Processed request latency: {finish_time}\n")

            except queue.Empty:
                idle_counter += 1
                if idle_counter >= max_idle_iterations:
                    print(f"[Worker {self.worker_id}] No requests received for a while. Exiting...")
                    # worker_status[gpu_id] = "dropped"
                    try:
                        remaining = rpc.rpc_sync("scheduler", notify_worker_dropped, args=(f"worker_{self.worker_id}",))
                        print(f"[Worker {self.worker_id}] Notified scheduler of dropping. {remaining} workers still active.")
                    except Exception as e:
                        print(f"[Worker {self.worker_id}] Failed to notify scheduler of dropping: {e}")
                    break
                continue

    def enqueue_request(self, request):
        """Adds a request to the queue."""
        self.request_queue.put(request)

    def shutdown(self):
        """Shuts down the worker."""
        print(f"[Worker {self.worker_id}] Shutting down.")
        self.running = False

# class NodeQueue:
#     """Manages a shared request queue for workers on a node."""
#     def __init__(self, num_workers):
#         self.request_queue = mp.Queue()  # ✅ Shared queue for all workers in this node
#         self.workers = [mp.Process(target=self.worker_process, args=(i,)) for i in range(num_workers)]
#         self.running = True

#     def worker_process(self, worker_id):
#         """Worker process that fetches requests from the shared queue."""
#         worker = NodeWorker(worker_id, self.request_queue)
#         worker.fetch_and_process_requests()

#     def enqueue_request(self, row):
#         """Adds a request to the shared queue."""
#         self.request_queue.put(row)
#         # print(f"[NodeQueue] Added request {request_id} to queue.")

#     def shutdown_workers(self):
#         """Shuts down all workers."""
#         self.running = False
#         for worker in self.workers:
#             worker.terminate()
#         print("[NodeQueue] All workers have shut down.")

class Scheduler:
    """Distributes requests to worker node queues using RPC."""
    def __init__(self, worker_ranks, cache, k_values, processor, clip_model, index, final_text_embeddings,cached_requests):
        self.worker_ranks = worker_ranks  # List of worker nodes
        self.current_worker = 0
        self.cache = cache
        self.k_values = k_values
        self.processor = processor
        self.clip_model = clip_model
        self.index = index
        self.cached_requests = cached_requests
        self.final_text_embeddings = final_text_embeddings
        self.new_cache_queue = queue.Queue()

    def add_new_cache(self, new_cache):
        self.new_cache_queue.put(new_cache)

    def process_request(self, selected_requests, start_time):
        """Assigns requests to node queues in a round-robin manner."""
        agg_k_distribution = {5: 0, 10: 0, 15: 0, 20: 0, 25: 0} 
        for idx, row in selected_requests.iterrows():
            while time.time() - start_time < row['seconds_from_start']:
                time.sleep(0.1)
            request_arrival_time = time.time()  # Start time of request processing

            row['start_time'] = request_arrival_time  # Add start_time to request
            while not self.new_cache_queue.empty():
                cache_data = self.new_cache_queue.get()  # Retrieves the dictionary from the queue
                new_cached_latents = cache_data['cached_latents']
                new_cached_prompt = cache_data['prompt']
                new_query_embedding = cache_data['query_embedding']
                while len(self.cache.item_map) + 5  > self.cache.max_size:
                    evicted_index = self.cache.evict()
                    if evicted_index is not None:
                        del self.cached_requests[evicted_index]
                        self.index , self.final_text_embeddings = evict_from_faiss(self.index, self.final_text_embeddings, evicted_index)

                
                num_embeddings = self.index.ntotal
                self.cached_requests.append(new_cached_prompt) 
                self.index.add(new_query_embedding)


                self.final_text_embeddings = np.concatenate((self.final_text_embeddings, new_query_embedding), axis=0)
                
                for idx, k in enumerate(self.k_values):
                    self.cache.insert(num_embeddings , 0, k, new_cached_latents[idx])
            prompt = row['prompt']
            # Process the prompt embedding
            texts = self.processor(text=[prompt], return_tensors="pt", truncation=True, padding=True, max_length=77).to(self.clip_model.device)
            with torch.no_grad():
                text_embedding = self.clip_model.get_text_features(**texts).cpu()
            query_embedding = text_embedding.numpy().reshape(1, -1)
            distances, indices = self.index.search(query_embedding, k=1)
            closest_prompt = self.cached_requests[indices[0][0]]
            closest_texts = self.processor(text=closest_prompt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(self.clip_model.device)
            with torch.no_grad():
                closest_text_embedding = self.clip_model.get_text_features(**closest_texts)
            text_embedding = text_embedding.to(self.clip_model.device)
            with torch.no_grad():
                text_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
                closest_text_norm = closest_text_embedding / closest_text_embedding.norm(dim=-1, keepdim=True) 
                text_similarity_scores = torch.matmul(text_norm, closest_text_norm.T)
            text_similarity_scores = torch.clamp(text_similarity_scores, min=0)
            text_embedding = text_embedding.cpu()
            assigned_worker = self.worker_ranks[self.current_worker]
            self.current_worker = (self.current_worker + 1) % len(self.worker_ranks)  
            print(f"[Scheduler] Assigning request {idx} to {assigned_worker}")
            print("text text similarity:", text_similarity_scores)
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
                best_candidate = self.cache.retrieve(closest_index, indices[0][0])
                
                if best_candidate:
                    score, (idex, k_i, latent) = best_candidate
                    row['cached'] = True
                    row['k'] = k_i
                    row['latent'] = latent.clone().to(dtype=torch.float32).cpu()
                    row['query_embedding'] = text_embedding.clone()
                    worker_rref = rpc.rpc_sync(f"{assigned_worker}", get_worker_rref)  # ✅ Correct
                    worker_rref.rpc_sync().enqueue_request(row)  # ✅ Call method remotely
                    agg_k_distribution[k_i] += 1
                else:
                    row['cached'] = None
                    row['k'] = None
                    row['latent'] = None
                    row['query_embedding'] = text_embedding.clone()
                    worker_rref = rpc.rpc_sync(f"{assigned_worker}", get_worker_rref)  # ✅ Correct
                    worker_rref.rpc_sync().enqueue_request(row)  # ✅ Call method remotely
                
                
            else:
                row['cached'] = None
                row['k'] = None
                row['latent'] = None
                row['query_embedding'] = text_embedding.clone()
                worker_rref = rpc.rpc_sync(f"{assigned_worker}", get_worker_rref)  # ✅ Correct
                worker_rref.rpc_sync().enqueue_request(row)  # ✅ Call method remotely
            print(agg_k_distribution)








        # self.processed_requests += 1
        # if self.processed_requests >= self.max_requests:
        #     print("[Scheduler] All requests processed. Stopping workers.")

        #     # ✅ Notify all nodes to shut down workers
        #     for node in self.node_ranks:
        #         node_queue_rref = rpc.remote(node, lambda: worker_rref)
        #         node_queue_rref.remote().shutdown_workers()

        #     time.sleep(5)  # Allow time for workers to exit
            # rpc.shutdown()  # Stop RPC communication

def worker_process(rank):
    """Starts a worker node process that runs NodeQueue."""
    rank = int(rank)
    global worker_rref  # Make it accessible for RPC
    gpu_id = (int(rank) - 1) % 4
    rpc.init_rpc(
        name=f"worker_{rank}",
        rank=rank,
        world_size=int(os.environ["WORLD_SIZE"]),
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions()
    )

    # ✅ Create worker and register RRef
    node_worker = NodeWorker(worker_id=rank, gpu_id=gpu_id)
    worker_rref = RRef(node_worker)

    # ✅ Expose worker_rref for remote access

    print(f"[Worker {rank}] RPC Initialized on GPU {gpu_id}, waiting for requests...")

    # ✅ Start processing requests
    node_worker.fetch_and_process_requests()

    rpc.shutdown()


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    mp.set_start_method("spawn", force=True)

    if int(os.environ["SLURM_PROCID"]) == 0:
        rpc.init_rpc(
        name=f"scheduler",
        rank=int(os.environ["SLURM_PROCID"]),
        world_size=int(os.environ["WORLD_SIZE"]),  # 1 scheduler + 4 worker nodes
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions()
        )
        metadata_df = pd.read_parquet(args.metadata_path)

        device = "cuda:0"
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        np.random.seed(42)
        if 'timestamp' in metadata_df.columns:
            # Sort by timestamp
            sorted_df = metadata_df.sort_values(by='timestamp')
            start_time = sorted_df['timestamp'].iloc[0]

            # Convert timestamp to seconds from start
            sorted_df['seconds_from_start'] = (sorted_df['timestamp'] - start_time).dt.total_seconds()

            # Select 1000 requests from the dataset
            selected_requests = sorted_df.iloc[50000:53000].copy()
            num_requests = len(selected_requests)

            # Define Poisson process parameters
            rate_per_minute = args.rate  # Requests per minute
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

            # Display the first few entries
            print(selected_requests[['timestamp', 'seconds_from_start']].head())

        else:
            print("No timestamp column found in the DataFrame.")

        image_directory = "/work1/talati/stilex/cache_diffusion/nohit"
        # Get a list of all image file paths in the directory
        image_paths = [os.path.join(image_directory, img_file) for img_file in os.listdir(image_directory) if img_file.endswith(('.png', '.jpg', '.jpeg'))]
        cached_requests = [extract_prompt(image_path) for image_path in image_paths]
        print("number of cached:", len(cached_requests))
        num_requests = len(selected_requests)
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
                text_embeddings_batch = clip_model.get_text_features(**texts).cpu()  # Text model inference
                batch_text_embeddings.append(text_embeddings_batch)
            batch_text_embeddings = torch.cat(batch_text_embeddings, dim=0)
            text_embeddings.append(batch_text_embeddings)

        # Concatenate all image embeddings into a single tensor
        final_text_embeddings = torch.cat(text_embeddings, dim=0)
        final_text_embeddings = final_text_embeddings.cpu()
        torch.save(final_text_embeddings, "final_text_embeddings.pt")

        # final_text_embeddings = torch.load("/work1/talati/stilex/cache_diffusion/final_text_embeddings.pt", map_location="cpu")
        print(f"Generated embeddings for {len(cached_requests)} images.")
        final_latents = torch.load("/work1/talati/stilex/cache_diffusion/cached_latents_1.pt", map_location="cpu")
        final_latents_1 = torch.load("/work1/talati/stilex/cache_diffusion/cached_latents_2.pt", map_location="cpu")
        final_latents_2 = torch.load("/work1/talati/stilex/cache_diffusion/cached_latents_3.pt", map_location="cpu")

        final_latents = torch.cat((final_latents,final_latents_1), dim=0)
        final_latents = torch.cat((final_latents,final_latents_2), dim=0)
        assert final_latents.shape[0] == 5 * final_text_embeddings.shape[0], \
        f"Assertion failed: {final_latents.shape[0]} != 5 * {final_text_embeddings.shape[0]}"
        embedding_dim = 768  # CLIP model output dimension
        index = faiss.IndexFlatL2(embedding_dim)  # Using FAISS for ANN search
        index.add(final_text_embeddings.numpy())  # Add text embeddings to FAISS index
        cache = KMinHeapCache(max_size=cache_size * 5, initial_embeddings=final_text_embeddings, latents=final_latents)
        k_values = [5, 10, 15, 20, 25]

        # ✅ Scheduler Process (Runs on First Node)
        node_ranks = [f"worker_{i}" for i in range(1, args.num_workers + 1)]
        global worker_status_tracker
        worker_status_tracker = WorkerStatusTracker(node_ranks)
        global scheduler_rref
        time.sleep(60)

        start_time = time.time()
        scheduler = Scheduler(node_ranks, cache, k_values, processor, clip_model, index, final_text_embeddings, cached_requests)
        scheduler_rref = RRef(scheduler)
        # # Example: Send test requests
        # prompts = ["Astronaut in a jungle", "Cyberpunk city skyline", "Fantasy castle"]
        # for i, prompt in enumerate(prompts):
        scheduler.process_request(selected_requests, start_time)
        # Keep scheduler alive until all workers have dropped
        print("[Scheduler] All requests dispatched. Waiting for workers to complete...")
        while not worker_status_tracker.all_workers_dropped():
            time.sleep(10)
            print(f"[Scheduler] Still waiting. Worker status: {worker_status_tracker.get_status()}")
        
        print("[Scheduler] All workers have dropped. Shutting down.")
        # Optional: Save any final statistics or results here
        
        rpc.shutdown()
    elif int(os.environ["SLURM_PROCID"]) >= 1:
        # ✅ Worker Process (Runs on Each Worker Node)
        worker_process(os.environ["SLURM_PROCID"])
