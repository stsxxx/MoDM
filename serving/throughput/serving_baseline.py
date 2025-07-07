import torch
import time
import queue
import torch.multiprocessing as mp
import pandas as pd
from diffusers import (
    StableDiffusion3Pipeline,
    FlowMatchEulerDiscreteScheduler,
    DiffusionPipeline
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


parser = argparse.ArgumentParser(description="model selection")
parser.add_argument("--image_directory", type=str, required=False, help="directory of generated images")
parser.add_argument("--large_model", type=str, required=True, help="which large model you wanna use")
parser.add_argument("--num_req", type=int, default=10000, required=True, help="number of requests")
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

def load_model(model_type, device):
    """Load the specified model type onto the given device."""
    if model_type == "sd3.5":
        return StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
        ).to(device)
    elif model_type == "flux":
        return DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    

# Request scheduler
def request_scheduler(req_queue,  selected_requests, start_time, worker_status, num_gpus, log_file="request_throughput_Baseline.csv"):

    minute = 0
    # with open(log_file, "w") as f:
    #     f.write("timestamp,request_rate,throughput\n")
    last_check_time = time.time()
    last_check_time_queue = last_check_time
    request_count_per_min = 0
    size_of_queues = 0
    throughput = 0        
    for _, row in selected_requests.iterrows():
        while time.time() - start_time < row['seconds_from_start']:
            time.sleep(0.1)
            
            
        request_arrival_time = time.time()  # Start time of request processing

        row['start_time'] = start_time  # Add start_time to request

        # prompt = row['prompt']


        req_queue.put(row.to_dict())


        request_count_per_min += 1
        current_time = time.time()
        
        if current_time - last_check_time_queue >= 60:
            elapsed_time = current_time - last_check_time_queue
            minute += 1
            new_size_of_queues =  req_queue.qsize() 
            throughput = size_of_queues + request_count_per_min - new_size_of_queues
            size_of_queues = new_size_of_queues
            # with open(log_file, "a") as f:
            #     f.write(f"{minute},{request_count_per_min / elapsed_time * 60},{throughput / elapsed_time * 60}\n")
            request_count_per_min = 0
            last_check_time_queue = current_time
            
    for _ in range(num_gpus):
        req_queue.put(None)  # Each worker will receive one None signal
        
    while True:
        # if req_queue.empty():
            # Check if all workers have finished or dropped
        all_done = all(status in ["finished", "dropped"] for status in worker_status.values())
        if all_done:
            print("[Scheduler] All workers have finished. Terminating.")
            break  # Exit scheduler loop

        time.sleep(1)  # Prevent busy waiting


# Worker Process
def worker(gpu_id, req_queue, latency_queue, worker_status, model_type):
    device = f"cuda:{gpu_id}"
    seed = 42 #any
    generator = torch.Generator(device).manual_seed(seed)
    model = load_model(model_type , device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(model.scheduler.config)
    idle_counter = 0
    max_idle_iterations = 100 # Set a threshold for termination
    while True:
        try:
            
            request = req_queue.get(timeout=10)
            
            if request is None:
                print(f"[Worker {gpu_id}] Received termination signal. Exiting...")
                worker_status[gpu_id] = "finished"
                break  
            idle_counter = 0
            prompt = request['prompt']
            clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:220]
            if model_type == 'sd3.5':
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(model.scheduler.config)
                timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [0], "cpu",0)[0]
                prompt_embeds, pooled_prompt_embeds, latents = model.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["latents"])
                model_outputs = model(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
                labels_batch = 0, current_latents=latents, height=1024, width=1024)
                
                generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                
                try:
                    model_outputs[1][0].save(generated_image_path)
                    
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    print("image name:", prompt)
                print(f"[Worker {gpu_id}] Added new image path to shared list: {generated_image_path}")
            elif model_type == 'flux':
                image = model(prompt = prompt, num_inference_steps = 50, height=1024, width=1024).images[0]
                generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                
                try:
                    image.save(generated_image_path)
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    print("image name:", prompt)
                print(f"[Worker {gpu_id}] Added new image path to shared list: {generated_image_path}")
                    
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

    metadata_df = pd.read_parquet('./metadata.parquet')
    if 'timestamp' in metadata_df.columns:
        sorted_df = metadata_df.sort_values(by='timestamp')
        
        # Set all seconds_from_start to zero
        sorted_df['seconds_from_start'] = 0

        # Display modified DataFrame
        print(sorted_df[['timestamp', 'seconds_from_start']].head())
    else:
        print("No timestamp column found in the DataFrame.")


    sorted_df = sorted_df.sort_values(by='seconds_from_start')
    selected_requests = sorted_df.iloc[50000:50000 + args.num_req].copy()
    # Force all timestamps to be zero
    selected_requests['seconds_from_start'] = 0
    num_requests = len(selected_requests)
    # selected_requests['seconds_from_start'] = generate_rapidly_increasing_seconds_from_start(
    # num_requests, min_rate=1, max_rate=9
    # )
    
 
    num_gpus = torch.cuda.device_count()
    req_queue = mp.Queue()

    latency_queue = mp.Queue()
    manager = mp.Manager()
    worker_status = manager.dict()
    
    start_time = time.time()

    scheduler = mp.Process(target=request_scheduler, args=(req_queue,  selected_requests, start_time, worker_status, num_gpus))

    scheduler.start()
    model_type = args.large_model
    workers = []
    for gpu_id in range(num_gpus):
        worker_status[gpu_id] = "starting"
        p = mp.Process(target=worker, args=(gpu_id, req_queue, latency_queue, worker_status, model_type))
        p.start()
        workers.append(p)

    scheduler.join()

    for p in workers:
        if p.is_alive():
            p.terminate()
    # Wait for all workers to finish
    for p in workers:
        p.join()
        


    print("[Main] All requests processed.")
    
    all_latencies = []
    while not latency_queue.empty():
        all_latencies.append(latency_queue.get())

    print("\n🚀 **Final Latency Report**")
    for i, latency in enumerate(all_latencies):
        print(f"{latency:.4f}")
        
    if all_latencies:
        total_time = all_latencies[-1] 
        throughput = args.num_req / total_time * 60
        print(f"\n📈 Total Time: {total_time:.4f} seconds")
        print(f"Throughput: {throughput:.2f} requests/min")
    else:
        print("No latencies recorded.")
    # if all_latencies:
    #     total_time = max(all_latencies)  # Last request latency
    #     throughput = len(all_latencies) / total_time * 60  # Requests per min

    #     print("\n🚀 **Final Latency Report**")
    #     for i, latency in enumerate(all_latencies):
    #         print(f"{latency:.4f}")

    #     print(f"\n📊 **Latency Summary**: Min = {min(all_latencies):.4f}s, Max = {max(all_latencies):.4f}s, Avg = {sum(all_latencies)/len(all_latencies):.4f}s")
    #     print(f"\n⏳ **Total Processing Time**: {total_time:.4f}s")
    #     print(f"⚡ **System Throughput**: {throughput:.2f} requests/min")
    # else:
    #     print("[Main] No latency data collected.")