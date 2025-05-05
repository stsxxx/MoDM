import torch
import time
import queue
import torch.multiprocessing as mp
import pandas as pd
from diffusers import StableDiffusion3Pipeline, StableDiffusionXLImg2ImgPipeline, DiffusionPipeline, SanaPipeline, StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
import faiss
import os
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import re
import heapq
from tqdm import tqdm
import argparse
import gc

# Cache Data Structure


parser = argparse.ArgumentParser(description="model selection")
parser.add_argument("--large_model", type=str, default='sd3.5',required=False, help="which large model you wanna use")
parser.add_argument("--small_model", type=str,default='sdxl', required=False, help="which small model you wanna use")
parser.add_argument("--num_req", type=int, default=10000, required=True, help="number of requests")
parser.add_argument("--image_directory", type=str, required=False, help="directory of generated images")

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
def request_scheduler(req_queue,  selected_requests, start_time, worker_status, log_file="request_throughput_climbing_Baseline.csv"):

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
            with open(log_file, "a") as f:
                f.write(f"{minute},{request_count_per_min / elapsed_time * 60},{throughput / elapsed_time * 60}\n")
            request_count_per_min = 0
            last_check_time_queue = current_time

    while True:
        if req_queue.empty():
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
    model = load_model(device,model_type)
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(model.scheduler.config)
    idle_counter = 0
    max_idle_iterations = 100 # Set a threshold for termination
    while True:
        try:
            
            request = req_queue.get(timeout=10)
            idle_counter = 0
            prompt = request['prompt']
            clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:300]

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
    num_gpus = torch.cuda.device_count()

    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
    # profile latency
    test_prompt = "a dog riding a bike"
    large_model_latency = []
    if args.large_model == "sd3.5":
        large_model = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
        ).to(device)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(large_model.scheduler.config)
        for i in range(2):
            start_time = time.time()
            timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [0], "cpu",0)[0]
            prompt_embeds, pooled_prompt_embeds, latents = large_model.input_process(prompt = test_prompt,negative_prompt = None, callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["latents"])
            model_outputs = large_model(prompt = test_prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
            cached_timestep=None ,labels_batch = 0, current_latents=latents, height=1024, width=1024,num_inference_steps = 50)
            end_time = time.time()
            large_model_latency.append(end_time - start_time) 
        avg_latency_large = sum(large_model_latency) / len(large_model_latency)
        print(f"Average large model Latency: {avg_latency_large:.4f} seconds")
        # Release memory
        del large_model
        torch.cuda.empty_cache()
        gc.collect()

    elif args.large_model == "flux":
        large_model =  DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
        for i in range(2):
            start_time = time.time()
            image = large_model(prompt = test_prompt, num_inference_steps = 50, height=1024, width=1024).images[0]
            end_time = time.time()
            large_model_latency.append(end_time - start_time) 
        avg_latency_large = sum(large_model_latency) / len(large_model_latency)
        print(f"Average large model Latency: {avg_latency_large:.4f} seconds")
        # Release memory
        del large_model
        torch.cuda.empty_cache()
        gc.collect()

    small_model_latency = []
    if args.small_model == "sdxl":
        small_model = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to(device)
        for i in range(5):
            start_time = time.time()
            small_model(prompt = test_prompt, num_inference_steps = 50, height=1024, width=1024)
            end_time = time.time()
            small_model_latency.append(end_time - start_time) 
        avg_latency_small = sum(small_model_latency) / len(small_model_latency)
        print(f"Average small model Latency: {avg_latency_small:.4f} seconds")
        # Release memory
        del small_model
        torch.cuda.empty_cache()
        gc.collect()
    elif args.small_model == "sana":
        small_model =  SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            variant="bf16",
            torch_dtype=torch.bfloat16,
        ).to(device)
        for i in range(5):
            start_time = time.time()
            small_model(prompt = test_prompt, num_inference_steps = 50, height=1024, width=1024)
            end_time = time.time()
            small_model_latency.append(end_time - start_time) 
        avg_latency_small = sum(small_model_latency) / len(small_model_latency)
        print(f"Average small model Latency: {avg_latency_small:.4f} seconds")
        # Release memory
        del small_model
        torch.cuda.empty_cache()
        gc.collect()

    np.random.seed(42)

    metadata_df = pd.read_parquet('./metadata.parquet')

    if 'timestamp' in metadata_df.columns:
        # print("First few entries in the 'timestamp' column:")
        # print(metadata_df['timestamp'].head())
        
        # Optionally sort by date
        # metadata_df['timestamp'] = pd.to_datetime(metadata_df['timestamp'])  # Ensure it's in datetime format
        sorted_df = metadata_df.sort_values(by='timestamp')
        start_time = sorted_df['timestamp'].iloc[0]
        
        # Add a new column for seconds relative to the start time
        sorted_df['seconds_from_start'] = (sorted_df['timestamp'] - start_time).dt.total_seconds()

        # Display the modified DataFrame
        print(sorted_df[['timestamp', 'seconds_from_start']].head())
        # print("Sorted DataFrame by timestamp:")
        # print(sorted_df.head())
    else:
        print("No timestamp column found in the DataFrame.")


    sorted_df = sorted_df.sort_values(by='seconds_from_start')
    selected_requests = sorted_df.iloc[50000:50000+args.num_req].copy()
    num_requests = len(selected_requests)
    large_throughput_per_gpu = 1 / avg_latency_large
    small_throughput_per_gpu = 1 / avg_latency_small 
    work_large = 0.3
    work_small = 0.7 * 0.8 * large_throughput_per_gpu / small_throughput_per_gpu
    gpus_for_large = min(np.round(work_large/(work_large+work_small) * num_gpus),num_gpus-1)

    time_gap = 1 / (large_throughput_per_gpu * gpus_for_large + small_throughput_per_gpu * (num_gpus - gpus_for_large))
    time_gap = max(time_gap-0.05, 0)
    print('time gap:', time_gap)

    selected_requests['seconds_from_start'] = generate_rapidly_increasing_seconds_from_start(
    num_requests, min_rate=1, max_rate=(1/time_gap)*60
    )
        # Print each value with its index
    for i, seconds in enumerate(selected_requests['seconds_from_start']):
        print(f"Request {i+1}: {seconds:.2f}")
 
    num_gpus = torch.cuda.device_count()
    req_queue = mp.Queue()

    latency_queue = mp.Queue()
    manager = mp.Manager()
    worker_status = manager.dict()
    
    start_time = time.time()

    scheduler = mp.Process(target=request_scheduler, args=(req_queue,  selected_requests, start_time, worker_status))

    scheduler.start()
    model_type = args.large_model
    workers = []
    for gpu_id in range(num_gpus):
        worker_status[gpu_id] = "starting"
        p = mp.Process(target=worker, args=(gpu_id, req_queue, latency_queue, worker_status,model_type))
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

    # print(f"\n📊 **Latency Summary**: Min = {min(all_latencies):.4f}s, Max = {max(all_latencies):.4f}s, Avg = {sum(all_latencies)/len(all_latencies):.4f}s")

    # print("[Main] All requests processed.")