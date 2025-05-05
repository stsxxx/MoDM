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
# === Argument Parsing for SLURM ===
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers per node")
parser.add_argument("--rate", type=float, default=4)
parser.add_argument("--metadata_path", type=str, required=True, help="path of metadata")
parser.add_argument("--image_directory", type=str, required=True, help="directory of generated images")

args = parser.parse_args()
directory = args.image_directory
os.makedirs(directory, exist_ok=True)
print(f"Directory '{directory}' created successfully.")

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
        log_file = f"SLO_LOG/baseline_slo_{args.rate}_3000.log"
        while self.running:
            try:
                request = self.request_queue.get(timeout=60)
                prompt = request['prompt']

                print(f"[Worker {self.worker_id}] Processing request: {prompt}")
                idle_counter = 0
                clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:220]
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
    def __init__(self, worker_ranks):
        self.worker_ranks = worker_ranks  # List of worker nodes
        self.current_worker = 0

    def process_request(self, selected_requests, start_time):
        """Assigns requests to node queues in a round-robin manner."""
        
        for idx, row in selected_requests.iterrows():
            while time.time() - start_time < row['seconds_from_start']:
                time.sleep(0.1)
            request_arrival_time = time.time()  # Start time of request processing

            row['start_time'] = request_arrival_time  # Add start_time to request
            assigned_worker = self.worker_ranks[self.current_worker]
            self.current_worker = (self.current_worker + 1) % len(self.worker_ranks)  
            print(f"[Scheduler] Assigning request {idx} to {assigned_worker}")

            worker_rref = rpc.rpc_sync(f"{assigned_worker}", get_worker_rref)  # ✅ Correct
            worker_rref.rpc_sync().enqueue_request(row)  # ✅ Call method remotely


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

    mp.set_start_method("spawn", force=True)

    if int(os.environ["SLURM_PROCID"]) == 0:
        rpc.init_rpc(
        name=f"scheduler",
        rank=int(os.environ["SLURM_PROCID"]),
        world_size=int(os.environ["WORLD_SIZE"]),  # 1 scheduler + 4 worker nodes
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions()
        )
        metadata_df = pd.read_parquet(args.metadata_path)

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
        

        # ✅ Scheduler Process (Runs on First Node)
        node_ranks = [f"worker_{i}" for i in range(1, args.num_workers + 1)]
        global worker_status_tracker
        worker_status_tracker = WorkerStatusTracker(node_ranks)
        time.sleep(60)

        start_time = time.time()
        scheduler = Scheduler(node_ranks)
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
