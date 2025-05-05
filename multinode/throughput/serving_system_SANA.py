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
    StableDiffusionXLImg2ImgPipeline,
    SanaPipeline
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
import PIL
import torch.distributed.rpc as rpc
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

cache_size = 100000
# === Argument Parsing for SLURM ===
parser = argparse.ArgumentParser()
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers per node")
parser.add_argument("--rate", type=float, default=4)
parser.add_argument("--num_small", type=float, default=1)
parser.add_argument("--image_directory", type=str, required=True, help="directory of generated images")
parser.add_argument("--metadata_path", type=str, required=True, help="path of metadata")

args = parser.parse_args()
directory = args.image_directory
os.makedirs(directory, exist_ok=True)
print(f"Directory '{directory}' created successfully.")

def extract_prompt(filename):
    """Extract prompt from filename by replacing underscores with spaces."""
    name = os.path.splitext(os.path.basename(filename))[0]  # Remove extension
    return re.sub(r'[_]+', ' ', name).strip()  # Convert underscores to spaces

def retrieve_best_image(text_embedding, final_image_embeddings, distribution, agg_k_distribution):
    """Retrieve the cached image with the highest cosine similarity and determine cache hit."""
    with torch.no_grad():
        text_norm = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        image_norm = final_image_embeddings / final_image_embeddings.norm(dim=-1, keepdim=True)
        similarity_scores = torch.matmul(text_norm, image_norm.T)
        similarity_scores = torch.clamp(similarity_scores, min=0)

    similarity_scores = similarity_scores.cpu().numpy().flatten()
    ranked_indices = np.argsort(-similarity_scores)
    highest_clip_score_index = ranked_indices[0]
    highest_clip_score = similarity_scores[highest_clip_score_index]

    k = None
    strength = None
    if highest_clip_score >= 0.3:
        k = 30
        strength = 0.4
        distribution[k] += 1
        agg_k_distribution[k] += 1 
    elif highest_clip_score >= 0.29:
        k = 25
        strength = 0.5
        distribution[k] += 1
        agg_k_distribution[k] += 1
    elif highest_clip_score >= 0.28:
        k = 15
        strength = 0.7
        distribution[k] += 1
        agg_k_distribution[k] += 1
    elif highest_clip_score >= 0.27:
        k = 10
        strength = 0.8
        distribution[k] += 1
        agg_k_distribution[k] += 1
    elif highest_clip_score >= 0.25:
        k = 5
        strength = 0.9
        distribution[k] += 1
        agg_k_distribution[k] += 1

    return highest_clip_score_index, highest_clip_score, k, strength, agg_k_distribution

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint, min_output, max_output):
        self.Kp = Kp  
        self.Ki = Ki  
        self.Kd = Kd  
        self.setpoint = setpoint  
        self.min_output = min_output  
        self.max_output = max_output 
        self.prev_error = 0  
        self.integral = 0  

    def compute(self, measured_value, dt):
        error = self.setpoint - measured_value  
        self.integral += error * dt  
        derivative = (error - self.prev_error) / dt if dt > 0 else 0 
        self.prev_error = error  

        # Compute PID output
        
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        output = np.clip(output, -self.max_output, self.max_output)
        return  output 

    
def get_current_time():
    return time.time()

def get_worker_rref():
    global worker_rref
    return worker_rref

def load_model(model_type, device):
    """Load the specified model type onto the given device."""
    if model_type == "large":
        return StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
        ).to(device)
    elif model_type == "small":
        return SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            variant="bf16",
    torch_dtype=torch.bfloat16,).to(device)

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
    def __init__(self, worker_id, gpu_id, model_type, num_worker, allocation):
        self.worker_id = worker_id
        self.hit_queue = queue.Queue()
        self.miss_queue = queue.Queue()
        self.control_queue = queue.Queue()
        self.running = True  # Used for graceful shutdown
        self.device = f"cuda:{gpu_id}"
        self.model_type = model_type
        self.allocation = allocation
        self.num_worker = num_worker
        self.num_small = None

    def fetch_and_process_requests(self):
        """Continuously fetches requests from the shared queue."""
        seed = 42 #any
        generator = torch.Generator(self.device).manual_seed(seed)
        model = load_model(self.model_type, self.device)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(model.scheduler.config)
        idle_counter = 0
        max_idle_iterations = 60 # Set a threshold for termination
        log_file = f"Throughput_LOG/Ours_throughput_SANA_1.5.log"
        while self.running:
            try:
                if not self.control_queue.empty():
                    self.allocation, self.num_small = self.control_queue.get()
                    small_start = self.num_worker - self.num_small + 1
                    if self.model_type != self.allocation[self.worker_id-1]:
                        if self.model_type == "large":
                            if self.miss_queue.empty():
                                print(f"[Worker {self.worker_id}] Switching to {self.allocation[self.worker_id-1]} model")
                                self.model_type = self.allocation[self.worker_id-1]
                                model = load_model(self.model_type, self.device)
                            else:
                                print(f"miss queue not empty, wait until empty")
                        else:
                            if small_start > self.num_worker:
                                print(f"[Worker {self.worker_id}] Switching to {self.allocation[self.worker_id-1]} model")
                                self.model_type = self.allocation[self.worker_id-1]
                                model = load_model(self.model_type, self.device)
                            else:
                                start_worker = 0
                                while not self.hit_queue.empty():
                                    hit_request = self.hit_queue.get()
                                    worker_rref = rpc.rpc_sync(f"worker_{(start_worker % self.num_small) + small_start}", get_worker_rref)  # ✅ Correct
                                    worker_rref.rpc_sync().enqueue_request_hit(hit_request)  # ✅ Call method remotely
                                    start_worker += 1
                                print(f"[Worker {self.worker_id}] Switching to {self.allocation[self.worker_id-1]} model")
                                self.model_type = self.allocation[self.worker_id-1]
                                model = load_model(self.model_type, self.device)

                if self.model_type == 'large':
                    try:
                        request = self.miss_queue.get_nowait()
                        idle_counter = 0
                    except queue.Empty:
                        if self.model_type != self.allocation[self.worker_id-1]:
                            print(f"[Worker {self.worker_id}] Switching to {self.allocation[self.worker_id-1]} model")
                            self.model_type = self.allocation[self.worker_id-1]
                            model = load_model(self.model_type, self.device)
                        request = self.hit_queue.get(timeout=10)
                        idle_counter = 0
                else:
                    request = self.hit_queue.get(timeout=10)
                    idle_counter = 0

                prompt = request['prompt']
                clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:220]

                print(f"[Worker {self.worker_id}] Processing request: {prompt}")
                if request['k'] is None:
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

                        scheduler_rref = rpc.rpc_sync(f"scheduler", get_scheduler_ref)  # ✅ Correct
                        scheduler_rref.rpc_sync().add_new_cache(generated_image_path)  # ✅ Call method remotely

                    except Exception as e:
                        print(f"Failed to save image: {e}")
                        print("image name:", prompt)
                    end_time = rpc.rpc_sync("scheduler", get_current_time)
                    finish_time = end_time - request['start_time']
                    with open(log_file,"a") as f:
                        f.write(f"[Worker {self.worker_id}] Processed request latency: {finish_time}\n")
                else:
                    try:
                        init_image = Image.open(request['retrieved_image_path']).convert("RGB")
                    except (OSError, IOError) as e:
                        print(f"[Worker {self.worker_id}] Error loading image {request['retrieved_image_path']}: {e}")
                        if self.model_type == "large":
                            wait_time = request['strength'] * 94.5
                            time.sleep(wait_time)
                            end_time = rpc.rpc_sync("scheduler", get_current_time)
                            finish_time = end_time - request['start_time']
                            with open(log_file,"a") as f:
                                f.write(f"[Worker {self.worker_id}] Processed request latency: {finish_time}\n")
                            continue
                        else:
                            wait_time = request['strength'] * 6.5
                            time.sleep(wait_time)
                            end_time = rpc.rpc_sync("scheduler", get_current_time)
                            finish_time = end_time - request['start_time']
                            with open(log_file,"a") as f:
                                f.write(f"[Worker {self.worker_id}] Processed request latency: {finish_time}\n")
                            continue
                    if self.model_type == "large":
                        model_ouputs = model.edit(prompt=prompt,
                            image=init_image,
                            strength=request['strength'],
                            guidance_scale=7.5,
                            negative_prompt=None)
                        generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                        try:
                            model_ouputs.images[0].save(generated_image_path)
                            scheduler_rref = rpc.rpc_sync(f"scheduler", get_scheduler_ref)  # ✅ Correct
                            scheduler_rref.rpc_sync().add_new_cache(generated_image_path)  # ✅ Call method remotely
                        except Exception as e:
                            print(f"Failed to save image: {e}")
                            print("image name:", prompt)
                        end_time = rpc.rpc_sync("scheduler", get_current_time)
                        finish_time = end_time - request['start_time']
                        with open(log_file,"a") as f:
                            f.write(f"[Worker {self.worker_id}] Processed request latency: {finish_time}\n")
                            
                    else:
                        model_ouputs = model.edit(prompt=prompt,
                        image=init_image,
                        strength=request['strength'],
                        guidance_scale=7.5,
                        generator=generator)
 
                        generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                        try:
                            model_ouputs.images[0].save(generated_image_path)
                            scheduler_rref = rpc.rpc_sync(f"scheduler", get_scheduler_ref)  # ✅ Correct
                            scheduler_rref.rpc_sync().add_new_cache(generated_image_path)  # ✅ Call method remotely
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

    def enqueue_request_hit(self, request):
        """Adds a request to the queue."""
        self.hit_queue.put(request)

    def enqueue_request_miss(self, request):
        """Adds a request to the queue."""
        self.miss_queue.put(request)

    def enqueue_control(self, new_allocation, num_small):
        """Adds a request to the queue."""
        self.control_queue.put((new_allocation, num_small))

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
    def __init__(self, worker_ranks, cached_image_paths, processor, clip_model,  final_image_embeddings, num_worker, num_small):
        self.worker_ranks = worker_ranks  # List of worker nodes
        self.current_worker_m = 0
        self.current_worker_h = 0

        self.cached_image_paths = cached_image_paths

        self.processor = processor
        self.clip_model = clip_model
        self.num_worker = num_worker
        self.num_small = num_small
        self.num_large = num_worker - num_small
        self.final_image_embeddings = final_image_embeddings
        self.new_cache_queue = queue.Queue()

    def add_new_cache(self, new_cache):
        self.new_cache_queue.put(new_cache)

    def process_request(self, selected_requests, start_time):
        """Assigns requests to node queues in a round-robin manner."""
        k_distribution = {5: 0, 10: 0, 15: 0, 25: 0, 30: 0}  # Track requests by k values
        agg_k_distribution = {5: 0, 10: 0, 15: 0, 25: 0, 30: 0} 
        hit_count = 0
        request_count = 0
        num_of_large = self.num_large
        last_check_time = time.time()
        large_model_throughput = 1 / (91.5/60)  # Requests/min
        small_model_throughput = 1 / (6.5/60)  # Requests/min
        Kp, Ki, Kd = 0.6, 0.05, 0.05  # PID tuning parameters (adjusts model allocation)
        pid_controller = PIDController(Kp, Ki, Kd, setpoint=0, min_output=-self.num_worker, max_output=self.num_worker)
        for idx, row in selected_requests.iterrows():
            while time.time() - start_time < row['seconds_from_start']:
                time.sleep(0.1)
            # request_arrival_time = time.time()  # Start time of request processing

            row['start_time'] = start_time  # Add start_time to request
            batch_new_image_paths = []
            while not self.new_cache_queue.empty():
                batch_new_image_paths.append(self.new_cache_queue.get())
            if batch_new_image_paths:
                new_images = []
                for image_path in batch_new_image_paths:
                    # Wait a bit to ensure the file is fully written
                    time.sleep(0.1)
                    try:
                        img = Image.open(image_path).convert("RGB")
                        new_images.append(img)
                    except (OSError, IOError) as e:
                        print(f"Error loading image {image_path}: {e}")
            else:
                new_images = []
            if new_images:
                new_image_tensors = self.processor(images=new_images, return_tensors="pt").to(self.clip_model.device)

                with torch.no_grad():
                    new_image_embeddings = self.clip_model.get_image_features(**new_image_tensors)

                # Concatenate new embeddings and add new paths
                self.final_image_embeddings = torch.cat((self.final_image_embeddings, new_image_embeddings), dim=0)
                self.cached_image_paths.extend(batch_new_image_paths)
            
            # Evict old entries if cache exceeds the size
            if self.final_image_embeddings.size(0) > cache_size:
                # Calculate the number of items to evict
                num_to_evict = self.final_image_embeddings.size(0) - cache_size

                # Remove the oldest embeddings and cached paths
                self.final_image_embeddings = self.final_image_embeddings[num_to_evict:]
                self.cached_image_paths = self.cached_image_paths[num_to_evict:]

            prompt = [row['prompt']]
            # Process the prompt embedding
            texts = self.processor(text=prompt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(self.clip_model.device)
            with torch.no_grad():
                text_embedding = self.clip_model.get_text_features(**texts)

            best_index, best_score, k, strength, agg_k_distribution= retrieve_best_image(text_embedding, self.final_image_embeddings, k_distribution, agg_k_distribution)
            print(agg_k_distribution)

            if k is not None:
                hit_count += 1  # Increment hit count
                row['retrieved_image_index'] = best_index
                row['retrieved_image_score'] = best_score
                row['retrieved_image_path'] = self.cached_image_paths[best_index]  # Retrieve path based on best index
                row['k'] = k
                row['strength'] = strength
                # hit_queue.put(row.to_dict())
                if self.num_small == 0:
                    self.current_worker_m = self.current_worker_m % self.num_large
                    assigned_worker = self.worker_ranks[self.current_worker_m]
                    self.current_worker_m = (self.current_worker_m + 1) % self.num_large
                    worker_rref = rpc.rpc_sync(f"{assigned_worker}", get_worker_rref)  # ✅ Correct
                    worker_rref.rpc_sync().enqueue_request_hit(row)  # ✅ Call method remotely
                else:
                    self.current_worker_h = self.current_worker_h % self.num_small
                    assigned_worker = self.worker_ranks[self.num_worker - self.current_worker_h - 1]
                    self.current_worker_h = (self.current_worker_h + 1) % self.num_small
                    worker_rref = rpc.rpc_sync(f"{assigned_worker}", get_worker_rref)  # ✅ Correct
                    worker_rref.rpc_sync().enqueue_request_hit(row)  # ✅ Call method remotely
            else:
                row['retrieved_image_index'] = None
                row['retrieved_image_score'] = best_score
                row['retrieved_image_path'] = None
                row['k'] = None
                row['strength'] = None
                # nonhit_queue.put(row.to_dict())
                self.current_worker_m = self.current_worker_m % self.num_large
                assigned_worker = self.worker_ranks[self.current_worker_m]
                self.current_worker_m = (self.current_worker_m + 1) % self.num_large
                worker_rref = rpc.rpc_sync(f"{assigned_worker}", get_worker_rref)  # ✅ Correct
                worker_rref.rpc_sync().enqueue_request_miss(row)  # ✅ Call method remotely




            request_count += 1

            current_time = time.time()

            if current_time - last_check_time >= 180:
                elapsed_time = current_time - last_check_time
                # request_rate = (request_count / elapsed_time * 60) if elapsed_time > 0 else 0
                hit_rate = hit_count / request_count if request_count > 0 else 0
                k_rates = {k: count / hit_count for k, count in k_distribution.items()}  # Normalize k counts

                # Combine all metrics into a single dictionary
                # metrics = {
                #     "request_rate": request_rate,
                #     "hit_rate": hit_rate,
                #     "k_rates": k_rates
                # }
                dt = elapsed_time / 60
                request_count = 0
                hit_count = 0
                nonhit_workload = (1 - hit_rate)
                k_rate = 0
                for k, rate in k_rates.items():
                    print(f"Rate for k={k}: {rate:.2%}")
                    k_rate += rate * (1 - k / 50)
                hit_workload = hit_rate * k_rate * large_model_throughput / small_model_throughput
                total_workload = hit_workload + nonhit_workload

                min_large_models = int(round(nonhit_workload / total_workload * self.num_worker))
                num_large = min_large_models
                # while num_large <= self.num_worker:
                # # Check if this allocation satisfies the hit workload constraint
                #     available_throughput = (
                #         num_large * large_model_throughput
                #         - nonhit_workload
                #         + (self.num_worker - num_large) * small_model_throughput
                #     )
                #     if 0.8 * available_throughput >= hit_workload:
                #         num_large += 1  # Try the next larger allocation
                #         continue
                #     else:
                #         num_large -= 1  # Step back to the largest valid configuration
                #         break
                num_large = max(1, min(num_large, self.num_worker-1))
                print("exact number of large:", num_large)
                pid_controller.setpoint = num_large
                num_large_add = pid_controller.compute(num_of_large, dt)
                print("delta:",num_large_add )
                num_of_large += num_large_add
                print("real number of large:",num_of_large )
                N_large = int(round(num_of_large))
                print("allocated number of large:",N_large )
                self.num_large = max(1, min(N_large, self.num_worker-1))
                self.num_small = self.num_worker - self.num_large
                print(f"[Global Monitor] Allocating {self.num_large} large models and {self.num_small} small models.")
                model_allocation = ["large"] * self.num_large + ["small"] * self.num_small
                for worker in self.worker_ranks:
                    worker_rref = rpc.rpc_sync(f"{worker}", get_worker_rref)  # ✅ Correct
                    worker_rref.rpc_sync().enqueue_control(new_allocation=model_allocation, num_small=self.num_small)  # ✅ Call method remotely
                for k in k_distribution.keys():
                    k_distribution[k] = 0  # Reset k distribution
                last_check_time = current_time
            time.sleep(1.5)

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
    num_large = args.num_workers - args.num_small
    small_start = args.num_workers - args.num_small + 1
    model_allocation = ["large"] * int(num_large) + ["small"] * int(args.num_small)
    # ✅ Create worker and register RRef
    if rank >= small_start:
        node_worker = NodeWorker(worker_id=rank, gpu_id=gpu_id, model_type='small', num_worker=args.num_workers, allocation=model_allocation)
    else:
        node_worker = NodeWorker(worker_id=rank, gpu_id=gpu_id, model_type='large', num_worker=args.num_workers, allocation=model_allocation)
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
        world_size=int(os.environ["WORLD_SIZE"]),  # 1 scheduler + 16 worker nodes
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions()
        )
        metadata_df = pd.read_parquet(args.metadata_path)

        device = "cuda:0"
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        np.random.seed(42)
        if 'timestamp' in metadata_df.columns:
            sorted_df = metadata_df.sort_values(by='timestamp')
            
            # Set all seconds_from_start to zero
            sorted_df['seconds_from_start'] = 0

            # Display modified DataFrame
            print(sorted_df[['timestamp', 'seconds_from_start']].head())
        else:
            print("No timestamp column found in the DataFrame.")


        sorted_df = sorted_df.sort_values(by='seconds_from_start')
        selected_requests = sorted_df.iloc[50000:55000].copy()
        # Force all timestamps to be zero
        selected_requests['seconds_from_start'] = 0

        image_directory = "/work1/talati/stilex/cache_diffusion/nohit"
        # Get a list of all image file paths in the directory
        cached_image_paths = [os.path.join(image_directory, img_file) for img_file in os.listdir(image_directory) if img_file.endswith(('.png', '.jpg', '.jpeg'))]
            # Batch size
        # batch_size = 128
        # num_batches = (len(cached_image_paths) + batch_size - 1) // batch_size

        # # Placeholder to store all image embeddings
        # image_embeddings = []

        # # Process each batch
        # for batch_idx in tqdm(range(num_batches)):
        #     # Select batch slice
        #     batch_start = batch_idx * batch_size
        #     batch_end = min((batch_idx + 1) * batch_size, len(cached_image_paths))
        #     batch_image_paths = cached_image_paths[batch_start:batch_end]

        #     # Load and preprocess images
        #     images = [Image.open(image_path).convert("RGB") for image_path in batch_image_paths]
        #     image_tensors = processor(images=images, return_tensors="pt").to(device)

        #     with torch.no_grad():
        #         # Generate image embeddings
        #         image_embeddings_batch = clip_model.get_image_features(**image_tensors)
        #         image_embeddings.append(image_embeddings_batch)

        # # Concatenate all image embeddings into a single tensor
        # final_image_embeddings = torch.cat(image_embeddings, dim=0)
        # torch.save(final_image_embeddings, "final_image_embeddings.pt")
        # print(f"Generated embeddings for {len(cached_image_paths)} images.")
        final_image_embeddings = torch.load("final_image_embeddings.pt", map_location=device)

        print(final_image_embeddings.shape) 

        # ✅ Scheduler Process (Runs on First Node)
        node_ranks = [f"worker_{i}" for i in range(1, args.num_workers + 1)]
        global worker_status_tracker
        worker_status_tracker = WorkerStatusTracker(node_ranks)
        global scheduler_rref
        time.sleep(60)

        start_time = time.time()
        scheduler = Scheduler(node_ranks, cached_image_paths, processor, clip_model, final_image_embeddings, int(args.num_workers), int(args.num_small))
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
