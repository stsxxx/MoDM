import torch
import time
import queue
import multiprocessing as mp
import pandas as pd
from diffusers import StableDiffusion3Pipeline, StableDiffusionXLImg2ImgPipeline, DiffusionPipeline, SanaPipeline, StableDiffusionXLPipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import re
import gc
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import numpy as np
from math import floor
import json
import csv


parser = argparse.ArgumentParser(description="MoDM parameters")
parser.add_argument("--large_model", type=str, required=True, help="which large model you wanna use")
parser.add_argument("--small_model", type=str, required=True, help="which small model you wanna use")
parser.add_argument("--num_req", type=int, default=10000, required=True, help="number of requests")
parser.add_argument("--cache_size", type=int, default=10000, help="cache size")
parser.add_argument("--warm_up_size", type=int, default=1000, required=True, help="number of warmup requests")
parser.add_argument("--cache_directory", type=str, required=False, help="directory of cached images")
parser.add_argument("--image_directory", type=str, required=False, default='./MoDM_images', help="directory of generated images")
parser.add_argument("--dataset", type=str, default='diffusiondb', required=False, help="dataset")
args = parser.parse_args()


directory = args.image_directory
os.makedirs(directory, exist_ok=True)
print(f"Directory '{directory}' created successfully.")


def load_model(model_type, device):
    """Load the specified model type onto the given device."""
    if model_type == "sd3.5":
        return StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16
        ).to(device)
    elif model_type == "sdxl":
        return StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to(device)
    elif model_type == "flux":
        return DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    elif model_type == "sana":
        return SanaPipeline.from_pretrained(
            "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
            variant="bf16",
            torch_dtype=torch.bfloat16,
        ).to(device)
    else:
        print("unknown model.")

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

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def precompute_timesteps_for_labels_flux(scheduler, labels, device, index,mu,sigmas):
    timesteps = []
    for label in labels:
        if label == 0:
            # For label 0, use full 50 iterations
            scheduler.set_timesteps(num_inference_steps=50, device=device,mu=mu, sigmas=sigmas)
            timesteps.append(scheduler.timesteps)
        elif label == 1:
            # For label 1, use last 40 iterations
            
            scheduler.set_timesteps(num_inference_steps=50, device=device, mu=mu, sigmas=sigmas)
            timesteps.append(scheduler.timesteps[index:])         
        else:
            timesteps.append([])  

    return timesteps

def extract_prompt(filename):
    """Extract prompt from filename by replacing underscores with spaces."""
    name = os.path.splitext(os.path.basename(filename))[0]  # Remove extension
    return re.sub(r'[_]+', ' ', name).strip()  # Convert underscores to spaces


if __name__ == "__main__":
    if args.cache_directory and not os.path.exists(args.cache_directory):
        print(f"The pre-cache directory {args.cache_directory} does not exist.")
        
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    warm_up_size = args.warm_up_size

    if args.dataset == 'diffusiondb':
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
        selected_requests = sorted_df.iloc[50000-warm_up_size:50000 + args.num_req].copy()
        # Force all timestamps to be zero
        selected_requests['seconds_from_start'] = 0
        num_requests = len(selected_requests)
            # Print each value with its index
        for i, seconds in enumerate(selected_requests['seconds_from_start']):
            print(f"Request {i+1}: {seconds:.2f}")

    elif args.dataset == "MJHQ":
        meta_data_path = "./MoDM_cache/MJHQ/meta_data.json"
        # Load metadata
        with open(meta_data_path, "r") as f:
            meta_data = json.load(f)
        meta_keys = list(meta_data.keys())
        # Select last 500 from each 3000-chunk
        selected_keys = []
        chunk_size = 3000
        num_chunks = len(meta_data) // chunk_size # 10 chunks
        for i in range(num_chunks):
            start = int((i + 1) * chunk_size - (warm_up_size + args.num_req)/ 10)
            end = int((i + 1) * chunk_size - args.num_req / 10)
            selected_keys.extend(meta_keys[start:end])
        for i in range(num_chunks):
            start = int((i + 1) * chunk_size - args.num_req / 10)
            end = int((i + 1) * chunk_size)
            selected_keys.extend(meta_keys[start:end])
        # Create subset dictionary
        selected_requests = {key: meta_data[key] for key in selected_keys}
        for key in selected_requests:
            selected_requests[key]['seconds_from_start'] = 0 
        selected_requests = pd.DataFrame.from_dict(selected_requests, orient='index')


    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    final_image_embeddings = None
    cached_image_paths = []

    if args.cache_directory and os.path.exists(args.cache_directory):

        image_directory = args.cache_directory

        # Get a list of all image file paths in the directory
        cached_image_paths = [os.path.join(image_directory, img_file) for img_file in os.listdir(image_directory) if img_file.endswith(('.png', '.jpg', '.jpeg'))]

        # Batch size
        batch_size = 128
        num_batches = (len(cached_image_paths) + batch_size - 1) // batch_size

        # Placeholder to store all image embeddings
        image_embeddings = []

        # Process each batch
        for batch_idx in tqdm(range(num_batches)):
            # Select batch slice
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(cached_image_paths))
            batch_image_paths = cached_image_paths[batch_start:batch_end]

            # Load and preprocess images
            images = [Image.open(image_path).convert("RGB") for image_path in batch_image_paths]
            image_tensors = processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                # Generate image embeddings
                image_embeddings_batch = model.get_image_features(**image_tensors)
                image_embeddings.append(image_embeddings_batch)

        # Concatenate all image embeddings into a single tensor
        final_image_embeddings = torch.cat(image_embeddings, dim=0)
        torch.save(final_image_embeddings, f"final_image_embeddings_{args.dataset}_{args.large_model}.pt")
        print(f"Generated embeddings for {len(cached_image_paths)} images.")

    seed = 42 #any
    generator = torch.Generator(device).manual_seed(seed)
    large_model = load_model(args.large_model, device)
    small_model = load_model(args.small_model, device)


    start_time = time.time()

    if warm_up_size != 0:
        
        for i, row in selected_requests.head(warm_up_size).iterrows():

            prompt = row['prompt']
            clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:230]
            # nonhit_queue.put(row.to_dict())
            
            if args.large_model == 'sd3.5':
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(large_model.scheduler.config)
                timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [0], "cpu",0)[0]
                prompt_embeds, pooled_prompt_embeds, latents = large_model.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["latents"])
                model_outputs = large_model(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
                labels_batch = 0, current_latents=latents, height=1024, width=1024)
                
                generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                new_images = model_outputs[1][0]
                try:
                    new_images.save(generated_image_path)
                    # shared_new_images.put(generated_image_path)
                    
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    print("image name:", prompt)

            elif args.large_model == 'flux':

                model_outputs = large_model(prompt = prompt,generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None, pre_computed_timesteps = None, num_inference_steps =50,
                latents=None, height=1024, width=1024, hit=False,)

                generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                new_images = model_outputs[1].images[0]
                try:
                    new_images.save(generated_image_path)  
                    # shared_new_images.put(generated_image_path)
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    print("image name:", prompt)

            new_image_tensors = processor(images=new_images, return_tensors="pt").to(device)

            with torch.no_grad():
                new_image_embeddings = model.get_image_features(**new_image_tensors)

            if final_image_embeddings is not None:
                # Concatenate new embeddings and add new paths
                final_image_embeddings = torch.cat((final_image_embeddings, new_image_embeddings), dim=0)
                cached_image_paths.append(generated_image_path)
            else:
                final_image_embeddings = new_image_embeddings.clone()
                cached_image_paths.append(generated_image_path)
                    
            # Evict old entries if cache exceeds the size
            if final_image_embeddings is not None and final_image_embeddings.size(0) >= warm_up_size:
                break
            if final_image_embeddings is not None and final_image_embeddings.size(0) > args.cache_size:
                # Calculate the number of items to evict
                num_to_evict = final_image_embeddings.size(0) - args.cache_size

                # Remove the oldest embeddings and cached paths
                final_image_embeddings = final_image_embeddings[num_to_evict:]
                cached_image_paths = cached_image_paths[num_to_evict:]

        torch.cuda.empty_cache()
        gc.collect()


    k_distribution = {5: 0, 10: 0, 15: 0, 25: 0, 30: 0}  
    agg_k_distribution = {5: 0, 10: 0, 15: 0, 25: 0, 30: 0} 


    prompt_to_path = {}

    start_time = time.time()

    # print(selected_requests)
    for i, row in selected_requests.iloc[warm_up_size:].iterrows():

        s_time = time.time()

        prompt = row['prompt']        
        clean_prompt = re.sub(r'[^\w\-_\.]', '_', prompt)[:230]
        host_generated_image_path = f"Generated_images/{clean_prompt}.png"
        prompt_to_path[prompt] = host_generated_image_path

        texts = processor(text=prompt, return_tensors="pt", truncation=True, padding=True, max_length=77).to(device)
        with torch.no_grad():
            text_embedding = model.get_text_features(**texts)

        if final_image_embeddings is not None:
            best_index, best_score, k, strength, agg_k_distribution= retrieve_best_image(text_embedding, final_image_embeddings, k_distribution, agg_k_distribution)
            if k is not None:
                print("hit")
                print(agg_k_distribution)

                init_image = Image.open(cached_image_paths[best_index]).convert("RGB")

                if args.small_model == "sdxl":
                    model_ouputs = small_model(prompt=prompt,
                        image=init_image,
                        strength=strength,
                        guidance_scale=7.5,
                        negative_prompt=None)
                    generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                    try:
                        model_ouputs.images[0].save(generated_image_path)
                        # shared_new_images.put(generated_image_path)
                    except Exception as e:
                        print(f"Failed to save image: {e}")
                        print("image name:", prompt)
                        
                elif args.small_model == "sana":
                    model_ouputs = small_model.edit(prompt=prompt,
                        image=init_image,
                        strength=strength,
                        guidance_scale=7.5,
                        generator=generator)
                    generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                    try:
                        model_ouputs.images[0].save(generated_image_path)
                        # shared_new_images.put(generated_image_path)
                    except Exception as e:
                        print(f"Failed to save image: {e}")
                        print("image name:", prompt)
                torch.cuda.empty_cache()
                gc.collect()
            else:
                if args.large_model == 'sd3.5':
                    scheduler = FlowMatchEulerDiscreteScheduler.from_config(large_model.scheduler.config)
                    timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [0], "cpu",0)[0]
                    prompt_embeds, pooled_prompt_embeds, latents = large_model.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["latents"])
                    model_outputs = large_model(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
                    callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
                    labels_batch = 0, current_latents=latents, height=1024, width=1024)
                    
                    generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                    new_images = model_outputs[1][0]
                    try:
                        new_images.save(generated_image_path)
                        # shared_new_images.put(generated_image_path)
                        
                    except Exception as e:
                        print(f"Failed to save image: {e}")
                        print("image name:", prompt)

                elif args.large_model == 'flux':

                    model_outputs = large_model(prompt = prompt,generator=generator, callback_on_step_end=None,
                    callback_on_step_end_tensor_inputs=None, pre_computed_timesteps = None, num_inference_steps =50,
                    latents=None, height=1024, width=1024, hit=False,)

                    generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                    new_images = model_outputs[1].images[0]
                    try:
                        new_images.save(generated_image_path)  
                        # shared_new_images.put(generated_image_path)
                    except Exception as e:
                        print(f"Failed to save image: {e}")
                        print("image name:", prompt)



                new_image_tensors = processor(images=new_images, return_tensors="pt").to(device)

                with torch.no_grad():
                    new_image_embeddings = model.get_image_features(**new_image_tensors)

                if final_image_embeddings is not None:
                    # Concatenate new embeddings and add new paths
                    final_image_embeddings = torch.cat((final_image_embeddings, new_image_embeddings), dim=0)
                    cached_image_paths.append(generated_image_path)
                else:
                    final_image_embeddings = new_image_embeddings.clone()
                    cached_image_paths.append(generated_image_path)
                        
                # Evict old entries if cache exceeds the size

                if final_image_embeddings is not None and final_image_embeddings.size(0) > args.cache_size:
                    # Calculate the number of items to evict
                    num_to_evict = final_image_embeddings.size(0) - args.cache_size

                    # Remove the oldest embeddings and cached paths
                    final_image_embeddings = final_image_embeddings[num_to_evict:]
                    cached_image_paths = cached_image_paths[num_to_evict:]

                torch.cuda.empty_cache()
                gc.collect()        

        else:
            if args.large_model == 'sd3.5':
                scheduler = FlowMatchEulerDiscreteScheduler.from_config(large_model.scheduler.config)
                timesteps_batch = precompute_timesteps_for_labels_35(scheduler, [0], "cpu",0)[0]
                prompt_embeds, pooled_prompt_embeds, latents = large_model.input_process(prompt = prompt,negative_prompt = None, generator=generator, callback_on_step_end=None,
            callback_on_step_end_tensor_inputs=["latents"])
                model_outputs = large_model(prompt = prompt, prompt_embeds = prompt_embeds, pooled_prompt_embeds = pooled_prompt_embeds, generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=["current_latents"], timesteps_batch = timesteps_batch,
                labels_batch = 0, current_latents=latents, height=1024, width=1024)
                
                generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                new_images = model_outputs[1][0]
                try:
                    new_images.save(generated_image_path)
                    # shared_new_images.put(generated_image_path)
                    
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    print("image name:", prompt)

            elif args.large_model == 'flux':

                model_outputs = large_model(prompt = prompt,generator=generator, callback_on_step_end=None,
                callback_on_step_end_tensor_inputs=None, pre_computed_timesteps = None, num_inference_steps =50,
                latents=None, height=1024, width=1024, hit=False,)

                generated_image_path = f"{args.image_directory}/{clean_prompt}.png"
                new_images = model_outputs[1].images[0]
                try:
                    new_images.save(generated_image_path)  
                    # shared_new_images.put(generated_image_path)
                except Exception as e:
                    print(f"Failed to save image: {e}")
                    print("image name:", prompt)


            new_image_tensors = processor(images=new_images, return_tensors="pt").to(device)

            with torch.no_grad():
                new_image_embeddings = model.get_image_features(**new_image_tensors)

            if final_image_embeddings is not None:
                # Concatenate new embeddings and add new paths
                final_image_embeddings = torch.cat((final_image_embeddings, new_image_embeddings), dim=0)
                cached_image_paths.append(generated_image_path)
            else:
                final_image_embeddings = new_image_embeddings.clone()
                cached_image_paths.append(generated_image_path)
                    
            # Evict old entries if cache exceeds the size

            if final_image_embeddings is not None and final_image_embeddings.size(0) > args.cache_size:
                # Calculate the number of items to evict
                num_to_evict = final_image_embeddings.size(0) - args.cache_size

                # Remove the oldest embeddings and cached paths
                final_image_embeddings = final_image_embeddings[num_to_evict:]
                cached_image_paths = cached_image_paths[num_to_evict:]
                
            torch.cuda.empty_cache()
            gc.collect()


        e_time = time.time()
        latency = time.time() - s_time
        with open("latency_log.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([latency])


    end_time = time.time()
    duration = end_time - start_time

    print(agg_k_distribution)

    throughput = args.num_req / duration * 60
    print(f"\n📈 Total Time: {duration:.4f} seconds")
    print(f"Throughput: {throughput:.2f} requests/min")
    with open("../Logs/prompt_to_path.json", "w") as f:
        json.dump(prompt_to_path, f, indent=2)
            


