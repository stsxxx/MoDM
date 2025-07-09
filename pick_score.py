import torch
import os
import re
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

# Load PickScore model
device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)

def get_image_paths(directory):
    """Get all image file paths from a directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

def extract_prompt(filename):
    """Extract prompt from filename by replacing underscores with spaces."""
    name = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r'[_]+', ' ', name).strip()

def compute_pick_scores_batch(image_dir, batch_size=8):
    """Compute PickScores for all images in a directory using batch processing."""
    
    image_paths = get_image_paths(image_dir)
    if not image_paths:
        print(f"[Warning] No images found in {image_dir}")
        return None

    prompts = [extract_prompt(path) for path in image_paths]
    total_score = 0.0
    num_images = len(image_paths)

    for i in tqdm(range(0, num_images, batch_size), desc=f"Processing {os.path.basename(image_dir)}"):
        batch_paths = image_paths[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]
        batch_images = []
        valid_prompts = []

        for path, prompt in zip(batch_paths, batch_prompts):
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
                valid_prompts.append(prompt)
            except Exception as e:
                print(f"[Warning] Skipping image {path} due to error: {e}")

        # inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True, truncation=True).to(device)
        image_inputs = processor(
            images=batch_images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        text_inputs = processor(
            text=valid_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

            scores = (model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)).cpu()
            print(scores)
        total_score += scores.sum().item()

    avg_score = total_score / num_images if num_images > 0 else 0
    return avg_score

# === Directory configuration ===
directories = {
    "NIRVANA_MJHQ": "./images/NIRVANA_throughput_MJHQ",
    "NIRVANA_DiffusionDB": "./images/NIRVANA_throughput_diffusionDB",
    "Vanilla_MJHQ": "./images/Vanilla_throughput_MJHQ",
    "Vanilla_diffusionDB": "./images/Vanilla_throughput_diffusionDB",
    "MoDM_sdxl_MJHQ": "./images/MoDM_throughput_MJHQ_sdxl",
    "MoDM_sana_MJHQ": "./images/MoDM_throughput_MJHQ_sana",
    "MoDM_sdxl_diffusionDB": "./images/MoDM_throughput_diffusionDB_sdxl",
    "MoDM_sana_diffusionDB": "./images/MoDM_throughput_diffusionDB_sana",
}


# === Compute and report PickScores ===
results = {}
for name, path in directories.items():
    avg_score = compute_pick_scores_batch(path, batch_size=8)
    if avg_score is not None:
        results[name] = avg_score
        print(f"\n{name} Directory - Average PickScore: {avg_score:.4f}")

print("\n=== Final PickScore Results ===")
for name, score in results.items():
    print(f"{name}: {score:.4f}")
