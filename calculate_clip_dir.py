import torch
import os
import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def get_image_paths(directory):
    """Get all image file paths from a directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

def extract_prompt(filename):
    """Extract prompt from filename by replacing underscores with spaces."""
    name = os.path.splitext(os.path.basename(filename))[0]  # Remove extension
    return re.sub(r'[_]+', ' ', name).strip()  # Convert underscores to spaces

def compute_clip_scores_batch(image_dir, batch_size=16):
    """Compute CLIP scores for all images in a directory using batch processing."""
    
    # Get image paths and prompts
    image_paths = get_image_paths(image_dir)
    if not image_paths:
        print(f"[Warning] No images found in {image_dir}")
        return None  # No images in directory

    prompts = [extract_prompt(path) for path in image_paths]
    total_score = 0.0
    num_images = len(image_paths)

    # Process in batches
    for i in tqdm(range(0, num_images, batch_size), desc=f"Processing {os.path.basename(image_dir)}"):
        batch_paths = image_paths[i:i+batch_size]
        batch_prompts = prompts[i:i+batch_size]

        # Load and preprocess images
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(text=batch_prompts, images=images, return_tensors="pt",truncation=True, padding=True,max_length=77).to(device)

        # Compute CLIP logits
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: (batch_size, batch_size)

        # Extract diagonal values (matching text-image pairs)
        batch_scores = logits_per_image.diag().cpu().numpy()

        # Sum up scores
        total_score += batch_scores.sum()

    # Compute average CLIP score
    avg_clip_score = total_score / num_images if num_images > 0 else 0
    return avg_clip_score

# Define directories

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
# Compute and print average CLIP scores for each directory
results = {}
for name, path in directories.items():
    avg_score = compute_clip_scores_batch(path, batch_size=16)
    if avg_score is not None:
        results[name] = avg_score
        print(f"\n{name} Directory - Average CLIP Score: {avg_score:.4f}")

# Final Summary
print("\n=== Final CLIP Score Results ===")
for name, score in results.items():
    print(f"{name}: {score:.4f}")
