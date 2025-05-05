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
# base_path = "/data6/stilex/cache_hit_image/final_check"
# directories = {
#     "OG": os.path.join(base_path, "og"),
#     "Small": os.path.join(base_path, "small"),
#     "Large": os.path.join(base_path, "large")
# }
# base_path = "/data6/stilex/large_small"
# directories = {
#     "OG": os.path.join(base_path, "nohit"),
#     "k=10": os.path.join(base_path, "small_10"),
#     "k=15": os.path.join(base_path, "small_15"),
#     "k=20": os.path.join(base_path, "small_20"),
#     "k=25": os.path.join(base_path, "small_25"),
#     "k=30": os.path.join(base_path, "small_30"),
# }
# directories = {
#     "N": "/data6/stilex/serving/NIRVANA/nohit",
#     "O": "/data6/stilex/serving/nohit"
# }

directories = {
    # "NIRVANA": "/data6/stilex/throughput/NIRVANA",
    # "OURS": "/data6/stilex/throughput/Ours",
    # "VANILLA": "/data6/stilex/throughput/baseline"
    'flux': '/home/stilex/Diffusion_Opt/serving/throughput/Images_MoDM'
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
