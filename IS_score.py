#!/usr/bin/env python3
"""
Inception Score (IS) Calculator
Calculates IS score for images in a specified directory using pre-trained Inception-v3 model.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
import argparse
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

class ImageDataset(Dataset):
    """Custom dataset for loading images from directory"""
    
    def __init__(self, image_dir: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        self.image_files = []
        
        for file in os.listdir(image_dir):
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                self.image_files.append(file)
        
        if not self.image_files:
            raise ValueError(f"No valid image files found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            blank_image = Image.new('RGB', (299, 299), color='white')
            if self.transform:
                blank_image = self.transform(blank_image)
            return blank_image

def load_inception_model(device: torch.device):
    """Load pre-trained Inception-v3 model"""
    model = inception_v3(pretrained=True, transform_input=False)
    model.eval()
    model.to(device)
    return model

def get_predictions(model, dataloader, device: torch.device) -> np.ndarray:
    """Get predictions from Inception model for all images with debugging"""
    predictions = []
    
    print("Getting predictions from Inception model...")
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Get logits from inception model
            try:
                with torch.cuda.amp.autocast():
                    logits = model(batch)
                
                # Check for issues in logits
                if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
                    print(f"Warning: Found NaN/inf in logits at batch {i}")
                
                # Convert to probabilities with temperature scaling for stability
                # Use temperature=1.0 for standard softmax
                probs = F.softmax(logits / 1.0, dim=1)
                
                # Additional check after softmax
                if torch.any(torch.isnan(probs)) or torch.any(torch.isinf(probs)):
                    print(f"Warning: Found NaN/inf in probabilities at batch {i}")
                
                predictions.append(probs.cpu().numpy())
                
                if i % 50 == 0:  # Progress update every 50 batches
                    print(f"Processed batch {i+1}/{len(dataloader)}")
                    
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Create dummy probabilities if batch fails
                batch_size = batch.shape[0]
                dummy_probs = torch.ones(batch_size, 1000) / 1000  # Uniform distribution
                predictions.append(dummy_probs.cpu().numpy())
    
    all_predictions = np.concatenate(predictions, axis=0)
    print(f"Total predictions shape: {all_predictions.shape}")
    
    return all_predictions

def calculate_inception_score(predictions: np.ndarray, splits: int = 10) -> Tuple[float, float]:
    """
    Calculate Inception Score from predictions with improved numerical stability
    
    Args:
        predictions: Array of shape (N, num_classes) with probability distributions
        splits: Number of splits for calculating mean and std
    
    Returns:
        Tuple of (mean_is, std_is)
    """
    N = predictions.shape[0]
    
    # Debug: Check for issues in predictions
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions min/max: {predictions.min():.6f} / {predictions.max():.6f}")
    print(f"Predictions sum (should be ~1 per row): {predictions.sum(axis=1)[:5]}")
    
    # Check for NaN or inf values
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        print("WARNING: Found NaN or inf values in predictions!")
        # Replace NaN/inf with uniform distribution
        predictions = np.nan_to_num(predictions, nan=1.0/predictions.shape[1], 
                                  posinf=1.0, neginf=0.0)
    
    # Ensure probabilities are normalized and positive
    epsilon = 1e-16
    predictions = np.maximum(predictions, epsilon)
    predictions = predictions / predictions.sum(axis=1, keepdims=True)
    
    # Calculate marginal distribution p(y) with numerical stability
    marginal = np.mean(predictions, axis=0)
    marginal = np.maximum(marginal, epsilon)
    marginal = marginal / marginal.sum()  # Ensure normalized
    
    print(f"Marginal distribution sum: {marginal.sum()}")
    print(f"Marginal min/max: {marginal.min():.8f} / {marginal.max():.8f}")
    
    # Calculate IS for each split
    scores = []
    split_size = N // splits
    
    for i in range(splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i < splits - 1 else N
        
        split_predictions = predictions[start_idx:end_idx]
        
        # Calculate KL divergence for each image in split
        kl_divergences = []
        for j in range(split_predictions.shape[0]):
            p_yx = split_predictions[j]
            
            # Ensure numerical stability
            p_yx = np.maximum(p_yx, epsilon)
            p_yx = p_yx / p_yx.sum()  # Renormalize
            
            # Calculate KL divergence: KL(p_yx || marginal)
            # Using stable computation: sum(p * (log(p) - log(q)))
            log_p = np.log(p_yx)
            log_q = np.log(marginal)
            
            kl_div = np.sum(p_yx * (log_p - log_q))
            
            # Check for numerical issues
            if np.isnan(kl_div) or np.isinf(kl_div):
                print(f"Warning: Invalid KL divergence at split {i}, image {j}")
                kl_div = 0.0
            
            kl_divergences.append(kl_div)
        
        # Calculate mean KL divergence for this split
        mean_kl = np.mean(kl_divergences)
        
        print(f"Split {i+1}: Mean KL = {mean_kl:.6f}", end="")
        
        # IS score is exp(mean KL divergence)
        # Clip to prevent overflow
        mean_kl = np.clip(mean_kl, -50, 50)  # Prevent exp overflow
        is_score = np.exp(mean_kl)
        
        print(f", IS = {is_score:.6f}")
        
        scores.append(is_score)
    
    scores = np.array(scores)
    
    # Final check for valid scores
    valid_scores = scores[~np.isnan(scores) & ~np.isinf(scores)]
    
    if len(valid_scores) == 0:
        print("ERROR: All IS scores are invalid!")
        return float('nan'), float('nan')
    
    if len(valid_scores) < len(scores):
        print(f"Warning: {len(scores) - len(valid_scores)} invalid scores removed")
    
    return np.mean(valid_scores), np.std(valid_scores)

def main():
    parser = argparse.ArgumentParser(description='Calculate Inception Score for images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing images')
    parser.add_argument('--splits', type=int, default=10,
                        help='Number of splits for IS calculation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use: cuda, cpu, or auto')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")

    
    # Define transforms for Inception-v3
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception-v3 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
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

    
    # Load Inception model
    print("Loading Inception-v3 model...")
    model = load_inception_model(device)
    
    results = {}
    for name, path in directories.items():
        if not os.path.exists(path):
            print(f"[WARNING] Directory {path} does not exist, skipping...")
            continue
        print(f"\nProcessing directory: {name}")
        dataset = ImageDataset(path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        preds = get_predictions(model, dataloader, device)
        mean_is, std_is = calculate_inception_score(preds, splits=10)
        results[name] = (mean_is, std_is)
        print(f"\n{name}: IS = {mean_is:.4f} ± {std_is:.4f}")

    print("\n=== Final Inception Score Results ===")
    for name, (mean_is, std_is) in results.items():
        print(f"{name}: IS = {mean_is:.4f} ± {std_is:.4f}")
    
if __name__ == "__main__":
    main()