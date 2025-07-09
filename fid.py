import os
from pytorch_fid import fid_score

# Paths to the two directories
path1 = ""
path2 = ""
path3 = ""

# Check if directories exist
if not os.path.exists(path1):
    raise ValueError(f"Path '{path1}' does not exist.")
if not os.path.exists(path2):
    raise ValueError(f"Path '{path2}' does not exist.")

# Calculate FID score
fid = fid_score.calculate_fid_given_paths([path1, path2], batch_size=50, device="cuda", dims=2048)
print(f"FID Score: {fid}")
fid = fid_score.calculate_fid_given_paths([path3, path2], batch_size=50, device="cuda", dims=2048)
print(f"FID Score: {fid}")

