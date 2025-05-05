import os
from pytorch_fid import fid_score

# Paths to the two directories
path1 = "/data6/stilex/throughput/NIRVANA"
path2 = "/data6/stilex/throughput/baseline"
path3 = "/data6/stilex/throughput/Ours"

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


# import os
# from PIL import Image
# from pytorch_fid import fid_score

# # Paths to the directories
# path1 = "/data6/stilex/img_similarity/img"
# path2 = "/data6/stilex/img_similarity/og"
# path3 = "/data6/stilex/img_similarity/text"

# # Check if directories exist
# if not os.path.exists(path1):
#     raise ValueError(f"Path '{path1}' does not exist.")
# if not os.path.exists(path2):
#     raise ValueError(f"Path '{path2}' does not exist.")
# if not os.path.exists(path3):
#     raise ValueError(f"Path '{path3}' does not exist.")

# # Function to filter images by size
# def filter_images_by_size(directory, target_size=(512, 512)):
#     valid_images = []
#     for filename in os.listdir(directory):
#         img_path = os.path.join(directory, filename)
#         if os.path.isfile(img_path):
#             try:
#                 img = Image.open(img_path)
#                 if img.size == target_size:
#                     valid_images.append(img_path)
#                 else:
#                     print(f"Skipping {filename}: Incorrect size {img.size}")
#             except Exception as e:
#                 print(f"Error reading {img_path}: {e}")
#     return valid_images

# # Filter valid images in all directories
# valid_images_path1 = filter_images_by_size(path1)
# valid_images_path2 = filter_images_by_size(path2)
# valid_images_path3 = filter_images_by_size(path3)

# # Save the valid images to new directories for FID computation
# # You could optionally copy the valid images into temporary folders for the FID calculation
# temp_path1 = '/data6/stilex/img_similarity/filtered_img'
# temp_path2 = '/data6/stilex/img_similarity/filtered_og'
# temp_path3 = '/data6/stilex/img_similarity/filtered_text'

# # Ensure the temp directories exist
# os.makedirs(temp_path1, exist_ok=True)
# os.makedirs(temp_path2, exist_ok=True)
# os.makedirs(temp_path3, exist_ok=True)

# for valid_image in valid_images_path1:
#     os.rename(valid_image, os.path.join(temp_path1, os.path.basename(valid_image)))

# for valid_image in valid_images_path2:
#     os.rename(valid_image, os.path.join(temp_path2, os.path.basename(valid_image)))

# for valid_image in valid_images_path3:
#     os.rename(valid_image, os.path.join(temp_path3, os.path.basename(valid_image)))

# # Calculate FID score
# fid = fid_score.calculate_fid_given_paths([temp_path1, temp_path2], batch_size=50, device="cuda", dims=2048)
# print(f"FID Score (path1 vs path2): {fid}")
# fid = fid_score.calculate_fid_given_paths([temp_path3, temp_path2], batch_size=50, device="cuda", dims=2048)
# print(f"FID Score (path3 vs path2): {fid}")