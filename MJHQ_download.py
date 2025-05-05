from huggingface_hub import hf_hub_download

hf_hub_download(
  repo_id="playgroundai/MJHQ-30K", 
  filename="mjhq30k_imgs.zip", 
  local_dir="MJHQ",
  repo_type="dataset"
)