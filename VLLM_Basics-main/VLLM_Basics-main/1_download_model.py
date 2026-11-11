# Step 1: Download SmolLM2 model from Hugging Face and save it locally

from huggingface_hub import snapshot_download

model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
save_folder = "models/SmolLM2-360M-Instruct"

print("Downloading model:", model_name)

snapshot_download(repo_id=model_name, local_dir=save_folder, ignore_patterns=["onnx/*", "runs/*"])

print("Model saved in folder:", save_folder)
