vLLM Inference Demo with SmolLM2
This project shows how much faster vLLM is compared to normal Hugging Face transformers inference.

We use a small model, SmolLM2-360M-Instruct, so it runs even on a laptop GPU.

Files
File	What it does
1_download_model.py	Downloads SmolLM2 from Hugging Face into models/
2_run_normal.py	Runs the model with normal transformers and prints speed details
3_run_vllm.py	Runs the same model with vLLM and prints speed details
requirements.txt	Python packages list
What you need
An NVIDIA GPU (4 GB VRAM is enough)
Latest NVIDIA driver
Python 3.10 to 3.12
About 10 GB of free disk space
Important: vLLM works only on Linux. On Windows you must use WSL (Windows Subsystem for Linux). Scripts 1 and 2 work on Windows directly.

Option A: Linux users (simplest)
Run everything in one venv.

cd vllm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python 1_download_model.py
python 2_run_normal.py
python 3_run_vllm.py
If script 3 says "Failed to find C compiler", install it:

sudo apt install -y build-essential python3-dev
Option B: Windows users
Step 1: Run scripts 1 and 2 on Windows
Open PowerShell inside the vllm folder:

python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install huggingface_hub transformers accelerate

python 1_download_model.py
python 2_run_normal.py
Do not install vllm here. It will fail on Windows.

Step 2: Install Ubuntu in WSL
Open PowerShell as Administrator:

wsl --install -d Ubuntu
Restart your PC if asked. Then open Ubuntu from the Start menu and create a username and password.

Check that WSL can see your GPU:

nvidia-smi
You should see your GPU name. You do not need to install any NVIDIA driver inside Ubuntu, because the Windows driver is shared.

Step 3: Install vLLM inside Ubuntu
sudo apt update
sudo apt install -y build-essential python3-dev curl

# uv is a fast Python installer, it also gives us Python 3.12
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

uv venv --python 3.12 ~/venvs/vllm-demo
source ~/venvs/vllm-demo/bin/activate
uv pip install vllm
Why is the venv in ~/venvs and not in the project folder? Your Windows files appear in Linux under /mnt/c/..., and reading files from there is very slow. vLLM is about 5 GB, so we keep its venv on the Linux side for speed.

Step 4: Run script 3
Go to your project folder from Ubuntu. Your C drive is at /mnt/c. Change the path to match your own:

cd /mnt/c/Users/YOUR_NAME/Desktop/vllm
source ~/venvs/vllm-demo/bin/activate
python 3_run_vllm.py
The model was already downloaded in step 1, so vLLM uses the same models/ folder.

Sample results (RTX 3050 Laptop, 4 GB)
Normal transformers	vLLM
Model load time	5.6 sec	65.8 sec
Single request speed	9.6 tokens/sec	120.5 tokens/sec
Single request time	6.23 sec	0.53 sec
Multi load	10 requests in 36.5 sec	100 requests in 1.06 sec
Multi load throughput	12 tokens/sec	4,821 tokens/sec
Why is vLLM so fast?
PagedAttention: vLLM stores the model's memory (KV cache) in small pages, like an operating system does with RAM. No GPU memory is wasted, so more requests fit at once.
Continuous batching: Normal transformers answers requests one by one. vLLM puts many requests together in one batch and keeps adding new ones as old ones finish, so the GPU is always busy.
CUDA graphs and compiled kernels: At startup vLLM prepares optimized GPU code. This is why loading takes longer, but every request after that is much faster.
Common problems
Problem	Fix
pip install vllm fails on Windows	vLLM needs Linux. Use WSL (Option B)
Failed to find C compiler	sudo apt install -y build-essential python3-dev
An attempt has been made to start a new process...	Keep the line os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" at the top of 3_run_vllm.py
CUDA out of memory	Lower gpu_memory_utilization in 3_run_vllm.py, for example to 0.6
nvidia-smi not found in WSL	Update your NVIDIA driver on Windows
Download is very slow	Make sure you use the latest 1_download_model.py, it skips extra ONNX files
