import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_folder = "models/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_folder)
model = AutoModelForCausalLM.from_pretrained(model_folder).to(device)
load_time = time.time() - start

question = "Explain what a GPU is in simple words."
messages = [{"role": "user", "content": question}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(device)

start = time.time()
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
gen_time = time.time() - start

input_tokens = inputs["input_ids"].shape[1]
output_tokens = output.shape[1] - input_tokens
answer = tokenizer.decode(output[0][input_tokens:], skip_special_tokens=True)

print("\n===== ANSWER =====")
print(answer)

print("\n===== DETAILS (Normal Transformers) =====")
print("Device           :", device)
print("Model load time  :", round(load_time, 2), "sec")
print("Input tokens     :", input_tokens)
print("Output tokens    :", output_tokens)
print("Response time    :", round(gen_time, 2), "sec")
print("Tokens per second:", round(output_tokens / gen_time, 2))
if device == "cuda":
    print("GPU memory used  :", round(torch.cuda.max_memory_allocated() / 1024**3, 2), "GB")

print("\n===== MULTI LOAD (10 requests, one by one) =====")
total_tokens = 0
start = time.time()
for i in range(10):
    q = [{"role": "user", "content": "Give me one fun fact number " + str(i + 1)}]
    t = tokenizer.apply_chat_template(q, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(t, return_tensors="pt").to(device)
    out = model.generate(**inp, max_new_tokens=100, do_sample=False)
    total_tokens += out.shape[1] - inp["input_ids"].shape[1]
total_time = time.time() - start

print("Total requests   : 10")
print("Total time       :", round(total_time, 2), "sec")
print("Total tokens     :", total_tokens)
print("Tokens per second:", round(total_tokens / total_time, 2))
