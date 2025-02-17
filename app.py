from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading
import time
import gc
import os

app = Flask(__name__)

###############################################################################
# 1. CONFIGURATION
###############################################################################
# Dictionary of model names to their respective local paths (or Hugging Face repos).
MODEL_PATHS = {
    "saul_7b_instruct": "./model_directory/models--Equall--Saul-7B-Instruct-v1/snapshots/2133ba7923533934e78f73848045299dd74f08d2",
    "lawma_8b": "./model_directory/models--ricdomolm--lawma-8B/snapshots/cf7b9086448228ba981a9748012a97b616a70579",
    "lawma_70b": "./model_directory/models--ricdomolm--lawma-70b/snapshots/cf7b9086448228ba981a9748012a97b616a70579",
    "DeepSeek-V2-Lite": "./model_directory/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7"
}

# Dictionary of corresponding Hugging Face model IDs to use for downloading if needed.
MODEL_HF_IDS = {
    "saul_7b_instruct": "Equall/Saul-7B-Instruct-v1",
    "lawma_8b": "ricdomolm/lawma-8b",
    "lawma_70b": "ricdomolm/lawma-70b",
    "DeepSeek-V2-Lite": "deepseek-ai/DeepSeek-V2-Lite-Chat"
}

# Which GPU device each model *should prefer* (you can have multiple GPUs)
MODEL_DEVICE_MAPPING = {
    "saul_7b_instruct": "cuda:0",
    "lawma_8b": "cuda:0",
    "lawma_70b": "cuda:1",
    "DeepSeek-V2-Lite": "cuda:0"
}

# Dictionary to hold currently loaded models
loaded_models = {}

###############################################################################
# 2. HELPER FUNCTIONS
###############################################################################

def build_prompt(user_question: str) -> str:
    prefix = (
        "You are a helpful tax and legal advisor. "
        "Answer the following question in a clear and concise manner:\n"
    )
    return prefix + user_question

def get_total_gpu_memory(device_str: str) -> float:
    device_idx = int(device_str.split(":")[-1])
    props = torch.cuda.get_device_properties(device_idx)
    total_mem_bytes = props.total_memory
    total_mem_mb = total_mem_bytes / (1024**2)
    return total_mem_mb

def get_current_gpu_usage(device_str: str) -> float:
    torch.cuda.set_device(device_str)
    reserved = torch.cuda.memory_reserved(device=device_str)
    return reserved / (1024**2)

def check_if_enough_memory(device_str: str, needed_mb: float) -> bool:
    total_mb = get_total_gpu_memory(device_str)
    reserved_mb = get_current_gpu_usage(device_str)
    free_mb = total_mb - reserved_mb
    return (free_mb >= needed_mb)

def estimate_model_size_in_mb(model_path: str, dtype_bytes=2) -> float:
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_path)
    meta_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    param_count = sum(p.numel() for p in meta_model.parameters())
    total_bytes = param_count * dtype_bytes
    total_mb = total_bytes / (1024**2)
    return total_mb

def unload_model(model_name: str):
    if model_name not in loaded_models:
        return
    print(f"[unload_model] Unloading model '{model_name}'.")
    model_entry = loaded_models[model_name]
    del model_entry["model"]
    del model_entry["tokenizer"]
    del loaded_models[model_name]
    gc.collect()
    torch.cuda.empty_cache()

def unload_least_recently_used_model(device_str: str):
    device_models = { name: info for name, info in loaded_models.items() if info["device"] == device_str }
    if not device_models:
        raise RuntimeError(f"No models found on device {device_str} to unload.")
    oldest_model_name = min(device_models, key=lambda nm: device_models[nm]["last_access_time"])
    unload_model(oldest_model_name)

def load_model(model_name: str):
    if model_name in loaded_models:
        loaded_models[model_name]["last_access_time"] = time.time()
        return

    if model_name not in MODEL_DEVICE_MAPPING:
        raise ValueError(f"No device mapping found for model: {model_name}")
    device_to_use = MODEL_DEVICE_MAPPING[model_name]
    model_path = MODEL_PATHS.get(model_name)
    
    # Check if the local model directory exists
    if not os.path.exists(model_path):
        hf_model_id = MODEL_HF_IDS.get(model_name, model_name)
        print(f"Local model for '{model_name}' not found at '{model_path}'. Downloading from Hugging Face using id '{hf_model_id}' ...")
        model_path = hf_model_id  # Use the Hugging Face id for download

    print(f"[load_model] Attempting to load '{model_name}' onto {device_to_use} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    allocated = torch.cuda.memory_allocated(device=device_to_use) / (1024**2)
    loaded_models[model_name] = {
        "tokenizer": tokenizer,
        "model": model,
        "device": device_to_use,
        "last_access_time": time.time(),
        "memory_footprint": allocated
    }
    print(f"[load_model] '{model_name}' loaded on {device_to_use}. Current allocated: {allocated:.2f} MB")

###############################################################################
# 3. PRE-DOWNLOAD MODELS (Optional: Download all models if not present)
###############################################################################
# This loop iterates over all models and downloads them if the local folder doesn't exist.
for mname, local_path in MODEL_PATHS.items():
    if not os.path.exists(local_path):
        hf_id = MODEL_HF_IDS.get(mname, mname)
        print(f"Local model for '{mname}' not found at '{local_path}'. Downloading from Hugging Face using id '{hf_id}' ...")
        # Download tokenizer and model to cache_dir
        _ = AutoTokenizer.from_pretrained(hf_id, cache_dir="./model_directory")
        _ = AutoModelForCausalLM.from_pretrained(hf_id, cache_dir="./model_directory", torch_dtype=torch.float16, trust_remote_code=True)
        print(f"Downloaded '{mname}'.")

###############################################################################
# 4. FLASK ROUTES
###############################################################################
@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    model_name = data.get("model_name")
    if not model_name:
        return jsonify({"error": "model_name is required"}), 400
    if model_name not in MODEL_PATHS:
        return jsonify({"error": f"Unknown model_name: {model_name}"}), 400

    question = data.get("text", "")
    max_length = data.get("max_length")
    if max_length is None:
        return jsonify({"error": "max_length must be provided"}), 400

    try:
        load_model(model_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    tokenizer = loaded_models[model_name]["tokenizer"]
    model = loaded_models[model_name]["model"]
    model_device = loaded_models[model_name]["device"]

    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)

    try:
        start_time = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        response_time = time.time() - start_time
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if not generated_text.strip().endswith(('.', '!', '?')):
            generated_text += " ..."
        loaded_models[model_name]["last_access_time"] = time.time()
        return jsonify({
            "response": generated_text,
            "time_taken": response_time,
            "model": model_name
        })
    except Exception as e:
        print(f"Error during /generate with model {model_name}:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/generate_stream', methods=['POST'])
def generate_text_stream():
    data = request.get_json()
    model_name = data.get("model_name")
    if not model_name:
        return jsonify({"error": "model_name is required"}), 400
    if model_name not in MODEL_PATHS:
        return jsonify({"error": f"Unknown model_name: {model_name}"}), 400

    question = data.get("text", "")
    max_length = data.get("max_length")
    if max_length is None:
        return jsonify({"error": "max_length must be provided"}), 400

    try:
        load_model(model_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    tokenizer = loaded_models[model_name]["tokenizer"]
    model = loaded_models[model_name]["model"]
    model_device = loaded_models[model_name]["device"]

    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    def run_generation():
        try:
            model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )
        except Exception as e:
            print(f"Error during stream generation with model {model_name}:", str(e))
    
    thread = threading.Thread(target=run_generation)
    thread.start()
    loaded_models[model_name]["last_access_time"] = time.time()
    return Response(streamer, mimetype="text/plain")

###############################################################################
# 5. RUN FLASK
###############################################################################
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000)



## Only issue with this script is the 70 B model does not load as it is bigger than one A100

