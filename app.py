from flask import Flask, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
import threading
import time
import gc
import os
import sqlite3
import uuid
import json
from functools import wraps
import boto3

# New imports for search endpoint
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

###############################################################################
# 1. TOKEN CHECK
###############################################################################
# Demo token for the API (for demonstration purposes)
DEMO_TOKEN = "team-tax-1531"

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        # Check if token is provided in the Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
        # Fallback: check for token in query parameters
        if not token:
            token = request.args.get("token")
        # Fallback: check for token in JSON payload if available
        if not token and request.is_json:
            json_data = request.get_json(silent=True)
            if json_data:
                token = json_data.get("token")
        # If token is missing or does not match, return an error
        if token != DEMO_TOKEN:
            return jsonify({"error": "Unauthorized: Invalid or missing token"}), 401
        return f(*args, **kwargs)
    return decorated_function

###############################################################################
# 2. DATABASE SETUP (CHAT HISTORY)
###############################################################################
DB_FILE = "chat_sessions.db"

def init_db():
    """Initialize the local SQLite database for storing chat history."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            chat_id TEXT,
            username TEXT,
            timestamp TEXT,
            role TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def store_chat_message(chat_id, username, role, content):
    """Stores a chat message in the local SQLite database."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (chat_id, username, timestamp, role, content) VALUES (?, ?, ?, ?, ?)",
                   (chat_id, username, timestamp, role, content))
    conn.commit()
    conn.close()

def get_chat_history(chat_id):
    """Retrieves chat history for a given chat session."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, username, role, content FROM chat_history WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,))
    messages = cursor.fetchall()
    conn.close()
    return [{"timestamp": msg[0], "username": msg[1], "role": msg[2], "content": msg[3]} for msg in messages]


###############################################################################
# 3. CONFIGURATION
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

# Dictionary to hold currently loaded models
loaded_models = {}

###############################################################################
# 3. HELPER FUNCTIONS
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
    device_models = { name: info for name, info in loaded_models.items() if info.get("device") == device_str }
    if not device_models:
        raise RuntimeError(f"No models found on device {device_str} to unload.")
    oldest_model_name = min(device_models, key=lambda nm: device_models[nm]["last_access_time"])
    unload_model(oldest_model_name)

def load_model(model_name: str):
    if model_name in loaded_models:
        loaded_models[model_name]["last_access_time"] = time.time()
        return

    model_path = MODEL_PATHS.get(model_name)
    
    # Check if the local model directory exists
    if not os.path.exists(model_path):
        hf_model_id = MODEL_HF_IDS.get(model_name, model_name)
        print(f"Local model for '{model_name}' not found at '{model_path}'. Downloading from Hugging Face using id '{hf_model_id}' ...")
        model_path = hf_model_id  # Use the Hugging Face id for download

    print(f"[load_model] Attempting to load '{model_name}' ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",  # Automatically spread model across GPUs
        trust_remote_code=True
    )
    
    loaded_models[model_name] = {
        "tokenizer": tokenizer,
        "model": model,
        "last_access_time": time.time(),
    }
    print(f"[load_model] '{model_name}' loaded. It has been automatically partitioned across available GPUs.")

def perform_generation(model_name: str, prompt: str, max_tokens: int):
    """
    A common helper function to perform text generation.
    Returns generated_text, response_time, and the tokenizer used.
    """
    try:
        load_model(model_name)
    except Exception as e:
        raise Exception(str(e))
    
    tokenizer = loaded_models[model_name]["tokenizer"]
    model = loaded_models[model_name]["model"]
    inputs = tokenizer(prompt, return_tensors="pt")
    start_time = time.time()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    response_time = time.time() - start_time

    # Decode the output IDs to text
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    if full_output.startswith(prompt):
        generated_text = full_output[len(prompt):].strip()
    else:
        generated_text = full_output.strip()

    # Ensure the generated text ends in a punctuation mark
    if not generated_text.endswith(('.', '!', '?')):
        generated_text += " ..."
    
    loaded_models[model_name]["last_access_time"] = time.time()
    
    return generated_text, response_time, tokenizer

###############################################################################
# 4. PRE-DOWNLOAD MODELS (Optional: Download all models if not present)
###############################################################################
for mname, local_path in MODEL_PATHS.items():
    if not os.path.exists(local_path):
        hf_id = MODEL_HF_IDS.get(mname, mname)
        print(f"Local model for '{mname}' not found at '{local_path}'. Downloading from Hugging Face using id '{hf_id}' ...")
        _ = AutoTokenizer.from_pretrained(hf_id, cache_dir="./model_directory")
        _ = AutoModelForCausalLM.from_pretrained(hf_id, cache_dir="./model_directory", torch_dtype=torch.float16, trust_remote_code=True)
        print(f"Downloaded '{mname}'.")

###############################################################################
# 5. FLASK ROUTES
###############################################################################

# --- Existing Endpoints ---

@app.route('/generate', methods=['POST'])
@token_required
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

    prompt = build_prompt(question)
    try:
        generated_text, response_time, _ = perform_generation(model_name, prompt, max_length)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "response": generated_text,
        "time_taken": response_time,
        "model": model_name
    })

@app.route('/generate_stream', methods=['POST'])
@token_required
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
    prompt = build_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt")
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

# --- New Endpoints ---

@app.route('/v1/models', methods=['GET'])
@token_required
def list_models():
    """Return a list of available models."""
    models = [{"id": name, "object": "model"} for name in MODEL_PATHS.keys()]
    return jsonify({
        "data": models,
        "object": "list"
    })

@app.route('/v1/chat/completions', methods=['POST'])
@token_required
def chat_completions():
    data = request.get_json()
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "model is required"}), 400
    if model_name not in MODEL_PATHS:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    username = data.get("username")
    if not username:
        return jsonify({"error": "username is required"}), 400

    messages = data.get("messages")
    if not messages:
        return jsonify({"error": "messages field is required"}), 400

    # Generate or retrieve chat_id
    chat_id = data.get("chat_id")
    if not chat_id:
        chat_id = str(uuid.uuid4())  # Generate a new chat session ID

    # Store user message
    last_user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content")
            store_chat_message(chat_id, username, "user", last_user_message)
            break
    if not last_user_message:
        return jsonify({"error": "No user message found in messages"}), 400

    max_tokens = data.get("max_tokens", 200)

    prompt = build_prompt(last_user_message)

    try:
        generated_text, response_time, tokenizer = perform_generation(model_name, prompt, max_tokens)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Store assistant response
    store_chat_message(chat_id, "assistant", "assistant", generated_text)

    return jsonify({
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "chat_id": chat_id,
        "username": username,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(generated_text)),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(generated_text))
        },
        "time_taken": response_time
    })


@app.route('/v1/completions', methods=['POST'])
@token_required
def completions():
    data = request.get_json()
    model_name = data.get("model")
    if not model_name:
        return jsonify({"error": "model is required"}), 400
    if model_name not in MODEL_PATHS:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    prompt_text = data.get("prompt")
    if not prompt_text:
        return jsonify({"error": "prompt is required"}), 400

    max_tokens = data.get("max_tokens")
    if max_tokens is None:
        return jsonify({"error": "max_tokens must be provided"}), 400

    tax_optimize = data.get("tax_optimize", True)
    n = data.get("n", 2)

    if tax_optimize:
        # Perform search on Pinecone index
        search_results = search_pinecone(prompt_text, top_k=n)
        prompt, form_ids = create_detailed_prompt(prompt_text, search_results, tax_optimize=True, n=n)
    else:
        prompt = build_prompt(prompt_text)

    try:
        generated_text, response_time, tokenizer = perform_generation(model_name, prompt, max_tokens)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "text": generated_text,
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(generated_text)),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(generated_text))
        },
        "time_taken": response_time
    })

@app.route('/chat_history', methods=['GET'])
@token_required
def chat_history():
    chat_id = request.args.get("chat_id")
    if not chat_id:
        return jsonify({"error": "chat_id is required"}), 400

    history = get_chat_history(chat_id)
    if not history:
        return jsonify({"error": "No chat history found for the given chat_id"}), 404

    return jsonify({"chat_id": chat_id, "messages": history})

@app.route('/chat_sessions', methods=['GET'])
@token_required
def list_chat_sessions():
    """List all stored chat session IDs."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT chat_id FROM chat_history")
    sessions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify({"chat_sessions": sessions})


###############################################################################
# 5. FUNCTION TO PUSH CHAT HISTORY TO S3 (COMMENTED OUT)
###############################################################################
S3_BUCKET_NAME = "chat_history"

def upload_chat_to_s3(chat_id):
    """Uploads a chat session history to an S3 bucket."""
    s3_client = boto3.client("s3")
    chat_history = get_chat_history(chat_id)
    chat_data = json.dumps({"chat_id": chat_id, "messages": chat_history})

    # Uncomment below line to enable S3 upload
    # s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=f"chats/{chat_id}.json", Body=chat_data, ContentType="application/json")

    print(f"Chat history {chat_id} prepared for S3 upload.")



# --- New Search Endpoint ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY is not set. The search endpoint may not work.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define index name
INDEX_NAME = "tax-rag"

# Connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Error: Pinecone index '{INDEX_NAME}' does not exist.")
    index = None
else:
    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {INDEX_NAME}")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_pinecone(query_text, top_k=5):
    """
    Searches Pinecone index with the given query and returns the top results.
    """
    if not index:
        return {"error": "Pinecone index is not initialized."}
    
    # Generate query embedding
    query_embedding = embedding_model.encode(query_text).tolist()
    
    # Query Pinecone
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    # Format results
    formatted_results = [
        {
            "id": match.get("id"),
            "score": match.get("score"),
            "text": match.get("metadata", {}).get("text")
        }
        for match in results.get("matches", [])
    ]
    
    return {
        "query": query_text,
        "results": formatted_results
    }

@app.route('/search', methods=['POST'])
@token_required
def search():
    """
    API endpoint for searching the Pinecone index.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field in JSON payload"}), 400
    print('data', data)
    query_text = data["query"]
    top_k = data.get("top_k", 5)
    
    # Perform search
    response = search_pinecone(query_text, top_k)
    
    return jsonify(response)

def create_detailed_prompt(user_query, search_results, tax_optimize=True, n=2):
    """
    Create a detailed prompt for a tax-related application using user query and search results.
    
    Parameters:
    - user_query (str): The original query from the user.
    - search_results (dict): The search results from Pinecone.
    - tax_optimize (bool): Whether to optimize the prompt for tax purposes.
    - n (int): Number of top search results to use for context.

    Returns:
    - str: The detailed prompt ready for model input.
    - set: A set of form IDs used in the context.
    """
    def search_online(query):
        """
        Placeholder function to perform an online search.
        This should be replaced with an actual implementation.
        
        Parameters:
        - query (str): The query to search online.

        Returns:
        - list: Simulated search results.
        """
        # Simulated response for demonstration purposes
        return [
            {"title": "Self-Employment Taxes", "snippet": "Learn about the forms required for self-employment taxes."},
            {"title": "Freelancer Tax Guide", "snippet": "A guide to taxes for freelancers and self-employed individuals."},
            {"title": "Independent Contractor Tax Tips", "snippet": "Tips for managing taxes as an independent contractor."}
        ]
    
    if tax_optimize:
        prefix = "You are a helpful tax advisor and legal expert. Use the provided context to answer the user's query in a clear and concise manner.\n"
        
        # Extract relevant context from search results, limiting to 'n' results
        context_segments = []
        form_ids = set()
        
        for result in search_results.get("results", [])[:n]:
            form_id = result.get("id", "")
            form_name = form_id.split(".md")[0] if ".md" in form_id else form_id
            form_ids.add(form_name)
            context_text = result.get("text", "").strip()
            context_segments.append(f"Form {form_name}: {context_text}")
        
        # Construct the context section for the prompt
        context_section = "\n".join(context_segments)
        
    else:
        prefix = "You are a helpful advisor. Use the provided online information to answer the user's query.\n"
        
        # Call the search_online function to get additional context
        online_results = search_online(user_query)
        
        # Format the search results for inclusion in the prompt, limiting to 'n' results
        context_segments = [f"{res['title']}: {res['snippet']}" for res in online_results[:n]]
        context_section = "\n".join(context_segments)
        
        form_ids = set()  # No specific forms are tracked in this mode

    # Build the complete prompt
    detailed_prompt = (
        f"{prefix}"
        f"User Query: {user_query}\n"
        f"Related Context:\n"
        f"{context_section}\n"
        "Note: The above information is extracted from relevant forms or online sources. Use it to formulate your response."
    )
    
    return detailed_prompt, form_ids

###############################################################################
# 6. RUN FLASK
###############################################################################
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000)
