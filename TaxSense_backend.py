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
import re
from functools import wraps
import boto3
import hashlib
import io
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from pinecone import Pinecone
import datetime

app = Flask(__name__)

DEMO_TOKEN = "team-tax-1531"

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
        if not token:
            token = request.args.get("token")
        if not token and request.is_json:
            json_data = request.get_json(silent=True)
            if json_data:
                token = json_data.get("token")
        if token != DEMO_TOKEN:
            return jsonify({"error": "Unauthorized: Invalid or missing token"}), 401
        return f(*args, **kwargs)
    return decorated_function

DB_FILE = "chat_sessions.db"

def init_db():
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
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chat_history (chat_id, username, timestamp, role, content) VALUES (?, ?, ?, ?, ?)",
                   (chat_id, username, timestamp, role, content))
    conn.commit()
    conn.close()

def get_chat_history(chat_id):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, username, role, content FROM chat_history WHERE chat_id = ? ORDER BY timestamp ASC", (chat_id,))
    messages = cursor.fetchall()
    conn.close()
    return [{"timestamp": msg[0], "username": msg[1], "role": msg[2], "content": msg[3]} for msg in messages]

MODEL_PATHS = {
    "saul_7b_instruct": "./model_directory/models--Equall--Saul-7B-Instruct-v1/snapshots/2133ba7923533934e78f73848045299dd74f08d2",
    "lawma_8b": "./model_directory/models--ricdomolm--lawma-8B/snapshots/cf7b9086448228ba981a9748012a97b616a70579",
    "lawma_70b": "./model_directory/models--ricdomolm--lawma-70b/snapshots/cf7b9086448228ba981a9748012a97b616a70579",
    "DeepSeek-V2-Lite": "./model_directory/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/85864749cd611b4353ce1decdb286193298f64c7",
    "TaxSense": "./model_directory/models--zainnobody--TaxSense"
}

MODEL_CONFIGS = {
    "TaxSense": {
        "hf_id": "zainnobody/TaxSense",
        "use_peft": True
    },
    "saul_7b_instruct": {
        "hf_id": "Equall/Saul-7B-Instruct-v1"
    },
    "lawma_8b": {
        "hf_id": "ricdomolm/lawma-8b"
    },
    "lawma_70b": {
        "hf_id": "ricdomolm/lawma-70b"
    },
    "DeepSeek-V2-Lite": {
        "hf_id": "deepseek-ai/DeepSeek-V2-Lite-Chat"
    }
}

loaded_models = {}

def build_prompt(user_question: str) -> str:
    prefix = (
        "You are a helpful tax and legal advisor. "
        "Answer the following question in a clear and concise manner:\n"
    )
    return prefix + user_question

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

def load_model(model_name: str):
    if model_name in loaded_models:
        loaded_models[model_name]["last_access_time"] = time.time()
        return

    model_path = MODEL_PATHS.get(model_name)
    if not os.path.exists(model_path):
        hf_model_id = MODEL_HF_IDS.get(model_name, model_name)
        print(f"Local model for '{model_name}' not found at '{model_path}'. Downloading from HF using id '{hf_model_id}' ...")
        model_path = hf_model_id

    print(f"[load_model] Loading '{model_name}' ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    loaded_models[model_name] = {
        "tokenizer": tokenizer,
        "model": model,
        "last_access_time": time.time(),
    }
    print(f"[load_model] '{model_name}' loaded across available GPUs.")

def perform_generation(model_name: str, prompt: str, max_tokens: int):
    load_model(model_name)
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
    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if full_output.startswith(prompt):
        generated_text = full_output[len(prompt):].strip()
    else:
        generated_text = full_output.strip()
    if not generated_text.endswith(('.', '!', '?')):
        generated_text += " ..."
    loaded_models[model_name]["last_access_time"] = time.time()
    return generated_text, response_time, tokenizer

# Pre-download models if not present locally.
for mname, local_path in MODEL_PATHS.items():
    if not os.path.exists(local_path):
        hf_id = MODEL_CONFIGS.get(mname, {}).get("hf_id", mname)
        print(f"Local model for '{mname}' not found at '{local_path}'. Downloading from '{hf_id}'...")
        _ = AutoTokenizer.from_pretrained(hf_id, cache_dir="./model_directory")
        _ = AutoModelForCausalLM.from_pretrained(hf_id, cache_dir="./model_directory", torch_dtype=torch.float16, trust_remote_code=True)
        print(f"Downloaded '{mname}'.")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    print("Warning: PINECONE_API_KEY is not set. Pinecone search endpoint may not work.")

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "tax-rag"
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Error: Pinecone index '{INDEX_NAME}' does not exist.")
    index = None
else:
    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {INDEX_NAME}")

def search_pinecone(query_text, top_k=5):
    if not index:
        return {"error": "Pinecone index is not initialized."}
    query_embedding = EMBEDDING_MODEL.encode(query_text).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    formatted_results = [
        {
            "id": match.get("id"),
            "score": match.get("score"),
            "text": match.get("metadata", {}).get("text")
        }
        for match in results.get("matches", [])
    ]
    return {"query": query_text, "results": formatted_results}

def create_detailed_prompt(user_query, search_results, tax_optimize=True, n=2):
    if tax_optimize:
        prefix = "You are a helpful tax advisor and legal expert. Use the provided context to answer the user's query in a clear and concise manner.\n"
        context_segments = []
        form_ids = set()
        for result in search_results.get("results", [])[:n]:
            form_id = result.get("id", "")
            form_name = form_id.split(".md")[0] if ".md" in form_id else form_id
            form_ids.add(form_name)
            context_text = result.get("text", "").strip()
            context_segments.append(f"Form {form_name}: {context_text}")
        context_section = "\n".join(context_segments)
    else:
        prefix = "You are a helpful advisor. Use the provided online information to answer the user's query.\n"
        context_section = ""
        form_ids = set()
    detailed_prompt = (
        f"{prefix}"
        f"User Query: {user_query}\n"
        f"Related Context:\n"
        f"{context_section}\n"
        "Note: The above information is extracted from relevant sources. Use it to formulate your response."
    )
    return detailed_prompt, form_ids

# Initialize embedding model (same as used for Pinecone)
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, chunk_size=500):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end
    return chunks

def compute_md5(byte_data):
    return hashlib.md5(byte_data).hexdigest()

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_pdf_context(username, chat_id, question, top_k=2):
    folder_path = os.path.join("pdf_uploads", username, chat_id)
    if not os.path.isdir(folder_path):
        return []
    all_chunks = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".meta.json"):
            meta_path = os.path.join(folder_path, fname)
            with open(meta_path, "r", encoding="utf-8") as mfile:
                meta_data = json.load(mfile)
                chunk_entries = meta_data.get("chunks", [])
                all_chunks.extend(chunk_entries)
    if not all_chunks:
        return []
    q_emb = EMBEDDING_MODEL.encode([question])[0]
    scored = []
    for entry in all_chunks:
        c_emb = np.array(entry["embedding"])
        c_text = entry["chunk"]
        score = cos_sim(q_emb, c_emb)
        scored.append((score, c_text))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_contexts = [t[1] for t in scored[:top_k]]
    return top_contexts

# Endpoints:

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

@app.route('/v1/completions', methods=['POST'])
@token_required
def completions():
    # Generates text completion using the provided prompt and, if enabled, appends Pinecone search references.
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

    tax_optimize = data.get("tax_optimize", False)
    if tax_optimize:
        search_results = search_pinecone(prompt_text, top_k=data.get("n", 2))
        prompt, _ = create_detailed_prompt(prompt_text, search_results, tax_optimize=True, n=data.get("n", 2))
    else:
        prompt = build_prompt(prompt_text)

    try:
        generated_text, response_time, tokenizer = perform_generation(model_name, prompt, max_tokens)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    if tax_optimize:
        references_list = []
        for i, result in enumerate(search_results.get("results", [])):
            snippet = result["text"][:150].replace("\n", " ")
            form_id = result["id"].replace(".md", "")
            references_list.append(f"- Form {form_id} snippet {i+1}: \"{snippet}...\"")

        if references_list:
            references_str = "\n".join(references_list)
            generated_text += (
                "\n\nThis response was compiled using the following form context:\n" +
                references_str
            )
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

    chat_id = data.get("chat_id") or str(uuid.uuid4())
    last_user_message = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_message = msg.get("content")
            store_chat_message(chat_id, username, "user", last_user_message)
            break
    if not last_user_message:
        return jsonify({"error": "No user message found in messages"}), 400

    max_tokens = data.get("max_tokens", 200)

    use_pdf_context = data.get("use_pdf_context", False)
    tax_optimize = data.get("tax_optimize", False)

    if use_pdf_context:
        pdf_context_chunks = retrieve_pdf_context(username, chat_id, last_user_message, top_k=2)
        if pdf_context_chunks:
            print("\n[RAG] Using PDF context from user's uploads:")
            for idx, chunk in enumerate(pdf_context_chunks, start=1):
                snippet = chunk[:100].replace("\n", " ")
                print(f"  Chunk {idx}: {snippet}...")
            rag_text = "\n\n".join(pdf_context_chunks)
            prompt = (
                "You are a helpful tax and legal advisor. Using the following PDF context and the user's question, "
                "provide a helpful and concise answer.\n\n"
                f"PDF Context:\n{rag_text}\n\n"
                f"User Question:\n{last_user_message}\n"
            )
        else:
            prompt = build_prompt(last_user_message)
    elif tax_optimize:
        search_results = search_pinecone(last_user_message, top_k=data.get("n", 2))
        prompt, _ = create_detailed_prompt(last_user_message, search_results, tax_optimize=True, n=data.get("n", 2))
    else:
        prompt = build_prompt(last_user_message)

    print("\n[Prompt to Model]:\n" + prompt)
    print("----------------------------------------------------\n")

    try:
        generated_text, response_time, tokenizer = perform_generation(model_name, prompt, max_tokens)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # If use_pdf_context, show the PDF references
    if use_pdf_context:
        pdf_context_chunks = retrieve_pdf_context(username, chat_id, last_user_message, top_k=2)
        if pdf_context_chunks:
            references_list = []
            for i, chunk in enumerate(pdf_context_chunks):
                snippet = chunk[:150].replace("\n", " ")
                references_list.append(f"- PDF snippet {i+1}: \"{snippet}...\"")

            references_str = "\n".join(references_list)
            generated_text += (
                "\n\nThis response was compiled using the following PDF snippets:\n" +
                references_str
            )

    # If tax_optimize, show the Pinecone references
    elif tax_optimize:
        references_list = []
        for i, result in enumerate(search_results.get("results", [])):
            snippet = result["text"][:150].replace("\n", " ")
            form_id = result.get("id", "")
            form_id = clean_form_id_v2(form_id)
            references_list.append(f"- Form {form_id} snippet {i+1}: \"{snippet}...\"")

        if references_list:
            references_str = "\n".join(references_list)
            generated_text += (
                "\n\nThis response was compiled using the following form context:\n" +
                references_str
            )

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

def clean_form_id_v2(form_id):
    form_id = form_id.replace(".md", "") 
    if re.search(r'_\d+$', form_id):
        form_id = re.sub(r'_\d+$', '', form_id)
    return form_id


@app.route("/upload_pdf", methods=["POST"])
@token_required
def upload_pdf():
    if "pdf" not in request.files:
        return jsonify({"error": "'pdf' file is required"}), 400
    if "username" not in request.form:
        return jsonify({"error": "'username' field is required"}), 400
    if "chat_id" not in request.form:
        return jsonify({"error": "'chat_id' field is required"}), 400

    pdf_file = request.files["pdf"]
    username = request.form["username"].strip()
    chat_id = request.form["chat_id"].strip()

    base_folder = os.path.join("pdf_uploads", username, chat_id)
    os.makedirs(base_folder, exist_ok=True)

    filename = pdf_file.filename
    save_path = os.path.join(base_folder, filename)
    pdf_file.save(save_path)

    # Compute MD5 hash for the file
    with open(save_path, "rb") as f:
        data_bytes = f.read()
    filehash = compute_md5(data_bytes)
    meta_json_path = save_path + ".meta.json"

    # If meta file exists with the same hash, skip processing
    if os.path.exists(meta_json_path):
        with open(meta_json_path, "r", encoding="utf-8") as meta_f:
            existing = json.load(meta_f)
            if existing.get("filehash") == filehash:
                return jsonify({
                    "message": f"PDF '{filename}' already processed. (same hash)",
                    "skipped_chunking": True
                })

    try:
        pdf_stream = io.BytesIO(data_bytes)
        reader = PdfReader(pdf_stream)
        all_text = []
        for page in reader.pages:
            ptxt = page.extract_text()
            if ptxt:
                all_text.append(ptxt)
        full_text = "\n".join(all_text)
        chunks = chunk_text(full_text, chunk_size=500)
        chunk_embeddings = EMBEDDING_MODEL.encode(chunks)
        chunk_entries = []
        for i, c in enumerate(chunks):
            chunk_entries.append({
                "chunk": c,
                "embedding": chunk_embeddings[i].tolist()
            })
        meta_data = {
            "filename": filename,
            "filehash": filehash,
            "chunks": chunk_entries
        }
        with open(meta_json_path, "w", encoding="utf-8") as mf:
            json.dump(meta_data, mf)
        return jsonify({
            "message": f"PDF '{filename}' uploaded and processed.",
            "chunks": len(chunks),
            "chat_id": chat_id,
            "username": username
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search', methods=['POST'])
@token_required
def search():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field in JSON payload"}), 400
    query_text = data["query"]
    top_k = data.get("top_k", 5)
    response = search_pinecone(query_text, top_k)
    return jsonify(response)


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
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT chat_id FROM chat_history")
    sessions = [row[0] for row in cursor.fetchall()]
    conn.close()
    return jsonify({"chat_sessions": sessions})

@app.route('/v1/models', methods=['GET'])
@token_required
def list_models():
    models = [{"id": name, "object": "model"} for name in MODEL_PATHS.keys()]
    return jsonify({"data": models, "object": "list"})

REFINING_DATA_DIR = "refining_data"
os.makedirs(REFINING_DATA_DIR, exist_ok=True)

@app.route("/feedback", methods=["POST"])
@token_required
def feedback():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    chat_id = data.get("chat_id")
    feedback_value = data.get("feedback")
    conversation = data.get("conversation", [])
    username = data.get("username", "")
    use_pdf = data.get("use_pdf_context", False)
    tax_opt = data.get("tax_optimize", False)

    if not chat_id or not feedback_value:
        return jsonify({"error": "Missing 'chat_id' or 'feedback'"}), 400

    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp_str}_{chat_id}_{feedback_value}.json"
    save_path = os.path.join(REFINING_DATA_DIR, filename)

    record = {
        "timestamp": timestamp_str,
        "chat_id": chat_id,
        "username": username,
        "feedback": feedback_value,
        "use_pdf_context": use_pdf,
        "tax_optimize": tax_opt,
        "conversation": conversation
    }
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=2))
        return jsonify({"message": "Feedback saved successfully.", "file": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



###############################################################################
# 7. RUN FLASK
###############################################################################
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=6000)
