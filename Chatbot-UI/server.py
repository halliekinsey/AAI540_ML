from fastapi import FastAPI
from pydantic import BaseModel
import pinecone
import requests
from sentence_transformers import SentenceTransformer
import os

# Initialize FastAPI
app = FastAPI()

# LM Studio API URL
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Define the index name and dimension
index_name = "tax-rag"
dimension = 128  # Replace with the dimensionality of your data

# Check if the index exists, create it if not
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' does not exist. Creating it...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",  # Use the metric suitable for your use case ('cosine', 'euclidean', etc.)
        spec=pinecone.ServerlessSpec(
            cloud="aws",  # Replace with your cloud provider, e.g., 'gcp' or 'aws'
            region="us-west-2"  # Replace with your desired region
        )
    )
else:
    print(f"Index '{index_name}' already exists.")

# Connect to the index
index = pc.Index(index_name)

print("Index is ready to use!")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class ChatRequest(BaseModel):
    message: str

def retrieve_relevant_context(query):
    """Search Pinecone for the most relevant tax information."""
    query_embedding = embedding_model.encode(query).tolist()
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
    return context if context else "No relevant tax information found."

@app.post("/chat")
def chat_with_model(request: ChatRequest):
    """Handles chat requests from the frontend using LM Studio API."""
    
    # Retrieve relevant context from Pinecone
    context = retrieve_relevant_context(request.message)
    input_text = f"Context: {context}\nUser: {request.message}\nAssistant:"

    # Send request to LM Studio API
    payload = {
        "model": "deepseek-coder-v2-lite-instruct",  # Ensure this matches your LM Studio model
        "prompt": input_text,
        "temperature": 0.7,
        "max_tokens": 200
    }

    response = requests.post(LM_STUDIO_API_URL, json=payload)

    # Check for API response
    if response.status_code == 200:
        response_json = response.json()
        assistant_reply = response_json.get("choices", [{}])[0].get("text", "").strip()
    else:
        assistant_reply = "Error: Unable to retrieve a response from LM Studio."

    return {"response": assistant_reply}
