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
pc = pinecone.Pinecone(api_key=os.getenv("pcsk_4axbre_KSH6nL6LoUzz2Lsbdt4tVdjrpqw69NaT1i2oryTH4vPVJHHj1ZrtkXw3GvHU4Ar"))
index = pc.Index("tax-rag")

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
