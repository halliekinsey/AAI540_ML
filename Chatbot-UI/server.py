from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pinecone
import requests
from sentence_transformers import SentenceTransformer
import os
from io import BytesIO
import markdown
from pdfminer.high_level import extract_text
from docx import Document

# Initialize FastAPI
app = FastAPI()

# LM Studio
LM_STUDIO_API_URL = "http://localhost:1234/v1/completions"

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Defining index name & correct embedding dimension
index_name = "tax-rag"
dimension = 384

# Check if Pinecone index exists
existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Index '{index_name}' does not exist. Creating it...")
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-west-2")
    )
else:
    print(f"Index '{index_name}' already exists.")

# Connect to Pinecone index
index = pc.Index(index_name)
print("✅ Pinecone index is ready!")

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class ChatRequest(BaseModel):
    message: str


def retrieve_relevant_context(query):
    """Search Pinecone for the most relevant IRS tax information."""
    query_embedding = embedding_model.encode(query).tolist()
    
    try:
        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
        if "matches" not in results or not results["matches"]:
            return "No relevant tax information found."
        
        context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
        return context
    except Exception as e:
        print(f"⚠️ Error querying Pinecone: {e}")
        return "Error retrieving tax information."


def convert_to_markdown(text):
    """Convert extracted text to Markdown format."""
    return markdown.markdown(text)


def extract_text_from_file(file: UploadFile):
    """Extract text from user-uploaded files."""
    contents = file.file.read()
    
    if file.filename.endswith(".pdf"):
        text = extract_text(BytesIO(contents))
    elif file.filename.endswith(".docx"):
        doc = Document(BytesIO(contents))
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = contents.decode("utf-8")
    
    return convert_to_markdown(text)

# Adding upload ability processing 
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """Accepts user-uploaded documents and finds similar IRS instructions."""
    if not file.filename.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(status_code=400, detail="Only PDF, DOCX, and TXT files are supported.")

    doc_text = extract_text_from_file(file)
    
    # Convert text to embedding
    doc_embedding = embedding_model.encode(doc_text).tolist()
    
    # Query Pinecone for similar IRS instructions
    results = index.query(vector=doc_embedding, top_k=5, include_metadata=True)
    
    matched_docs = "\n".join([match["metadata"]["text"] for match in results["matches"]])
    
    return {
        "uploaded_doc": doc_text[:500],
        "related_irs_instructions": matched_docs if matched_docs else "No matching IRS instructions found."
    }


@app.post("/chat/")
def chat_with_model(request: ChatRequest):
    """Handles chat requests from the frontend using LM Studio API."""
    
    # Retrieve context from Pinecone
    context = retrieve_relevant_context(request.message)
    input_text = f"Context: {context}\nUser: {request.message}\nAssistant:"

    # Send request to LM Studio
    payload = {
        "model": "deepseek-coder-v2-lite-instruct",
        "prompt": input_text,
        "temperature": 0.7,
        "max_tokens": 200
    }

    try:
        response = requests.post(LM_STUDIO_API_URL, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        response_json = response.json()
        
        assistant_reply = response_json.get("choices", [{}])[0].get("text", "").strip()
        return {"response": assistant_reply}

    except requests.exceptions.RequestException as e:
        print(f"⚠️ LM Studio API Error: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with LM Studio API.")

