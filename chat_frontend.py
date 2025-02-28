import gradio as gr
import requests
import uuid
import time
import os
import io
from PyPDF2 import PdfReader

# ----------------------
#  CONFIG
# ----------------------

API_HOST = "http://0.0.0.0:6000"  # or your server address
TOKEN = "team-tax-1531"

# For the /upload_pdf endpoint:
UPLOAD_PDF_URL = f"{API_HOST}/upload_pdf"

# For the /chat_sessions and /chat_history endpoints:
CHAT_SESSIONS_URL = f"{API_HOST}/chat_sessions"
CHAT_HISTORY_URL = f"{API_HOST}/chat_history"

# For the /v1/chat/completions endpoint:
CHAT_COMPLETIONS_URL = f"{API_HOST}/v1/chat/completions"

# The list of model names recognized by your backend:
MODEL_OPTIONS = [
    "saul_7b_instruct",
    "lawma_8b",
    "lawma_70b",
    "DeepSeek-V2-Lite"
]

# ----------------------
#  HELPER FUNCTIONS
# ----------------------

def get_headers():
    """Build the headers with your token for auth."""
    return {"Authorization": f"Bearer {TOKEN}"}

def fetch_chat_sessions():
    """Calls GET /chat_sessions to retrieve all chat IDs."""
    try:
        resp = requests.get(CHAT_SESSIONS_URL, headers=get_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("chat_sessions", [])
    except Exception as e:
        print("Error fetching chat sessions:", e)
        return []

def load_chat_history(chat_id):
    """Calls GET /chat_history?chat_id=..., returns conversation as (role, content)."""
    if not chat_id:
        return [], ""

    try:
        url = f"{CHAT_HISTORY_URL}?chat_id={chat_id}"
        resp = requests.get(url, headers=get_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        messages = data.get("messages", [])
        chat_history = [(m["role"], m["content"]) for m in messages]
        return chat_history, data.get("chat_id", chat_id)
    except Exception as e:
        return [], chat_id

def generate_chat_response(
    user_message,
    chat_history,
    username,
    chat_id,
    selected_model,
    use_pdf_context,
    tax_optimize
):
    """
    Called when user sends a new message.
    We append that message to chat_history, then send entire conversation to /v1/chat/completions.
    """
    if not user_message.strip():
        # Blank message
        return "", chat_history, chat_id

    # Append user's new message to local chat_history
    chat_history.append(("user", user_message))

    # Convert local chat history to the format the backend expects
    messages_for_api = [
        {"role": role, "content": content} for (role, content) in chat_history
    ]

    payload = {
        "model": selected_model,
        "messages": messages_for_api,
        "max_tokens": 200,
        "username": username,
        "chat_id": chat_id,
        "use_pdf_context": use_pdf_context,
        "tax_optimize": tax_optimize
    }

    try:
        resp = requests.post(CHAT_COMPLETIONS_URL, json=payload, headers=get_headers(), timeout=60)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        chat_history.append(("assistant", error_msg))
        return "", chat_history, chat_id

    # Check for a valid response
    if "choices" not in data:
        error_msg = f"Error: unexpected response format: {data}"
        chat_history.append(("assistant", error_msg))
        return "", chat_history, chat_id

    assistant_msg = data["choices"][0]["message"]["content"]
    new_chat_id = data.get("chat_id", chat_id)

    # Append assistant's response
    chat_history.append(("assistant", assistant_msg))

    return "", chat_history, new_chat_id

def new_chat():
    """Start a new empty conversation with a fresh chat_id."""
    return [], str(uuid.uuid4())

def refresh_sessions():
    """Refresh the session list in the dropdown."""
    sessions = fetch_chat_sessions()
    return gr.update(choices=sessions)

def load_selected_chat_from_dropdown(chat_id):
    """Load chat history from the selected chat_id in the dropdown."""
    if not chat_id:
        return [], ""
    chat_history, final_id = load_chat_history(chat_id)
    return chat_history, final_id

# ----------------------
#  PDF UPLOAD HANDLING
# ----------------------

def upload_pdf(pdf_path, username, chat_id):
    """
    Gradio callback: 
    - Opens the PDF file from 'pdf_path'
    - Uploads it via POST /upload_pdf with multipart/form-data
    - The backend will chunk & embed the PDF for RAG
    """
    if not pdf_path:
        return "No PDF selected or invalid path."

    if not username.strip():
        return "Please enter a username before uploading."

    if not chat_id.strip():
        return "Please enter a valid chat ID before uploading."

    file_name = os.path.basename(pdf_path)
    try:
        with open(pdf_path, "rb") as f:
            files = {"pdf": (file_name, f, "application/pdf")}
            data = {
                "username": username,
                "chat_id": chat_id
            }
            # Make the request
            resp = requests.post(UPLOAD_PDF_URL, files=files, data=data, headers=get_headers())
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                return f"Error uploading PDF: {result['error']}"
            return (
                f"Successfully uploaded PDF '{file_name}'. "
                f"Chunks processed: {result.get('chunks')}. "
                f"Message: {result.get('message', '')}"
            )
    except Exception as e:
        return f"Exception during upload: {str(e)}"

# ----------------------
#  BUILDING THE UI
# ----------------------

def build_interface():
    with gr.Blocks(title="Tax & Legal Chat", theme="default") as demo:
        gr.Markdown("""
        # Tax & Legal Chat  
        **Left panel** for sessions & PDF indexing.  
        **Right panel** for conversation with chosen model.
        """)

        with gr.Row():
            #
            # LEFT COLUMN: Sessions + PDF Upload
            #
            with gr.Column(scale=1):
                gr.Markdown("### Sessions")
                load_sessions_btn = gr.Button("Refresh Session List")
                chat_sessions_dropdown = gr.Dropdown(label="Existing Sessions", choices=[], interactive=True)
                load_chat_btn = gr.Button("Load Selected Chat")

                gr.Markdown("---")
                new_chat_btn = gr.Button("New Chat", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### PDF Upload (for RAG)")
                pdf_input = gr.File(label="Select PDF", type="filepath")
                pdf_status = gr.Markdown()
                upload_pdf_btn = gr.Button("Upload & Index PDF")

            #
            # RIGHT COLUMN: Chat
            #
            with gr.Column(scale=3):
                # Basic user info
                with gr.Row():
                    username = gr.Textbox(label="Username", value="alice", interactive=True)
                    selected_model = gr.Dropdown(
                        label="Select Model",
                        choices=MODEL_OPTIONS,
                        value=MODEL_OPTIONS[0],
                        interactive=True
                    )

                chat_id_box = gr.Textbox(label="Chat ID", value=str(uuid.uuid4()), interactive=True)

                # The chat itself
                chatbot = gr.Chatbot(label="Conversation", type="tuples")

                # For user message
                user_input = gr.Textbox(label="Your Message:", placeholder="Ask your tax/legal question...")

                # Checkboxes to control PDF context or Pinecone
                with gr.Row():
                    use_pdf_context = gr.Checkbox(label="Use PDF context", value=False)
                    tax_optimize = gr.Checkbox(label="Tax Optimize (Pinecone)", value=False)

                send_btn = gr.Button("Send", variant="primary")

                # We'll store the entire (role, content) conversation in a state
                chat_history_state = gr.State([])

        # ----------------------
        #  CALLBACK WIRES
        # ----------------------

        # (1) Refresh session list
        load_sessions_btn.click(
            fn=refresh_sessions,
            inputs=[],
            outputs=chat_sessions_dropdown
        )

        # (2) Load selected chat
        load_chat_btn.click(
            fn=load_selected_chat_from_dropdown,
            inputs=[chat_sessions_dropdown],
            outputs=[chatbot, chat_id_box]
        ).then(
            fn=lambda x: x,  # pass same chat as state
            inputs=[chatbot],
            outputs=[chat_history_state]
        )

        # (3) New Chat
        new_chat_btn.click(
            fn=new_chat,
            inputs=[],
            outputs=[chatbot, chat_id_box]
        ).then(
            fn=lambda: [],
            inputs=[],
            outputs=[chat_history_state]
        )

        # (4) Upload & Index PDF
        upload_pdf_btn.click(
            fn=upload_pdf,
            inputs=[pdf_input, username, chat_id_box],
            outputs=[pdf_status]
        )

        # (5) Send user message -> get assistant reply
        send_btn.click(
            fn=generate_chat_response,
            inputs=[
                user_input,         # new message
                chat_history_state, # entire conversation so far
                username,
                chat_id_box,
                selected_model,
                use_pdf_context,
                tax_optimize
            ],
            outputs=[
                user_input,         # reset user msg input
                chatbot,
                chat_id_box
            ],
        )

    return demo


# ----------------------
#  MAIN
# ----------------------

if __name__ == "__main__":
    demo_app = build_interface()
    demo_app.launch(server_name="0.0.0.0", server_port=7860, share=True)
