import gradio as gr
import requests
import uuid
import time
import os
import io
import re
from PyPDF2 import PdfReader

API_HOST = "http://0.0.0.0:6000"
TOKEN = "team-tax-1531"

UPLOAD_PDF_URL = f"{API_HOST}/upload_pdf"
CHAT_SESSIONS_URL = f"{API_HOST}/chat_sessions"
CHAT_HISTORY_URL = f"{API_HOST}/chat_history"
CHAT_COMPLETIONS_URL = f"{API_HOST}/v1/chat/completions"

def get_headers():
    return {"Authorization": f"Bearer {TOKEN}"}

try:
    resp = requests.get(f"{API_HOST}/v1/models", headers=get_headers(), timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or not isinstance(data["data"], list):
        raise RuntimeError("Unexpected response format from /v1/models.")
    MODEL_OPTIONS = [m["id"] for m in data["data"] if isinstance(m, dict) and "id" in m]
    if not MODEL_OPTIONS:
        raise RuntimeError("No models available from the backend.")
    DEFAULT_MODEL = "DeepSeek-V2-Lite" if "DeepSeek-V2-Lite" in MODEL_OPTIONS else MODEL_OPTIONS[0]
except requests.exceptions.RequestException as e:
    print(f"Error: Failed to fetch models from backend - {e}")
    raise RuntimeError("Critical failure: Unable to retrieve model list.")
except Exception as e:
    print(f"Unexpected error: {e}")
    raise RuntimeError("Unexpected error while fetching model list.")

def fetch_chat_sessions():
    try:
        resp = requests.get(CHAT_SESSIONS_URL, headers=get_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("chat_sessions", [])
    except Exception as e:
        print("Error fetching chat sessions:", e)
        return []

def load_chat_history(chat_id):
    if not chat_id:
        return [], "", None

    try:
        url = f"{CHAT_HISTORY_URL}?chat_id={chat_id}"
        resp = requests.get(url, headers=get_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        messages = data.get("messages", [])
        chat_history = [(m["role"], m["content"]) for m in messages]
        return chat_history, data.get("chat_id", chat_id), messages
    except Exception as e:
        return [], chat_id, None

def fetch_user_sessions(username):
    all_sessions = fetch_chat_sessions()
    user_chat_ids = []

    for sid in all_sessions:
        _, _, raw_msgs = load_chat_history(sid)
        if raw_msgs is None:
            continue
        for m in raw_msgs:
            if m.get("username") == username:
                user_chat_ids.append(sid)
                break

    return user_chat_ids

def load_selected_chat_from_dropdown(chat_id):
    if not chat_id:
        return [], ""
    chat_history, final_id, _ = load_chat_history(chat_id)
    return chat_history, final_id

def new_chat():
    return [], ""


def refresh_sessions(username):
    sessions = fetch_user_sessions(username)
    return gr.update(choices=sessions)


def generate_chat_id_from_question(question: str) -> str:
    processed = re.sub(r'[^a-zA-Z0-9\s-]', '', question.strip().lower())
    processed = processed.replace(' ', '-')
    processed = processed[:50]
    if not processed:
        processed = str(uuid.uuid4())[:8]
    return processed

def generate_chat_response(
    user_message,
    chat_history,
    username,
    chat_id,
    selected_model,
    use_pdf_context,
    tax_optimize,
    max_tokens
):
    if not user_message.strip():
        return "", chat_history, chat_id
    if not chat_id:
        chat_id = generate_chat_id_from_question(user_message)
    chat_history.append(("user", user_message))
    messages_for_api = [
        {"role": role, "content": content} for (role, content) in chat_history
    ]

    payload = {
        "model": selected_model,
        "messages": messages_for_api,
        "max_tokens": max_tokens,
        "username": username,
        "chat_id": chat_id,
        "use_pdf_context": use_pdf_context,
        "tax_optimize": tax_optimize
    }

    try:
        resp = requests.post(CHAT_COMPLETIONS_URL, json=payload, headers=get_headers(), timeout=420)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        chat_history.append(("assistant", error_msg))
        return "", chat_history, chat_id

    if "choices" not in data:
        error_msg = f"Error: unexpected response format: {data}"
        chat_history.append(("assistant", error_msg))
        return "", chat_history, chat_id

    assistant_msg = data["choices"][0]["message"]["content"]
    new_chat_id = data.get("chat_id", chat_id)

    chat_history.append(("assistant", assistant_msg))

    return "", chat_history, new_chat_id

def upload_pdf(pdf_path, username, chat_id):
    if not pdf_path:
        return "No PDF selected or invalid path."

    if not username.strip():
        return "Please enter a username before uploading."

    if not chat_id.strip():
        chat_id = str(uuid.uuid4())

    file_name = os.path.basename(pdf_path)
    try:
        with open(pdf_path, "rb") as f:
            files = {"pdf": (file_name, f, "application/pdf")}
            data = {"username": username, "chat_id": chat_id}
            resp = requests.post(UPLOAD_PDF_URL, files=files, data=data, headers=get_headers())
            resp.raise_for_status()
            result = resp.json()
            if "error" in result:
                return f"Error uploading PDF: {result['error']}"
            return (
                f"Successfully uploaded PDF '{file_name}'. "
                f"Chunks processed: {result.get('chunks')}. "
                f"Message: {result.get('message', '')}\n"
                f"(Chat ID now: {chat_id})"
            )
    except Exception as e:
        return f"Exception during upload: {str(e)}"

def on_pdf_change(pdf_path, username, chat_id, pdf_loaded_state):

    status_msg = upload_pdf(pdf_path, username, chat_id)
    if "Successfully uploaded PDF" in status_msg or "already processed" in status_msg:
        # success => enable PDF usage
        return (
            status_msg,
            True, 
            gr.update(value=True, interactive=True),
            gr.update(value=False, interactive=False),
            chat_id if chat_id else str(uuid.uuid4())
        )
    else:
        return (
            status_msg,
            False,
            gr.update(),
            gr.update(),
            chat_id
        )

def on_tax_optimize_change(tax_value, pdf_status, use_pdf_context):

    if tax_value:
        return (
            "",
            gr.update(value=False, interactive=False)
        )
    else:
        # User toggled it off => we will keep whatever the pdf_status is, but re-enable pdf_context in principle
        # We'll manage actual enabling in the next step if PDF is loaded
        return (pdf_status, gr.update(interactive=True))

def on_use_pdf_change(pdf_context_value, pdf_status, tax_opt):
    if pdf_context_value:
        return (
            pdf_status,
            gr.update(value=False, interactive=False, info="First remove the PDF to use this")
        )
    else:
        return (
            pdf_status,
            gr.update(interactive=True, info="Uses our database to refine your result.")
        )

def on_pdf_loaded_change(pdf_loaded, current_use_pdf_value):
    if pdf_loaded:
        return gr.update(
            interactive=True,
            info="PDF is ready to provide context!"
        )
    else:
        return gr.update(
            value=False,
            interactive=False,
            info="Select and upload a PDF first."
        )

def send_feedback(feedback_value, chat_history, username, chat_id, use_pdf_context, tax_optimize):
    if not chat_id:
        return "Cannot send feedback without a chat_id. Please chat first."

    payload = {
        "chat_id": chat_id,
        "username": username,
        "feedback": feedback_value,     # "UP" or "DOWN"
        "conversation": chat_history,  
        "use_pdf_context": use_pdf_context,
        "tax_optimize": tax_optimize
    }

    url = f"{API_HOST}/feedback"
    try:
        resp = requests.post(url, json=payload, headers=get_headers(), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            return f"Error in feedback: {data['error']}"
        return f"Feedback saved: {data.get('message', 'OK')}"
    except Exception as e:
        return f"Exception sending feedback: {str(e)}"

def build_interface():
    with gr.Blocks(
        theme="IBM_Carbon_Theme",
        title="Tax & Legal Chat"
    ) as demo:

        gr.Markdown(
            """
<div style="text-align:center; margin-bottom:20px;">
    <h1 style="margin-bottom:0.2em;">TaxSense: Automated Filing Assistant</h1>
    <p>By <a href="https://github.com/zainnobody">Zain Ali</a>, <a href="https://github.com/halliekinsey">Hallie Kinsey</a>, and Akram Mahmoud.<br>
        This system can help you with IRS tax filing guidance <em>and</em> general documents.<br>
        Upload a PDF to provide custom context, or simply chat about any topic.
    </p>
</div>
            """,
            elem_id="header_centered"
        )

        with gr.Row():

            with gr.Column(scale=1):
                gr.Markdown("### Sessions")
                username = gr.Textbox(label="Username", value="zain", interactive=True)

                load_sessions_btn = gr.Button("Refresh Session List")
                chat_sessions_dropdown = gr.Dropdown(label="Existing Sessions", choices=[], interactive=True)
                load_chat_btn = gr.Button("Load Selected Chat")

                gr.Markdown("---")
                new_chat_btn = gr.Button("New Chat", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### PDF Upload (auto-index)")

                pdf_input = gr.File(label="Select PDF", type="filepath")
                pdf_status = gr.Markdown()


            with gr.Column(scale=3):
                pdf_loaded_state = gr.State(False)

                with gr.Row():
                    selected_model = gr.Dropdown(
                        label="Select Model",
                        choices=MODEL_OPTIONS,
                        value=DEFAULT_MODEL,
                        interactive=True
                    )
                    max_tokens_box = gr.Number(
                        label="Max Tokens",
                        value=200,
                        precision=0
                    )
                    chat_id_box = gr.Textbox(
                        label="Chat ID (auto-generated on 1st message)",
                        value="",
                        interactive=True
                    )

                chatbot = gr.Chatbot(label="Conversation", type="tuples")

                user_input = gr.Textbox(
                    label="Your Message:",
                    placeholder="Ask any question, or get help with your doc..."
                )

                with gr.Row():
                    use_pdf_context = gr.Checkbox(
                        label="Use PDF context",
                        value=False,
                        interactive=False, 
                        info="Select and upload a PDF first."
                    )
                    tax_optimize = gr.Checkbox(
                        label="Tax Optimize (Pinecone)",
                        value=False,
                        interactive=True,
                        info="Uses our database to refine your result."
                    )
                with gr.Row():
                    thumbs_up_btn = gr.Button("👍 Thumbs Up")
                    thumbs_down_btn = gr.Button("👎 Thumbs Down")
                    feedback_status = gr.Markdown("")
                send_btn = gr.Button("Send", variant="primary")

                chat_history_state = gr.State([])

        load_sessions_btn.click(
            fn=refresh_sessions,
            inputs=[username],
            outputs=chat_sessions_dropdown
        )
        load_chat_btn.click(
            fn=load_selected_chat_from_dropdown,
            inputs=[chat_sessions_dropdown],
            outputs=[chatbot, chat_id_box]
        ).then(
            fn=lambda x: x,
            inputs=[chatbot],
            outputs=[chat_history_state]
        )
        new_chat_btn.click(
            fn=new_chat,
            inputs=[],
            outputs=[chatbot, chat_id_box]
        ).then(
            fn=lambda: [],
            inputs=[],
            outputs=[chat_history_state]
        )
        pdf_input.change(
            fn=on_pdf_change,
            inputs=[pdf_input, username, chat_id_box, pdf_loaded_state],
            outputs=[pdf_status, pdf_loaded_state, use_pdf_context, tax_optimize, chat_id_box]
        )

        tax_optimize.change(
            fn=on_tax_optimize_change,
            inputs=[tax_optimize, pdf_status, use_pdf_context],
            outputs=[pdf_status, use_pdf_context]
        )

        use_pdf_context.change(
            fn=on_use_pdf_change,
            inputs=[use_pdf_context, pdf_status, tax_optimize],
            outputs=[pdf_status, tax_optimize]
        )
        pdf_loaded_state.change(
            fn=on_pdf_loaded_change,
            inputs=[pdf_loaded_state, use_pdf_context],
            outputs=use_pdf_context
        )

        send_btn.click(
            fn=generate_chat_response,
            inputs=[
                user_input,
                chat_history_state,
                username,
                chat_id_box,
                selected_model,
                use_pdf_context,
                tax_optimize,
                max_tokens_box
            ],
            outputs=[
                user_input,
                chatbot,
                chat_id_box
            ]
        )

        thumbs_up_btn.click(
            fn=send_feedback,
            inputs=[
                gr.State("UP"),      
                chat_history_state,
                username,
                chat_id_box,
                use_pdf_context,
                tax_optimize
            ],
            outputs=[feedback_status]
        )

        thumbs_down_btn.click(
            fn=send_feedback,
            inputs=[
                gr.State("DOWN"),
                chat_history_state,
                username,
                chat_id_box,
                use_pdf_context,
                tax_optimize
            ],
            outputs=[feedback_status]
        )


    return demo

if __name__ == "__main__":
    demo_app = build_interface()
    demo_app.launch(server_name="0.0.0.0", server_port=7860, share=True)
