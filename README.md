# TaxSense: A RAG-Powered Tax Advisory System

TaxSense is a Retrieval-Augmented Generation (RAG) system that delivers informed U.S. tax guidance by combining fine-tuned language models, user-uploaded PDFs, and a Pinecone-indexed corpus of IRS documents. Its conversational interface (powered by Gradio) provides clear, reliable tax advice while maintaining secure chat logging and easy data backup.

Our project features a custom-trained model, **TaxSense**, built specifically for this application. For more details about its development and performance, explore our model on Hugging Face at [zainnobody/TaxSense](https://huggingface.co/zainnobody/TaxSense).


## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
   - [Local Launch](#local-launch)
   - [Docker Deployment](#docker-deployment)
6. [Backend & API Endpoints](#backend--api-endpoints)
7. [Frontend Dashboard](#frontend-dashboard)
8. [Admin Service for Data Backup & Sync](#admin-service-for-data-backup--sync)
9. [Technical Deep Dive](#technical-deep-dive)
10. [Authors & Credits](#authors--credits)
11. [License](#license)


## Overview

TaxSense (also referred to as LegalSense in some documents) is an end-to-end tax advisory system that leverages Retrieval-Augmented Generation to answer complex tax-related questions. By sourcing context from official IRS filings, custom user PDFs, and external data indexed via Pinecone, TaxSense ensures that its guidance is both detailed and accurate.


## Features

- **Retrieval-Augmented Generation (RAG):** Combines live query processing with embedded IRS document context to produce traceable, high-quality responses.
- **Fine-Tuned Models:** Supports multiple models (e.g., DeepSeek-V2-Lite, Saul-7B-Instruct, Lawma variants) selectable at runtime.
- **PDF Upload & Processing:** Upload custom PDFs to extract and index context automatically.
- **Pinecone-Based Tax Optimization:** Retrieve relevant IRS forms and instructions to augment answers.
- **Chat Management:** Secure conversation logging via SQLite and session management.
- **Admin Service:** Automated backups and S3 synchronization for key data directories.
- **Docker Deployment:** Ready-to-run Dockerfiles for simplified deployment across environments.


## Project Structure

```
.
├── README.md                         <-- This documentation file
├── TaxSense Detailed Report.ipynb    <-- Technical deep-dive notebook (design, testing, rationale)
├── TaxSense_backend.py               <-- Flask-based backend exposing API endpoints
├── TaxSense_frontend.py              <-- Gradio-based user interface for chat, PDF upload, and model selection
├── admin.py                          <-- Admin service for data backup and AWS S3 sync
├── data                              <-- CSV files with tax questions & answers, and supporting data
├── further-docs                      <-- Additional documentation and testing reports (backend/admin endpoints)
├── logs                              <-- Development logs and model performance notes
├── miscellaneous                     <-- Exploratory notebooks and experiments (e.g., initial data exploration)
├── requirements.txt                  <-- Python dependencies and version information
├── sample-files                      <-- Example chat logs, PDFs, and SQLite files
└── supporting_media                  <-- Diagrams, screenshots, and media assets (e.g., TaxSense architecture, S3 screenshot)
```


## Installation & Setup

### Prerequisites

- **Python 3.9+**
- **pip** (or conda)
- For GPU acceleration, ensure PyTorch is installed with CUDA support.
- (Optional) AWS credentials for S3 backup and Pinecone API key for tax optimization features.

### Environment Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/halliekinsey/AAI540_ML.git
   cd TaxSense
   ```

2. **Create and Activate a Virtual Environment (Recommended):**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables:**

   - For Pinecone-based tax optimization:

     ```bash
     export PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
     ```

   - (Optional) For AWS backup, set your AWS credentials:

     ```bash
     export AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY"
     export AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_KEY"
     ```


## Usage

### Local Launch

1. **Start the Backend:**

   ```bash
   python TaxSense_backend.py
   ```
   
   The backend runs on `http://0.0.0.0:6000` by default.

2. **Start the Frontend:**

   In a separate terminal, run:

   ```bash
   python TaxSense_frontend.py
   ```
   
   The Gradio UI launches on `http://127.0.0.1:7860` (or `0.0.0.0:7860`).

3. **Interact with TaxSense:**

   - Enter a username.
   - Optionally upload a PDF for custom context.
   - Select your preferred AI model.
   - Toggle "Tax Optimize" to enable Pinecone-based retrieval.
   - Chat with the assistant to get tax guidance.

### Docker Deployment

For containerized deployment, you can build separate Docker images for the backend and frontend:

1. **Backend Dockerfile Example:**

   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   ENV PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
   CMD ["python", "TaxSense_backend.py"]
   EXPOSE 6000
   ```

2. **Frontend Dockerfile Example:**

   ```dockerfile
   FROM python:3.10-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .
   CMD ["python", "TaxSense_frontend.py"]
   EXPOSE 7860
   ```

3. **Build and Run:**

   ```bash
   docker build -t taxsense-backend -f Dockerfile.backend .
   docker run -d -p 6000:6000 taxsense-backend

   docker build -t taxsense-frontend -f Dockerfile.frontend .
   docker run -d -p 7860:7860 taxsense-frontend
   ```


## Backend & API Endpoints

The backend is a Flask application exposing several RESTful endpoints:

- **Chat & Completion Endpoints:**
  - `POST /v1/chat/completions` – Handles conversational interactions.
  - `POST /v1/completions` – Returns text completions (OpenAI-style).

- **Search Endpoint:**
  - `POST /search` – Leverages Pinecone to retrieve relevant IRS document snippets based on a query.

- **PDF Upload:**
  - `POST /upload_pdf` – Allows uploading and processing of PDFs for context extraction.

- **Chat History:**
  - `GET /chat_history` – Retrieves conversation history from the local SQLite database.
  - `GET /chat_sessions` – Lists available chat sessions.

- **Models List:**
  - `GET /v1/models` – Returns a list of available model identifiers (configured in `MODEL_PATHS` and `MODEL_CONFIGS`).

- **Feedback:**
  - `POST /feedback` – Accepts user feedback for further analysis and fine-tuning.

For detailed code and additional context, refer to the inline comments in `TaxSense_backend.py` and the [Final Backend API Endpoints Testing Report](further-docs/backend_final_api_endpoints_test_results.md).


## Frontend Dashboard

The Gradio-based frontend in `TaxSense_frontend.py` provides a user-friendly interface with the following capabilities:

- **Session Management:** Create, load, and refresh chat sessions.
- **PDF Upload:** Drag and drop PDFs to add custom context.
- **Model Selection:** Choose from multiple AI models (e.g., DeepSeek-V2-Lite, Saul-7B-Instruct, Lawma variants).
- **Context Toggles:** Enable “Use PDF Context” or “Tax Optimize” (via Pinecone) to refine responses.
- **Feedback:** Thumbs up/down buttons for user feedback on the generated responses.


## Admin Service for Data Backup & Sync

The `admin.py` script provides endpoints for managing data backup and synchronization:

- **ZIP Data:** `GET /admin/zip_data` returns a ZIP archive of critical directories and files.
- **Unzip Data:** `POST /admin/unzip_data` processes an uploaded ZIP file to update local data.
- **S3 Push/Pull:**
  - `GET /admin/push_s3` – Uploads data to an Amazon S3 bucket.
  - `GET /admin/pull_s3` – Downloads data from S3 to local storage.

These endpoints facilitate automated backups and ensure data consistency across deployments. For further details, review `admin.py` and the [Admin Endpoints Testing Report](further-docs/admin_final_api_endpoints_test_results.md).


## Technical Deep Dive

For a detailed explanation of the design, model fine-tuning, PDF processing, Pinecone integration, and challenges encountered, refer to the "TaxSense Detailed Report.ipynb" notebook. This notebook covers:

- **Data Collection & Cleaning:** How tax Q&A data and IRS instructions were gathered and processed.
- **Contextual Information Extraction:** Splitting, indexing, and embedding PDF content.
- **Model Fine-Tuning:** Steps for preparing and training the TaxSense model using LoRA and PEFT.
- **Endpoint Implementation:** Detailed code walkthrough for chat, completions, search, and admin endpoints.
- **Performance & Resource Considerations:** Benchmarking various models (DeepSeek-V2-Lite, Saul-7B-Instruct, Lawma variants) for speed, quality, and scalability.


## Authors & Credits

TaxSense was created by:

- [Zain Ali](https://github.com/zainnobody)
- [Hallie Kinsey](https://github.com/halliekinsey)
- Akram Mahmoud

Special thanks to contributors and testers who helped refine the system.


## License

This project is licensed under the terms specified in the LICENSE file. Please review the license for details on permitted usage and distribution.


Enjoy seamless tax guidance with TaxSense!
