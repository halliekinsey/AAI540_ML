# Final Backend API Endpoints Testing Report

This document details the final testing of all API endpoints for our local AI model server running on `http://localhost:6000`. Each endpoint was tested for expected functionality and correctness using `cURL` commands. The results are documented, showing the responses received.

## Tested API Endpoints and Commands

### 1. List Available Models
**Command:**
```sh
curl -X GET http://localhost:6000/v1/models \
-H "Authorization: Bearer team-tax-1531"
```
**Expected Behavior:**  
Returns a list of available models on the server.

**Test Result:**
```json
{
  "data": [
    {"id": "saul_7b_instruct", "object": "model"},
    {"id": "lawma_8b", "object": "model"},
    {"id": "lawma_70b", "object": "model"},
    {"id": "DeepSeek-V2-Lite", "object": "model"}
  ],
  "object": "list"
}
```
**Success:** The API correctly returned the list of models.


2. Generate Text (Non-Chat) $2
**Command:**
```sh
curl -X POST http://localhost:6000/generate \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "model_name": "DeepSeek-V2-Lite",
  "text": "What is the process for filing taxes?",
  "max_length": 150
}'
```
**Expected Behavior:**  
Generates a response based on the provided input prompt.

**Test Result:**
```json
{
  "model": "DeepSeek-V2-Lite",
  "response": "Step 1: Gather necessary documents and information...\nStep 2: Choose the appropriate tax form...",
  "time_taken": 13.6
}
```
**Success:** The model generated a valid response explaining the tax filing process.


### 3. Generate Text as a Stream
**Command:**
```sh
curl -X POST http://localhost:6000/generate_stream \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "model_name": "DeepSeek-V2-Lite",
  "text": "Explain the steps for tax filing.",
  "max_length": 150
}'
```
**Expected Behavior:**  
Streams a response with tax filing steps.

**Test Result (Partial Output):**
```json
{
  "response": "1. Gather Necessary Documents...\n2. Determine Filing Status..."
}
```
**Success:** The response was streamed correctly, listing tax filing steps.


### 4. Start a New Chat Session
**Command:**
```sh
curl -X POST http://localhost:6000/v1/chat/completions \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "model": "DeepSeek-V2-Lite",
  "username": "z-ali",
  "messages": [
    {"role": "user", "content": "What tax deductions can I claim as a freelancer?"}
  ],
  "max_tokens": 200
}'
```
**Expected Behavior:**  
Starts a new chat session and returns a `chat_id`.

**Test Result:**
```json
{
  "chat_id": "5bb3bcc0-246c-4bc2-9f76-21be1f4f29bc",
  "choices": [{
    "message": {
      "content": "As a freelancer, you may claim deductions for home office expenses, self-employment tax, health insurance premiums, etc.",
      "role": "assistant"
    },
    "finish_reason": "stop"
  }],
  "usage": {"total_tokens": 233}
}
```
**Success:** The chat session was created, and a valid response was generated.


### 5. Continue an Existing Chat Session
**Command:**
```sh
curl -X POST http://localhost:6000/v1/chat/completions \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "model": "DeepSeek-V2-Lite",
  "username": "z-ali",
  "chat_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "messages": [
    {"role": "user", "content": "How do I maximize my deductions?"}
  ],
  "max_tokens": 200
}'
```
**Expected Behavior:**  
Continues a previous chat session.

**Test Result:**
```json
{
  "chat_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "choices": [{
    "message": {
      "content": "To maximize deductions, keep track of expenses, itemize where possible, and contribute to retirement plans.",
      "role": "assistant"
    }
  }]
}
```
**Success:** The chat continued successfully, and a valid response was generated.


### 6. Generate a Plain Completion (Non-Chat)
**Command:**
```sh
curl -X POST http://localhost:6000/v1/completions \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "model": "DeepSeek-V2-Lite",
  "prompt": "Provide guidance on small business tax planning.",
  "max_tokens": 200
}'
```
**Expected Behavior:**  
Returns a general response for tax planning.

**Test Result:**
```json
{
  "choices": [{
    "text": "Small businesses should understand their tax obligations, choose the right structure, and maximize deductions."
  }]
}
```
**Success:** The response contained useful tax planning guidance.


### 7. Search the Pinecone Index
**Command:**
```sh
curl -X POST http://localhost:6000/search \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "query": "freelancer tax deductions",
  "top_k": 3
}'
```
**Expected Behavior:**  
Returns top 3 relevant indexed documents.

**Test Result:**
```json
{
  "query": "freelancer tax deductions",
  "results": [
    {"id": "i1040sf.md_118", "score": 0.6055, "text": "...deductions on Schedule 1 (Form 1040)..."},
    {"id": "i1120c.md_176", "score": 0.5836, "text": "...Taxable income figured without the deduction..."},
    {"id": "i4562.md_56", "score": 0.5757, "text": "...trade or business deductions..."}
  ]
}
```
**Success:** The search query returned relevant indexed results.


### 8. Retrieve Chat History
**Command:**
```sh
curl -X GET "http://localhost:6000/chat_history?chat_id=a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
-H "Authorization: Bearer team-tax-1531"
```
**Expected Behavior:**  
Returns the chat history for a given `chat_id`.

**Test Result:**
```json
{
  "chat_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "messages": [
    {"content": "How do I maximize my deductions?", "role": "user"},
    {"content": "Keep track of expenses, itemize, and use tax credits.", "role": "assistant"}
  ]
}
```
**Success:** The chat history was retrieved correctly.


### 9. List All Chat Sessions
**Command:**
```sh
curl -X GET "http://localhost:6000/chat_sessions" \
-H "Authorization: Bearer team-tax-1531"
```
**Expected Behavior:**  
Returns all active chat session IDs.

**Test Result:**
```json
{
  "chat_sessions": [
    "92ce3f2a-a2b9-4418-9ae4-9144d9dc8510",
    "c05368d7-fff2-4319-bc65-a2cbcc47ed5b",
    "123e4567-e89b-12d3-a456-426614174000",
    "5bb3bcc0-246c-4bc2-9f76-21be1f4f29bc"
  ]
}
```
**Success:** All active chat sessions were listed.

