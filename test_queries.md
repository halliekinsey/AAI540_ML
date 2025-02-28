## All queries

### **1. List Available Models**
```sh
curl -X GET http://localhost:6000/v1/models \
-H "Authorization: Bearer team-tax-1531"
```

---

### **2. Generate Text (Non-Chat)**
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

---

### **3. Generate Text as a Stream**
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

---

### **4. Start a New Chat Session**
This command creates a new chat session for **z-ali** and returns a unique `chat_id` (save this ID for future calls).
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

*Example response (the returned chat_id is used in the next command):*
```json
{
  "id": "chatcmpl-1700000000",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "DeepSeek-V2-Lite",
  "chat_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "username": "z-ali",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "As a freelancer, you may claim deductions for business expenses such as equipment, software, and home office expenses."},
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 100,
    "total_tokens": 150
  },
  "time_taken": 1.2
}
```

---

### **5. Continue an Existing Chat Session**
Use the `chat_id` from the previous response to continue the chat.
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

---

### **6. Generate a Plain Completion (Non-Chat)**
```sh
curl -X POST http://localhost:6000/v1/completions \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "model": "DeepSeek-V2-Lite",
  "prompt": "Provide guidance on small business tax planning.",
  "max_tokens": 200,
  "tax_optimize": false
}'
```

---

### **7. Search the Pinecone Index**
```sh
curl -X POST http://localhost:6000/search \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "query": "freelancer tax deductions",
  "top_k": 3
}'
```

---

### **8. Retrieve Chat History**
Replace `<CHAT_ID>` with the actual chat session ID (e.g., `a1b2c3d4-e5f6-7890-abcd-ef1234567890`).
```sh
curl -X GET "http://localhost:6000/chat_history?chat_id=a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
-H "Authorization: Bearer team-tax-1531"
```

---

### **9. List All Chat Sessions**
```sh
curl -X GET "http://localhost:6000/chat_sessions" \
-H "Authorization: Bearer team-tax-1531"
```

---

### **10.Push Chat History to S3**
*This endpoint is not active by default. To test it, you would first uncomment the S3 upload line in the `upload_chat_to_s3()` function and optionally expose an endpoint for it (e.g., `/upload_chat_to_s3`).*
```sh
curl -X POST "http://localhost:6000/upload_chat_to_s3" \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "chat_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}'
```






## Logs:



```
(workbench-zaina-nc-wb1) (workbench-zaina-nc-wb1) root@6cd743e3-a6f3-41cb-8ea8-3cf1d46de239:~# curl -X GET http://localhost:6000/v1/models \
-H "Authorization: Bearer team-tax-1531"
{"data":[{"id":"saul_7b_instruct","object":"model"},{"id":"lawma_8b","object":"model"},{"id":"lawma_70b","object":"model"},{"id":"DeepSeek-V2-Lite","object":"model"}],"object":"list"}
(workbench-zaina-nc-wb1) (workbench-zaina-nc-wb1) root@6cd743e3-a6f3-41cb-8ea8-3cf1d46de239:~# curl -X POST http://localhost:6000/generate \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "model_name": "DeepSeek-V2-Lite",
  "text": "What is the process for filing taxes?",
  "max_length": 150
}'
{"model":"DeepSeek-V2-Lite","response":"Step 1: Gather necessary documents and information\nTo begin the process of filing taxes, you need to gather all the necessary documents and information required by the tax authority in your country. This typically includes:\n- Personal identification documents (e.g., Social Security number, passport)\n- Financial statements (e.g., W-2 forms, 1099 forms, bank statements)\n- Records of expenses and deductions (e.g., receipts, bills, charitable donation records)\n\nStep 2: Choose the appropriate tax form\nDetermine which tax form you need to file based on your income, filing status, and the tax authority's requirements. Common tax forms include:\n- Form 1 ...","time_taken":13.595934867858887}
(workbench-zaina-nc-wb1) (workbench-zaina-nc-wb1) root@6cd743e3-a6f3-41cb-8ea8-3cf1d46de239:~# curl -X POST http://localhost:6000/generate_stream \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "model_name": "DeepSeek-V2-Lite",
  "text": "Explain the steps for tax filing.",
  "max_length": 150
}'


As a helpful tax and legal advisor, I would guide you through the steps for tax filing as follows:

1. **Gather Necessary Documents and Information:**
   - Collect all necessary documents such as W-2 forms, 1099 forms, and any other income statements.
   - Gather receipts and records for deductions and credits you plan to claim.
   - If you have dependents, gather their Social Security numbers and any relevant information.

2. **Determine Filing Status:**
   - Decide on your filing status (e.g., single, married filing jointly, head of household, etc.).
   - This will affect the tax forms you need to complete and the standard deduction(workbench-zaina-nc-wb1) (workbench-zaina-nc-wb1) root@6cd743e3-a6f3-41cb-8ea8-3cf1d46de239:~# curl -X POST http://localhost:6curl -X POST http://localhost:6000/v1/chat/completions \
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
{"chat_id":"5bb3bcc0-246c-4bc2-9f76-21be1f4f29bc","choices":[{"finish_reason":"stop","index":0,"message":{"content":"As a freelancer, you are eligible to claim various tax deductions that can help reduce your taxable income. Here are some common tax deductions you can claim:\n\n1. Home office expenses: If you use a portion of your home exclusively for work, you can claim a portion of your rent, mortgage interest, utilities, and other expenses as a tax deduction.\n\n2. Self-employment tax: As a freelancer, you are responsible for paying both the employee and employer portions of Social Security and Medicare taxes. You can deduct half of the self-employment tax you pay from your gross income when calculating your adjusted gross income (AGI).\n\n3. Health insurance premiums: You can deduct the premiums you pay for health, dental, and long-term care insurance as an above-the-line deduction on Schedule 1 (Form 1040) of your tax return.\n\n4. Retirement plan contributions: Contributions to a SEP-IRA, ...","role":"assistant"}}],"created":1740626407,"id":"chatcmpl-1740626407","model":"DeepSeek-V2-Lite","object":"chat.completion","time_taken":18.0883207321167,"usage":{"completion_tokens":200,"prompt_tokens":33,"total_tokens":233},"username":"z-ali"}
(workbench-zaina-nc-wb1) (workbench-zaina-nc-wb1) root@6cd743e3-a6f3-41cb-8ea8-3cf1d46de239:~# curl -X POST http://localhost:6000/v1/chat/completions \
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
{"chat_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","choices":[{"finish_reason":"stop","index":0,"message":{"content":"As a tax and legal advisor, I would advise you to follow these steps to maximize your deductions:\n\n1. Understand the tax laws: Familiarize yourself with the tax laws and regulations in your country, as they will dictate what deductions you are eligible for.\n\n2. Keep track of your expenses: Maintain a detailed record of all your income and expenses throughout the year. This will help you identify eligible deductions and ensure you don't miss any opportunities to reduce your taxable income.\n\n3. Itemize your deductions: If you have significant expenses, consider itemizing your deductions instead of taking the standard deduction. This may result in a larger tax refund or a smaller tax bill.\n\n4. Maximize retirement contributions: Contribute to retirement accounts, such as a 401(k) or an IRA, to reduce your taxable income and take advantage of tax-deferred growth.\n\n5. Educate yourself on tax credits: Tax credits are ...","role":"assistant"}}],"created":1740626429,"id":"chatcmpl-1740626429","model":"DeepSeek-V2-Lite","object":"chat.completion","time_taken":18.08206868171692,"usage":{"completion_tokens":200,"prompt_tokens":29,"total_tokens":229},"username":"z-ali"}
(workbench-zaina-nc-wb1) (workbench-zaina-nc-wb1) root@6cd743e3-a6f3-41cb-8ea8-3cf1d46de239:~# curl -X POST http://localhost:6000/v1/completions \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "model": "DeepSeek-V2-Lite",
  "prompt": "Provide guidance on small business tax planning.",
  "max_tokens": 200,
  "tax_optimize": false
}'
{"choices":[{"finish_reason":"stop","index":0,"text":"As a small business owner, it is crucial to plan for your taxes effectively to minimize your tax liability and ensure compliance with tax laws. Here are some key strategies for small business tax planning:\n\n1. **Understand Your Tax Obligations:**\n   - Familiarize yourself with the tax requirements specific to your business structure (e.g., sole proprietorship, partnership, LLC, corporation).\n   - Stay updated on changes in tax laws that may affect your business.\n\n2. **Choose the Right Business Structure:**\n   - Consider the tax implications of each structure. For example, an S corporation may offer better tax savings for pass-through entities with multiple owners.\n   - Think about liability protection, ease of setup, and future growth potential when choosing your business structure.\n\n3. **Maximize Deductions:**\n   - Keep meticulous records of all business expenses to claim deductions.\n   - Take advantage of deductions for business use of home ..."}],"created":1740626452,"id":"cmpl-1740626452","model":"DeepSeek-V2-Lite","object":"text_completion","time_taken":17.884191751480103,"usage":{"completion_tokens":200,"prompt_tokens":30,"total_tokens":230}}
(workbench-zaina-nc-wb1) (workbench-zaina-nc-wb1) root@6cd743e3-a6f3-41cb-8ea8-3cf1d46de239:~# curl -X POST http://localhost:6000/search \
-H "Authorization: Bearer team-tax-1531" \
-H "Content-Type: application/json" \
-d '{
  "query": "freelancer tax deductions",
  "top_k": 3
}'
{"query":"freelancer tax deductions","results":[{"id":"i1040sf.md_118","score":0.605542183,"text":"law.\nDon't deduct the following taxes on this line.\n\u2022 Federal income taxes, including your self-employment\ntax. However, you can deduct one-half of self-employment tax\non Schedule 1 (Form 1040), line 15.\n\u2022 Estate and gift taxes.\n\u2022 Taxes assessed for improvements, such as paving and\nsewers.\n\u2022 Taxes on your home or personal-use property. You may\nbe able to deduct on line 32 expenses related to your home or\nprinciple residence, such as property taxes, if you use your"},{"id":"i1120c.md_176","score":0.583637357,"text":"\u2022 Taxable income figured without the deduction.\nThe deduction shall not exceed 50% of the Form W-2\nwages allocable to domestic production gross receipts\n(DPGR) of the specified cooperative for the tax year. See\nRev. Proc. 2021-11, 2021-6 I.R.B. 833, available at\nIRS.gov/irb/2021-6\\_IRB#REV-PROC-2021-11.\nReporting the deduction. Specified cooperatives may\nuse Form 8903, Domestic Production Activities\nDeduction, to compute the section 199A(g) deduction."},{"id":"i4562.md_56","score":0.575733125,"text":"income from any trade or business you actively conducted,\ncomputed without regard to any section 179 expense\ndeduction, the deduction for one-half of self-employment\ntaxes under section 164(f), or any net operating loss\ndeduction. Also, include all wages, salaries, tips, and other\ncompensation you earned as an employee (from Form 1040,\nline 1). Do not reduce this amount by unreimbursed\nemployee business expenses. If you are married filing a joint"}]}
(workbench-zaina-nc-wb1) (workbench-zaina-nc-wb1) root@6cd743e3-a6f3-41cb-8ea8-3cf1d46de239:~# curl -X GET "http://localhost:6000/chat_history?chat_id=a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
-H "Authorization: Bearer team-tax-1531"
{"chat_id":"a1b2c3d4-e5f6-7890-abcd-ef1234567890","messages":[{"content":"How do I maximize my deductions?","role":"user","timestamp":"2025-02-27 03:20:10","username":"z-ali"},{"content":"As a tax and legal advisor, I would advise you to follow these steps to maximize your deductions:\n\n1. Understand the tax laws: Familiarize yourself with the tax laws and regulations in your country, as they will dictate what deductions you are eligible for.\n\n2. Keep track of your expenses: Maintain a detailed record of all your income and expenses throughout the year. This will help you identify eligible deductions and ensure you don't miss any opportunities to reduce your taxable income.\n\n3. Itemize your deductions: If you have significant expenses, consider itemizing your deductions instead of taking the standard deduction. This may result in a larger tax refund or a smaller tax bill.\n\n4. Maximize retirement contributions: Contribute to retirement accounts, such as a 401(k) or an IRA, to reduce your taxable income and take advantage of tax-deferred growth.\n\n5. Educate yourself on tax credits: Tax credits are ...","role":"assistant","timestamp":"2025-02-27 03:20:29","username":"assistant"}]}
(workbench-zaina-nc-wb1) (workbench-zaina-nc-wb1) root@6cd743e3-a6f3-41cb-8ea8-3cf1d46de239:~# curl -X GET "http://localhost:6000/chat_sessions" \
-H "Authorization: Bearer team-tax-1531"
{"chat_sessions":["92ce3f2a-a2b9-4418-9ae4-9144d9dc8510","c05368d7-fff2-4319-bc65-a2cbcc47ed5b","123e4567-e89b-12d3-a456-426614174000","2eb1f9c9-fdda-4777-ad4e-430d03939633","5bb3bcc0-246c-4bc2-9f76-21be1f4f29bc","a1b2c3d4-e5f6-7890-abcd-ef1234567890"]}
```