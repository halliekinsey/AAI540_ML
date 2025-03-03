# Final Admin Endpoints Testing Report

This section documents the testing of admin backend API endpoints designed for managing data backups, synchronization, and file versioning. The tested endpoints cover zipping, unzipping, pushing, and pulling data, focusing on file-by-file source control for efficiency. While initially considering ZIP backups, individual file pushes have been found more effective.

For faster execution, we conducted the tests using command-line `cURL` rather than Python.

## Tested API Endpoints and Commands

### 1. Zipping Data
Creates a backup ZIP file of all generated data, excluding model data by default. This provides a snapshot of the current system state.

Command:
```bash
curl -H "Authorization: Bearer team-tax-1531" "http://localhost:9000/admin/zip_data?include_model_data=false" -o backup.zip
```

Expected Behavior:  
Downloads a ZIP file containing the backup.

Test Result:
```
100 27.4M  100 27.4M    0     0  7721k      0  0:00:03  0:00:03 --:--:-- 7721k
```
Success: The ZIP file was created and downloaded successfully.

### 2. Unzipping Data
Extracts the contents of the provided ZIP file, updating only new or modified files.

Command:
```bash
curl -X POST -H "Authorization: Bearer team-tax-1531" -F "file=@backup.zip" "http://localhost:9000/admin/unzip_data"
```

Expected Behavior:  
Extracts the ZIP file and updates necessary files.

Test Result:
```json
{"files_skipped":0,"files_updated":24,"message":"Unzip completed"}
```
Success: 24 files were updated, and no redundant files were extracted.

### 3. Pushing Data to S3
Uploads local data to the S3 bucket `tax-legal-data`. Metadata checks prevent re-uploading unchanged files.

#### Initial Push
Command:
```bash
curl -H "Authorization: Bearer team-tax-1531" "http://localhost:9000/admin/push_s3?include_model_data=false"
```

Expected Behavior:  
Uploads files to S3.

Test Result:
```json
{"bucket":"tax-legal-data","files_skipped":0,"files_uploaded":24,"message":"Push to S3 completed"}
```
Success: 24 files were uploaded.

#### Subsequent Push (No Changes)
Command:
```bash
curl -H "Authorization: Bearer team-tax-1531" "http://localhost:9000/admin/push_s3?include_model_data=false"
```

Expected Behavior:  
Skips files that have not changed.

Test Result:
```json
{"bucket":"tax-legal-data","files_skipped":24,"files_uploaded":0,"message":"Push to S3 completed"}
```
Success: No unnecessary uploads, confirming efficient file versioning.

### 4. Pulling Data from S3
Retrieves data from the S3 bucket to a local directory (`s3_data`), preserving folder structure.

Command:
```bash
curl -H "Authorization: Bearer team-tax-1531" "http://localhost:9000/admin/pull_s3?local_root=s3_data"
```

Expected Behavior:  
Downloads files while skipping already-updated files.

Test Result:
```json
{"bucket":"tax-legal-data","files_downloaded":24,"files_skipped":0,"local_root":"s3_data","message":"Pull from S3 completed"}
```
Success: All 24 files were downloaded.

### 5. Verifying Local Directory Structure
After pulling the data, we checked the directory structure using:

```bash
tree s3_data/
```

Expected Behavior:  
Lists the file structure of downloaded data.

Test Result:
```
s3_data/
├── chat_sessions.db
├── pdf_uploads
│   ├── alice
│   │   ├── 6b45ff29-40b9-45de-bc21-3d9652a69259
│   │   │   ├── i706.pdf
│   │   │   └── i706.pdf.meta.json
│   │   ├── aabeeee6-8fd6-4602-b792-da65b416c475
│   │   │   ├── i706.pdf
│   │   │   └── i706.pdf.meta.json
│   │   └── abcd-1234
│   │       ├── i706.pdf
│   │       └── i706.pdf.meta.json
│   ├── alzain
│   │   ├── 843687b1-2982-44e7-ac10-644f239e0a5f
│   │   │   ├── i706.pdf
│   │   │   └── i706.pdf.meta.json
│   │   ├── e34b19ab-2329-4c70-adf2-14dfef36eaae
│   │   │   ├── i706.pdf
│   │   │   └── i706.pdf.meta.json
│   │   └── f1503d4b-5945-4379-91cc-af8b3266b686
│   │       ├── i706.pdf
│   │       └── i706.pdf.meta.json
│   └── zain
│       ├── 39a724ba-923d-470f-89c1-e093cb7cfe4b
│       │   ├── i706.pdf
│       │   └── i706.pdf.meta.json
│       ├── f3cbf185-9843-4a84-afef-241f6f072a4e
│       │   ├── i706.pdf
│       │   └── i706.pdf.meta.json
│       └── f766bc49-6bc3-4e98-aafd-08a81ef3d23b
│           ├── i706.pdf
│           └── i706.pdf.meta.json
└── refining_data
    ├── 20250301_085430_what-is-i701_UP.json
    ├── 20250301_085959_what-is-the-best-form-to-file-tax-as-a-single-in-u_UP.json
    ├── 20250301_090104_what-is-the-best-form-to-file-tax-as-a-single-in-u_UP.json
    ├── 20250301_090222_crazy-tax_DOWN.json
    └── 20250301_090239_crazy-tax_DOWN.json
```
Success: The folder structure was maintained exactly as expected.
