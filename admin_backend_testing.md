### Admin Backend Endpoints Testing

This section outlines our workflow for verifying the admin backend endpoints. We implemented endpoints for zipping, unzipping, pushing, and pulling data, with an emphasis on file-by-file source control. Initially, we considered pushing a ZIP file as a backup; however, pushing individual files has proven more effective.

Also, we utilized the command line rather than Python for this testing, as it proved to be quicker.

#### Zipping Data

*This endpoint creates a backup ZIP file of all generated data (excluding model data by default). This ZIP can serve as a complete snapshot if needed.*

**Command:**
```bash
curl -H "Authorization: Bearer team-tax-1531" "http://localhost:9000/admin/zip_data?include_model_data=false" -o backup.zip
```

**Sample Output:**
```
100 27.4M  100 27.4M    0     0  7721k      0  0:00:03  0:00:03 --:--:-- 7721k
```



#### Unzipping Data

*This endpoint extracts the contents of a provided ZIP file into the local filesystem, updating only new or modified files.*

**Command:**
```bash
curl -X POST -H "Authorization: Bearer team-tax-1531" -F "file=@backup.zip" "http://localhost:9000/admin/unzip_data"
```

**Sample Output:**
```json
{"files_skipped":0,"files_updated":24,"message":"Unzip completed"}
```



#### Pushing Data to S3

*This endpoint uploads local data (excluding model data by default) to the S3 bucket `tax-legal-data`. By pushing files individually with metadata checks, unchanged files are not re-uploaded.*

**Initial Push Command:**
```bash
curl -H "Authorization: Bearer team-tax-1531" "http://localhost:9000/admin/push_s3?include_model_data=false"
```

**Initial Push Output:**
```json
{"bucket":"tax-legal-data","files_skipped":0,"files_uploaded":24,"message":"Push to S3 completed"}
```

**Subsequent Push Command (after data is already in S3):**
```bash
curl -H "Authorization: Bearer team-tax-1531" "http://localhost:9000/admin/push_s3?include_model_data=false"
```

**Subsequent Push Output:**
```json
{"bucket":"tax-legal-data","files_skipped":24,"files_uploaded":0,"message":"Push to S3 completed"}
```

*For additional context, refer to the S3 bucket screenshot located in the `supporting_media` directory:*

![S3 Bucket Screenshot](supporting_media/Screenshot%202025-03-01%20at%201.37.57%E2%80%AFAM.png)



#### Pulling Data from S3

*This endpoint retrieves data from the S3 bucket back to a local directory (default: `s3_data`), preserving the original folder structure.*

**Command:**
```bash
curl -H "Authorization: Bearer team-tax-1531" "http://localhost:9000/admin/pull_s3?local_root=s3_data"
```

**Sample Output:**
```json
{"bucket":"tax-legal-data","files_downloaded":24,"files_skipped":0,"local_root":"s3_data","message":"Pull from S3 completed"}
```



#### Viewing the Local Directory Structure

After pulling the data, you can review the directory structure with the `tree` command:

```bash
tree s3_data/
```

**Example Output:**
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
