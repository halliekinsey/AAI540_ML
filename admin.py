import os
import io
import zipfile
import hashlib
import datetime
import boto3

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from functools import wraps

app = Flask(__name__)

DEMO_TOKEN = "team-tax-1531"

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
        if not token:
            token = request.args.get("token")
        if token != DEMO_TOKEN:
            return jsonify({"error": "Unauthorized: Invalid or missing token"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Directories / Files to include and optional model directory.
DATA_DIRECTORIES = {
    "pdf_uploads": "pdf_uploads",
    "refining_data": "refining_data"
}
DATA_FILES = [
    "chat_sessions.db"
]
MODEL_DATA_DIR = "model_directory"


def compute_md5_bytes(data_bytes):
    return hashlib.md5(data_bytes).hexdigest()

def compute_md5_file(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def zip_all_data(include_model_data=False):
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, dir_path in DATA_DIRECTORIES.items():
            if os.path.isdir(dir_path):
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        full_path = os.path.join(root, file)
                        arcname = os.path.relpath(full_path, ".")
                        zf.write(full_path, arcname)
        for file in DATA_FILES:
            if os.path.exists(file):
                zf.write(file, file)
        if include_model_data and os.path.isdir(MODEL_DATA_DIR):
            for root, _, files in os.walk(MODEL_DATA_DIR):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, ".")
                    zf.write(full_path, arcname)
    mem_zip.seek(0)
    return mem_zip

@app.route("/admin/zip_data", methods=["GET"])
@token_required
def zip_data_endpoint():
    include_model = request.args.get("include_model_data", "false").lower() == "true"
    zip_file = zip_all_data(include_model)
    filename = "data_backup.zip"
    return send_file(
        zip_file,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename
    )

@app.route("/admin/unzip_data", methods=["POST"])
@token_required
def unzip_data_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file_obj = request.files["file"]
    try:
        with zipfile.ZipFile(file_obj) as zf:
            files_updated = 0
            files_skipped = 0
            for zip_info in zf.infolist():
                extracted_path = os.path.join(".", zip_info.filename)
                os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                file_bytes = zf.read(zip_info.filename)
                new_md5 = compute_md5_bytes(file_bytes)
                if os.path.exists(extracted_path):
                    local_md5 = compute_md5_file(extracted_path)
                    if local_md5 == new_md5:
                        files_skipped += 1
                        continue
                with open(extracted_path, "wb") as f:
                    f.write(file_bytes)
                files_updated += 1
        return jsonify({
            "message": "Unzip completed",
            "files_updated": files_updated,
            "files_skipped": files_skipped
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def upload_file_to_s3(s3_client, bucket_name, local_path, s3_key):
    local_md5 = compute_md5_file(local_path)
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        remote_md5 = response.get("Metadata", {}).get("md5hash", "")
        if remote_md5 == local_md5:
            return False
    except s3_client.exceptions.ClientError:
        pass
    s3_client.upload_file(
        Filename=local_path,
        Bucket=bucket_name,
        Key=s3_key,
        ExtraArgs={"Metadata": {"md5hash": local_md5, "last_modified": datetime.datetime.utcnow().isoformat()}}
    )
    return True

def upload_directory(s3_client, bucket_name, local_dir, s3_prefix=""):
    files_uploaded = 0
    files_skipped = 0
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, ".")
            s3_key = os.path.join(s3_prefix, rel_path).replace("\\", "/")
            if upload_file_to_s3(s3_client, bucket_name, local_path, s3_key):
                files_uploaded += 1
            else:
                files_skipped += 1
    return files_uploaded, files_skipped

@app.route("/admin/push_s3", methods=["GET"])
@token_required
def push_to_s3_endpoint():
    include_model = request.args.get("include_model_data", "false").lower() == "true"
    bucket_name = "tax-legal-data"
    s3_client = boto3.client("s3")
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except s3_client.exceptions.ClientError:
        try:
            s3_client.create_bucket(Bucket=bucket_name)
        except Exception as e:
            return jsonify({"error": f"Failed to create bucket: {str(e)}"}), 500
    total_uploaded = 0
    total_skipped = 0
    for label, dir_path in DATA_DIRECTORIES.items():
        if os.path.isdir(dir_path):
            uploaded, skipped = upload_directory(s3_client, bucket_name, dir_path, s3_prefix="")
            total_uploaded += uploaded
            total_skipped += skipped
    for file in DATA_FILES:
        if os.path.exists(file):
            if upload_file_to_s3(s3_client, bucket_name, file, file):
                total_uploaded += 1
            else:
                total_skipped += 1
    if include_model and os.path.isdir(MODEL_DATA_DIR):
        uploaded, skipped = upload_directory(s3_client, bucket_name, MODEL_DATA_DIR, s3_prefix="")
        total_uploaded += uploaded
        total_skipped += skipped
    return jsonify({
        "message": "Push to S3 completed",
        "files_uploaded": total_uploaded,
        "files_skipped": total_skipped,
        "bucket": bucket_name
    })


def pull_directory_from_s3(s3_client, bucket_name, local_root):
    os.makedirs(local_root, exist_ok=True)
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name)
    files_downloaded = 0
    files_skipped = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_path = os.path.join(local_root, key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                head = s3_client.head_object(Bucket=bucket_name, Key=key)
                remote_md5 = head.get("Metadata", {}).get("md5hash", "")
            except Exception as e:
                remote_md5 = ""
            if os.path.exists(local_path):
                local_md5 = compute_md5_file(local_path)
                if local_md5 == remote_md5:
                    files_skipped += 1
                    continue
            s3_client.download_file(bucket_name, key, local_path)
            files_downloaded += 1
    return files_downloaded, files_skipped

@app.route("/admin/pull_s3", methods=["GET"])
@token_required
def pull_from_s3_endpoint():
    bucket_name = "tax-legal-data"
    # Optional: allow local root to be specified via query string (default "s3_data")
    local_root = request.args.get("local_root", "s3_data")
    s3_client = boto3.client("s3")
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except s3_client.exceptions.ClientError as e:
        return jsonify({"error": f"Bucket {bucket_name} does not exist: {str(e)}"}), 404
    downloaded, skipped = pull_directory_from_s3(s3_client, bucket_name, local_root)
    return jsonify({
        "message": "Pull from S3 completed",
        "local_root": local_root,
        "files_downloaded": downloaded,
        "files_skipped": skipped,
        "bucket": bucket_name
    })

if __name__ == '__main__':
    port = int(os.environ.get("ADMIN_PORT", 9000))
    app.run(host="0.0.0.0", port=port)
