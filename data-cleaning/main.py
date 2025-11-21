import os
import subprocess
from extract_doc_json import extract_json
from clean_json import extract_hierarchical_sections
from s3_store_load import create_s3_bucket, upload_to_s3, load_from_s3
from flattening_chunking import flatten_chunk_json
from DynamoDB_store_load import upload_to_dynamodb
from helper import save_json, load_json


# ----------- Settings -----------
disease_category = "breast"
# Set doc ID as "YOUR_DOC_ID", as found at `https://docs.google.com/document/d/YOUR_DOC_ID/edit`
DOCUMENT_ID = "1sWQwqcSH23B30CKCVOaQ2kb4D4qES6YfPqmgJYR5rnY"
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "saved_data")


# 1. Extract raw JSON from Google Doc
raw_json = extract_json(DOCUMENT_ID)
save_json(disease_category, "raw", save_dir, raw_json)

# 2. Format the JSON into hierarchical sections
clean_json = extract_hierarchical_sections(raw_json, disease_category)
save_json(disease_category, "clean", save_dir, clean_json)

# 3. Save raw and cleaned JSON to S3
profile_name="YOUR_AWS_PROFILE"
bucket_name = "YOUR_S3_BUCKET"
region="YOUR_AWS_REGION"

# Refresh AWS SSO credentials, equivalent to running:
# aws sso login --profile PROFILE_NAME
subprocess.run(["aws", "sso", "login", "--profile", profile_name], check=True)

# Create S3 bucket and upload files
create_s3_bucket(bucket_name, profile_name, region)
raw_json_path = os.path.join(save_dir, f"{disease_category}_raw.json")
clean_json_path = os.path.join(save_dir, f"{disease_category}_clean.json")
upload_to_s3(disease_category, bucket_name, profile_name, raw_json_path, clean_json_path)

# 4. Connect to S3 & load cleaned JSON file
loaded_clean_json = load_from_s3(disease_category, bucket_name, profile_name, file_type="clean")

# 5. Flatten and chunk the JSON data
flattened_chunks = flatten_chunk_json(loaded_clean_json, model_name="text-embedding-3-large", max_tokens = 500, overlap_tokens = 100)
save_json(disease_category, "chunks", save_dir, flattened_chunks)

# 6. Upload flattened chunks to DynamoDB
table_name = "YOUR_DYNAMODB_TABLEs"
flattened_json = load_json(disease_category, "chunks", save_dir)
upload_to_dynamodb(profile_name, table_name, flattened_json)