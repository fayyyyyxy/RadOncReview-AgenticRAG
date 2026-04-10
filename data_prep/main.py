import os
import subprocess
from extract_doc_json import extract_json
from clean_json import extract_hierarchical_sections
from s3_store_load import create_s3_bucket, upload_json_to_s3, upload_file_to_s3, load_from_s3
from flattening_chunking import flatten_chunk_json
from DynamoDB_store_load import upload_to_dynamodb
from helper import save_json
from embedding import add_embeddings, build_faiss_index


# ----------------- Settings -----------------
PROFILE_NAME="YOUR_AWS_PROFILE"
S3_BUCKET_NAME = "YOUR_S3_BUCKET"
REGION="YOUR_AWS_REGION"
DYNAMODB_TABLE_NAME = "YOUR_DYNAMODB_TABLE"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
FAISS_INDEX_FILE = "radonc_faiss_index.bin"

# Chunking parameters
max_tokens = 300
overlap_tokens = 45

# Setup save directory for intermediate files
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "saved_data")
os.makedirs(save_dir, exist_ok=True)


DOCUMENTS_TO_PROCESS = [
    {
        "disease_category": "Breast",
        "document_id": "1sWQwqcSH23B30CKCVOaQ2kb4D4qES6YfPqmgJYR5rnY"
    },
    
    {
        "disease_category": "Thorax",
        "document_id": "1oKD3L5ieCk03FWU6fCnj8aiHKRPJD-q6IpjXpQCuexw"
    },
    {
        "disease_category": "Hematologic",
        "document_id": "1gKy2Hpx7FxInjOpKIBkTFJWpqhJ3I-gSXz9eRwq-NSY"
    },
    {
        "disease_category": "BrainMets|Oligo|Immuno",
        "document_id": "1CfbqB4YnaPB8U3r2LykLv2v3bRLJyYQV0tvX4Js2Mog"
    },
    {
        "disease_category": "CNS|Pediatrics",
        "document_id": "17O0LOemBhckXGuuPBCh6u8vqBfc6lg88r46B8YctMXU"
    },
    {
        "disease_category": "Gastrointestinal",
        "document_id": "13NEZCS6s13MVLixabbO2vjY73zHxJ37qE16gBbApSdY"
    },
    {
        "disease_category": "Genitourinary",
        "document_id": "1j15zXLBPWwqty60Slm2jnHEiqaoT2iw5Gapp4iMWJsw"
    },
    {
        "disease_category": "Gynecology",
        "document_id": "1X-MmBeoIl3IECEGIUVV4sFz_AR_s5AEQb8Xsx4szmJg"   
    },
    {
        "disease_category": "Head&Neck|Skin",
        "document_id": "1STZuiggtbkDIuuNMpDVSsqT2KMyp1017y8qV5Gz_GGc"   
    },
    {
        "disease_category": "Hematologic",
        "document_id": "1gKy2Hpx7FxInjOpKIBkTFJWpqhJ3I-gSXz9eRwq-NSY"
    },
    {
        "disease_category": "Sarcoma",
        "document_id": "1eal6YYRhPGwh4_R5MPQdioLZLapJBSIAXZjuO3IeU6M"
    }
]
# --------------------------------------------


def process_document(doc_config, current_faiss_id):
    """
    Process a single document: extract, clean, chunk, embed, and store.
    Return the next available FAISS_ID for the subsequent document.
    """
    disease_category = doc_config["disease_category"]
    document_id = doc_config["document_id"]
    document_url = f"https://docs.google.com/document/d/{document_id}" 
    print(f"\n--- Starting Ingestion for Document: {disease_category.upper()} ---")

    # 1. Extract raw JSON from Google Doc
    raw_json = extract_json(document_id)

    # 2. Format the JSON into hierarchical sections
    clean_json = extract_hierarchical_sections(raw_json, disease_category)

    # 3. Save raw and cleaned JSON to S3
    # Refresh AWS SSO credentials
    subprocess.run(["aws", "sso", "login", "--profile", PROFILE_NAME], check=True)
    
    s3_raw_json_key = f"raw/{disease_category}_raw.json"
    s3_clean_json_key = f"clean/{disease_category}_clean.json"
    upload_json_to_s3(raw_json, S3_BUCKET_NAME, PROFILE_NAME, s3_raw_json_key)
    upload_json_to_s3(clean_json, S3_BUCKET_NAME, PROFILE_NAME, s3_clean_json_key)

    save_json(disease_category, "clean", save_dir, load_from_s3(S3_BUCKET_NAME, PROFILE_NAME, s3_clean_json_key))
    save_json(disease_category, "raw", save_dir, load_from_s3(S3_BUCKET_NAME, PROFILE_NAME, s3_raw_json_key))

    # 4. Flatten and chunk the JSON data
    token_counter_model = "text-embedding-3-large" # For estimating tokens in chunking only 
    flattened_chunks = flatten_chunk_json(clean_json, token_counter_model, max_tokens, overlap_tokens, current_faiss_id, document_url=document_url)
    save_json(disease_category, "chunks", save_dir, flattened_chunks)

    # The FAISS_ID for the next document starts after the last ID used here
    next_faiss_id = flattened_chunks[-1]["FAISS_ID"] + 1
    
    # 5. Upload flattened chunks to DynamoDB
    upload_to_dynamodb(PROFILE_NAME, DYNAMODB_TABLE_NAME, flattened_chunks, region=REGION)

    # 6. Add embeddings
    chunk_embeddings = add_embeddings(flattened_chunks, EMBEDDING_MODEL, PROFILE_NAME, REGION) 
    save_json(disease_category, "chunk_embeddings", save_dir, chunk_embeddings)

    return next_faiss_id


def main():
    # Refresh AWS SSO credentials, equivalent to running:
    # aws sso login --profile PROFILE_NAME
    subprocess.run(["aws", "sso", "login", "--profile", PROFILE_NAME], check=True)

    # Create S3 bucket (if it doesn't exist)
    create_s3_bucket(S3_BUCKET_NAME, PROFILE_NAME, REGION)

    # Process all documents sequentially
    current_faiss_id = 0
    all_chunk_embeddings = []
    
    for doc in DOCUMENTS_TO_PROCESS:
        try:
            # Process document and get its chunks and the ID for the next document
            #doc_chunks, current_faiss_id = process_document(doc, current_faiss_id)
            current_faiss_id = process_document(doc, current_faiss_id) 
            #all_chunk_embeddings.extend(doc_chunks)
        except Exception as e:
            print(f"Failed to process document {doc['disease_category']}: {e}")
            continue 

    # 7. Build FAISS Index for ALL documents and upload it to S3
    # if all_chunk_embeddings:
    #     local_faiss_path = os.path.join(save_dir, FAISS_INDEX_FILE)
    #     build_faiss_index(all_chunk_embeddings, local_faiss_path)
        
    #     index_file_to_upload = [(local_faiss_path, FAISS_INDEX_FILE)]
    #     upload_file_to_s3(index_file_to_upload, S3_BUCKET_NAME, PROFILE_NAME)
    #     print(f"FAISS Index successfully uploaded to s3://{S3_BUCKET_NAME}/{FAISS_INDEX_FILE}")
        
    # else:
    #     print("No documents processed successfully. FAISS index not built.")


if __name__ == "__main__":
    main()