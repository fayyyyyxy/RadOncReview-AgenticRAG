#!/usr/bin/env python3
"""
Debug FAISS search and DynamoDB retrieval chain.
"""
import boto3
import json
import faiss
import numpy as np
from decimal import Decimal

PROFILE = "YOUR_AWS_PROFILE"
REGION = "YOUR_AWS_REGION"
BUCKET = "YOUR_S3_BUCKET"
TABLE_NAME = "YOUR_DYNAMODB_TABLE"
FAISS_KEY = "radonc_faiss_index.bin"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

def main():
    print("=" * 60)
    print("  FAISS + DynamoDB Integration Test")
    print("=" * 60)
    
    session = boto3.Session(profile_name=PROFILE)
    s3 = session.client("s3", region_name=REGION)
    bedrock = session.client("bedrock-runtime", region_name=REGION)
    dynamodb = session.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)
    
    # Step 1: Load FAISS index
    print("\n[1] Loading FAISS index...")
    local_path = "/tmp/debug_faiss.bin"
    s3.download_file(BUCKET, FAISS_KEY, local_path)
    index = faiss.read_index(local_path)
    print(f"    Loaded {index.ntotal} vectors, dimension {index.d}")
    
    # Step 2: Generate embedding for test query
    print("\n[2] Generating embedding for query...")
    query = "What is the standard dose for breast cancer?"
    print(f"    Query: {query}")
    
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": query})
    )
    result = json.loads(response["body"].read())
    embedding = np.array(result["embedding"], dtype="float32")
    print(f"    Embedding dimension: {len(embedding)}")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embedding.reshape(1, -1))
    
    # Step 3: Search FAISS
    print("\n[3] Searching FAISS index...")
    top_k = 10
    distances, indices = index.search(embedding.reshape(1, -1), top_k)
    
    print(f"    Top {top_k} results:")
    print(f"    Indices: {indices[0]}")
    print(f"    Indices type: {type(indices[0])}, dtype: {indices.dtype}")
    print(f"    Distances: {distances[0]}")
    
    # Step 4: Fetch from DynamoDB
    print("\n[4] Fetching from DynamoDB...")
    
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        faiss_id = int(idx)  # Convert numpy.int64 to Python int
        
        print(f"\n    Result {i+1}:")
        print(f"      FAISS_ID: {faiss_id} (original type: {type(idx).__name__})")
        print(f"      Distance: {dist:.4f}")
        
        # Try to fetch from DynamoDB
        try:
            response = table.get_item(Key={"FAISS_ID": faiss_id})
            if "Item" in response:
                item = response["Item"]
                path = item.get("path", "N/A")
                text = item.get("text", "N/A")[:100]
                print(f"      Path: {path}")
                print(f"      Text: {text}...")
            else:
                print(f"      ERROR: Item not found in DynamoDB!")
        except Exception as e:
            print(f"      ERROR: {e}")
    
    print("\n" + "=" * 60)
    print("  Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
