import boto3
import json
import faiss
import numpy as np

def add_embeddings(chunks, model_id, aws_profile, region):
    """
    Add embedding vectors to flattened chunk entries using Amazon embedding model.
    """

    session = boto3.Session(profile_name=aws_profile)
    bedrock = session.client("bedrock-runtime", region_name=region)

    for item in chunks:
        text = item["text"]

        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps({"inputText": text})
        )
        response_body = json.loads(response["body"].read())

        embedding = response_body["embedding"]  # Titan v2 returns this

        item["embedding"] = embedding   # attach vector to the JSON

    return chunks


def build_faiss_index(chunks, faiss_path):
    """
    Creates the FAISS Index from the embeddings, serializes it to a local file.
    """
    
    # Extract embeddings and ensure they are float32 NumPy array (FAISS requirement)
    embeddings = [np.array(item["embedding"], dtype='float32') for item in chunks]
    
    if not embeddings:
        print("No embeddings found to build FAISS index.")
        return

    vectors = np.vstack(embeddings)
    
    # 1. Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)
    
    # 2. Build the FAISS index
    dim = vectors.shape[1]
    # IndexFlatIP (Inner Product) + Normalized Vectors = Exact Cosine Similarity
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # 3. Serialize to local file
    faiss.write_index(index, faiss_path)
    print(f"FAISS Index serialized to local path: {faiss_path}")