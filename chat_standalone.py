#!/usr/bin/env python3
"""
Standalone chat script with all retrieval logic inline.
No external module dependencies.
"""
import boto3
import json
import faiss
import numpy as np
import os

# Configuration
PROFILE = "YOUR_AWS_PROFILE"
REGION = "YOUR_AWS_REGION"
BUCKET = "YOUR_S3_BUCKET"
TABLE_NAME = "YOUR_DYNAMODB_TABLE"
FAISS_KEY = "radonc_faiss_index.bin"
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"
LLM_MODEL = "meta.llama3-70b-instruct-v1:0"

# Global variables
session = None
s3 = None
bedrock = None
dynamodb = None
table = None
faiss_index = None


def init_aws():
    """Initialize AWS clients."""
    global session, s3, bedrock, dynamodb, table, faiss_index
    
    print("Connecting to AWS...")
    session = boto3.Session(profile_name=PROFILE)
    s3 = session.client("s3", region_name=REGION)
    bedrock = session.client("bedrock-runtime", region_name=REGION)
    dynamodb = session.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)
    
    # Load FAISS index
    print("Loading FAISS index...")
    local_path = "/tmp/chat_faiss_index.bin"
    s3.download_file(BUCKET, FAISS_KEY, local_path)
    faiss_index = faiss.read_index(local_path)
    print(f"Loaded {faiss_index.ntotal} vectors")
    print("Ready!\n")


def get_embedding(text):
    """Generate embedding for text."""
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": text})
    )
    result = json.loads(response["body"].read())
    embedding = np.array(result["embedding"], dtype="float32")
    faiss.normalize_L2(embedding.reshape(1, -1))
    return embedding


def search_faiss(query, top_k=5):
    """Search FAISS index and return chunk IDs with scores."""
    embedding = get_embedding(query)
    distances, indices = faiss_index.search(embedding.reshape(1, -1), top_k)
    
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx >= 0 and dist > 0.3:  # Threshold
            results.append((int(idx), float(dist)))
    return results


def fetch_chunks(faiss_ids):
    """Fetch chunks from DynamoDB."""
    chunks = []
    for fid in faiss_ids:
        try:
            response = table.get_item(Key={"FAISS_ID": fid})
            if "Item" in response:
                item = response["Item"]
                # Convert Decimals to native types
                chunk = {
                    "faiss_id": int(item["FAISS_ID"]),
                    "path": item.get("path", ""),
                    "text": item.get("text", ""),
                }
                chunks.append(chunk)
        except Exception as e:
            print(f"  Warning: Could not fetch ID {fid}: {e}")
    return chunks


def generate_answer(query, chunks):
    """Generate answer using Llama 3."""
    if not chunks:
        return "I could not find relevant information to answer your question."
    
    # Build context
    context = "\n\n".join([
        f"[{i+1}] Source: {c['path']}\n{c['text']}"
        for i, c in enumerate(chunks[:5])
    ])
    
    prompt = f"""<s>[INST] You are a radiation oncology expert. Answer the question based on the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Answer based on the context provided
2. Cite sources using [1], [2], etc.
3. If the context doesn't fully answer the question, say so
4. Be concise but accurate

Answer: [/INST]"""
    
    response = bedrock.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps({
            "prompt": prompt,
            "max_gen_len": 1024,
            "temperature": 0.1
        })
    )
    
    result = json.loads(response["body"].read())
    return result.get("generation", "Error generating response")


def process_query(query):
    """Process a single query."""
    print("  Searching...")
    
    # Search
    results = search_faiss(query, top_k=10)
    
    if not results:
        return "No relevant documents found.", []
    
    print(f"  Found {len(results)} relevant chunks")
    
    # Fetch chunks
    faiss_ids = [r[0] for r in results]
    chunks = fetch_chunks(faiss_ids)
    
    if not chunks:
        return "Could not retrieve document content.", []
    
    print(f"  Retrieved {len(chunks)} chunks")
    print("  Generating answer...")
    
    # Generate answer
    answer = generate_answer(query, chunks)
    
    return answer, chunks


def main():
    print("=" * 60)
    print("  RadOnc RAG Chat (Standalone Version)")
    print("=" * 60)
    
    try:
        init_aws()
    except Exception as e:
        print(f"Error connecting to AWS: {e}")
        return
    
    print("Type your questions. Type 'quit' to exit.")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not query:
            continue
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        try:
            answer, chunks = process_query(query)
            print(f"\nAgent: {answer}")
            
            if chunks:
                print(f"\n[Sources: {len(chunks)} documents]")
                for i, c in enumerate(chunks[:3]):
                    print(f"  [{i+1}] {c['path']}")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
