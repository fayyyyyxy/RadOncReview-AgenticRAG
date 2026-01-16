"""
Quick test script to verify AWS connection and run a real query.
Make sure to run first: aws sso login --profile YOUR_AWS_PROFILE
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import default_config


def test_aws_connection():
    """Test AWS connection"""
    print("=" * 60)
    print("Testing AWS Connection")
    print("=" * 60)
    
    import boto3
    
    profile = default_config.aws.profile_name
    region = default_config.aws.region
    bucket = default_config.aws.s3_bucket
    
    print(f"\nConfiguration:")
    print(f"  Profile: {profile}")
    print(f"  Region: {region}")
    print(f"  S3 Bucket: {bucket}")
    print(f"  DynamoDB Table: {default_config.aws.dynamodb_table}")
    
    try:
        session = boto3.Session(profile_name=profile)
        
        # Test S3
        print(f"\n[S3] Testing connection...")
        s3 = session.client("s3", region_name=region)
        response = s3.list_objects_v2(Bucket=bucket, MaxKeys=10)
        
        print(f"  OK - S3 connection successful!")
        print(f"  Found {response.get('KeyCount', 0)} objects:")
        for obj in response.get('Contents', []):
            print(f"    - {obj['Key']} ({obj['Size']} bytes)")
        
        # Test DynamoDB
        print(f"\n[DynamoDB] Testing connection...")
        dynamodb = session.resource("dynamodb", region_name=region)
        table = dynamodb.Table(default_config.aws.dynamodb_table)
        
        # Get table info
        item_count = table.item_count
        print(f"  OK - DynamoDB connection successful!")
        print(f"  Table: {table.table_name}")
        print(f"  Item count: {item_count}")
        
        # Test Bedrock
        print(f"\n[Bedrock] Testing connection...")
        bedrock = session.client("bedrock-runtime", region_name=region)
        print(f"  OK - Bedrock client created!")
        print(f"  Model: {default_config.llm.model_id}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR - Connection failed: {e}")
        print("\nPlease ensure:")
        print("  1. Run 'aws sso login --profile YOUR_AWS_PROFILE'")
        print("  2. AWS credentials are not expired")
        print("  3. You have the correct IAM permissions")
        return False


def test_faiss_index():
    """Test FAISS index loading"""
    print("\n" + "=" * 60)
    print("Testing FAISS Index")
    print("=" * 60)
    
    import boto3
    import faiss
    
    try:
        session = boto3.Session(profile_name=default_config.aws.profile_name)
        s3 = session.client("s3", region_name=default_config.aws.region)
        
        # Download FAISS index
        local_path = "/tmp/test_faiss_index.bin"
        print(f"\nDownloading FAISS index...")
        s3.download_file(
            default_config.aws.s3_bucket,
            default_config.aws.faiss_index_key,
            local_path
        )
        
        # Load index
        index = faiss.read_index(local_path)
        print(f"  OK - FAISS index loaded!")
        print(f"  Vector count: {index.ntotal}")
        print(f"  Vector dimension: {index.d}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR - FAISS test failed: {e}")
        return False


def test_embedding():
    """Test Embedding generation"""
    print("\n" + "=" * 60)
    print("Testing Embedding Generation")
    print("=" * 60)
    
    import boto3
    import json
    
    try:
        session = boto3.Session(profile_name=default_config.aws.profile_name)
        bedrock = session.client("bedrock-runtime", region_name=default_config.aws.region)
        
        test_text = "What is the dose for breast cancer radiation therapy?"
        
        print(f"\nTest text: {test_text}")
        print(f"Model: {default_config.embedding.model_id}")
        
        response = bedrock.invoke_model(
            modelId=default_config.embedding.model_id,
            body=json.dumps({"inputText": test_text})
        )
        
        result = json.loads(response["body"].read())
        embedding = result["embedding"]
        
        print(f"  OK - Embedding generated!")
        print(f"  Vector dimension: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR - Embedding test failed: {e}")
        return False


def test_llm():
    """Test LLM generation"""
    print("\n" + "=" * 60)
    print("Testing LLM Generation")
    print("=" * 60)
    
    import boto3
    import json
    
    try:
        session = boto3.Session(profile_name=default_config.aws.profile_name)
        bedrock = session.client("bedrock-runtime", region_name=default_config.aws.region)
        
        test_prompt = "What is radiation therapy? Answer in one sentence."
        model_id = default_config.llm.model_id
        
        print(f"\nTest prompt: {test_prompt}")
        print(f"Model: {model_id}")
        
        # Llama 3 format
        body = {
            "prompt": f"<s>[INST] {test_prompt} [/INST]",
            "max_gen_len": 100,
            "temperature": 0.1
        }
        
        response = bedrock.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )
        
        result = json.loads(response["body"].read())
        generated_text = result.get("generation", "")
        
        print(f"  OK - LLM response generated!")
        print(f"  Response: {generated_text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"\nERROR - LLM test failed: {e}")
        return False


def test_real_query():
    """Test real query with the full pipeline"""
    print("\n" + "=" * 60)
    print("Testing Real Query")
    print("=" * 60)
    
    try:
        from agent import create_agent
        
        print("\nCreating Agent (real mode)...")
        agent = create_agent(use_mock=False)
        
        test_queries = [
            "What is the standard dose for breast cancer?",
        ]
        
        for query in test_queries:
            print(f"\n{'─' * 50}")
            print(f"Query: {query}")
            print('─' * 50)
            
            response = agent.query(query)
            
            print(f"\nAnswer:\n{response.answer[:500]}...")
            print(f"\nConfidence: {response.confidence_score:.2%}")
            print(f"Citations: {len(response.citations)}")
            print(f"Latency: {response.latency_ms:.0f}ms")
            
            if response.citations:
                print(f"\nSources:")
                for c in response.citations[:3]:
                    print(f"  [{c.citation_id}] {c.source_path}")
        
        return True
        
    except Exception as e:
        print(f"\nERROR - Query test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "RadOnc Agentic RAG - AWS Connection Test".center(60))
    print("=" * 60)
    
    results = {}
    
    # Test 1: AWS connection
    results["AWS Connection"] = test_aws_connection()
    
    if not results["AWS Connection"]:
        print("\nAWS connection failed, skipping remaining tests")
        return
    
    # Test 2: FAISS index
    results["FAISS Index"] = test_faiss_index()
    
    # Test 3: Embedding
    results["Embedding"] = test_embedding()
    
    # Test 4: LLM
    results["LLM"] = test_llm()
    
    # Test 5: Real query
    if results["FAISS Index"] and results["Embedding"] and results["LLM"]:
        results["Real Query"] = test_real_query()
    else:
        print("\nSkipping real query test (dependencies failed)")
        results["Real Query"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "-"
        print(f"  [{icon}] {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed'}")


if __name__ == "__main__":
    main()
