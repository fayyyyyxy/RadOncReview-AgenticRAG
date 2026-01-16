#!/usr/bin/env python3
"""
Diagnostic script to check data synchronization between FAISS index and DynamoDB.
"""
import boto3
import json
import faiss
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import default_config


def main():
    print("=" * 60)
    print("  Data Sync Diagnostic Tool")
    print("=" * 60)
    
    config = default_config.aws
    
    print(f"\nConfiguration:")
    print(f"  Profile: {config.profile_name}")
    print(f"  Region: {config.region}")
    print(f"  S3 Bucket: {config.s3_bucket}")
    print(f"  DynamoDB Table: {config.dynamodb_table}")
    print(f"  FAISS Index: {config.faiss_index_key}")
    
    session = boto3.Session(profile_name=config.profile_name)
    
    # ============ Check FAISS Index ============
    print("\n" + "-" * 60)
    print("[1] Checking FAISS Index...")
    
    try:
        s3 = session.client("s3", region_name=config.region)
        local_path = "/tmp/diag_faiss_index.bin"
        
        s3.download_file(config.s3_bucket, config.faiss_index_key, local_path)
        index = faiss.read_index(local_path)
        
        print(f"    FAISS vectors: {index.ntotal}")
        print(f"    FAISS dimension: {index.d}")
        print(f"    Expected FAISS_IDs: 0 to {index.ntotal - 1}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return
    
    # ============ Check DynamoDB Table ============
    print("\n" + "-" * 60)
    print("[2] Checking DynamoDB Table...")
    
    try:
        dynamodb = session.resource("dynamodb", region_name=config.region)
        table = dynamodb.Table(config.dynamodb_table)
        
        # Get table info
        print(f"    Table name: {table.table_name}")
        print(f"    Item count (approx): {table.item_count}")
        print(f"    Key schema: {table.key_schema}")
        
        # Scan a few items to see the structure
        print("\n    Scanning first 5 items...")
        response = table.scan(Limit=5)
        items = response.get("Items", [])
        
        if items:
            print(f"    Found {len(items)} items")
            print("\n    Sample item keys:")
            for item in items[:3]:
                faiss_id = item.get("FAISS_ID", "N/A")
                path = item.get("path", "N/A")
                print(f"      FAISS_ID: {faiss_id} (type: {type(faiss_id).__name__})")
                print(f"      Path: {path[:50]}...")
                print()
        else:
            print("    WARNING: No items found in table!")
            
    except Exception as e:
        print(f"    ERROR: {e}")
        return
    
    # ============ Check ID Matching ============
    print("-" * 60)
    print("[3] Checking FAISS_ID matching...")
    
    # Get all FAISS_IDs from DynamoDB
    print("    Scanning all FAISS_IDs from DynamoDB...")
    
    all_faiss_ids = set()
    scan_kwargs = {"ProjectionExpression": "FAISS_ID"}
    
    try:
        while True:
            response = table.scan(**scan_kwargs)
            for item in response.get("Items", []):
                fid = item.get("FAISS_ID")
                if fid is not None:
                    # Convert Decimal to int
                    all_faiss_ids.add(int(fid))
            
            if "LastEvaluatedKey" not in response:
                break
            scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
        
        print(f"    DynamoDB FAISS_IDs found: {len(all_faiss_ids)}")
        
        if all_faiss_ids:
            min_id = min(all_faiss_ids)
            max_id = max(all_faiss_ids)
            print(f"    DynamoDB ID range: {min_id} to {max_id}")
            
            # Check for gaps
            expected_ids = set(range(index.ntotal))
            missing_in_dynamo = expected_ids - all_faiss_ids
            extra_in_dynamo = all_faiss_ids - expected_ids
            
            print(f"\n    FAISS expects IDs: 0 to {index.ntotal - 1}")
            print(f"    Missing in DynamoDB: {len(missing_in_dynamo)} IDs")
            print(f"    Extra in DynamoDB: {len(extra_in_dynamo)} IDs")
            
            if missing_in_dynamo:
                sample_missing = sorted(list(missing_in_dynamo))[:10]
                print(f"    Sample missing IDs: {sample_missing}")
            
            if extra_in_dynamo:
                sample_extra = sorted(list(extra_in_dynamo))[:10]
                print(f"    Sample extra IDs: {sample_extra}")
        
    except Exception as e:
        print(f"    ERROR scanning: {e}")
    
    # ============ Test Direct Lookup ============
    print("\n" + "-" * 60)
    print("[4] Testing direct lookup...")
    
    test_ids = [0, 1, 2, 100, 200, 500]
    
    for test_id in test_ids:
        try:
            response = table.get_item(Key={"FAISS_ID": test_id})
            if "Item" in response:
                item = response["Item"]
                print(f"    ID {test_id}: FOUND - {item.get('path', 'N/A')[:40]}...")
            else:
                print(f"    ID {test_id}: NOT FOUND")
        except Exception as e:
            print(f"    ID {test_id}: ERROR - {e}")
    
    # ============ Summary ============
    print("\n" + "=" * 60)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    if len(all_faiss_ids) == 0:
        print("\n  PROBLEM: DynamoDB table is EMPTY!")
        print("  SOLUTION: Re-run the data preparation pipeline:")
        print("            cd data_prep && python main.py")
    elif len(missing_in_dynamo) > 0:
        print(f"\n  PROBLEM: {len(missing_in_dynamo)} FAISS IDs not in DynamoDB")
        print("  SOLUTION: Data is out of sync. Re-run data_prep/main.py")
    elif len(all_faiss_ids) != index.ntotal:
        print(f"\n  PROBLEM: Count mismatch - FAISS: {index.ntotal}, DynamoDB: {len(all_faiss_ids)}")
        print("  SOLUTION: Re-run data_prep/main.py to rebuild both")
    else:
        print("\n  STATUS: Data appears to be in sync!")


if __name__ == "__main__":
    main()
