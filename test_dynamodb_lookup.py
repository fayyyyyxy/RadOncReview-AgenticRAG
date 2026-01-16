#!/usr/bin/env python3
"""
Quick test to debug DynamoDB FAISS_ID lookup issue.
"""
import boto3
from decimal import Decimal

PROFILE = "YOUR_AWS_PROFILE"
REGION = "YOUR_AWS_REGION"
TABLE_NAME = "YOUR_DYNAMODB_TABLE"

def main():
    print("Testing DynamoDB FAISS_ID lookup...")
    print("=" * 50)
    
    session = boto3.Session(profile_name=PROFILE)
    dynamodb = session.resource("dynamodb", region_name=REGION)
    table = dynamodb.Table(TABLE_NAME)
    
    test_ids = [0, 1, 100, 216, 250, 251]
    
    for fid in test_ids:
        print(f"\n--- Testing FAISS_ID = {fid} ---")
        
        # Method 1: Using int
        print(f"  Method 1 (int): ", end="")
        try:
            response = table.get_item(Key={"FAISS_ID": fid})
            if "Item" in response:
                print(f"FOUND - {response['Item'].get('path', 'N/A')[:40]}")
            else:
                print("NOT FOUND (no Item in response)")
        except Exception as e:
            print(f"ERROR - {e}")
        
        # Method 2: Using Decimal
        print(f"  Method 2 (Decimal): ", end="")
        try:
            response = table.get_item(Key={"FAISS_ID": Decimal(fid)})
            if "Item" in response:
                print(f"FOUND - {response['Item'].get('path', 'N/A')[:40]}")
            else:
                print("NOT FOUND (no Item in response)")
        except Exception as e:
            print(f"ERROR - {e}")
        
        # Method 3: Using string
        print(f"  Method 3 (str): ", end="")
        try:
            response = table.get_item(Key={"FAISS_ID": str(fid)})
            if "Item" in response:
                print(f"FOUND - {response['Item'].get('path', 'N/A')[:40]}")
            else:
                print("NOT FOUND (no Item in response)")
        except Exception as e:
            print(f"ERROR - {e}")
    
    # Check actual key type in table
    print("\n" + "=" * 50)
    print("Checking actual data type in DynamoDB...")
    
    response = table.scan(Limit=1)
    if response.get("Items"):
        item = response["Items"][0]
        faiss_id = item.get("FAISS_ID")
        print(f"  Sample FAISS_ID value: {faiss_id}")
        print(f"  Type: {type(faiss_id)}")
        print(f"  Repr: {repr(faiss_id)}")


if __name__ == "__main__":
    main()
