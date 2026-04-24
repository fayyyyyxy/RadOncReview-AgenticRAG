import boto3
from botocore.exceptions import ClientError


def upload_to_dynamodb(profile_name, table_name, flattened_chunks, region="YOUR_AWS_REGION"):
    """ Store flattened chunks into DynamoDB table. Using FAISS_ID as the Partition Key (PK)."""
    session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
    
    dynamodb = session.resource("dynamodb", region_name=region)
    try:
        # Define KeySchema with FAISS_ID (Number) as the HASH key
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "FAISS_ID", "KeyType": "HASH"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "FAISS_ID", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST"
        )
        table.wait_until_exists()
        print(f"Table '{table_name}' created!")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            table = dynamodb.Table(table_name)
            print("Table already exists — skipping creation.")
        else:
            raise e


    with table.batch_writer() as batch:
        for item in flattened_chunks:
            sk = item["path"] + "#" + str(item["chunk_id"])

            batch.put_item(
                Item={
                    "FAISS_ID": item["FAISS_ID"],
                    "SK": sk,
                    **item
                }
            )
    print(f"Uploaded {len(flattened_chunks)} items to DynamoDB table '{table_name}'.")