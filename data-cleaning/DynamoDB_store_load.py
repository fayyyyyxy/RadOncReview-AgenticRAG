import boto3
from botocore.exceptions import ClientError


def upload_to_dynamodb(profile_name, table_name, flattened_chunks):
    """ Store flattened chunks into DynamoDB table. """
    session = boto3.Session(profile_name=profile_name)
    dynamodb = session.resource("dynamodb")

    try:
        table = dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "PK", "KeyType": "HASH"},
                {"AttributeName": "SK", "KeyType": "RANGE"}
            ],
            AttributeDefinitions=[
                {"AttributeName": "PK", "AttributeType": "S"},
                {"AttributeName": "SK", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST"
        )
        table.wait_until_exists()
        print("Table created!")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            table = dynamodb.Table(table_name)
            print("Table already exists — skipping creation.")
        else:
            raise e


    for item in flattened_chunks:
        pk = item["category"] # disease category as partition key
        sk = item["path"] + "#" + str(item["chunk_id"])

        table.put_item(
            Item={
                "PK": pk,
                "SK": sk,
                **item
            }
        )