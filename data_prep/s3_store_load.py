import boto3
import json
import subprocess
from io import BytesIO

def create_s3_bucket(bucket_name, profile_name, region):
    """ 
        Create S3 bucket if it doesn't exist.
        It is assumed that the AWS CLI is configured with the given profile.
        Equivalent to running:
            aws s3 mb s3://BUCKET_NAME --region REGION --profile PROFILE_NAME
    """
    cmd = [
        "aws", "s3", "mb", f"s3://{bucket_name}",
        "--region", region,
        "--profile", profile_name
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"[SUCCESS] Bucket '{bucket_name}' created.")
    else:
        print("[WARNING]", result.stderr.strip())


def upload_json_to_s3(json_data, bucket_name, profile_name, s3_key):
    """ Upload JSON data directly to an S3 bucket. """
    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3")
    
    # Serialize JSON to a string, then encode it to bytes
    data_bytes = json.dumps(json_data, indent=4).encode("utf-8")
    
    # Use BytesIO to create an in-memory file-like object
    in_memory_file = BytesIO(data_bytes)

    # Use upload_fileobj for streaming from memory
    s3.upload_fileobj(in_memory_file, bucket_name, s3_key)
    print(f"Uploaded: s3://{bucket_name}/{s3_key}")


def upload_file_to_s3(files, bucket_name, profile_name):
    """ Upload file(s) to an S3 bucket. """

    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3")

    for local_path, s3_key in files:
        s3.upload_file(local_path, bucket_name, s3_key)
        print("Uploaded:", f"s3://{bucket_name}/{s3_key}")


def load_from_s3(bucket_name, profile_name, s3_key):
    """ Load a file from S3 bucket. """

    session = boto3.Session(profile_name=profile_name)
    s3 = session.client("s3")
    obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
    data = obj["Body"].read()
    if s3_key.endswith(".json"):
        return json.loads(data.decode("utf-8"))
    return data