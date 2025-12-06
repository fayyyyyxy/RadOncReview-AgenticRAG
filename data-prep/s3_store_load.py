import boto3
import json
import subprocess

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
        print(f"[SUCCESS] Bucket '{bucket_name}' created (or already exists).")
    else:
        print(f"[ERROR] Failed to create bucket '{bucket_name}'.")
        print("stderr:", result.stderr.strip())


def upload_to_s3(files, bucket_name, profile_name):
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