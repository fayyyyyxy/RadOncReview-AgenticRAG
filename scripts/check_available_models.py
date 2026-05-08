"""
Detect which Bedrock models are available for your AWS account.
"""
import sys
import os

# Add project root to path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
import json
from config import create_boto3_session, default_config

PROFILE = default_config.aws.profile_name
REGION = default_config.aws.region

# List of common models to test
MODELS_TO_TEST = [
    ("amazon.titan-text-express-v1", "Titan Text Express"),
    ("amazon.titan-text-lite-v1", "Titan Text Lite"),
    ("anthropic.claude-3-sonnet-20240229-v1:0", "Claude 3 Sonnet"),
    ("anthropic.claude-3-haiku-20240307-v1:0", "Claude 3 Haiku"),
    ("meta.llama3-70b-instruct-v1:0", "Llama 3 70B"),
    ("meta.llama3-8b-instruct-v1:0", "Llama 3 8B"),
    ("meta.llama4-scout-17b-instruct-v1:0", "Llama 4 Scout 17B"),
    ("meta.llama4-maverick-17b-instruct-v1:0", "Llama 4 Maverick 17B"),
    ("mistral.mistral-7b-instruct-v0:2", "Mistral 7B"),
]

def test_model(bedrock_client, model_id, model_name):
    """Test if a single model is available"""
    try:
        # Build request based on model type
        if "anthropic" in model_id:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hi"}]
            }
        elif "titan" in model_id:
            body = {
                "inputText": "Hi",
                "textGenerationConfig": {"maxTokenCount": 10}
            }
        elif "meta" in model_id or "llama" in model_id.lower():
            body = {
                "prompt": "Hi",
                "max_gen_len": 10
            }
        elif "mistral" in model_id:
            body = {
                "prompt": "<s>[INST] Hi [/INST]",
                "max_tokens": 10
            }
        else:
            body = {"inputText": "Hi"}

        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(body)
        )
        return True, "AVAILABLE"

    except bedrock_client.exceptions.AccessDeniedException:
        return False, "ACCESS DENIED"
    except bedrock_client.exceptions.ValidationException as e:
        if "not authorized" in str(e).lower():
            return False, "ACCESS DENIED"
        return False, f"VALIDATION ERROR: {str(e)[:50]}"
    except Exception as e:
        error_msg = str(e)
        if "not authorized" in error_msg.lower() or "deny" in error_msg.lower():
            return False, "ACCESS DENIED"
        return False, f"ERROR: {error_msg[:50]}"


def main():
    print("=" * 60)
    print("Detecting Available Bedrock Models")
    print("=" * 60)
    print(f"\nProfile: {PROFILE}")
    print(f"Region: {REGION}\n")

    try:
        session = create_boto3_session()
        bedrock = session.client("bedrock-runtime", region_name=REGION)
    except Exception as e:
        print(f"ERROR - Cannot create Bedrock client: {e}")
        return

    print("Testing model availability...\n")

    available_models = []

    for model_id, model_name in MODELS_TO_TEST:
        is_available, status = test_model(bedrock, model_id, model_name)
        status_icon = "[+]" if is_available else "[-]"
        print(f"  {status_icon} {model_name:25} {status}")
        if is_available:
            available_models.append((model_id, model_name))

    print("\n" + "=" * 60)

    if available_models:
        print("Available models:")
        for model_id, model_name in available_models:
            print(f"  - {model_name}: {model_id}")

        print(f"\nRecommendation: Set model_id in config.py to:")
        print(f'   model_id: str = "{available_models[0][0]}"')
    else:
        print("No available models found")
        print("\nPlease contact your AWS administrator to enable Bedrock model access")


if __name__ == "__main__":
    main()
