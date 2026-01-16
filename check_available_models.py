"""
检测哪些 Bedrock 模型可用
"""
import boto3
import json

PROFILE = "YOUR_AWS_PROFILE"
REGION = "YOUR_AWS_REGION"

# 常用模型列表
MODELS_TO_TEST = [
    # Amazon Titan
    ("amazon.titan-text-premier-v1:0", "Amazon Titan Premier"),
    ("amazon.titan-text-express-v1", "Amazon Titan Express"),
    ("amazon.titan-text-lite-v1", "Amazon Titan Lite"),
    
    # Claude
    ("anthropic.claude-3-sonnet-20240229-v1:0", "Claude 3 Sonnet"),
    ("anthropic.claude-3-haiku-20240307-v1:0", "Claude 3 Haiku"),
    ("anthropic.claude-instant-v1", "Claude Instant"),
    
    # Llama
    ("meta.llama3-70b-instruct-v1:0", "Llama 3 70B"),
    ("meta.llama3-8b-instruct-v1:0", "Llama 3 8B"),
    
    # Mistral
    ("mistral.mistral-7b-instruct-v0:2", "Mistral 7B"),
    ("mistral.mixtral-8x7b-instruct-v0:1", "Mixtral 8x7B"),
]

def test_model(bedrock_client, model_id, model_name):
    """测试单个模型是否可用"""
    try:
        # 根据模型类型构建请求
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
        return True, "✅ 可用"
        
    except bedrock_client.exceptions.AccessDeniedException:
        return False, "❌ 无权限"
    except bedrock_client.exceptions.ValidationException as e:
        if "not authorized" in str(e).lower():
            return False, "❌ 无权限"
        return False, f"⚠️ 验证错误: {str(e)[:50]}"
    except Exception as e:
        error_msg = str(e)
        if "not authorized" in error_msg.lower() or "deny" in error_msg.lower():
            return False, "❌ 无权限"
        return False, f"⚠️ 错误: {error_msg[:50]}"


def main():
    print("=" * 60)
    print("🔍 检测可用的 Bedrock 模型")
    print("=" * 60)
    print(f"\nProfile: {PROFILE}")
    print(f"Region: {REGION}\n")
    
    try:
        session = boto3.Session(profile_name=PROFILE)
        bedrock = session.client("bedrock-runtime", region_name=REGION)
    except Exception as e:
        print(f"❌ 无法创建 Bedrock 客户端: {e}")
        return
    
    print("测试模型可用性...\n")
    
    available_models = []
    
    for model_id, model_name in MODELS_TO_TEST:
        is_available, status = test_model(bedrock, model_id, model_name)
        print(f"  {model_name:25} {status}")
        if is_available:
            available_models.append((model_id, model_name))
    
    print("\n" + "=" * 60)
    
    if available_models:
        print("✅ 可用的模型:")
        for model_id, model_name in available_models:
            print(f"  - {model_name}: {model_id}")
        
        print(f"\n💡 建议: 在 config.py 中将 model_id 设置为:")
        print(f'   model_id: str = "{available_models[0][0]}"')
    else:
        print("❌ 没有找到可用的模型")
        print("\n请联系你的 AWS 管理员开通 Bedrock 模型访问权限")


if __name__ == "__main__":
    main()
