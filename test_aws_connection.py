"""
Quick test script to verify AWS connection and run a real query.
确保先运行: aws sso login --profile YOUR_AWS_PROFILE
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import default_config


def test_aws_connection():
    """测试 AWS 连接"""
    print("=" * 60)
    print("🔧 测试 AWS 连接")
    print("=" * 60)
    
    import boto3
    
    profile = default_config.aws.profile_name
    region = default_config.aws.region
    bucket = default_config.aws.s3_bucket
    
    print(f"\n配置信息:")
    print(f"  Profile: {profile}")
    print(f"  Region: {region}")
    print(f"  S3 Bucket: {bucket}")
    print(f"  DynamoDB Table: {default_config.aws.dynamodb_table}")
    
    try:
        session = boto3.Session(profile_name=profile)
        
        # 测试 S3
        print(f"\n📦 测试 S3 连接...")
        s3 = session.client("s3", region_name=region)
        response = s3.list_objects_v2(Bucket=bucket, MaxKeys=10)
        
        print(f"  ✅ S3 连接成功!")
        print(f"  找到 {response.get('KeyCount', 0)} 个对象:")
        for obj in response.get('Contents', []):
            print(f"    - {obj['Key']} ({obj['Size']} bytes)")
        
        # 测试 DynamoDB
        print(f"\n📊 测试 DynamoDB 连接...")
        dynamodb = session.resource("dynamodb", region_name=region)
        table = dynamodb.Table(default_config.aws.dynamodb_table)
        
        # 获取表信息
        item_count = table.item_count
        print(f"  ✅ DynamoDB 连接成功!")
        print(f"  表名: {table.table_name}")
        print(f"  项目数量: {item_count}")
        
        # 测试 Bedrock
        print(f"\n🤖 测试 Bedrock 连接...")
        bedrock = session.client("bedrock-runtime", region_name=region)
        print(f"  ✅ Bedrock 客户端创建成功!")
        print(f"  模型: {default_config.llm.model_id}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 连接失败: {e}")
        print("\n请确保:")
        print("  1. 运行了 'aws sso login --profile YOUR_AWS_PROFILE'")
        print("  2. AWS 凭证未过期")
        print("  3. 有正确的 IAM 权限")
        return False


def test_faiss_index():
    """测试 FAISS 索引加载"""
    print("\n" + "=" * 60)
    print("🔍 测试 FAISS 索引")
    print("=" * 60)
    
    import boto3
    import faiss
    
    try:
        session = boto3.Session(profile_name=default_config.aws.profile_name)
        s3 = session.client("s3", region_name=default_config.aws.region)
        
        # 下载 FAISS 索引
        local_path = "/tmp/test_faiss_index.bin"
        print(f"\n下载 FAISS 索引...")
        s3.download_file(
            default_config.aws.s3_bucket,
            default_config.aws.faiss_index_key,
            local_path
        )
        
        # 加载索引
        index = faiss.read_index(local_path)
        print(f"  ✅ FAISS 索引加载成功!")
        print(f"  向量数量: {index.ntotal}")
        print(f"  向量维度: {index.d}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAISS 测试失败: {e}")
        return False


def test_embedding():
    """测试 Embedding 生成"""
    print("\n" + "=" * 60)
    print("📐 测试 Embedding 生成")
    print("=" * 60)
    
    import boto3
    import json
    
    try:
        session = boto3.Session(profile_name=default_config.aws.profile_name)
        bedrock = session.client("bedrock-runtime", region_name=default_config.aws.region)
        
        test_text = "What is the dose for breast cancer radiation therapy?"
        
        print(f"\n测试文本: {test_text}")
        print(f"模型: {default_config.embedding.model_id}")
        
        response = bedrock.invoke_model(
            modelId=default_config.embedding.model_id,
            body=json.dumps({"inputText": test_text})
        )
        
        result = json.loads(response["body"].read())
        embedding = result["embedding"]
        
        print(f"  ✅ Embedding 生成成功!")
        print(f"  向量维度: {len(embedding)}")
        print(f"  前5个值: {embedding[:5]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Embedding 测试失败: {e}")
        return False


def test_real_query():
    """测试真实查询"""
    print("\n" + "=" * 60)
    print("🎯 测试真实查询")
    print("=" * 60)
    
    try:
        from agent import create_agent
        
        print("\n创建 Agent (真实模式)...")
        agent = create_agent(use_mock=False)
        
        test_queries = [
            "What is the standard dose for breast cancer?",
            "What are the side effects of radiation therapy?",
        ]
        
        for query in test_queries:
            print(f"\n{'─' * 50}")
            print(f"📝 查询: {query}")
            print('─' * 50)
            
            response = agent.query(query)
            
            print(f"\n💡 回答:\n{response.answer[:500]}...")
            print(f"\n📊 置信度: {response.confidence_score:.2%}")
            print(f"📚 引用数: {len(response.citations)}")
            print(f"⏱️ 延迟: {response.latency_ms:.0f}ms")
            
            if response.citations:
                print(f"\n引用来源:")
                for c in response.citations[:3]:
                    print(f"  [{c.citation_id}] {c.source_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 查询测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "🏥 RadOnc Agentic RAG - AWS 连接测试 🏥".center(60))
    print("=" * 60)
    
    results = {}
    
    # 测试 1: AWS 连接
    results["AWS 连接"] = test_aws_connection()
    
    if not results["AWS 连接"]:
        print("\n⚠️ AWS 连接失败，跳过后续测试")
        return
    
    # 测试 2: FAISS 索引
    results["FAISS 索引"] = test_faiss_index()
    
    # 测试 3: Embedding
    results["Embedding"] = test_embedding()
    
    # 测试 4: 真实查询
    if results["FAISS 索引"] and results["Embedding"]:
        results["真实查询"] = test_real_query()
    else:
        print("\n⚠️ 跳过真实查询测试（依赖项失败）")
        results["真实查询"] = False
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 测试总结")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n{'🎉 所有测试通过!' if all_passed else '⚠️ 部分测试失败'}")


if __name__ == "__main__":
    main()
