"""
Generate test questions for KB retrieval evaluation.

Logic:
  For each disease site folder in S3:
    For each of 5 clinical topics:
      1. Retrieve top K chunks from Bedrock KB
      2. Call Nova Pro with those chunks → generate 3 QA pairs
  Output: test_questions.csv
"""

import sys
import os

# Add project root to path when running this script directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
import json
import csv
import time
import re
from config import default_config, create_boto3_session


AWS_PROFILE   = default_config.aws.profile_name
AWS_REGION    = default_config.aws.region
S3_BUCKET     = default_config.aws.s3_bucket
KB_ID         = default_config.aws.knowledge_base_id
DS_ID         = default_config.aws.knowledge_base_datasource_id
S3_PREFIX     = "reading-lists/"
NOVA_MODEL_ID = "us.amazon.nova-pro-v1:0"
OUTPUT_CSV    = "test_questions.csv"

CHUNKS_PER_TOPIC     = 80   # chunks retrieved per topic
QUESTIONS_PER_TOPIC  = 3    # questions generated per topic

# 5 Clinical topics targeting KB trial paper content
TOPICS = [
    {
        "name": "treatment_outcomes_survival",
        "query_suffix": "overall survival local control biochemical recurrence radiation",
        "instruction": "This topic covers clinical outcomes from radiation oncology trials, including overall survival, progression-free survival, local control, and biochemical recurrence rates."
    },
    {
        "name": "dose_fractionation",
        "query_suffix": "radiation dose Gy fractions hypofractionation schedule",
        "instruction": "This topic covers radiation dose and fractionation used in trials, including total dose, fractionation schedules, hypofractionation rationale, and comparisons between different regimens."
    },
    {
        "name": "toxicity_side_effects",
        "query_suffix": "toxicity acute late adverse events radiation complications grade",
        "instruction": "This topic covers radiation-related toxicity reported in trials, including acute and late side effects, toxicity grades, organ-specific complications, and quality of life impact."
    },
    {
        "name": "trial_comparison",
        "query_suffix": "randomized trial hazard ratio p-value treatment arms comparison",
        "instruction": "This topic covers randomized trial design and results, including treatment arm comparisons, primary endpoints, hazard ratios, statistical significance, and clinical conclusions."
    },
    {
        "name": "patient_selection",
        "query_suffix": "eligibility criteria patient selection risk stratification staging",
        "instruction": "This topic covers patient selection in radiation oncology trials, including eligibility criteria, risk group stratification, staging, tumor characteristics, and which subgroups benefited most."
    },
]

CSV_COLUMNS = [
    "disease_site",
    "topic",
    "question",
    "expected_answer",
    "source_document"
]


def get_disease_site_folders(s3_client) -> list:
    """List all top-level disease site folders under reading-lists/ in S3."""
    response = s3_client.list_objects_v2(
        Bucket=S3_BUCKET,
        Prefix=S3_PREFIX,
        Delimiter="/"
    )
    folders = []
    for prefix in response.get("CommonPrefixes", []):
        folder = prefix["Prefix"].replace(S3_PREFIX, "").rstrip("/")
        if folder:
            folders.append(folder)
    return sorted(folders)


def retrieve_chunks(bedrock_agent, disease_site: str, topic: dict) -> list:
    """Retrieve top CHUNKS_PER_TOPIC chunks from KB for a disease site + topic."""
    query = f"{disease_site} {topic['query_suffix']}"
    try:
        retrieval_config = {
            "vectorSearchConfiguration": {
                "numberOfResults": CHUNKS_PER_TOPIC,
                "overrideSearchType": "HYBRID"
            }
        }

        filters = []
        if DS_ID:
            filters.append({
                "equals": {
                    "key": "x-amz-bedrock-kb-data-source-id",
                    "value": DS_ID
                }
            })
        filters.append({
            "startsWith": {
                "key": "x-amz-bedrock-kb-source-uri",
                "value": f"s3://{S3_BUCKET}/{S3_PREFIX}{disease_site}/"
            }
        })
        retrieval_config["vectorSearchConfiguration"]["filter"] = (
            {"andAll": filters} if len(filters) > 1 else filters[0]
        )

        response = bedrock_agent.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration=retrieval_config
        )

        chunks = []
        for r in response.get("retrievalResults", []):
            text = r.get("content", {}).get("text", "")
            s3_uri = r.get("location", {}).get("s3Location", {}).get("uri", "")

            # Get document title and top folder from S3 path
            path_parts = s3_uri.replace(
                f"s3://{S3_BUCKET}/{S3_PREFIX}", ""
            ).split("/")
            doc_title  = path_parts[-1].replace(".pdf", "").replace(".docx", "") if path_parts else s3_uri
            top_folder = path_parts[0] if path_parts else disease_site

            chunks.append({
                "text":      text,
                "doc_title": doc_title,
                "folder":    top_folder,
            })
        return chunks

    except Exception as e:
        print(f"\n    KB retrieve failed: {e}")
        return []


def format_chunks_for_prompt(chunks: list) -> str:
    """Format chunks into numbered context string for the LLM prompt."""
    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Chunk {i+1}] Source: {chunk['folder']} / {chunk['doc_title']}\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_qa_pairs(
    bedrock_runtime,
    disease_site: str,
    topic: dict,
    chunks: list
) -> list:
    """Call Nova Pro with retrieved chunks to generate QA pairs."""

    context = format_chunks_for_prompt(chunks)

    prompt = f"""You are a radiation oncology expert creating test questions to evaluate a clinical RAG system.

Disease site: {disease_site}
Topic focus: {topic['name']}

The following are excerpts from actual clinical trial papers and radiation oncology literature for {disease_site}:

{context}

{topic['instruction']}

Based ONLY on the content above, generate exactly {QUESTIONS_PER_TOPIC} question-answer pairs.

Requirements:
- Each question MUST come from a DIFFERENT source document — do not generate multiple questions from the same paper
- Each question should have a different clinical focus — do not repeat the same question structure across the {QUESTIONS_PER_TOPIC} questions
- Questions should be phrased as a radiation oncology clinician would naturally ask in clinic
- Questions MUST be phrased as a question ending with "?" — not a statement
- Answers should be grounded in the text and include relevant clinical data when available
- For source_document, copy the exact Source name from the chunk header (e.g. "Gastrointestinal / German Rectal Trial, NEJM 2004")
- Do NOT use any knowledge outside the provided chunks

Return ONLY a valid JSON array, no other text, no markdown fences:
[
  {{
    "question": "...",
    "answer": "...",
    "source_document": "..."
  }}
]"""

    # Nova Pro uses Converse API format
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {
            "maxTokens": 2000,
            "temperature": 0.3
        }
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId=NOVA_MODEL_ID,
            body=json.dumps(request_body)
        )
        response_body = json.loads(response["body"].read())

        # Parse Nova Pro response
        raw = response_body["output"]["message"]["content"][0]["text"].strip()

        # Strip markdown fences if present
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip()

        qa_pairs = json.loads(raw)
        return qa_pairs[:QUESTIONS_PER_TOPIC]

    except json.JSONDecodeError as e:
        print(f"\n    JSON parse failed: {e} | raw: {raw[:150]}")
        return []
    except Exception as e:
        print(f"\n    Nova Pro call failed: {e}")
        return []


def is_valid_question(qa: dict) -> bool:
    """Validate a QA pair — must be a proper question with non-empty answer."""
    question = qa.get("question", "").strip()
    answer   = qa.get("answer", "").strip()
    return (
        question.endswith("?")   # must be a question
        and len(question) > 20   # not too short
        and len(answer) > 10     # answer must not be empty
    )


def generate_with_retry(
    bedrock_runtime,
    disease_site: str,
    topic: dict,
    chunks: list,
    max_retries: int = 3
) -> list:
    """
    Generate QA pairs with automatic retry until QUESTIONS_PER_TOPIC valid questions are collected or max_retries is reached.
    Deduplicates by question text to avoid repeats across retries.
    """
    valid_pairs  = []
    seen_questions = set()
    attempts     = 0

    while len(valid_pairs) < QUESTIONS_PER_TOPIC and attempts < max_retries:
        attempts += 1
        pairs = generate_qa_pairs(bedrock_runtime, disease_site, topic, chunks)

        for qa in pairs:
            q_text = qa.get("question", "").strip()
            if (is_valid_question(qa)
                    and q_text not in seen_questions
                    and len(valid_pairs) < QUESTIONS_PER_TOPIC):
                valid_pairs.append(qa)
                seen_questions.add(q_text)

        if len(valid_pairs) < QUESTIONS_PER_TOPIC:
            print(f"\n   Only {len(valid_pairs)} valid question(s), retrying ({attempts}/{max_retries})...",
                  end=" ", flush=True)
            time.sleep(0.3)

    return valid_pairs


def main():
    # AWS clients
    session         = create_boto3_session()
    s3              = session.client("s3",                    region_name=AWS_REGION)
    bedrock_agent   = session.client("bedrock-agent-runtime", region_name=AWS_REGION)
    bedrock_runtime = session.client("bedrock-runtime",       region_name=AWS_REGION)

    # Get disease site folders from S3
    print("\nFetching disease site folders from S3...")
    folders = get_disease_site_folders(s3)
    print(f"Found {len(folders)} folders: {folders}")
    print(f"\nPlan: {len(folders)} sites × {len(TOPICS)} topics × {QUESTIONS_PER_TOPIC} questions"
          f" = {len(folders) * len(TOPICS) * QUESTIONS_PER_TOPIC} total questions\n")

    # Open CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        total_questions = 0
        total_calls     = len(folders) * len(TOPICS)
        call_count      = 0

        for disease_site in folders:
            print(f"\n── {disease_site} ──────────────────────────────")

            for topic in TOPICS:
                call_count += 1
                print(f"  [{call_count:2d}/{total_calls}] {topic['name']:<35}", end=" ", flush=True)

                # Step 1: Retrieve chunks from KB
                chunks = retrieve_chunks(bedrock_agent, disease_site, topic)
                if not chunks:
                    print("→ 0 chunks, skipping")
                    continue
                print(f"→ {len(chunks):2d} chunks", end=" ", flush=True)

                # Step 2: Generate QA pairs with Nova Pro (with retry for invalid questions)
                qa_pairs = generate_with_retry(bedrock_runtime, disease_site, topic, chunks)
                if not qa_pairs:
                    print("→ failed")
                    continue
                print(f"→ {len(qa_pairs)} questions ✓")

                # Step 3: Write to CSV immediately (safe even if script stops mid-run)
                for qa in qa_pairs:
                    writer.writerow({
                        "disease_site":    disease_site,
                        "topic":           topic["name"],
                        "question":        qa.get("question", ""),
                        "expected_answer": qa.get("answer", ""),
                        "source_document": qa.get("source_document", ""),
                    })
                    total_questions += 1
                csvfile.flush()
                # Small delay to avoid Bedrock throttling
                time.sleep(0.5)

    print(f"✓ Generated {total_questions} questions → {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":
    main()
