# Agentic RAG Engine for Radiation Oncology

An autonomous Retrieval Augmented Generation (RAG) system for radiation oncology treatment guidelines retrieval. This system connects to AWS services (S3, DynamoDB, Bedrock) to provide accurate, citation-backed answers to clinical questions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Query                                   │
│            "What is the standard dose for breast cancer?"           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Input Guardrails                                │
│               • PHI Detection & Filtering                           │
│               • Input Validation                                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Query Analyzer                                  │
│               • Intent Classification                               │
│               • Disease Category Detection                          │
│               • Treatment Modality Extraction                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Task Planner                                    │
│               • Query Decomposition                                 │
│               • Retrieval Strategy Selection                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Retriever Manager                                 │
│    ┌───────────┐    ┌───────────┐    ┌───────────────────┐          │
│    │   FAISS   │    │ DynamoDB  │    │  Amazon Titan     │          │
│    │  (S3)     │    │ (Chunks)  │    │  (Embeddings)     │          │
│    └───────────┘    └───────────┘    └───────────────────┘          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Response Synthesizer                               │
│               • Llama 3 70B (via Bedrock)                           │
│               • Citation Generation                                 │
│               • Self-Critique & Validation                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Output Guardrails                                │
│               • Safety Verification                                 │
│               • Medical Disclaimer Addition                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Final Response                                 │
│          Answer + Citations + Confidence Score                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

Before using this system, ensure you have:

1. **Python 3.9+** installed
2. **AWS CLI** configured with SSO access
3. **Data Pipeline Completed** - The `data_prep` pipeline must have been run to populate:
   - S3 bucket with FAISS index (`radonc_faiss_index.bin`)
   - DynamoDB table with chunk data (`YOUR_DYNAMODB_TABLE`)

## Setup

### Step 1 — Install Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Download spaCy model (for text processing)
python -m spacy download en_core_web_sm
```

### Step 2 — Set up external services

Before running the data preparation pipeline, you must complete two one-time setup steps:

**Google Docs API** — needed to pull treatment guideline documents from Google Docs.
Follow the setup guide: [`data_prep/setup-guides/GoogleDocsAPI-setup.pdf`](data_prep/setup-guides/GoogleDocsAPI-setup.pdf)

This will produce two credential files that must be placed inside `data_prep/`:
- `credentials.json` — OAuth2 client credentials downloaded from Google Cloud Console
- `token.json` — generated automatically on first run via browser-based OAuth flow

**AWS CLI** — needed to access S3, DynamoDB, and Bedrock.
Follow the setup guide: [`data_prep/setup-guides/AWS-CLI-setup.pdf`](data_prep/setup-guides/AWS-CLI-setup.pdf)


Once installed, configure your SSO profile (first time only):
```bash
aws configure sso --profile YOUR_AWS_PROFILE
# You will be prompted for:
# - SSO start URL (get from your AWS admin)
# - SSO region: YOUR_AWS_REGION
# - Account and role selection
```

Then log in (required every session):
```bash
aws sso login --profile YOUR_AWS_PROFILE
```

### Step 3 — Verify AWS configuration

Edit `config.py` if your AWS settings differ from the defaults:
```python
@dataclass
class AWSConfig:
    profile_name: str = "YOUR_AWS_PROFILE"
    region: str = "YOUR_AWS_REGION"
    s3_bucket: str = "YOUR_S3_BUCKET"
    dynamodb_table: str = "YOUR_DYNAMODB_TABLE"
    faiss_index_key: str = "radonc_faiss_index.bin"
```

### Step 4 — Run the data preparation pipeline

> ⚠️ This step only needs to be run **once** (or whenever the source documents are updated).
```bash
python data_prep/main.py
```

## Usage

### Interactive Chat (Recommended)

Start an interactive chat session:

```bash
python chat.py
```

Example session:

```
============================================================
  RadOnc Agentic RAG - Interactive Chat
============================================================

Connecting to AWS...
Connected successfully!

Ask questions about radiation oncology treatment guidelines.
Type 'quit' or 'exit' to end the session.
Type 'trace' to show reasoning trace for the last query.
------------------------------------------------------------

You: What is the standard dose for breast cancer?

Thinking...
[10:38:14] GUARDRAIL: Checking input safety
[10:38:14] ANALYZE: Intent: dose_fractionation, Disease: Breast
[10:38:14] PLAN: Created 1 sub-tasks
[10:38:15] RETRIEVE: Found 10 relevant chunks
[10:38:17] SYNTHESIZE: Generating answer...

Agent: Based on the guidelines, hypofractionation is recommended for 
whole breast irradiation. The standard regimens are 40 Gy in 15 fractions 
(START B) or 42.5 Gy in 16 fractions (Whelan) [1][2].

[Confidence: 85.0% | Citations: 3 | Latency: 3200ms]

Sources:
  [1] Breast > Fractionation
  [2] Breast > Whole Breast Irradiation

You: trace

--- Reasoning Trace ---
  [10:38:14] GUARDRAIL: Checking input safety
  [10:38:14] ANALYZE: Intent: dose_fractionation, Disease: Breast
  [10:38:14] PLAN: Created 1 sub-tasks: Simple query
  [10:38:15] RETRIEVE: Task completed, found 10 chunks
  [10:38:17] SYNTHESIZE: Generated answer with 3 citations
  [10:38:17] CRITIQUE: Complete: True, Accurate: True, Safe: True
------------------------------

You: quit
Goodbye!
```

### Streamlit Web UI

```bash
streamlit run ui_streamlit.py
```

### REST API

Start the API server:

```bash
python api.py
# Or with hot reload:
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Submit a question |
| `/feedback` | POST | Submit feedback |
| `/categories` | GET | List disease categories |
| `/modalities` | GET | List treatment modalities |

Example API call:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "query": "What is the standard dose for breast cancer?",
    "include_reasoning": true,
    "include_citations": true
  }'
```

### Python API

Use the agent programmatically:

```python
from agent import create_agent

# Create agent (connects to AWS)
agent = create_agent(use_mock=False)

# Query
response = agent.query("What is the standard dose for breast cancer?")

# Access results
print(response.answer)
print(f"Confidence: {response.confidence_score:.1%}")

for citation in response.citations:
    print(f"[{citation.citation_id}] {citation.source_path}")

# View reasoning trace
for step in response.reasoning_trace:
    print(step)
```

## Project Structure

```
RadOncReview-AgenticRAG/
├── config.py                   # Configuration settings (AWS, LLM, retrieval)
├── models.py                   # Data models (QueryAnalysis, AgentResponse, etc.)
├── agent.py                    # Main agent orchestrator
├── query_analyzer.py           # Query intent & entity extraction
├── task_planner.py             # Task decomposition & planning
├── retriever.py                # FAISS search & DynamoDB retrieval
├── synthesizer.py              # LLM response generation & self-critique
├── guardrails.py               # Input/output safety checks
├── llm_client.py               # AWS Bedrock LLM client
├── chat.py                     # Interactive CLI chat interface
├── api.py                      # REST API server (FastAPI)
├── ui_streamlit.py             # Streamlit web UI
├── demo.py                     # Demo & component testing scripts
├── check_available_models.py   # Detect available Bedrock models
├── requirements.txt            # Python dependencies
│
└── data_prep/
    ├── main.py                 # Data preparation pipeline (run once)
    ├── extract_doc_json.py     # Pull raw JSON from Google Docs API
    ├── clean_json.py           # Format raw JSON into hierarchical sections
    ├── flattening_chunking.py  # Flatten hierarchy + sentence-level chunking
    ├── embedding.py            # Generate embeddings + build FAISS index
    ├── DynamoDB_store_load.py  # Upload/load chunks to/from DynamoDB
    ├── s3_store_load.py        # Upload/load files to/from S3
    ├── helper.py               # Utility functions
    └── setup-guides/
        ├── GoogleDocsAPI-setup.pdf
        └── AWS-CLI-setup.pdf
```

## Supported Query Types

| Intent | Example Questions |
|--------|-------------------|
| Dose/Fractionation | "What is the dose for breast IMRT?" |
| Treatment Protocol | "What is the standard treatment for stage III lung cancer?" |
| Side Effects | "What are the side effects of whole brain radiation?" |
| Comparison | "Compare SBRT vs conventional fractionation" |
| Indications | "When is proton therapy indicated?" |
| Contraindications | "What are contraindications for brachytherapy?" |
| Staging | "How is T2N1M0 breast cancer treated?" |

## Supported Disease Categories

- Breast
- Thorax (Lung, Esophagus)
- Hematologic (Lymphoma, Leukemia)
- CNS (Brain tumors)
- GI (Gastric, Pancreatic, Colorectal)
- GU (Prostate, Bladder)
- Gynecologic (Cervical, Uterine)
- Head & Neck
- Pediatric
- Sarcoma
- Skin (Melanoma)

## Troubleshooting

### AWS Connection Issues

```bash
# Re-login to AWS SSO
aws sso login --profile YOUR_AWS_PROFILE

# Verify connection
aws s3 ls s3://YOUR_S3_BUCKET --profile YOUR_AWS_PROFILE
```

### Model Access Denied

Run the model checker to see available models:

```bash
python check_available_models.py
```

Update `config.py` with an available model.

### No Results Found

1. Check if data pipeline has been run (`data_prep/main.py`)
2. Verify DynamoDB table exists and has data
3. Verify FAISS index exists in S3
