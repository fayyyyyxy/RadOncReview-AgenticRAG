# Agentic RAG Engine for Radiation Oncology

An autonomous Retrieval Augmented Generation (RAG) system for radiation oncology treatment guidelines retrieval.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Query                                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Input Guardrails                                  │
│              (PHI Filter, Length Check, Safety)                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Query Analyzer                                    │
│     (Intent Classification, Entity Extraction, Complexity Check)    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Task Planner                                      │
│         (Query Decomposition, Strategy Selection, Planning)         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Retriever Manager                                   │
│                                                                      │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│   │   Dense     │  │   Sparse    │  │   Hybrid    │                │
│   │  (FAISS)    │  │  (Keyword)  │  │   (RRF)     │                │
│   └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                      │
│   ┌─────────────────────────────────────────────────┐              │
│   │              DynamoDB (Metadata)                 │              │
│   └─────────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 Response Synthesizer                                 │
│            (LLM Generation with Citations)                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Self-Critic                                       │
│       (Completeness, Accuracy, Safety Evaluation)                   │
│                         │                                            │
│              ┌──────────┴──────────┐                                │
│              ▼                     ▼                                │
│         [Pass]              [Needs Retry]                           │
│              │                     │                                │
│              │           ┌────────▼────────┐                        │
│              │           │  Replan & Retry │                        │
│              │           └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   Output Guardrails                                  │
│           (Safety Check, Disclaimer Addition)                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Final Response                                  │
│         (Answer + Citations + Confidence + Reasoning)               │
└─────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
agentic_rag/
├── __init__.py          # Package initialization
├── config.py            # Configuration settings
├── models.py            # Data models and types
├── llm_client.py        # LLM client (AWS Bedrock)
├── query_analyzer.py    # Query analysis and classification
├── task_planner.py      # Task decomposition and planning
├── retriever.py         # Retrieval manager (FAISS + DynamoDB)
├── synthesizer.py       # Response synthesis and self-critique
├── guardrails.py        # Safety and compliance checks
├── agent.py             # Main agent orchestration
├── api.py               # REST API endpoints
├── demo.py              # Demo and testing scripts
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run Demo (Mock Mode)

Test the system without AWS credentials:

```bash
# Run all demos
python demo.py --mode all

# Interactive mode
python demo.py --mode interactive

# Specific component demo
python demo.py --mode analysis
python demo.py --mode planning
python demo.py --mode guardrails
python demo.py --mode pipeline
```

### 3. Configure AWS Credentials

Ensure your AWS profile is configured:

```bash
aws configure --profile YOUR_AWS_PROFILE
```

### 4. Run with Real AWS Backend

```python
from agentic_rag import create_agent

# Create agent with real AWS connections
agent = create_agent(use_mock=False)

# Query the system
response = agent.query("What is the standard dose for breast cancer IMRT?")

print(response.answer)
print(f"Confidence: {response.confidence_score:.2%}")
for citation in response.citations:
    print(f"[{citation.citation_id}] {citation.source_path}")
```

### 5. Run API Server

```bash
# Development mode
python api.py

# Or with uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API endpoints:
- `GET /health` - Health check
- `POST /query` - Query guidelines
- `POST /feedback` - Submit feedback
- `GET /categories` - List disease categories
- `GET /modalities` - List treatment modalities

## 🔧 Configuration

Edit `config.py` to customize:

```python
@dataclass
class LLMConfig:
    model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    max_tokens: int = 4096
    temperature: float = 0.1

@dataclass
class RetrieverConfig:
    top_k: int = 10
    similarity_threshold: float = 0.5
    use_hybrid_search: bool = True

@dataclass
class AgentConfig:
    max_iterations: int = 5
    enable_self_critique: bool = True
    confidence_threshold: float = 0.7
```

## 📊 Key Features

### Query Analysis
- Intent classification (dose, protocol, side effects, comparison, etc.)
- Disease category detection (Breast, Thorax, CNS, etc.)
- Treatment modality extraction (IMRT, SBRT, Brachytherapy, etc.)
- Complexity assessment for multi-step queries

### Retrieval Strategies
- **Dense Search**: FAISS vector similarity using Titan embeddings
- **Sparse Search**: Keyword-based matching with BM25-style scoring
- **Hybrid Search**: Reciprocal Rank Fusion (RRF) combining both
- **Multi-hop**: Iterative refinement for complex queries

### Self-Critique
- Completeness evaluation
- Accuracy verification against citations
- Safety assessment for medical recommendations
- Automatic retry with improved retrieval

### Guardrails
- PHI (Protected Health Information) filtering
- Input length and content validation
- Output safety checks
- Automatic disclaimer addition

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_query_analyzer.py -v
```

## 📝 Example Queries

```python
# Simple factual query
"What is the standard dose for whole breast radiation?"

# Comparison query
"Compare SBRT vs conventional fractionation for early-stage lung cancer"

# Complex multi-part query
"What are the indications, contraindications, and dose fractionation 
options for stereotactic radiosurgery in brain metastases?"

# Side effects query
"What are the acute and late toxicities of thoracic radiation?"

# Staging query
"What is the treatment approach for T2N1M0 breast cancer?"
```

## 🔒 Security Considerations

1. **PHI Protection**: Automatic detection and filtering of patient identifiers
2. **API Authentication**: API key validation for all endpoints
3. **Content Safety**: Output guardrails prevent harmful recommendations
4. **Audit Logging**: All queries and responses can be logged for compliance

## 📈 Next Steps

1. **AWS AgentCore Integration**: Deploy on AWS AgentCore for production scaling
2. **Web UI Development**: Build React/Tailwind interface
3. **Feedback Loop**: Implement RLHF for continuous improvement
4. **Evaluation Framework**: Add automated benchmarking and metrics
5. **Additional Guidelines**: Ingest NCCN, ASTRO, ESTRO guidelines

## 📄 License

[Your License Here]

## 🤝 Contributing

[Contribution Guidelines]
