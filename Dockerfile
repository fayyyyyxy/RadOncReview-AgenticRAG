FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Cross-Encoder model into image so cold start doesn't need to fetch it
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

COPY api.py config.py chat.py ./
COPY core/ ./core/
COPY evaluation/ ./evaluation/
COPY calibration/ ./calibration/
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
