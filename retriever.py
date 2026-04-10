"""
Retriever Manager: Handles all retrieval operations.
Supports dense (vector), sparse (keyword), and hybrid search strategies.
"""
import boto3
import json
import faiss
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
from sentence_transformers import CrossEncoder

from models import RetrievedChunk, SubTask, RetrievalStrategy, TaskStatus
from config import RetrieverConfig, EmbeddingConfig, AWSConfig


class CrossEncoderReranker:
    """Cross-Encoder model for reranking retrieved documents."""
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", preload: bool = False):
        self.model_name = model_name
        self._model = None

        # Preload model if requested
        if preload:
            self._load_model()
    
    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                print(f"Loading Cross-Encoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
                print("Cross-Encoder loaded successfully")
            except ImportError:
                print("Warning: sentence-transformers not installed. Install with: pip install sentence-transformers")
            except Exception as e:
                print(f"Warning: Failed to load cross-encoder: {e}")
    
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 10) -> List[RetrievedChunk]:
        """Rerank chunks using cross-encoder."""
        self._load_model()
        
        if self._model is None or not chunks:
            return chunks[:top_k]
        
        try:
            # Create query-document pairs
            pairs = [[query, c.text] for c in chunks]
            
            # Get cross-encoder scores
            scores = self._model.predict(pairs, show_progress_bar=False)
            
            # Combine chunks with scores and sort
            scored_chunks = list(zip(chunks, scores))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Update relevance scores and return top_k
            results = []
            for chunk, score in scored_chunks[:top_k]:
                chunk.relevance_score = float(score)
                results.append(chunk)
            
            return results
        except Exception as e:
            print(f"Cross-encoder reranking failed: {e}")
            return chunks[:top_k]


class RetrieverManager:
    """Manages retrieval from the FAISS index and DynamoDB."""
    
    def __init__(
        self, 
        retriever_config: RetrieverConfig,
        embedding_config: EmbeddingConfig,
        aws_config: AWSConfig
    ):
        self.config = retriever_config
        self.embedding_config = embedding_config
        self.aws_config = aws_config
        
        # Initialize AWS clients with explicit region
        self.session = boto3.Session(profile_name=aws_config.profile_name)
        self.bedrock = self.session.client("bedrock-runtime", region_name=aws_config.region)
        self.s3 = self.session.client("s3", region_name=aws_config.region)
        self.dynamodb = self.session.resource("dynamodb", region_name=aws_config.region)
        self.table = self.dynamodb.Table(aws_config.dynamodb_table)

        # Bedrock Knowledge Base client
        self.bedrock_agent = self.session.client(
            "bedrock-agent-runtime",
            region_name=aws_config.region
        )
        self.kb_id = aws_config.knowledge_base_id
        self.kb_enabled = aws_config.knowledge_base_enabled
        
        # FAISS index (loaded lazily)
        self._faiss_index = None
        self._faiss_index_loaded = False
        
        # Cross-Encoder for reranking
        self._cross_encoder = CrossEncoderReranker(preload=True)
    
    
    # Bedrock Knowledge Base retrieval
    def retrieve_from_knowledge_base(
        self,
        query: str,
        top_k: int = 10
    ) -> List[RetrievedChunk]:
        """Retrieve from Bedrock Knowledge Base (reading list PDFs/docs in S3)."""
        if not self.kb_enabled:
            return []
        try:
            retrieval_config = {
                "vectorSearchConfiguration": {
                    "numberOfResults": top_k,
                    "overrideSearchType": "HYBRID"
                }
            }

            # Filter to specific data source if configured — leave empty to search all
            if self.aws_config.knowledge_base_datasource_id:
                retrieval_config["vectorSearchConfiguration"]["filter"] = {
                    "equals": {
                        "key": "x-amz-bedrock-kb-data-source-id",
                        "value": self.aws_config.knowledge_base_datasource_id 
                    }
                }

            response = self.bedrock_agent.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration=retrieval_config
            )

            results = []
            for i, r in enumerate(response.get("retrievalResults", [])):
                content = r.get("content", {}).get("text", "")
                s3_uri = r.get("location", {}).get("s3Location", {}).get("uri", "")
                score = r.get("score", 0.0)

                # Derive title and category from S3 path
                # e.g. s3://YOUR_S3_BUCKET/reading-lists/GI/Rectum/German Rectal Trial.pdf
                path_parts = s3_uri.replace(
                    "s3://YOUR_S3_BUCKET/reading-lists/", ""
                ).split("/")
                disease = path_parts[0] if path_parts else "Unknown"
                doc_title = (
                    path_parts[-1].replace(".pdf", "").replace(".docx", "")
                    if path_parts else s3_uri
                )

                results.append(RetrievedChunk(
                    faiss_id=-(i + 1),       # negative IDs — no collision with FAISS IDs
                    chunk_id=i,
                    text=content,
                    path=f"{disease} > {doc_title}",
                    category=disease,
                    similarity_score=float(score),
                    source_type="clinical_trial",
                    document_title=doc_title,
                    s3_location=s3_uri
                ))
            return results

        except Exception as e:
            print(f"Bedrock KB retrieval failed: {e}")
            return []


    def _load_faiss_index(self) -> faiss.Index:
        """Load FAISS index from S3."""
        if self._faiss_index_loaded:
            return self._faiss_index
        
        # Download from S3 to temp file
        local_path = "/tmp/faiss_index.bin"
        self.s3.download_file(
            self.aws_config.s3_bucket,
            self.aws_config.faiss_index_key,
            local_path
        )
        
        self._faiss_index = faiss.read_index(local_path)
        self._faiss_index_loaded = True
        print(f"Loaded FAISS index with {self._faiss_index.ntotal} vectors")
        return self._faiss_index
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Amazon Titan."""
        response = self.bedrock.invoke_model(
            modelId=self.embedding_config.model_id,
            body=json.dumps({"inputText": text})
        )
        response_body = json.loads(response["body"].read())
        embedding = np.array(response_body["embedding"], dtype="float32")
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embedding.reshape(1, -1))
        return embedding
    
    def _fetch_chunks_from_dynamodb(self, faiss_ids: List[int]) -> List[Dict]:
        """Fetch chunk metadata from DynamoDB by FAISS IDs."""
        chunks = []
        
        # Batch get (DynamoDB supports up to 100 items per batch)
        for i in range(0, len(faiss_ids), 100):
            batch_ids = faiss_ids[i:i+100]
            
            for fid in batch_ids:
                try:
                    # Convert to Python int (handles numpy.int64)
                    fid_int = int(fid)
                    response = self.table.get_item(Key={"FAISS_ID": fid_int})
                    if "Item" in response:
                        # Convert Decimal values back to native Python types
                        item = self._convert_decimals(response["Item"])
                        chunks.append(item)
                except Exception as e:
                    print(f"Error fetching FAISS_ID {fid}: {e}")
        
        return chunks
    
    def _convert_decimals(self, obj):
        """Convert Decimal objects to int/float for JSON serialization."""
        from decimal import Decimal
        
        if isinstance(obj, dict):
            return {k: self._convert_decimals(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimals(v) for v in obj]
        elif isinstance(obj, Decimal):
            if obj % 1 == 0:
                return int(obj)
            else:
                return float(obj)
        return obj
    
    def dense_search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Perform dense (vector similarity) search."""
        top_k = top_k or self.config.top_k
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search FAISS index
        index = self._load_faiss_index()
        distances, indices = index.search(query_embedding.reshape(1, -1), top_k)
        
        # Filter by similarity threshold
        valid_results = [
            (int(idx), float(dist)) 
            for idx, dist in zip(indices[0], distances[0])
            if idx >= 0 and dist >= self.config.similarity_threshold
        ]
        
        if not valid_results:
            return []
        
        # Fetch metadata from DynamoDB
        faiss_ids = [idx for idx, _ in valid_results]
        chunks_data = self._fetch_chunks_from_dynamodb(faiss_ids)
        
        # Create RetrievedChunk objects
        results = []
        id_to_score = {idx: score for idx, score in valid_results}
        
        for chunk in chunks_data:
            faiss_id = int(chunk["FAISS_ID"])
            
            # Extract headings
            headings = {k: v for k, v in chunk.items() if k.startswith("heading_")}
            
            results.append(RetrievedChunk(
                faiss_id=faiss_id,
                chunk_id=int(chunk.get("chunk_id", 0)),
                text=chunk.get("text", ""),
                path=chunk.get("path", ""),
                category=chunk.get("category", ""),
                headings=headings,
                similarity_score=id_to_score.get(faiss_id, 0.0),
                source_type="review_notes",                        # 
                document_title=chunk.get("category", ""),          # 
                s3_location=chunk.get("document_url", "") 
            ))
        
        # Sort by similarity score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results
    
    def sparse_search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Perform sparse (keyword-based) search using DynamoDB scan with filtering."""
        top_k = top_k or self.config.top_k
        
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        # Scan DynamoDB and score by keyword matches
        # Note: In production, use OpenSearch/Elasticsearch for better keyword search
        all_chunks = []
        
        try:
            # Scan with filter for category if detected
            scan_kwargs = {"Limit": 1000}  # Limit scan for performance
            response = self.table.scan(**scan_kwargs)
            all_chunks.extend(response.get("Items", []))
            
            # Handle pagination
            while "LastEvaluatedKey" in response and len(all_chunks) < 5000:
                scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
                response = self.table.scan(**scan_kwargs)
                all_chunks.extend(response.get("Items", []))
        except Exception as e:
            print(f"DynamoDB scan error: {e}")
            return []
        
        # Score chunks by keyword matches
        scored_chunks = []
        for chunk in all_chunks:
            text = chunk.get("text", "").lower()
            path = chunk.get("path", "").lower()
            
            # Count keyword matches
            score = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Weight title/path matches higher
                if keyword_lower in path:
                    score += 2
                if keyword_lower in text:
                    score += text.count(keyword_lower)
            
            if score > 0:
                scored_chunks.append((chunk, score))
        
        # Sort by score and take top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_chunks = scored_chunks[:top_k]
        
        # Convert to RetrievedChunk objects
        results = []
        max_score = max(score for _, score in top_chunks) if top_chunks else 1
        
        for chunk, score in top_chunks:
            headings = {k: v for k, v in chunk.items() if k.startswith("heading_")}
            
            results.append(RetrievedChunk(
                faiss_id=int(chunk["FAISS_ID"]),
                chunk_id=int(chunk.get("chunk_id", 0)),
                text=chunk.get("text", ""),
                path=chunk.get("path", ""),
                category=chunk.get("category", ""),
                headings=headings,
                similarity_score=score / max_score  # Normalize to 0-1
            ))
        
        return results
    
    def hybrid_search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        """Combine dense and sparse search results."""
        top_k = top_k or self.config.top_k
        
        # Get results from both methods
        dense_results = self.dense_search(query, top_k=top_k )
        sparse_results = self.sparse_search(query, top_k=top_k )
        
    
        # Merge results using Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60  # RRF constant
        
        # Score dense results
        for rank, chunk in enumerate(dense_results):
            rrf_scores[chunk.faiss_id] = rrf_scores.get(chunk.faiss_id, 0) + 1 / (k + rank + 1)
        
        # Score sparse results
        for rank, chunk in enumerate(sparse_results):
            rrf_scores[chunk.faiss_id] = rrf_scores.get(chunk.faiss_id, 0) + 1 / (k + rank + 1)
        
        # Create lookup for chunks
        chunk_lookup = {c.faiss_id: c for c in dense_results + sparse_results}
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Return top_k unique results
        results = []
        for fid in sorted_ids[:top_k]:
            chunk = chunk_lookup[fid]
            chunk.relevance_score = rrf_scores[fid]
            results.append(chunk)
        
        return results
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query for sparse search."""
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
            'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
            'if', 'or', 'because', 'until', 'while', 'what', 'which', 'who',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-zA-Z0-9]+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Also extract medical terms (keep capitalized terms and acronyms)
        medical_terms = re.findall(r'\b[A-Z]{2,}\b', query)  # Acronyms like SBRT, IMRT
        keywords.extend([t.lower() for t in medical_terms])
        
        return list(set(keywords))
    
    def execute_task(self, task: SubTask) -> SubTask:
        """Execute a retrieval task and update its status."""
        task.status = TaskStatus.IN_PROGRESS
        
        # Use task-specific top_k if provided, otherwise use default
        top_k = task.top_k if task.top_k else None
        
        try:
            # Select retrieval method based on strategy
            if task.retrieval_strategy == RetrievalStrategy.DENSE:
                chunks = self.dense_search(task.query, top_k=top_k)
            elif task.retrieval_strategy == RetrievalStrategy.SPARSE:
                chunks = self.sparse_search(task.query, top_k=top_k)
            elif task.retrieval_strategy == RetrievalStrategy.HYBRID:
                chunks = self.hybrid_search(task.query, top_k=top_k)
            elif task.retrieval_strategy == RetrievalStrategy.MULTI_HOP:
                # Multi-hop: first dense, then refine
                initial_chunks = self.dense_search(task.query, top_k)
                if initial_chunks:
                    # Use top result to refine search
                    refined_query = f"{task.query} {initial_chunks[0].path}"
                    chunks = self.hybrid_search(refined_query, top_k=top_k)
                else:
                    chunks = []
            # Bedrock Knowledge Base
            elif task.retrieval_strategy == RetrievalStrategy.KNOWLEDGE_BASE:
                chunks = self.retrieve_from_knowledge_base(
                    task.query,
                    top_k=top_k or self.config.top_k
                )
            else:
                # Fallback
                chunks = self.dense_search(task.query, top_k=top_k)
            
            # Store results in task
            task.retrieved_chunks = [
                {
                    "faiss_id": c.faiss_id,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "path": c.path,
                    "category": c.category,
                    "similarity_score": c.similarity_score,
                    "relevance_score": c.relevance_score,
                    "source_type": c.source_type,
                    "document_title": c.document_title,
                    "s3_location": c.s3_location
                }
                for c in chunks
            ]
            task.relevance_scores = [c.similarity_score for c in chunks]
            task.status = TaskStatus.COMPLETED
            
            # Generate summary of retrieved content
            if chunks:
                task.result = f"Retrieved {len(chunks)} relevant chunks. " \
                             f"Top result: {chunks[0].path} (score: {chunks[0].similarity_score:.3f})"
            else:
                task.result = "No relevant chunks found."
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.result = f"Retrieval failed: {str(e)}"
        
        return task
    
    def rerank(
        self, 
        query: str, 
        chunks: List[RetrievedChunk],
        top_k: Optional[int] = None
    ) -> List[RetrievedChunk]:
        """Rerank chunks using Cross-Encoder model."""
        top_k = top_k or self.config.rerank_top_k
        
        # Use Cross-Encoder for reranking
        return self._cross_encoder.rerank(query, chunks, top_k)


class MockRetrieverManager:
    """Mock retriever for testing without AWS."""
    
    def __init__(self, *args, **kwargs):
        self.call_count = 0
    
    def dense_search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        self.call_count += 1
        return [
            RetrievedChunk(
                faiss_id=i,
                chunk_id=1,
                text=f"Mock chunk {i} for query: {query[:50]}",
                path=f"Category > Section {i}",
                category="Mock",
                similarity_score=0.9 - (i * 0.05)
            )
            for i in range(min(top_k, 5))
        ]
    
    def sparse_search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        return self.dense_search(query, top_k)
    
    def hybrid_search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        return self.dense_search(query, top_k)
    
    def execute_task(self, task: SubTask) -> SubTask:
        task.status = TaskStatus.COMPLETED
        task.retrieved_chunks = [{"text": "Mock result", "path": "Mock > Path"}]
        task.result = "Mock retrieval completed"
        return task
    
    def rerank(self, query: str, chunks: List[RetrievedChunk], top_k: int = 5) -> List[RetrievedChunk]:
        return chunks[:top_k]
