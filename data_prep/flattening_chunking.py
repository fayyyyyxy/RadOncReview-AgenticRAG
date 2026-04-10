from typing import List, Dict
import tiktoken
from spacy.lang.en import English


def flatten_json(node: Dict, category: str, headings=None):
    """
    Recursively flatten hierarchical JSON up to any heading level.
    Only include headings that exist.
    """
    if headings is None:
        headings = {}
    flattened = []

    # Update headings dict with existing headings
    for key in node:
        if key.startswith("heading_"):
            headings[key] = node[key]

    # Build path from category + all existing headings
    sorted_heading_keys = sorted(headings.keys(), key=lambda x: int(x.split("_")[1]))
    path_parts = [category] + [headings[k] for k in sorted_heading_keys]
    full_path = " > ".join(path_parts)
    
    # If node has content, add a flattened entry
    if "content" in node:
        entry = {
            "category": category,
        }
        # Include only existing headings
        for k, v in headings.items():
            entry[k] = v
        entry["path"] =  full_path
        entry["content"] = node["content"]
        flattened.append(entry)

    # Recursively process children
    for child in node.get("children", []):
        flattened.extend(flatten_json(child, category, headings.copy()))

    return flattened


def chunk_by_sentences(text: str, max_tokens: int , overlap_tokens: int, model_name: str) -> List[str]:
    # tokenizer for token counting
    encoding = tiktoken.encoding_for_model(model_name)

    # sentence splitter
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = nlp(text)

    sentences = [sent.text.strip() for sent in doc.sents]

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        sent_tokens = len(encoding.encode(sent))

        # If adding this sentence exceeds max_tokens → finalize current chunk
        if current_tokens + sent_tokens > max_tokens:
            # finalize chunk
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)

            # ---- OVERLAP ----
            # add last few sentences based on token overlap
            overlap_list = []
            overlap_count = 0
            for s in reversed(current_chunk):
                t = len(encoding.encode(s))
                if overlap_count + t > overlap_tokens:
                    break
                overlap_list.insert(0, s) 
                overlap_count += t

            current_chunk = overlap_list[:]  # start next chunk with overlap
            current_tokens = sum(len(encoding.encode(s)) for s in current_chunk)

        current_chunk.append(sent)
        current_tokens += sent_tokens

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def flatten_chunk_json(json_data, model_name, max_tokens, overlap_tokens, start_faiss_id, document_url=""):
    """ 
    Flatten, chunk, and assign a unique sequential FAISS_ID to each chunk.
    start_faiss_id ensures IDs are unique when running multiple ingestion jobs.
    """

    flattened_data = flatten_json(json_data, json_data["Category"])
    flattened_chunks = []
    faiss_id_counter = start_faiss_id

    for entry in flattened_data:
        content = entry.pop("content")  # remove original full content
        chunks = chunk_by_sentences(content, max_tokens, overlap_tokens, model_name)
        # Each chunk becomes a separate entry
        for i, chunk in enumerate(chunks):
            chunk_entry = entry.copy()
            # Add FAISS_ID and chunk_id
            chunk_entry["FAISS_ID"] = faiss_id_counter
            chunk_entry["chunk_id"] = i + 1
            # Add Path to each chunk's text for context
            chunk_entry["text"] = f"Title: {entry['path']}\n\nContent: {chunk}"
            chunk_entry["document_url"] = document_url  # url
            flattened_chunks.append(chunk_entry)
            faiss_id_counter += 1

    return flattened_chunks