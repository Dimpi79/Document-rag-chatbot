import os
import time
import uuid
from typing import List, Dict, Tuple

from dotenv import load_dotenv
import cohere
import google.generativeai as genai

from db_pinecone import upsert_vectors, query_vectors

load_dotenv()
try:
    import streamlit as st
    SECRETS = st.secrets
except Exception:
    SECRETS = {}

COHERE_API_KEY = os.getenv("COHERE_API_KEY") or SECRETS.get("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or SECRETS.get("GEMINI_API_KEY")

if not COHERE_API_KEY:
    raise ValueError("Missing COHERE_API_KEY in .env")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

co = cohere.Client(COHERE_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("models/gemini-2.5-flash-preview-09-2025")

# ---------- Chunking ----------
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """
    Simple character-based chunking (fast + OK for assessment).
    chunk_size ~ 800-1200; overlap ~ 10-15%
    """
    text = text.replace("\r", "\n").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks

# ---------- Embeddings (Cohere 1024-d) ----------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Cohere embedding model (choose a 1024-d model).
    If your account/model name differs, change model string here.
    """
    resp = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return resp.embeddings

def embed_query(query: str) -> List[float]:
    resp = co.embed(
        texts=[query],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    return resp.embeddings[0]

# ---------- Indexing ----------
def index_document(doc_text: str, source_name: str = "uploaded_doc") -> Dict:
    """
    Chunks -> embeddings -> upsert into Pinecone
    """
    t0 = time.time()
    chunks = chunk_text(doc_text)
    if not chunks:
        return {"ok": False, "message": "No text to index."}

    embeddings = embed_texts(chunks)

    doc_id = str(uuid.uuid4())[:8]
    vectors = []
    for i, (ch, emb) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"{doc_id}-{i}",
            "values": emb,
            "metadata": {
                "source": source_name,
                "doc_id": doc_id,
                "chunk_id": i,
                "text": ch
            }
        })

    upsert_vectors(vectors)
    return {
        "ok": True,
        "doc_id": doc_id,
        "chunks_indexed": len(chunks),
        "seconds": round(time.time() - t0, 3)
    }

# ---------- Retrieval + Rerank ----------
def retrieve_and_rerank(query: str, top_k: int = 8, rerank_top_n: int = 4) -> List[Dict]:
    qvec = embed_query(query)
    res = query_vectors(qvec, top_k=top_k)
    matches = res.get("matches", []) if isinstance(res, dict) else res.matches

    # Prepare docs for rerank
    docs = []
    for m in matches:
        md = m.get("metadata", {}) if isinstance(m, dict) else m.metadata
        docs.append(md.get("text", ""))

    if not docs:
        return []

    rr = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docs,
        top_n=min(rerank_top_n, len(docs))
    )

    # Map rerank results back to metadata
    reranked = []
    for r in rr.results:
        i = r.index
        m = matches[i]
        md = m.get("metadata", {}) if isinstance(m, dict) else m.metadata
        reranked.append({
            "rank_score": float(r.relevance_score),
            "source": md.get("source", "unknown"),
            "doc_id": md.get("doc_id", ""),
            "chunk_id": md.get("chunk_id", -1),
            "text": md.get("text", "")
        })
    return reranked

# ---------- Answer with citations ----------
def answer_with_citations(query: str, contexts: List[Dict]) -> Tuple[str, List[Dict], Dict]:
    """
    Returns:
      answer_text, contexts_used, stats
    """
    t0 = time.time()

    if not contexts:
        return (
            "I couldn’t find relevant context in the indexed documents. Try indexing a document first or rephrase your query.",
            [],
            {"seconds": round(time.time() - t0, 3)}
        )

    # Build numbered context block
    ctx_lines = []
    for i, c in enumerate(contexts, start=1):
        ctx_lines.append(f"[{i}] (source={c['source']}, chunk={c['chunk_id']})\n{c['text']}\n")

    prompt = f"""
You are a RAG assistant. Answer the user question ONLY using the provided sources.
Rules:
- Every factual sentence MUST end with at least one citation like [1] or [1][2].
- Do NOT use outside knowledge. If not found in Sources, reply exactly: "Not found in the provided documents."
- Keep the answer concise (4-8 lines max).

Question: {query}

Sources:
{chr(10).join(ctx_lines)}
"""

    resp = gemini.generate_content(prompt)
    answer = resp.text.strip() if hasattr(resp, "text") and resp.text else str(resp)

    return answer, contexts, {"seconds": round(time.time() - t0, 3)}
