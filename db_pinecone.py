import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()
try:
    import streamlit as st
    SECRETS = st.secrets
except Exception:
    SECRETS = {}

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or SECRETS.get("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX") or SECRETS.get("PINECONE_INDEX") or "mini-rag-index"

if not PINECONE_API_KEY:
   raise ValueError("Missing PINECONE_API_KEY (set it in Streamlit Secrets or .env)")
    
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def upsert_vectors(vectors: list[dict]):
    # vectors: [{ "id": str, "values": [float...], "metadata": {...} }, ...]
    index.upsert(vectors=vectors)

def query_vectors(vector: list[float], top_k: int = 8, metadata_filter: dict | None = None):
    return index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter or {}
    )
