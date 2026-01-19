# Mini RAG Assistant (Pinecone + Cohere + Gemini)

A small RAG app where users upload a PDF (or paste text), the app chunks + embeds the content into a hosted vector DB (Pinecone), retrieves the most relevant chunks, reranks them, and answers queries using an LLM with inline citations.

## Demo Features
- Upload PDF / paste text input
- Chunking + metadata storage for citations
- Hosted Vector DB (Pinecone) indexing + retrieval
- Retriever (top-k) + Reranker (Cohere)
- LLM answer with inline citations like [1], [2]
- Shows top sources/snippets used for the answer
- Basic timing shown for the query

---

## Tech Stack
- **Frontend/UI:** Streamlit
- **Vector DB (hosted):** Pinecone (cosine, **1024-d**)
- **Embeddings:** Cohere (`embed-english-v3.0`) *(1024 dimension)*
- **Reranker:** Cohere (`rerank-english-v3.0`)
- **LLM:** Gemini (`models/gemini-2.5-flash-preview-09-2025`)
- **PDF Parsing:** pypdf

---

## RAG Configuration
- **Chunk size:** ~900 characters
- **Overlap:** ~120 characters (≈10–15%)
- **Retriever:** top-k = 8 (default)
- **Reranker:** top-n = 4 (default)
- **Metadata stored:** source, doc_id, chunk_id, text

---

## Project Structure
mini_rag_assessment/
app.py # Streamlit UI
rag.py # chunking, embeddings, retrieval, rerank, citations prompt
db_pinecone.py # pinecone init + upsert/query helpers
eval.md # 5 QA evaluation
requirements.txt
.env.example
.gitignore
README.md


---

## Setup (Local)
### 1) Create venv + install deps
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

2. Create a .env file (do NOT commit it). Use .env.example as reference:
PINECONE_API_KEY=your_key
PINECONE_INDEX=mini-rag-index

COHERE_API_KEY=your_key
GEMINI_API_KEY=your_key

3. Run the app
streamlit run app.py

4. How to Use:

a. Upload a PDF (or paste text) in the sidebar and click Index document
b. Enter a query and click Ask
c. The answer will include inline citations like [1] [2]
d. Scroll down to view the matched source chunks

5. Evaluation

See eval.md for 5 sample queries and expected answers.
indexed document used during evaluation: Research_Paper.pdf ( ANN-Based Error Correction...)

6. Deployed using Streamlit Cloud:

Push repo to GitHub
Create a new Streamlit app from the repo
Add secrets ( PINECONE_API_KEY, COHERE_API_KEY, GEMINI_API_KEY, PINECONE_INDEX)
Deploy and paste the live URL here

Notes / Tradeoffs

PDF text extraction may fail on scanned/image-based PDFs; paste-text mode works reliably.
Chunking is character-based for simplicity and speed; can be upgraded to token-based chunking.