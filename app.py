import time
import streamlit as st
from pypdf import PdfReader

from rag import index_document, retrieve_and_rerank, answer_with_citations

st.set_page_config(page_title="Mini RAG Assistant", layout="wide")
st.title("Mini RAG Assistant (Pinecone + Cohere + Gemini)")

with st.sidebar:
    st.header("Index a document")
    mode = st.radio("Input type", ["Paste text", "Upload PDF"], index=0)

    source_name = st.text_input("Source name (for citations)", value="demo_doc")

    doc_text = ""
    if mode == "Paste text":
        doc_text = st.text_area("Paste content to index", height=220)
    else:
        pdf = st.file_uploader("Upload PDF", type=["pdf"])
        if pdf is not None:
            reader = PdfReader(pdf)
            pages_text = []
            failed_pages = 0

            for i, p in enumerate(reader.pages):
                try:
                    txt = p.extract_text()
                    pages_text.append(txt or "")
                except Exception:
                    failed_pages += 1
                    pages_text.append("")

            doc_text = "\n".join(pages_text)

            if failed_pages > 0:
                st.warning(
                    f"Could not extract text from {failed_pages} page(s). "
                    "If the PDF is scanned/image-based, use Paste text mode."
                )


    if st.button("Index document", type="primary"):
        if not doc_text.strip():
            st.error("Please provide text/PDF content first.")
        else:
            out = index_document(doc_text, source_name=source_name)
            if out["ok"]:
                st.success(f"Indexed ✅ doc_id={out['doc_id']} | chunks={out['chunks_indexed']} | {out['seconds']}s")
            else:
                st.error(out["message"])

st.divider()

st.subheader("Ask a question")
query = st.text_input("Query", placeholder="Ask something based on your indexed document...")
col1, col2 = st.columns([1, 1])
with col1:
    top_k = st.slider("Retriever top-k", 3, 15, 8)
with col2:
    top_n = st.slider("Reranker top-n", 2, 8, 4)

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Type a question first.")
    else:
        t0 = time.time()
        contexts = retrieve_and_rerank(query, top_k=top_k, rerank_top_n=top_n)
        answer, used, stats = answer_with_citations(query, contexts)
        st.caption(f"Total time: {round(time.time()-t0, 3)}s | LLM time: {stats.get('seconds')}s")

        st.markdown("### Answer")
        st.write(answer)

        st.markdown("### Sources (top reranked)")
        for i, c in enumerate(used, start=1):
            st.markdown(f"**[{i}]** source=`{c['source']}` | chunk=`{c['chunk_id']}` | score=`{c['rank_score']:.3f}`")
            st.code(c["text"][:1200])
