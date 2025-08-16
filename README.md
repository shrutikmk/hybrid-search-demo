# arXiv Hybrid Search Demo (BM25 + Embeddings + Reranking)

> A public, end-to-end demo of hybrid retrieval over **arXiv** papers using BM25 + dense embeddings (Matryoshka or Qwen embeddings) + (optional) ColBERT/cross-encoder reranking, with lightweight evals and a simple RAG answerer.

This repository is a **public analogue** of an internal internship project.  
All data here comes from **public arXiv metadata and abstracts** via the arXiv API.  
No proprietary code, data, or configs are included.

---

## TL;DR
- **Ingest**: fetch arXiv paper metadata/abstracts for chosen categories (e.g., cs.AI, cs.LG).  
- **Index**: build BM25 + dense vector indexes (Elasticsearch/Qdrant or FAISS).  
- **Retrieve**: hybrid (BM25 ∪ dense) → rerank (ColBERT or cross-encoder).  
- **Evaluate**: small IR eval set (nDCG@10, Recall@50) + query generation script.  
- **RAG**: minimal LLM answerer (pluggable—local vLLM or API).

---

## Why this matters
- Mirrors real-world **document retrieval** needs: long abstracts, jargon, entity overlap.
- Demonstrates **hybrid search** tradeoffs and **reranking** benefits on a clean, public dataset.
- Shows practical **DS/ML engineering**: ingestion, indexing, evals, dashboards, and a demo CLI.

---

## Data Source & Privacy
- **arXiv API** (public): titles, authors, abstracts, categories, links.  
- We store only metadata/abstracts needed for retrieval.  
- Rate-limit friendly; no scraping of PDFs.

---

## Architecture (at a glance)

1. **Ingestion**: arXiv API → JSONL/Parquet (`data/`).
2. **Indexing**:
   - **BM25**: Elasticsearch (or Whoosh) index.
   - **Dense**: Sentence Transformers (e.g., `tomaarsen/mpnet-base-nli-matryoshka`) or `Qwen/Qwen3-Embedding-8B` → Qdrant/FAISS.
3. **Hybrid Retrieval**: top-k from BM25 and dense → union/merge.
4. **Reranking (optional)**:
   - **ColBERT** (Late Interaction) or
   - **Cross-encoder** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
5. **RAG (optional)**: pass top chunks to an LLM (local via vLLM or API) for an answer.
6. **Eval**: synthetic/hand-curated query set → nDCG@k, Recall@k.

*(Add your diagram to `/docs/architecture.png` and link it here.)*

---

## Features (WIP checklist)

- [x] arXiv ingestion (categories, date ranges, paging, dedupe)
- [x] BM25 index + search
- [x] Dense embeddings + vector index (Qdrant/FAISS)
- [ ] Hybrid merge + scoring heuristics
- [ ] Rerankers: ColBERT / cross-encoder toggle
- [ ] Lightweight evaluation (nDCG@10, Recall@50)
- [ ] Minimal RAG answerer (local vLLM or API)
- [ ] Streamlit demo (search UI + inspect results)
- [ ] Docker compose for one-command bring-up

---

## Getting Started

### 1) Environment
```bash
# Python 3.10+
python -m venv .venv && source .venv/bin/activate
pip install -U pip

# Core deps
pip install arxiv pandas pyarrow tqdm pydantic "qdrant-client>=1.9.0" faiss-cpu

# Retrieval models
pip install "sentence-transformers>=2.7.0" fastembed

# Optional: Elasticsearch (BM25)
# - If using managed ES/OpenSearch, skip local install and set env vars instead.
pip install elasticsearch

# Optional rerankers
pip install colbert-ai  # or a cross-encoder:
pip install transformers torch --extra-index-url https://download.pytorch.org/whl/cu121

# Optional UI + RAG
pip install streamlit uvicorn fastapi
