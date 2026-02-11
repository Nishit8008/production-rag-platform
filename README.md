#  Production RAG Platform

A Retrieval-Augmented Generation (RAG) system built with:

- FastAPI (Backend)
- Streamlit (Frontend)
- FAISS (Vector Store)
- Google Gemini Embeddings
- LangChain (LCEL)

---
##  Architecture

Streamlit → FastAPI → RAG Pipeline → FAISS → Gemini LLM

---
##  Project Structure

backend/
  core/
    ingestion.py
    rag_pipeline.py
    vector_store.py
  main.py

frontend/
  streamlit_app.py
  
---

##  Setup

### 1️ Clone the repo
git clone <repo_url>
cd rag-platform

### 2️ Backend setup

cd backend
pip install -r requirements.txt
uvicorn main:app --reload

### 3️ Frontend setup

cd frontend
pip install -r requirements.txt
streamlit run streamlit_app.py

---

##  Features

- Document ingestion
- Semantic search with FAISS
- Context-aware generation
- Source document transparency
- Modular RAG pipeline
