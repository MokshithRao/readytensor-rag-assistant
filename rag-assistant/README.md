# RAG Assistant (Starter)

A simple Retrieval-Augmented Generation (RAG) assistant built with LangChain and FAISS/Chroma.
It ingests a small custom document set (PDF, TXT, MD), builds an embedding index, and serves a
question-answering interface via CLI or a minimal Gradio UI.

---

## Features
- LangChain-based pipeline (prompt → retriever → response).
- FAISS vector store (default) with local persistence.
- Sentence-Transformers embeddings (default: `all-MiniLM-L6-v2`).
- Pluggable LLM: OpenAI or local Ollama.
- CLI and Gradio UI.
- Clean repo layout with eval notes and a smoke test.
- Optional LangSmith logging via env var.

## Quickstart

### 1) Python Environment
- Python 3.10–3.12 recommended
- Create and activate a virtual environment, then:
```bash
pip install -r requirements.txt
```

### 2) Configure
Copy `.env.example` to `.env` and set at least one LLM provider:
```bash
# for OpenAI
OPENAI_API_KEY=sk-...

# or for local Ollama
#   Install Ollama: https://ollama.com/download
#   Pull a model:   ollama pull llama3.1
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

Optional: enable LangSmith (observability)
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_...
LANGCHAIN_PROJECT=rag-assistant
```

### 3) Put Data
Drop a handful of `.pdf`, `.txt`, or `.md` files into `data/raw/`.

### 4) Build the Vector Store
```bash
python ingest.py --data_dir data/raw --store_dir store/faiss_index
```

### 5) Ask Questions (CLI)
```bash
python cli.py --store_dir store/faiss_index
```

### 6) Minimal UI (Gradio)
```bash
python app.py --store_dir store/faiss_index --port 7860
```

### 7) Smoke Test
```bash
pytest -q
```

---

## Repo Structure
```
rag-assistant-starter/
├── app.py                    # Gradio UI
├── cli.py                    # CLI app (Typer)
├── ingest.py                 # Build/persist FAISS index
├── rag_chain.py              # Prompt + retriever + LLM chain
├── utils.py                  # Helpers (env, loading, etc.)
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/                  # Put your source docs here
│   └── processed/
├── store/                    # Persisted vector store
├── eval/
│   └── eval_instructions.md
├── tests/
│   └── test_smoke.py
├── examples/
│   └── seed_questions.txt
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

