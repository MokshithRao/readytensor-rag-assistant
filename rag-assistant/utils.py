from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

load_dotenv()

@dataclass
class Settings:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    k: int = int(os.getenv("K", "4"))
    search_type: str = os.getenv("SEARCH_TYPE", "similarity")
    store_type: str = os.getenv("STORE_TYPE", "faiss")
    llm_provider: str = os.getenv("LLM_PROVIDER", "openai")  # openai|ollama
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1")

def get_settings() -> Settings:
    return Settings()
