from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from utils import get_settings

SUPPORTED_EXTS = [".txt", ".md", ".pdf"]

def load_documents(data_dir: str):
    docs = []
    data_path = Path(data_dir)
    for ext in SUPPORTED_EXTS:
        if ext in [".txt", ".md"]:
            loader = DirectoryLoader(data_dir, glob=f"**/*{ext}", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
            docs.extend(loader.load())
        elif ext == ".pdf":
            # DirectoryLoader can't autoselect PyPDFLoader easily; iterate
            for p in data_path.rglob("*.pdf"):
                docs.extend(PyPDFLoader(str(p)).load())
    return docs

def main():
    parser = argparse.ArgumentParser(description="Ingest documents and build FAISS index.")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory with source docs")
    parser.add_argument("--store_dir", type=str, default="store/faiss_index", help="Where to persist FAISS index")
    args = parser.parse_args()

    settings = get_settings()
    print(f"[ingest] Loading documents from: {args.data_dir}")
    docs = load_documents(args.data_dir)
    if not docs:
        raise SystemExit("No documents found. Add .pdf/.txt/.md files under data/raw/.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[ingest] Docs: {len(docs)} → Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(args.store_dir, exist_ok=True)
    index_path = os.path.join(args.store_dir, "index.faiss")
    store_path = os.path.join(args.store_dir, "index.pkl")
    vectorstore.save_local(args.store_dir)
    print(f"[ingest] Saved FAISS index to {args.store_dir}")

if __name__ == "__main__":
    main()
