from __future__ import annotations

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# LLM providers
from utils import get_settings
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly using the provided context. "
    "If the answer is not in the context, say you don't know. "
    "Cite sources with their metadata (file name) when possible.\n\n"
    "{context}"
)

def load_vectorstore(store_dir: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=get_settings().embedding_model)
    return FAISS.load_local(store_dir, embeddings, allow_dangerous_deserialization=True)

def get_llm():
    s = get_settings()
    if s.llm_provider.lower() == "ollama":
        return ChatOllama(model=s.ollama_model, temperature=0.2)
    # default openai
    return ChatOpenAI(model=s.openai_model, temperature=0.2)

def build_chain(store_dir: str):
    s = get_settings()
    vs = load_vectorstore(store_dir)
    retriever = vs.as_retriever(search_type=s.search_type, search_kwargs={"k": s.k})

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Question: {input}")
    ])

    llm = get_llm()
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(retriever, doc_chain)

    return rag

    # Small adapter: accept {"input": "..."} OR {"question": "..."}
    def input_adapter(user_input: dict):
        return {"question": user_input.get("question") or user_input.get("input")}

    return input_adapter | rag

def format_docs(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"[{src}]\n{d.page_content}")
    return "\n\n".join(parts)
