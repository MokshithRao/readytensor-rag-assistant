from __future__ import annotations

import os
import typer
from rich.console import Console
from rich.panel import Panel

from rag_chain import build_chain

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def chat(store_dir: str = typer.Option("store/faiss_index", help="Vector store directory")):
    """Simple CLI loop for Q&A."""
    chain = build_chain(store_dir)
    console.print(Panel.fit("RAG Assistant CLI — ask questions about your documents. Type 'exit' to quit."))

    while True:
        q = typer.prompt("You")
        if q.strip().lower() in {"exit", "quit"}:
            break
        # FIX: pass "question" instead of "input"
        result = chain.invoke({"input": q})
        answer = result.get("answer") or result.get("output_text") or str(result)
        console.print(Panel(answer, title="Assistant"))

if __name__ == "__main__":
    app()
