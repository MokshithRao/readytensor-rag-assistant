from __future__ import annotations

import argparse
import gradio as gr
from rag_chain import build_chain

def make_app(store_dir: str):
    chain = build_chain(store_dir)

    def respond(question: str):
        if not question.strip():
            return "Please enter a question."
        result = chain.invoke({"input": question})
        return result.get("answer") or result.get("output_text") or str(result)

    with gr.Blocks(title="RAG Assistant") as demo:
        gr.Markdown("# RAG Assistant")
        gr.Markdown("Ask questions grounded in your uploaded corpus.")
        inp = gr.Textbox(label="Question", placeholder="What is this document about?")
        out = gr.Markdown(label="Answer")
        btn = gr.Button("Ask")
        btn.click(fn=respond, inputs=inp, outputs=out)
    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_dir", type=str, default="store/faiss_index")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = make_app(args.store_dir)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=True)

if __name__ == "__main__":
    main()
