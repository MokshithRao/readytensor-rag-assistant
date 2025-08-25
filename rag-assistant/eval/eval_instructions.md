# Lightweight RAG Evaluation

This folder sketches a simple approach to evaluate retrieval quality and end-to-end answers.

## 1) Sanity Checks
- Confirm index size: number of chunks, average tokens per chunk.
- Spot-check top-3 results for a few seed questions in `examples/seed_questions.txt`.

## 2) Automated Retrieval Metrics (optional)
Use [`ragas`](https://github.com/explodinggradients/ragas) to compute:
- **Context Precision/Recall** (are retrieved chunks relevant?)
- **Faithfulness** (is the answer grounded in retrieved context?)

Example (pseudo-code):
```python
# pip install ragas datasets evaluate
from ragas.metrics import faithfulness, context_recall, context_precision
# prepare dataset of (question, ground_truth, retrieved_context, answer)
# run metrics and record scores
```

## 3) Prompt Variants
Experiment with:
- `k` (retriever depth), `search_type` = similarity vs mmr
- Chunk size/overlap in `configs/config.yaml`
- System prompt phrasing (cite sources, don't answer if unknown)

## 4) Report
Add a short write-up (1–2 pages) to your publication:
- Dataset & licenses
- Preprocessing & chunking choices
- Retriever params & LLM
- Representative Q&A with sources
- Limitations (coverage, hallucination, multilinguals) and mitigations
