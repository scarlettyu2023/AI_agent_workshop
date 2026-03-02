# Topic 5 — Retrieval-Augmented Generation (RAG)

**CS @ UVA — Agentic AI (Spring 2026)**
Exercises Completed: 1–7

Assignment Specification:
[https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic5RAG/rag.html](https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/Topic5RAG/rag.html)

---

# Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Pipeline Architecture](#pipeline-architecture)
4. [How to Run](#how-to-run)
5. [Exercise 1 — RAG vs No-RAG](#exercise-1--rag-vs-no-rag)
6. [Exercise 2 — Using GPT-4o Mini](#exercise-2--using-gpt-4o-mini-api-based-rag)
7. [Exercise 3 — Different Corpora Comparison](#exercise-3--different-corpora-comparison)
8. [Exercise 4 — Effect of Top-K](#exercise-4--effect-of-top-k)
9. [Exercise 5 — Handling Unanswerable Questions](#exercise-5--handling-unanswerable-questions)
10. [Exercise 6 — Query Phrasing Sensitivity](#exercise-6--query-phrasing-sensitivity)
11. [Exercise 7 — Chunk Overlap Experiment](#exercise-7--chunk-overlap-experiment)
12. [Key Takeaways](#key-takeaways)
13. [Reproducibility](#reproducibility)

---

# Overview

This project implements a full Retrieval-Augmented Generation (RAG) pipeline from scratch using:

* Sentence-Transformers (all-MiniLM-L6-v2) for embeddings
* FAISS (IndexFlatIP with cosine similarity)
* OpenAI API for generation
* Configurable chunking with overlap
* Automatic device detection (CUDA > MPS > CPU)

The system supports:

* Direct LLM answering (no retrieval)
* RAG answering (top-K retrieval + grounded generation)
* Top-K experiments
* Query phrasing sensitivity analysis
* Chunk overlap experiments with dynamic index rebuilding

---

# Repository Structure

```
Topic5RAG/
│
├── manual_rag_pipeline_universal_scarlett.ipynb   # Main notebook (Exercises 1–7)
├── Corpora/                                       # Text corpora used for retrieval
│   └── ModelTService/                             # Example corpus folder
├── results/                                       # Saved experiment outputs
│   └── ex7_results.pkl                            # Chunk overlap experiment results
├── README.md                                      # Project documentation
```

Directory Table of Contents:

| Path                                         | Description                      |
| -------------------------------------------- | -------------------------------- |
| manual_rag_pipeline_universal_scarlett.ipynb | Full RAG pipeline implementation |
| Corpora/                                     | Raw text corpora                 |
| results/                                     | Serialized experiment results    |
| README.md                                    | Project documentation            |

---

# Pipeline Architecture

## 1. Chunking

* Fixed-size chunks (default: 512 characters)
* Configurable overlap (0–256 tested)
* Attempts paragraph or sentence boundary alignment
* Metadata tracking (source file, chunk index, char offsets)

## 2. Embeddings

* Model: sentence-transformers/all-MiniLM-L6-v2
* 384-dimensional embeddings
* L2-normalized for cosine similarity

## 3. Vector Index

* FAISS IndexFlatIP
* Cosine similarity via normalized vectors

## 4. Retrieval

* Embed query
* FAISS search (top-K)
* Return (chunk, similarity score)

## 5. Generation

* Structured prompt template
* Context injected with source metadata
* Model instructed to answer only from context

---

# How to Run

Install dependencies:

```
pip install torch transformers sentence-transformers faiss-cpu pymupdf accelerate openai
```

Open and run:

```
manual_rag_pipeline_universal_scarlett.ipynb
```

---

# Exercise 1 — RAG vs No-RAG

Goal: Compare direct LLM answers vs retrieval-augmented answers.

Findings:

* Direct LLM often hallucinated or produced vague answers.
* RAG answers were grounded in corpus evidence.
* Retrieval significantly improved factual precision.

---

# Exercise 2 — Using GPT-4o Mini (API-Based RAG)

Goal: Replace the open generation model with GPT-4o Mini via OpenAI API and compare performance.

Procedure:

* Keep the same retrieval pipeline (chunking + embeddings + FAISS).
* Swap generation step to OpenAI API model.
* Compare latency and answer quality.

Findings:

* API model produced more coherent and better-structured answers.
* Latency increased compared to fully local generation.
* Retrieval grounding remained the dominant factor for factual accuracy.

Conclusion:
Model choice affects fluency, but retrieval quality remains the primary driver of correctness.

---

# Exercise 3 — Different Corpora Comparison

Goal: Compare RAG behavior across different corpora (e.g., Model T manual vs Congressional Record).

Procedure:

* Load different corpus folders.
* Rebuild embeddings and index.
* Run identical query structure adapted to each corpus.

Findings:

* Technical manuals benefited from smaller chunk sizes.
* Legislative documents required slightly higher top-K to capture multi-paragraph reasoning.
* Corpus structure strongly influences optimal chunking strategy.

Conclusion:
RAG configuration is corpus-dependent.

---

---

# Exercise 4 — Effect of Top-K

Tested: top_k = 1, 3, 5, 10

| top_k | Observation          |
| ----- | -------------------- |
| 1     | Often incomplete     |
| 3     | Improved coverage    |
| 5     | Best balance         |
| 10    | Increased redundancy |

Conclusion: top_k = 5 provided best tradeoff.

---

# Exercise 5 — Handling Unanswerable Questions

Goal: Evaluate how the system behaves when the answer does not exist in the corpus.

Test Categories:

* Off-topic question
* Related but missing detail
* False premise question

Procedure:

* Run RAG pipeline normally.
* Compare baseline prompt vs stricter refusal prompt.
* Measure whether the model fabricates an answer.

Findings:

* Without strict instructions, model occasionally attempted partial speculation.
* With explicit refusal instruction, the model reliably declined when context was insufficient.

Conclusion:
Prompt design is critical for reducing hallucination in unanswerable cases.

---

# Exercise 6 — Query Phrasing Sensitivity

Testing multiple phrasings of the same question showed:

* Retrieval rankings change with phrasing.
* Keyword-heavy queries improved precision.
* Natural language queries sometimes broadened context.

Conclusion: Query rewriting materially affects retrieval quality.

---

# Exercise 7 — Chunk Overlap Experiment

Fixed chunk size: 512
Overlaps tested: 0, 64, 128, 256

Procedure:

* Rebuild chunks and FAISS index for each overlap
* Run same query set
* Measure retrieval quality, generation quality, and rebuild time

Results Summary:

| Overlap | Retrieval Quality                 | Cost                  |
| ------- | --------------------------------- | --------------------- |
| 0       | Boundary evidence sometimes split | Lowest cost           |
| 64      | Improved continuity               | Slightly larger index |
| 128     | Strong evidence coverage          | Moderate cost         |
| 256     | Marginal improvement              | Highest redundancy    |

Conclusion: 128 overlap provided best tradeoff.

---

# Key Takeaways

1. RAG reduces hallucination.
2. Top-K has an optimal midpoint.
3. Query phrasing impacts retrieval rankings.
4. Chunk overlap improves boundary handling.
5. Overlap exhibits diminishing returns.

---

# Reproducibility

To fully reproduce:

1. Place corpus under `Corpora/`
2. Install dependencies
3. Run the notebook sequentially
4. Experiment outputs saved under `results/`

---

End of README

