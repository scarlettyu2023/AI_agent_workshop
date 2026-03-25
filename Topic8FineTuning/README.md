# 🦙 Topic 8: Fine-Tuning an LLM

> Teaching a small language model to speak SQL — from scratch, on a laptop, in under 20 minutes.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Files](#files)
3. [Setup](#setup)
4. [What the Code Does](#what-the-code-does)
5. [Results](#results)
6. [Discussion](#discussion)

---

## 🔍 Overview

This project fine-tunes `meta-llama/Llama-3.2-1B` to translate natural language questions into SQL queries, using the `b-mc2/sql-create-context` dataset (78,577 examples from WikiSQL and Spider). Fine-tuning is done remotely via the [Tinker API](https://thinkingmachines.ai) using LoRA (Low-Rank Adaptation), so all GPU computation runs on Tinker's servers while the training loop runs locally on a laptop.

The core question the exercise addresses: **when should you fine-tune instead of using RAG?** Text-to-SQL is a *skill*, not a fact lookup — the model needs to learn to compose valid SQL syntax and map natural language concepts to column names it has never seen. RAG can retrieve similar examples but cannot teach the model to generalize to novel schemas. Fine-tuning bakes the skill into the model's weights.

---

## 📁 Files

| File | Description |
|------|-------------|
| 🐍 `sql_finetune.py` | Main script: loads data, evaluates base model, fine-tunes, evaluates fine-tuned model, tests novel schemas |
| 🔍 `sql_matches.py` | Execution-based SQL comparison utility (provided by instructor) |
| 🗄️ `sql_create_context_v4.json` | Dataset: 78,577 (question, schema, SQL) triples (not committed — too large) |
| 📄 `output_task1.txt` | Terminal output from the final run |

---

## ⚙️ Setup

```bash
pip install tinker transformers python-dotenv
export TINKER_API_KEY=your_key_here
```

Place `sql_create_context_v4.json` and `sql_matches.py` in the same directory, then:

```bash
python sql_finetune.py | tee output_task1.txt
```

---

## 🧠 What the Code Does

**📦 Step 1 — Load & split data**
78,577 examples are shuffled with a fixed seed (42) and split into 200 held-out test examples and 78,377 training examples.

**📊 Step 2 — Evaluate base model**
Before any training, the unmodified Llama-3.2-1B is evaluated on all 200 test questions. Each question is fed as:
```
Table schema:
CREATE TABLE head (age INTEGER, ...)
Question: How many heads of departments are older than 56?
SQL:
```
The model generates SQL and the result is compared to the expected answer using execution-based evaluation: both queries are run on an in-memory SQLite database built from the schema, and the result sets are compared.

**🏷️ Step 3 — Tokenize training data**
Each training example is converted into a `Datum` with a loss-weight mask: prompt tokens get weight 0, completion tokens (the SQL answer) get weight 1. This ensures the model is only trained to predict the SQL output, not the schema or question.

**🔥 Step 4 — Fine-tune**
One epoch over all 78,377 training examples, batch size 256 (~307 batches), learning rate 5e-4, LoRA rank 32. The training loop calls `forward_backward()` and `optim_step()` on Tinker's servers, which handle all GPU computation remotely.

**✅ Step 5 — Evaluate fine-tuned model**
The same 200 held-out test questions are evaluated again with the fine-tuned model.

**🧪 Step 6 — Novel schema questions**
Five hand-crafted questions with schemas not seen during training (employees, products, students, orders, courses/enrollments) test out-of-distribution generalization at Easy, Medium, and Hard difficulty levels.

---

## 📈 Results

### Accuracy: Before and After Fine-Tuning

| Model | Correct | Accuracy |
|-------|---------|----------|
| 🤖 Base Llama-3.2-1B | 91 / 200 | 45.5% |
| ✨ Fine-tuned (1 epoch, LoRA rank 32) | 183 / 200 | **91.5%** |
| 🚀 **Improvement** | +92 | **+46.0pp** |

The result matches the exercise's expected range (~87% fine-tuned vs ~37% base). Our base model measured slightly higher than expected (~45% vs ~37%), likely because the execution-based evaluator accepts semantically equivalent queries that differ in syntax.

### 📉 Training Loss Curve

Loss dropped from ~0.128 at step 10 to ~0.029 at step 307 — a steep initial drop in the first ~50 batches as the model rapidly learned the SQL format, followed by gradual refinement. No divergence or instability was observed.

```
0.13 |█
0.10 |███
0.07 |██████
0.05 |████████████
0.03 |█████████████████████████████████  ← converged
     └──────────────────────────────────▶ batch step (307)
```

### 🧪 Novel Schema Questions (Out-of-Distribution)

| Question | Difficulty | Result | Note |
|----------|------------|--------|------|
| Names of employees in engineering dept | 🟢 Easy | ✅ Correct | Case difference handled by evaluator |
| Count products costing more than $50 | 🟢 Easy | ✅ Correct | Exact match |
| Highest score in science class | 🟡 Medium | ✅ Correct | Extra clause ignored by execution-based eval |
| Top 3 customers by total order amount | 🟡 Medium | ❌ Wrong | Column order swapped — logically correct, evaluator strict |
| Students enrolled per department (JOIN) | 🔴 Hard | ✅ Correct | Different alias style, same result |

**🎯 Novel schema accuracy: 4/5 (80%)**

---

## 💬 Discussion

### 🧩 What did the model actually learn?
The fine-tuned model clearly learned SQL *structure and syntax* — it correctly generates SELECT, WHERE, GROUP BY, ORDER BY, JOIN, and aggregation queries on schemas it has never seen. The Hard-1 JOIN question passing is particularly striking: the model generalized JOIN logic to a completely novel pair of tables. This is evidence of genuine skill internalization, not pattern memorization.

### ⚠️ Where does it still fail?
The main failure mode on novel schemas is **hallucination of extra conditions** — the model adds WHERE clauses (`AND category = 'electronics'`, `AND name = 'John'`) or extra columns not mentioned in the question. These are plausible-looking values from training data, applied inappropriately to new schemas. This is a generalization gap: the model knows SQL syntax but sometimes over-applies patterns from similar training examples.

Medium-2 (orders TOP-3) is a special case — the generated SQL is logically correct but fails because column order in SELECT is swapped. This is a limitation of the strict execution-based evaluator, not a model error.

### 🆚 Why does fine-tuning beat RAG here?
RAG would struggle because the task requires *compositional generation* — combining SQL syntax knowledge with the specific column names and logic of each schema. Retrieving similar (question, SQL) pairs helps with syntax but doesn't teach the model to map novel column names to the right SQL clauses. Fine-tuning bakes this compositional ability into the weights, which is why accuracy jumps from 45% to 91% after one epoch.

### 🚫 Prompt engineering at inference time doesn't substitute for training distribution
During experimentation, adding an instruction prefix to the novel schema prompts ("Only use tables and columns from the schema. Do not add extra conditions...") consistently made results *worse*. The model was trained on the bare `Table schema / Question / SQL:` format, so deviating from it at inference time introduced noise the model had no basis to interpret. To reliably improve out-of-distribution performance, the instruction prefix would need to be included in training examples too.

### 🤏 Small models + fine-tuning can surprise you
Llama-3.2-1B is tiny by modern standards (1.2B parameters), but one epoch of LoRA fine-tuning on 78k examples was enough to take it from 45% to 91.5% on a complex compositional task. All base model weights remained frozen — only the LoRA adapter (~1.6% of total parameters) was trained. This confirms the exercise's key insight: fine-tuning teaches skills efficiently even on small models, and LoRA makes it practical without requiring a GPU cluster.
