# Task 1. MMLU Sequential vs Parallel Timing (Local Transformers Inference)

## Setup

* Environment: Google Colab
* Hardware:

  * Sequential run: CPU-only
  * Parallel run: Tesla T4 GPU (CUDA enabled)
* Model: `allenai/OLMo-2-0425-1B`
* Programs:

  * `llama_mmlu_eval_topic1.py` — Subject: **abstract_algebra** (100 questions)
  * `llama_mmlu_eval_topic2.py` — Subject: **astronomy** (152 questions)

---

## Sequential Execution (CPU)

**Command:**

```bash
time { python llama_mmlu_eval_topic1.py ; python llama_mmlu_eval_topic2.py ; }
```

**Results:**

* Subject 1: abstract_algebra

  * Questions: 100
  * Accuracy: 32.00%
  * Script-reported runtime: ~6.5 minutes
* Subject 2: astronomy

  * Questions: 152
  * Accuracy: 39.47%
  * Script-reported runtime: ~10.6 minutes
* **Total wall-clock time (`real`): 19m22.641s**

**Observation:**
The total wall-clock time is slightly larger than the sum of the two evaluation phases. This overhead is primarily caused by **each script independently loading the full model and tokenizer**, as well as dataset initialization and I/O. This highlights an inefficiency of local, script-based inference: even when running the same model back-to-back, the startup cost is paid per process.

---

## Parallel Execution (GPU)

**Command:**

```bash
time { python llama_mmlu_eval_topic1.py & python llama_mmlu_eval_topic2.py & wait ; }
```

**Results:**

* Both programs detected and used the same GPU (Tesla T4, CUDA)
* Both processes loaded their own copy of the model into GPU memory
* **Total wall-clock time (`real`): 2m4.327s**

**Observation:**
In the parallel GPU run, both programs executed inference concurrently on the same GPU. Despite duplicating the model in memory, the total wall-clock time was dramatically lower than the sequential CPU run because GPU acceleration dominated the computation cost.

However, both processes still **competed for the same GPU resources**, which limits scalability. While parallel execution improved throughput in this case, true scaling would require either batching requests within a single model instance or running multiple model replicas behind a scheduler or load balancer.

---

## Key Takeaway

This experiment demonstrates the difference between **process-level parallelism** and **system-level throughput**. Sequential local inference incurs repeated model startup costs, while parallel execution can significantly reduce wall-clock time when sufficient hardware acceleration is available. Ultimately, performance is bounded by the compute capacity of the underlying hardware and how efficiently model instances are shared or replicated across clients.

## Task 2. OpenAI API Smoke Test

### What this does
Runs a minimal request to confirm the OpenAI Python SDK is installed, the API key is set, and the model can be called successfully.

### Key setup
Set `OPENAI_API_KEY` (env var). Then run the script/notebook cell.

### Code explanation
- `client = OpenAI()`: creates an authenticated API client (reads `OPENAI_API_KEY`) used to send requests.
- `client.chat.completions.create(...)`: sends a chat prompt to `gpt-4o-mini` and returns the model output in `response` (with `max_tokens=5` limiting output length).

### Expected result
A short reply such as `Working!`, which confirms the end-to-end request/response pipeline works.
