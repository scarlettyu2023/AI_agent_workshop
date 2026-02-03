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

## Task 3. Manual Tool Handling with Custom Calculator Agent

This project demonstrates how tool calling works “under the hood” in an LLM-based agent by manually implementing the full loop: tool schema → model tool request → Python execution → result injection → final model response.

Instead of relying on LangGraph’s built-in tools, I define and dispatch my own tools from scratch.

## Features

- Manual agent loop with OpenAI tool calling
- Custom `get_weather` tool (simulated API)
- Custom `calculator` tool with **geometric functions**
- JSON-based argument parsing and result formatting
- Support for **multiple tool calls in a single iteration**
- Demonstration of strategies to force LLMs to use tools

## Tools Implemented

### 1. Weather Tool
Returns simulated weather data for a given city.

### 2. Calculator Tool
Supports both arithmetic and geometric operations:

**Arithmetic**
- `add`
- `subtract`
- `multiply`
- `divide`

**Geometric**
- `area_circle(r)`
- `circumference_circle(r)`
- `area_rectangle(width, height)`
- `hypotenuse(a, b)`

All inputs are parsed using `json.loads`, and outputs are returned as JSON using `json.dumps`.

---

## Example System Output

### Test 1 — Single Tool Call

User: What's the weather like in San Francisco?
--- Iteration 1 ---
LLM wants to call 1 tool(s)
Tool: get_weather
Args: {'location': 'San Francisco'}
Result: Sunny, 72°F
--- Iteration 2 ---
Assistant: The weather in San Francisco is sunny with a temperature of 72°F.

### Test 2 — No Tool Call
User: Say hello!
--- Iteration 1 ---
Assistant: Hello! How can I assist you today?

### Test 3 — Multiple Tool Calls
User: What's the weather in New York and London?
--- Iteration 1 ---
LLM wants to call 2 tool(s)
Tool: get_weather
Args: {'location': 'New York'}
Result: Cloudy, 55°F
Tool: get_weather
Args: {'location': 'London'}
Result: Rainy, 48°F
--- Iteration 2 ---
Assistant:
New York: Cloudy, 55°F
London: Rainy, 48°F

### Calculator Example (Geometric)
User: Use the calculator to find the area of a circle with radius 3
--- Iteration 1 ---
LLM wants to call 1 tool(s)
Tool: calculator
Args: {'operation': 'area_circle', 'operands': [3]}
Result: {"result": 28.274333882308138}
--- Iteration 2 ---
Assistant: The area of a circle with radius 3 is approximately 28.27.

## Forcing the Model to Use Tools

Some smaller models (e.g., `gpt-4.1-mini`) attempt to perform calculations internally instead of calling the calculator tool.

### Strategies Tested

### 1. System Prompt Enforcement
Add a strict rule:
> "All mathematical operations must use the calculator tool. Do not compute results yourself."

### 2. Forced Tool Selection
Change:
```python
tool_choice="auto"
To:
tool_choice={"type": "function", "function": {"name": "calculator"}}
This guarantees the model always routes through the calculator.
3. Tool-Oriented Query Framing
Rephrase prompts to:
"Use the calculator tool to compute: ..."
This significantly increases tool call reliability.
```


## Task 4. Tool Calling Experiments (LangGraph)

I ran the LangGraph tool-handling sample locally and extended it with three custom tools:
1) `calculator` (arithmetic + geometry + trig; returns JSON),
2) `count_letter` (counts occurrences of a letter in text; returns JSON),
3) `text_metrics` (custom tool: basic text stats; returns JSON).

For cleaner style, I replaced tool execution `if/else` dispatch with a `tool_map` lookup:
- `tools = [...]`
- `tool_map = {t.name: t for t in tools}`
- dispatch via `tool_map[function_name].invoke(function_args)`.

### Multi-tool in one turn
The query “Are there more i’s than s’s in ‘Mississippi riverboats’?” typically triggers two `count_letter` tool calls in a single model turn.

### Sequential chaining across iterations
The query “What is the sin of the difference between the number of i’s and s’s in ‘Mississippi riverboats’?” triggers two letter-count calls first, then uses the calculator (`subtract`, then `sin`) in a later outer-loop iteration.

### All tools
I used a combined query that required weather lookup, letter counting, text metrics, and calculator geometry/trig in one workflow. Terminal traces are included in my portfolio.

