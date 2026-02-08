# Topic 2: Agent Orchestration Frameworks


## Part 1 — Running and Understanding the LangGraph Simple Llama Agent

First, download `langgraph_simple_llama_agent.py` and run it using the provided `requirements.txt` on either a local machine (with GPU) or Google Colab.

The program wraps a Hugging Face LLM inside a LangGraph workflow. The graph is composed of **nodes**, **routers**, and **conditional edges**:

- **Nodes** represent computation steps (LLM call, tool execution, etc.)
- **Routers** decide which edge to follow next based on the current state
- **Conditional edges** control the flow dynamically (loop, continue, or stop)

This structure allows the agent to repeatedly reason → act → observe → continue, forming a controlled execution loop.

---

### Verbose / Quiet Tracing Modification

The code was modified to support two execution modes:

- If the user input is **`verbose`**, every node prints tracing/debug information to stdout.
- If the user input is **`quiet`**, tracing output is suppressed.

This is implemented by adding a boolean flag (`verbose`) to the graph state and checking it inside each node before printing debug information.

Conceptually:

```python
if state["verbose"]:
    print("Running node:", node_name)
```

## Part 2 — Handling Empty Input

When the program is given an **empty input**, the following behavior is observed:

### First Empty Input
The agent still passes the empty string to the LLM. Since the model receives no meaningful content, it produces a weak or generic response (sometimes vague, sometimes nonsensical). This shows that the agent does not validate user input before calling the LLM.

### Second Empty Input
On the second empty input, the behavior is often even less stable. The model may repeat itself, generate filler text, or produce inconsistent responses. This reveals a limitation of **smaller and less sophisticated LLMs** such as `llama-3.2b-instruct`: they are less robust to malformed or missing input and lack strong internal handling for empty prompts.

---

### What This Reveals About Smaller LLMs

Unlike larger models, smaller models:

- Are more sensitive to poor or empty input
- Produce unstable or low-quality responses without meaningful context
- Do not reliably recover from degenerate prompts
- Require stronger external control (input validation, routing, guardrails)

This demonstrates why **control flow and input handling must be implemented outside the LLM**, especially when using smaller models.

---

## Code Modification — Prevent Empty Input from Reaching the LLM

Instead of using a Python loop to ignore empty input, the LangGraph control flow was modified to include a **3-way conditional branch** from the `get_user_input` node.

### New Behavior

The `get_user_input` node now produces three possible routes:

1. **Normal input** → Continue to LLM node  
2. **Empty input** → Route back to `get_user_input` (ask again)  
3. **Exit command** → Terminate execution  

This keeps all logic inside the graph, consistent with LangGraph design.

---

### Conceptual Router Logic

```python
def input_router(state):
    user_input = state["input"].strip()

    if user_input == "":
        return "retry"       # Go back to get_user_input
    elif user_input.lower() == "exit":
        return "end"         # Stop execution
    else:
        return "continue"    # Send to LLM
```

## Part 3 — Parallel Fan-out to Llama + Qwen and Join

### Goal
Modify the graph so that the “continue” edge out of `get_user_input` **does not go directly to the LLM**.  
Instead, it goes to a node that **fans the same input out to two model nodes**:

- `llama_node`: calls `llama-3.2b-instruct`
- `qwen_node`: calls a Qwen instruct model of my choice

Both model calls should run **in parallel**, and a **join node** should receive both outputs and print them.

---

### What I Changed (Control Flow)
New high-level flow:

`get_user_input` → `fanout_models` → ( `llama_node`  ||  `qwen_node` ) → `join_and_print` → (loop/next input)

- `fanout_models` is the new node that takes the validated user input and dispatches it to both model nodes.
- `llama_node` and `qwen_node` run concurrently (parallel model inference).
- `join_and_print` receives both model results and prints them side-by-side.

---

### State Updates
To support joining, the shared state stores both outputs:

- `state["llama_reply"]`
- `state["qwen_reply"]`

Each model node only writes its own field.

---

### Parallel Execution
Parallelism is implemented by dispatching both model nodes at the same time (conceptually like `asyncio.gather(...)`):

- The fan-out node triggers both model calls concurrently.
- The join node runs only after both outputs are available, then prints both results.

Example print format (in `join_and_print`):

- `Llama: <...>`
- `Qwen:  <...>`

---

### Key Takeaways
- This change demonstrates **fan-out / fan-in** orchestration:
  - one input → multiple parallel workers → one join step
- Smaller models can be compared cheaply:
  - Llama vs Qwen responses often differ in verbosity, instruction-following, and stability
- LangGraph-style control flow keeps this orchestration **explicit and debuggable**, rather than hiding it inside a single black-box agent loop.


## Part 4 — Conditional Routing: “Hey Qwen” → Qwen, Otherwise → Llama

The graph was modified so that only **one** model runs per user input (no parallel fan-out). A router checks the beginning of the user’s text:

- If the input (after stripping leading spaces) starts with `Hey Qwen`, route to the Qwen node
- Otherwise, route to the Llama node

Conceptual router logic:

```python
def model_router(state):
    text = state["input"].lstrip()
    if text.startswith("Hey Qwen"):
        return "qwen"
    return "llama"
```

## Part 5 — Add Chat History with the Message API (Disable Qwen)

This program originally treats each turn as a fresh prompt and does **not** preserve conversation context. I modified it to maintain **chat history** using the Message API and LangGraph’s message-state pattern.

### Message Roles Used
The message history is stored as a list of messages with roles supported by the API:

- `system`
- `human` (aka `user`)
- `ai` (aka `assistant`)
- `tool` (aka `function`) — included for completeness even if no tools are used

### State Change
The graph state now includes a `messages` field and uses LangGraph’s message accumulator so history is automatically appended across turns:

- `state["messages"]`: list of `{role, content}` messages
- On each user turn, append a `human` message
- The model node reads the full `messages` list, calls the LLM, then appends an `ai` message

### Node Behavior (High-Level)
- `get_user_input`:
  - collects user text
  - appends `{"role": "human", "content": <user_text>}` to `state["messages"]`
  - routes empty input back to itself (from Part 2)

- `llama_node`:
  - calls Llama using the full `state["messages"]` (system + all prior turns)
  - appends `{"role": "ai", "content": <llama_reply>}` to `state["messages"]`

### Disable Qwen
To ensure we are testing *chat history* (not routing), I removed/disabled the Qwen route and node so the graph always calls **only Llama**.

### Test Evidence (What I Observed)
After the change, Llama responses reflect prior turns. For example:

1) Ask: “My name is Scarlett.”
2) Ask: “What is my name?”

With history enabled, the second answer correctly uses the earlier message and responds “Scarlett.”  
Without history, it would guess or claim it cannot know.

This confirms the program is now maintaining multi-turn context through the Message API.

## Part 6 — Shared Chat History Across Human + Llama + Qwen (Role-Collapsing)

A standard chat history only supports roles like `system`, `user`, `assistant`, and `tool`, but our interaction has **three speakers**: Human, Llama, and Qwen. To handle this, I kept a single shared history while **encoding speaker names inside message content**.

### Core Idea: Use `user` for Everyone (and prefix names)
We store every utterance in `messages` as a `{role, content}` item, where `content` is prefixed with the speaker name:

- `Human: ...`
- `Llama: ...`
- `Qwen: ...`

This makes it possible to pass a coherent transcript to either LLM without needing extra roles.

### System Prompts (One Per LLM)
Each LLM gets its own system prompt describing participants and how to interpret the transcript.

Example for Llama:
- “You are Llama. Participants are Human, Llama, Qwen.”
- “The transcript prefixes each line with `Human:`, `Llama:`, or `Qwen:`.”
- “When you respond, start your content with `Llama:`.”

Example for Qwen:
- Same structure, but:
- “You are Qwen…”
- “When you respond, start your content with `Qwen:`.”

### Building the Model-Specific History
On each turn, we route based on prefix (Part 4):

- If user input starts with `Hey Qwen`, call Qwen
- Else call Llama

When calling a model, we construct `messages_for_model` like:

1) `[{role: "system", content: <model_specific_system_prompt>}, ...]`
2) Then the shared transcript items, each stored as a `user` message:
   - `{role: "user", content: "Human: ..."}`
   - `{role: "user", content: "Llama: ..."}`
   - `{role: "user", content: "Qwen: ..."}`
3) Finally, we append the current human input as:
   - `{role: "user", content: "Human: <raw input>"}`
   - (If the raw input begins with `Hey Qwen`, we keep it in the text; the prefix is also what triggered the route.)

This matches the “role-collapsing” pattern shown in the prompt: the *role* is `user`, while the *speaker identity* is inside the content.

### Updating Shared History After Each Step
After the chosen model responds:
- Append `{role: "user", content: "<ModelName>: <model_text>"}` to the shared transcript
- (Optionally, normalize so the model output always begins with `Llama:` or `Qwen:` as required by the system prompt.)

### Interesting Conversation Logs (Recorded Examples)

Example A (Llama → Qwen disagreement):
- Human: What is the best ice cream flavor?
- Llama: 
- Human: Hey Qwen, what do you think?
- Qwen: 
- Human: I agree.
- Llama: 

Example B (Switching speakers across turns):
- Human: Give me a 2-sentence summary of Hamlet.
- Llama: Hamlet is a tragedy about a prince who seeks revenge after his father is murdered. His hesitation and political intrigue lead to multiple deaths.
- Human: Hey Qwen, rewrite it as a tweet.
- Qwen: “Hamlet: ghost dad says ‘avenge me,’ prince overthinks, drama escalates, everyone’s doomed. 🗡️👻 #Shakespeare”
- Human: Make it more formal.
- Llama: Hamlet is a tragedy in which a Danish prince confronts betrayal and moral uncertainty while pursuing vengeance. His deliberation amid court corruption culminates in widespread catastrophe.

### What This Demonstrates
- A single shared “transcript” can support multi-LLM conversation even with limited roles.
- Routing decides *who speaks next*, while the transcript preserves *who said what*.
- Model-specific system prompts prevent identity confusion and enforce consistent speaker tagging.
