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

## Part 6 — Integrating Chat History with Llama/Qwen Switching (3 Speakers, 4 Roles)

This agent maintains a single shared transcript while allowing switching between two LLMs (Llama and Qwen). The challenge is that the chat message API only supports roles: `system`, `user`/`human`, `assistant`/`ai`, and `tool`, but our dialog has three “speakers” (Human, Llama, Qwen).

### Role-collapsing strategy (speaker name in content)
To represent 3 speakers using a 4-role API, the implementation stores the conversation as a transcript where each line is a `user` message but the `content` includes a speaker prefix:

- `Human: ...`
- `Llama: ...`
- `Qwen: ...`

When calling either model, we pass:
1) A model-specific `system` message explaining the participants and the transcript format.
2) The transcript messages as `user` role entries with speaker tags in the text.

This matches the pattern from the prompt, e.g. Qwen receives prior context like:
- `Human: ...`
- `Llama: ...`

and Llama similarly receives:
- `Human: ...`
- `Qwen: ...`
- `Human: ...`

### Model-specific system prompts
Each model receives a different system prompt stating:
- who the participants are (Human, Llama, Qwen),
- that each turn in the transcript is prefixed with the speaker name,
- and that the model must respond with its own prefix (either `Llama:` or `Qwen:`).

### Switching (routing) policy
Routing is controlled by a prefix rule:
- If input begins with `Hey Qwen,` → route to Qwen
- If input begins with `Hey Llama,` → route to Llama
- Otherwise → default to Llama

This makes the control flow explicit and keeps switching behavior outside the LLM.

### Recorded conversations (evidence)
Below are excerpts demonstrating history + switching:

1) Human asks Llama:
- Human: What is the best ice cream flavor?
- Llama: ... chocolate chip cookie dough ...

2) Human switches to Qwen:
- Human: Hey Qwen, what do you think?
- Qwen: ... chocolate chip cookie dough ...

3) Prefix routing is strict:
- Human: Hi Qwen, as a llm, do you like llama?
Because this does not start with `Hey Qwen`, it defaults to Llama. This shows the router is deterministic but brittle to trigger phrasing.

4) Small-model identity confusion can occur:
- Qwen: “As a llama, I believe that llamas are kind...”
This illustrates that smaller LLMs can sometimes drift in speaker identity even with transcript tagging, motivating stronger system prompts and stricter output normalization.

The final dialog also demonstrates multi-turn context being preserved across speaker switches (e.g., Llama commenting on what Qwen said, and continuing the same topic).

## Part 7 — Checkpointing and Crash Recovery

LangGraph provides built-in **checkpointing** that allows an agent to recover from crashes without losing progress. Instead of storing conversation state only in memory, the graph persists its state to durable storage after each step. If the process is interrupted, the agent can restart from the last checkpoint.

### Implementation

A persistent SQLite checkpointer was added to the graph:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
graph = builder.compile(checkpointer=checkpointer)
```

Each conversation uses a stable `thread_id` so that all checkpoints belong to the same conversation timeline:

```python
config = {"configurable": {"thread_id": "conversation-001"}}
```

As long as `checkpoints.db` and the `thread_id` remain unchanged, the agent will resume from the saved state.

---

### Crash Recovery Test

The following experiment was performed:

1. Start the agent and begin a conversation.
2. Enter several messages so that chat history is created.
3. Kill the program during or after a conversation turn.
4. Restart the program **without deleting `checkpoints.db` or `.thread_id`**.
5. Continue the conversation.

---

### Observed Output (Evidence)

After restarting, the agent resumed with full memory:

* The agent still remembered the user’s name (“Scarlett Yu”).
* The agent remembered the previously mentioned course (“CS6501”).
* The conversation continued seamlessly with no reset.

Example excerpt:

```
[thread_id=4a4260f5-12d5-4d47-b59c-3133d44859f1]
Checkpoint DB: checkpoints.db

> what is my name?
Llama: nice to meet you again, Scarlett Yu! Your name is Scarlett Yu.

> which class i am studying?
Llama: ... you mentioned earlier that you were studying CS6501 ...
```

This confirms that the conversation state was restored from the checkpoint database rather than memory.

---

### What This Demonstrates

* Checkpoints persist graph state to durable storage.
* The agent can recover after crash or forced termination.
* Conversation history is preserved across restarts.
* LangGraph enables fault-tolerant, long-running agent execution.



