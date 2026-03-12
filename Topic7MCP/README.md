# Topic 7: MCP Tool Integration with Ai2 Asta

**CS 6501 — Workshop on Building AI Agents, Spring 2026**

This directory contains exercises for Topic 7, covering Model Context Protocol (MCP) and Agent-to-Agent (A2A) communication. The MCP exercises use Ai2's Asta tool server, which wraps the Semantic Scholar database of 225M+ academic papers.

---

## Table of Contents

| File | Description |
|------|-------------|
| [exercise_a.py](#exercise-a) | Discover Asta MCP tools via `tools/list` |
| [exercise_b.py](#exercise-b) | Direct Asta tool calls — three focused drills |
| [exercise_c.py](#exercise-c) | Asta-powered research chatbot with GPT-4o mini |
| [exercise_d.py](#exercise-d) | Citation network explorer agent — autonomous MCP pipeline |

---

## Exercise A

**File:** `exercise_a.py`

**Goal:** Interrogate the Asta MCP server directly using a `tools/list` JSON-RPC request, then print each tool's name, description, required parameters, and optional parameters.

**What it does:**
- POSTs a `tools/list` JSON-RPC message to `https://asta-tools.allen.ai/mcp/v1`
- Parses the SSE (Server-Sent Events) response by extracting the `data:` line
- Iterates over all tools and prints their schema in a readable format

**Key discovery:** The Asta server responds with SSE format (`text/event-stream`) rather than plain JSON, so the `Accept` header must include `text/event-stream` and the response body must be parsed by finding the `data:` prefixed line.

**Tools discovered on the Asta server:**

| Tool | Required Params | Purpose |
|------|----------------|---------|
| `get_paper` | `paper_id` | Fetch full metadata for a paper by ID |
| `get_paper_batch` | `ids` | Fetch metadata for multiple papers at once |
| `get_citations` | `paper_id` | Get papers that cite a given paper |
| `search_authors_by_name` | `name` | Search for authors by name |
| `get_author_papers` | `author_id` | Get all papers by a specific author |
| `search_papers_by_relevance` | `keyword` | Keyword/semantic search over 225M+ papers |
| `search_paper_by_title` | `title` | Search for a paper by its title |
| `snippet_search` | `query` | Find text snippets matching a query |

**Answers to lesson plan questions:**
- *Which tool to find all papers about "transformer attention mechanisms"?* → `search_papers_by_relevance` (using `keyword="transformer attention mechanisms"`)
- *Which tool to find who else published in the same area as a specific author?* → `search_authors_by_name` to find the author, then `get_author_papers` to retrieve their work, then `search_papers_by_relevance` with related keywords to find others in the field

**Sample output:**
```
Tool: search_papers_by_relevance
  Description: Search for papers by keyword relevance.
  Required: keyword (string)
  Optional: fields (string), limit (integer), publication_date_range (string), venues (string)
```

---

## Exercise B

**File:** `exercise_b.py`

**Goal:** Call three specific Asta tools directly (without a chatbot) to practice different access patterns.

### Drill 1 — `search_papers_by_relevance`

Searches for recent papers about large language model agents. Because the API returns exactly one paper per call regardless of the `limit` parameter, the drill issues five calls with varied keyword phrasings and deduplicates results by `paperId`.

**Sample output:**
```
1. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents (2024)
   Authors: Qiusi Zhan, Zhixiang Liang, Zifan Ying et al.
2. From Prompt to Action: A Comprehensive Review of LLM Autonomous Agents (2025)
   Authors: Zainab Rafique, Muhammad Wasim, Mudassar Hussain et al.
...
```

### Drill 2 — `get_citations`

Fetches papers that cite the original BERT paper (`ARXIV:1810.04805`), filtered to 2023 onward. Like Drill 1, the API returns one result per call, so the drill queries 10 narrow quarterly date windows (`2023 Q1` through `2025 H2`) to collect 10 distinct citing papers.

**Sample output:**
```
1. Quantifying the Academic Quality of Children's Videos using Machine Comprehension (2023)
2. MEMD-ABSA: a multi-element multi-domain dataset for aspect-based sentiment analysis (2023)
...
10. Enhancing large language models for knowledge graph question answering (2026)
```

### Drill 3 — `get_paper` + `snippet_search`

The lesson plan specified `get_references` for this drill, but **Asta does not expose a `get_references` tool**. This drill adapts by using two available tools instead:

1. `get_paper` on the ReAct paper (`ARXIV:2210.03629`) to retrieve full metadata including title, year, citation count, reference count, fields of study, authors, and abstract.
2. `snippet_search` to find passages in the literature that discuss and describe the ReAct framework, showing how other papers cite and characterize it.

**ReAct paper metadata retrieved:**
- Title: *ReAct: Synergizing Reasoning and Acting in Language Models*
- Year: 2022 | Citations: 6,281 | References: 63
- Authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao

**Sample snippet result:**
```
From: Exploring Advanced Large Language Models with LLMsuite (score: 0.633)
"ReAct, developed by researchers from Princeton University and Google, stands for
Reasoning + Acting. This framework enhances large language models by combining
reasoning with actionable outputs..."
```

---

## Exercise C

**File:** `exercise_c.py`

**Goal:** Build a research chatbot that fetches Asta tool schemas dynamically at startup and uses GPT-4o mini to decide which tools to call, executing them via MCP and looping until a final answer is produced.

**Architecture:**
1. At startup, call `tools/list` and convert all 8 Asta schemas to OpenAI function-calling format (renaming `inputSchema` -> `parameters`)
2. Pass the tools to GPT-4o mini with `tool_choice="auto"`
3. When the model emits `tool_calls`, extract name and arguments, POST a `tools/call` to Asta, and append the result as a `tool` message
4. Loop until the model returns a plain text response with no tool calls
5. Maintain full conversation history across turns for multi-turn context

**Key design decisions:**
- Tool results are truncated to ~3000 characters before being fed back to the model to keep token costs manageable
- Errors from MCP (tool errors, network failures) are returned as strings so the model can acknowledge them gracefully rather than crashing
- All tool calls are printed as `[tool call] name(args)` so the model's decisions are observable

**Sample session:**

Query: *"Who wrote Attention is All You Need and what else have they published?"*

The model automatically chained 9 tool calls: one `search_papers_by_relevance` to find the paper and extract author IDs, then one `get_author_papers` call per author. No tool-routing code was written -- the model read the schemas and decided the sequence on its own.

Query: *"Tell me about the ReAct paper and its impact"*

The model called four different tools unprompted: `search_paper_by_title`, `search_papers_by_relevance` (twice with different keywords), and `snippet_search` -- demonstrating that it can select from the full tool palette based on what each query needs.

**Observed limitation:** When asked about BERT citations, the model searched for "BERT" by keyword and retrieved a Sentence-BERT paper ID rather than the original BERT paper (`ARXIV:1810.04805`). The model guessed a paper ID from search results rather than knowing the canonical ID. This could be fixed by including known paper IDs in the system prompt.

**What changed vs Exercise B?** In Exercise B, every tool call required writing explicit code to choose the tool, pass the right parameters, and parse the response. In Exercise C, zero tool-specific code was written after startup -- the model read the MCP schemas and handled all routing decisions. Adding a new Asta tool tomorrow would require no code changes.

---

## Exercise D

**File:** `exercise_d.py`

**Goal:** Build an autonomous agent that takes a seed paper ID, collects a full citation neighborhood via direct MCP calls (no LLM tool routing), and uses GPT-4o mini solely to write a structured markdown report.

**How to run:**
```bash
python exercise_d.py ARXIV:2210.03629
```

To save the report to a file, redirect stdout:
```bash
python exercise_d.py ARXIV:2210.03629 > react_report.md
```

Other paper IDs to try:
```bash
python exercise_d.py ARXIV:1706.03762   # Attention Is All You Need
python exercise_d.py ARXIV:1810.04805   # BERT
python exercise_d.py ARXIV:2201.11903   # Chain-of-Thought Prompting
python exercise_d.py ARXIV:2005.14165   # GPT-3
```

**Architecture — 4 fixed data-collection steps, then 1 generation step:**

| Step | MCP Tool(s) Used | What It Collects |
|------|-----------------|------------------|
| 1 | `get_paper` | Seed paper title, abstract, authors, citation count, fields of study |
| 2 | `search_papers_by_relevance` + `snippet_search` + `get_paper` | 5 foundational related works (adapted from missing `get_references`) |
| 3 | `get_citations` x6 | 5 recent citing papers via quarterly date windows (2022 onward) |
| 4 | `get_author_papers` x N | Each seed paper author's most-cited other work |
| 5 | GPT-4o mini | Writes the final markdown report from all collected data |

**Key design decision:** The LLM has no role in deciding which tools to call or in what order — all MCP calls are hardcoded in sequence. The LLM only sees the final assembled data and writes prose. This is the opposite of Exercise C, where the LLM controlled all tool routing.

**Adaptation for missing `get_references`:** Since Asta does not expose a `get_references` tool, Step 2 uses `search_papers_by_relevance` with the seed paper's title as a keyword to find related foundational works, then fills remaining slots using `snippet_search` results resolved to full metadata via `get_paper`.

**Sample run — ReAct paper (`ARXIV:2210.03629`):**

```
[1/4] Fetching seed paper metadata: ARXIV:2210.03629
[2/4] Finding foundational works (get_references unavailable -- using keyword search)
  Found 1 foundational works
[3/4] Fetching recent citing papers (2022 onward)
  Found 5 recent citing papers
[4/4] Fetching author profiles
  Shunyu Yao: Referral Augmentation for Zero-Shot Information Retrieval
  Jeffrey Zhao: An Efficient Algorithm for Thresholding Monte Carlo Tree Search
  Dian Yu: Tree of Thoughts: Deliberate Problem Solving with Large Language Models
  Nan Du: Learning to Select the Best Forecasting Tasks for Clinical Outcome Prediction
  Izhak Shafran: An Efficient Algorithm for Thresholding Monte Carlo Tree Search
  Karthik Narasimhan: Reflexion: language agents with verbal reinforcement learning
  Yuan Cao: Catch Your Breath: Adaptive Computation for Self-Paced Sequence Production

Generating markdown report with GPT-4o mini...
```

**Sample report output (stdout):**

```markdown
# ReAct: Synergizing Reasoning and Acting in Language Models — Citation Network Report

## Summary
The seed paper "ReAct" (2022) explores the synergy between reasoning (e.g. chain-of-thought
prompting) and acting (e.g. action plan generation) in large language models, which had
previously been studied as separate topics...

## Foundational Works
- **CodeReviewQA** (2025) - Examines LLM limitations in practical software engineering tasks...

## Recent Developments
- **Demonstrate-Search-Predict** (2022) by Khattab et al. - Integrates retrieval mechanisms
  with language models, complementing the ReAct approach...
- **Reflexion** (2023) by Shinn et al. - Uses verbal reinforcement learning to enable
  language agents to learn from trial and error...

## Author Profiles
- **Karthik Narasimhan** - *Reflexion: language agents with verbal reinforcement learning*
  (2023) - Directly extends the ReAct line of work on language agents...
```

**Observed limitation:** Step 2 (foundational works) only found 1 paper instead of 5, because `search_papers_by_relevance` returns one result per call and the keyword search for the seed paper title returns the seed paper itself or very recent work rather than older foundational papers. A better substitute for `get_references` would require iterating over known reference IDs, which Asta does not currently expose.

**What changed vs Exercise C?** Exercise C let the LLM decide every tool call. Exercise D inverts this: all tool calls are deterministic and hardcoded, and the LLM is used only as a text generator at the end. This makes the agent more predictable and cheaper to run, but less flexible.


---

## Key Discoveries

**SSE transport:** The Asta MCP server uses Server-Sent Events (`text/event-stream`) rather than plain JSON responses. Clients must include `Accept: application/json, text/event-stream` in request headers and parse the `data:` line from the response body.

**One result per call:** Despite accepting a `limit` parameter, `search_papers_by_relevance` and `get_citations` each return exactly one result per call. Collecting multiple results requires multiple calls with varied inputs (different keywords or date windows).

**`get_references` is absent:** The lesson plan references a `get_references` tool that does not exist in Asta's current tool list. The available tools are `get_paper`, `get_paper_batch`, `get_citations`, `search_authors_by_name`, `get_author_papers`, `search_papers_by_relevance`, `search_paper_by_title`, and `snippet_search`.

**MCP schema = OpenAI tool schema:** The `inputSchema` field in each MCP tool definition is valid JSON Schema, identical in structure to what OpenAI's function-calling API expects under `parameters`. This makes converting MCP tools to OpenAI format a direct one-to-one mapping -- just renaming one key.

**Response structure varies by tool:** `get_citations` wraps the result under a `"citingPaper"` key; `snippet_search` returns a list of objects with `{ score, paper, snippet: { text } }` structure; `search_papers_by_relevance` returns a flat paper dict directly. Robust parsing requires handling each shape explicitly.

**LLM tool selection is emergent:** The model selects and sequences tools based solely on reading their names and descriptions -- no explicit routing logic is needed. For the "Attention is All You Need" query, it correctly inferred that it needed to find author IDs first before calling `get_author_papers`, and executed 9 sequential calls without any orchestration code.

---

*A2A exercises to be added as completed.*
