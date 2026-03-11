# Topic 7: MCP Tool Integration with Ai2 Asta

**CS 6501 — Workshop on Building AI Agents, Spring 2026**

This directory contains exercises for Topic 7, covering Model Context Protocol (MCP) and Agent-to-Agent (A2A) communication. The MCP exercises use Ai2's Asta tool server, which wraps the Semantic Scholar database of 225M+ academic papers.

---

## Table of Contents

| File | Description |
|------|-------------|
| [exercise_a.py](#exercise-a) | Discover Asta MCP tools via `tools/list` |
| [exercise_b.py](#exercise-b) | Direct Asta tool calls — three focused drills |

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

## Key Discoveries

**SSE transport:** The Asta MCP server uses Server-Sent Events (`text/event-stream`) rather than plain JSON responses. Clients must include `Accept: application/json, text/event-stream` in request headers and parse the `data:` line from the response body.

**One result per call:** Despite accepting a `limit` parameter, `search_papers_by_relevance` and `get_citations` each return exactly one result per call. Collecting multiple results requires multiple calls with varied inputs (different keywords or date windows).

**`get_references` is absent:** The lesson plan references a `get_references` tool that does not exist in Asta's current tool list. The available tools are `get_paper`, `get_paper_batch`, `get_citations`, `search_authors_by_name`, `get_author_papers`, `search_papers_by_relevance`, `search_paper_by_title`, and `snippet_search`.

**MCP schema = OpenAI tool schema:** The `inputSchema` field in each MCP tool definition is valid JSON Schema, identical in structure to what OpenAI's function-calling API expects under `parameters`. This makes converting MCP tools to OpenAI format a direct one-to-one mapping (Exercise C).

**Response structure varies by tool:** `get_citations` wraps the result under a `"citingPaper"` key; `snippet_search` returns a list of objects with `{ score, paper, snippet: { text } }` structure; `search_papers_by_relevance` returns a flat paper dict directly. Robust parsing requires handling each shape explicitly.

---

*More exercises (C, D, and A2A) to be added as completed.*
