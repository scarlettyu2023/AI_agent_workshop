"""
Exercise B: Direct Asta Tool Calls - Three Focused Drills
CS 6501 - Agentic AI, Topic 7: MCP

Drill 1 - search_papers_by_relevance: Find recent LLM agent papers
Drill 2 - get_citations: Trace impact of the BERT paper (2023+)
Drill 3 - get_paper + snippet_search: ReAct paper context
         (Note: Asta has no get_references tool)
"""

import requests
import json
import os

MCP_URL = "https://asta-tools.allen.ai/mcp/v1"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"]
}


def parse_sse(text: str) -> dict:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if data_str:
                return json.loads(data_str)
    raise ValueError(f"No data line found in SSE response:\n{text[:500]}")


def call_tool(tool_name: str, arguments: dict, call_id: int = 1):
    payload = {
        "jsonrpc": "2.0",
        "id": call_id,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments}
    }
    resp = requests.post(MCP_URL, headers=headers, json=payload)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "text/event-stream" in content_type or not resp.text.strip().startswith("{"):
        result = parse_sse(resp.text)
    else:
        result = resp.json()

    if "error" in result:
        raise RuntimeError(f"MCP error: {result['error']}")

    content = result["result"]["content"][0]
    if result["result"].get("isError"):
        raise RuntimeError(f"Tool error from '{tool_name}':\n{content['text']}")

    raw_text = content["text"].strip()
    if not raw_text:
        raise RuntimeError(f"Empty response from '{tool_name}'")

    return json.loads(raw_text)


def discover_tools() -> dict:
    payload = {"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}}
    resp = requests.post(MCP_URL, headers=headers, json=payload)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "text/event-stream" in content_type or not resp.text.strip().startswith("{"):
        result = parse_sse(resp.text)
    else:
        result = resp.json()
    return {t["name"]: t for t in result["result"]["tools"]}


def extract_papers(data) -> list:
    """Normalize any response shape into a flat list of paper dicts."""
    if isinstance(data, list):
        out = []
        for item in data:
            if isinstance(item, dict):
                out.append(item.get("citingPaper", item.get("citedPaper", item)))
        return out
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return extract_papers(data["data"])
        if "citingPaper" in data:
            return [data["citingPaper"]]
        if "citedPaper" in data:
            return [data["citedPaper"]]
        if "paperId" in data or "title" in data:
            return [data]
    return []


# ---------------------------------------------------------------------------
# Drill 0: Discover tools
# ---------------------------------------------------------------------------
def drill_0():
    print("=" * 60)
    print("DISCOVERY: tools/list - actual parameter names")
    print("=" * 60)
    tools = discover_tools()
    for name, t in tools.items():
        props = t.get("inputSchema", {}).get("properties", {})
        required = t.get("inputSchema", {}).get("required", [])
        print(f"\nTool: {name}")
        for pname, pinfo in props.items():
            req = " (required)" if pname in required else ""
            print(f"  Param: {pname} [{pinfo.get('type','?')}]{req}")
    return tools


# ---------------------------------------------------------------------------
# Drill 1: search_papers_by_relevance - LLM agent papers
# ---------------------------------------------------------------------------
def drill_1():
    print("\n" + "=" * 60)
    print("DRILL 1: search_papers_by_relevance - LLM Agent Papers")
    print("=" * 60)

    seen_ids = set()
    papers = []
    keywords = [
        "large language model agents",
        "LLM autonomous agents planning",
        "LLM tool use agents benchmark",
        "agentic AI language model reasoning",
        "multi-agent LLM framework",
    ]

    for kw in keywords:
        if len(papers) >= 5:
            break
        data = call_tool("search_papers_by_relevance", {
            "keyword": kw,
            "fields": "title,year,authors",
            "limit": 5
        })
        for p in extract_papers(data):
            pid = p.get("paperId", p.get("title", ""))
            if pid not in seen_ids:
                seen_ids.add(pid)
                papers.append(p)
                if len(papers) >= 5:
                    break

    print(f"\nFound {len(papers)} papers:\n")
    for i, paper in enumerate(papers, start=1):
        title = paper.get("title", "N/A")
        year = paper.get("year", "N/A")
        authors = paper.get("authors", [])
        author_names = ", ".join(a.get("name", "") for a in authors[:3])
        if len(authors) > 3:
            author_names += " et al."
        print(f"{i}. {title} ({year})")
        print(f"   Authors: {author_names}")


# ---------------------------------------------------------------------------
# Drill 2: get_citations - BERT paper (2023 onward)
# Asta returns one citation per call; use different well-known papers that
# cite BERT to work around the single-result limitation, then fall back to
# raw debug if still empty.
# ---------------------------------------------------------------------------
def drill_2():
    print("\n" + "=" * 60)
    print("DRILL 2: get_citations - BERT Paper (2023 onward)")
    print("=" * 60)
    # get_citations returns exactly one citing paper per call (a dict with
    # key "citingPaper"). We collect up to 10 distinct papers by querying
    # narrow quarterly windows across 2023-2025 to force different results.
    windows = [
        "2023-01-01:2023-03-31", "2023-04-01:2023-06-30",
        "2023-07-01:2023-09-30", "2023-10-01:2023-12-31",
        "2024-01-01:2024-03-31", "2024-04-01:2024-06-30",
        "2024-07-01:2024-09-30", "2024-10-01:2024-12-31",
        "2025-01-01:2025-06-30", "2025-07-01:",
    ]
    seen_ids = set()
    all_citations = []
    for window in windows:
        if len(all_citations) >= 10:
            break
        try:
            raw = call_tool("get_citations", {
                "paper_id": "ARXIV:1810.04805",
                "fields": "title,year,authors",
                "limit": 10,
                "publication_date_range": window
            })
            papers = extract_papers(raw)
            for p in papers:
                pid = p.get("paperId", p.get("title", ""))
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    all_citations.append(p)
        except RuntimeError as e:
            print(f"  [skipped {window}: {e}]")
    print(f"\nCollected {len(all_citations)} citing papers (2023+):\n")
    for i, paper in enumerate(all_citations[:10], start=1):
        title = paper.get("title", "N/A")
        year = paper.get("year", "N/A")
        authors = paper.get("authors", [])
        author_names = ", ".join(a.get("name", "") for a in authors[:3])
        if len(authors) > 3:
            author_names += " et al."
        print(f"  {i}. {title} ({year})")
        print(f"     Authors: {author_names}")

def drill_3():
    print("\n" + "=" * 60)
    print("DRILL 3: get_paper + snippet_search - ReAct Paper Context")
    print("  (Asta has no get_references; using get_paper + snippet_search)")
    print("=" * 60)

    # Step 1: full metadata
    data = call_tool("get_paper", {
        "paper_id": "ARXIV:2210.03629",
        "fields": "title,year,authors,abstract,citationCount,referenceCount,fieldsOfStudy"
    })

    print(f"\nTitle:       {data.get('title', 'N/A')}")
    print(f"Year:        {data.get('year', 'N/A')}")
    print(f"Citations:   {data.get('citationCount', 'N/A')}")
    print(f"References:  {data.get('referenceCount', 'N/A')}")
    fields = data.get("fieldsOfStudy") or []
    print(f"Fields:      {', '.join(fields)}")
    authors = data.get("authors", [])
    print(f"Authors:     {', '.join(a.get('name','') for a in authors)}")
    abstract = data.get("abstract", "")
    if abstract:
        print(f"\nAbstract:\n  {abstract[:350]}...")

    # Step 2: snippet_search
    # Structure: list of { score, paper: {title, ...}, snippet: {text: "..."} }
    print("\n--- snippet_search: 'ReAct reasoning acting language model' ---\n")
    snippets_data = call_tool("snippet_search", {
        "query": "ReAct reasoning acting language model",
        "limit": 5
    })

    items = snippets_data if isinstance(snippets_data, list) else []
    if isinstance(snippets_data, dict):
        for key in ("data", "results", "snippets", "items"):
            if key in snippets_data:
                items = snippets_data[key]
                break

    for i, item in enumerate(items[:5], start=1):
        if not isinstance(item, dict):
            continue

        paper_info = item.get("paper", {})
        paper_title = paper_info.get("title", "N/A") if isinstance(paper_info, dict) else "N/A"
        score = item.get("score", "N/A")

        # Snippet text is at item["snippet"]["text"]
        snippet_obj = item.get("snippet", {})
        if isinstance(snippet_obj, dict):
            snippet_text = snippet_obj.get("text", "N/A")
        elif isinstance(snippet_obj, str):
            snippet_text = snippet_obj
        else:
            snippet_text = "N/A"

        print(f"  {i}. From: {paper_title}")
        print(f"     Score: {score:.3f}" if isinstance(score, float) else f"     Score: {score}")
        print(f"     \"{snippet_text[:200]}\"")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tools = drill_0()
    drill_1()
    drill_2()
    drill_3()