"""
Exercise D: Citation Network Explorer Agent
CS 6501 - Agentic AI, Topic 7: MCP

Autonomous agent that builds a "citation neighborhood" for a seed paper
and produces a structured markdown report. No human in the loop.
The LLM's only role is final report generation -- all MCP calls are
made directly by the agent in a fixed order.

Usage:
    python exercise_d.py ARXIV:2210.03629
    python exercise_d.py ARXIV:1706.03762

Note on get_references: Asta does not expose a get_references tool.
Step 2 is adapted using search_papers_by_relevance with keywords
extracted from the seed paper's abstract, plus snippet_search to find
foundational works cited in the literature around this paper.
"""

import requests
import json
import os
import sys
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MCP_URL = "https://asta-tools.allen.ai/mcp/v1"

asta_headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"]
}

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------------------------------------------------------------------------
# MCP helpers
# ---------------------------------------------------------------------------
def parse_sse(text: str) -> dict:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if data_str:
                return json.loads(data_str)
    raise ValueError(f"No data line found in SSE response:\n{text[:500]}")


def mcp_call(tool_name: str, arguments: dict) -> dict:
    """Call an Asta MCP tool and return the parsed result dict/list."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments}
    }
    resp = requests.post(MCP_URL, headers=asta_headers, json=payload)
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
        raise RuntimeError(f"Tool error: {content['text']}")

    raw = content["text"].strip()
    if not raw:
        return {}

    return json.loads(raw)


def unwrap_paper(data) -> dict:
    """Extract a single paper dict from any response shape."""
    if isinstance(data, dict):
        if "citingPaper" in data:
            return data["citingPaper"]
        if "citedPaper" in data:
            return data["citedPaper"]
        if "paperId" in data or "title" in data:
            return data
    return {}


# ---------------------------------------------------------------------------
# Step 1: Seed paper metadata
# ---------------------------------------------------------------------------
def get_seed_paper(paper_id: str) -> dict:
    print(f"[1/4] Fetching seed paper metadata: {paper_id}")
    data = mcp_call("get_paper", {
        "paper_id": paper_id,
        "fields": "title,year,authors,abstract,citationCount,referenceCount,fieldsOfStudy"
    })
    return data


# ---------------------------------------------------------------------------
# Step 2: Foundational works
# Since get_references is unavailable, we search for related foundational
# papers using keywords from the seed paper's title and abstract.
# ---------------------------------------------------------------------------
def get_foundational_works(seed: dict) -> list:
    print("[2/4] Finding foundational works (get_references unavailable -- using keyword search)")

    title = seed.get("title", "")
    abstract = seed.get("abstract", "") or ""

    # Build search keywords from the title (most signal-dense)
    # Also try the abstract's first sentence for context
    first_sentence = abstract.split(".")[0] if abstract else ""

    keywords = [
        title,
        first_sentence[:80] if first_sentence else title,
    ]

    seen_ids = set()
    # Exclude the seed paper itself
    seen_ids.add(seed.get("paperId", ""))

    foundational = []

    for kw in keywords:
        if len(foundational) >= 5:
            break
        if not kw.strip():
            continue
        try:
            data = mcp_call("search_papers_by_relevance", {
                "keyword": kw,
                "fields": "title,year,authors,abstract,citationCount",
                "limit": 5
            })
            paper = unwrap_paper(data)
            if paper:
                pid = paper.get("paperId", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    foundational.append(paper)
        except RuntimeError as e:
            print(f"  [warning] keyword search failed: {e}")

    # Fill remaining slots with snippet_search for broader coverage
    if len(foundational) < 5:
        try:
            snippets = mcp_call("snippet_search", {
                "query": title,
                "limit": 10
            })
            items = snippets if isinstance(snippets, list) else []
            for item in items:
                if len(foundational) >= 5:
                    break
                paper_info = item.get("paper", {})
                pid = paper_info.get("corpusId", "") or paper_info.get("paperId", "")
                pid = str(pid)
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    # Fetch full metadata for this paper
                    try:
                        full = mcp_call("get_paper", {
                            "paper_id": f"CorpusId:{pid}",
                            "fields": "title,year,authors,abstract,citationCount"
                        })
                        if full.get("title"):
                            foundational.append(full)
                    except RuntimeError:
                        # Fall back to snippet paper info
                        foundational.append({
                            "title": paper_info.get("title", "N/A"),
                            "year": paper_info.get("year", "N/A"),
                            "authors": paper_info.get("authors", []),
                            "abstract": item.get("snippet", {}).get("text", ""),
                            "citationCount": None
                        })
        except RuntimeError as e:
            print(f"  [warning] snippet_search failed: {e}")

    print(f"  Found {len(foundational)} foundational works")
    return foundational[:5]


# ---------------------------------------------------------------------------
# Step 3: Recent citing papers (last 3 years)
# ---------------------------------------------------------------------------
def get_recent_citations(paper_id: str) -> list:
    print("[3/4] Fetching recent citing papers (2022 onward)")

    windows = [
        "2022-01-01:2022-12-31",
        "2023-01-01:2023-06-30",
        "2023-07-01:2023-12-31",
        "2024-01-01:2024-06-30",
        "2024-07-01:2024-12-31",
        "2025-01-01:",
    ]

    seen_ids = set()
    citations = []

    for window in windows:
        if len(citations) >= 5:
            break
        try:
            data = mcp_call("get_citations", {
                "paper_id": paper_id,
                "fields": "title,year,authors,abstract",
                "limit": 5,
                "publication_date_range": window
            })
            paper = unwrap_paper(data)
            if paper:
                pid = paper.get("paperId", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    citations.append(paper)
        except RuntimeError as e:
            print(f"  [warning] citation window {window} failed: {e}")

    print(f"  Found {len(citations)} recent citing papers")
    return citations[:5]


# ---------------------------------------------------------------------------
# Step 4: Author profiles — most notable other work per author
# ---------------------------------------------------------------------------
def get_author_profiles(seed: dict) -> list:
    print("[4/4] Fetching author profiles")

    authors = seed.get("authors", [])
    seed_paper_id = seed.get("paperId", "")
    profiles = []

    for author in authors:
        author_id = author.get("authorId", "")
        author_name = author.get("name", "Unknown")
        if not author_id:
            continue

        try:
            data = mcp_call("get_author_papers", {
                "author_id": author_id,
                "paper_fields": "title,year,citationCount,abstract",
                "limit": 10
            })

            # Unwrap: could be a single paper dict or need different handling
            papers = []
            if isinstance(data, dict) and "paperId" in data:
                papers = [data]
            elif isinstance(data, list):
                papers = data
            elif isinstance(data, dict):
                papers = list(data.values()) if data else []

            # Filter out the seed paper and sort by citation count
            other_papers = [
                p for p in papers
                if isinstance(p, dict) and p.get("paperId", "") != seed_paper_id
                and p.get("title")
            ]
            other_papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)

            top_paper = other_papers[0] if other_papers else None
            profiles.append({
                "name": author_name,
                "author_id": author_id,
                "top_paper": top_paper
            })
            print(f"  {author_name}: {top_paper.get('title', 'N/A') if top_paper else 'no other papers found'}")

        except RuntimeError as e:
            print(f"  [warning] could not fetch papers for {author_name}: {e}")
            profiles.append({"name": author_name, "author_id": author_id, "top_paper": None})

    return profiles


# ---------------------------------------------------------------------------
# Report generation — LLM writes the final markdown
# ---------------------------------------------------------------------------
def generate_report(seed: dict, foundational: list, citations: list, profiles: list) -> str:
    print("\nGenerating markdown report with GPT-4o mini...")

    def paper_summary(p: dict) -> str:
        if not p:
            return "N/A"
        title = p.get("title", "N/A")
        year = p.get("year", "N/A")
        authors = p.get("authors", [])
        author_str = ", ".join(a.get("name", "") for a in authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        abstract = (p.get("abstract") or "")[:300]
        citations_count = p.get("citationCount", "N/A")
        return f"Title: {title}\nYear: {year}\nAuthors: {author_str}\nCitations: {citations_count}\nAbstract excerpt: {abstract}"

    author_profile_text = ""
    for profile in profiles:
        top = profile.get("top_paper")
        if top:
            author_profile_text += f"\nAuthor: {profile['name']}\n{paper_summary(top)}\n"
        else:
            author_profile_text += f"\nAuthor: {profile['name']}\nNo other papers found.\n"

    prompt = f"""You are writing a structured research report in markdown format.

Here is the data you have collected. Write the report exactly following the structure below.
Use only the data provided -- do not invent citations or facts.

---
SEED PAPER:
{paper_summary(seed)}

FOUNDATIONAL WORKS (related papers found via keyword search):
{chr(10).join(paper_summary(p) for p in foundational)}

RECENT CITING PAPERS (2022 onward):
{chr(10).join(paper_summary(p) for p in citations)}

AUTHOR PROFILES (each author's most-cited other work):
{author_profile_text}
---

Write a markdown report with exactly these sections:

# [Paper Title] — Citation Network Report

## Summary
One paragraph summarizing the seed paper's contribution, methodology, and significance.

## Foundational Works
For each foundational work, one bullet with title, year, authors, and a one-sentence description of its relevance.

## Recent Developments
For each citing paper, one bullet with title, year, authors, and a one-sentence description of how it builds on or relates to the seed paper.

## Author Profiles
For each author, one bullet with their name and their most notable other work (title, year, brief description).
"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a precise academic research assistant. Write clean, well-structured markdown reports based only on the data provided."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000
    )

    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python exercise_d.py <paper_id>")
        print("Example: python exercise_d.py ARXIV:2210.03629")
        sys.exit(1)

    paper_id = sys.argv[1]
    print("=" * 60)
    print(f"Citation Network Explorer Agent")
    print(f"Seed paper: {paper_id}")
    print("=" * 60 + "\n")

    # All MCP calls happen here in fixed order -- no LLM routing
    seed        = get_seed_paper(paper_id)
    foundational = get_foundational_works(seed)
    citations   = get_recent_citations(paper_id)
    profiles    = get_author_profiles(seed)

    # LLM is used only for report generation
    report = generate_report(seed, foundational, citations, profiles)

    print("\n" + "=" * 60)
    print("REPORT")
    print("=" * 60 + "\n")
    print(report)


if __name__ == "__main__":
    main()