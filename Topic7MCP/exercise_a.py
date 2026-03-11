import requests
import os
import json

# Q1: Which tool would you use to find all papers about "transformer attention mechanisms"?
# A: search_papers_by_relevance
#
# Q2: Which tool would you use to find who else published in the same area as a specific author?
# A: search_authors_by_name (then analyze their papers or related authors)

URL = "https://asta-tools.allen.ai/mcp/v1"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"]
}

payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}

resp = requests.post(URL, headers=headers, json=payload)

raw = resp.text.strip()

# MCP server returns SSE, so extract JSON from data: line
json_line = None
for line in raw.splitlines():
    if line.startswith("data:"):
        json_line = line[len("data:"):].strip()

if json_line is None:
    raise RuntimeError("No JSON data found in response")

data = json.loads(json_line)

tools = data["result"]["tools"]

for tool in tools:
    print(f"\nTool: {tool['name']}")

    desc = tool["description"].strip().split("\n")[0]
    print(f"  Description: {desc}")

    schema = tool.get("inputSchema", {})
    props = schema.get("properties", {})
    required = schema.get("required", [])

    if required:
        req_str = ", ".join(f"{p} ({props[p]['type']})" for p in required)
        print(f"  Required: {req_str}")
    else:
        print("  Required: None")

    optional = [p for p in props if p not in required]

    if optional:
        opt_str = ", ".join(f"{p} ({props[p]['type']})" for p in optional)
        print(f"  Optional: {opt_str}")