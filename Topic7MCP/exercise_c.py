"""
Exercise C: Asta-Powered Research Chatbot
CS 6501 - Agentic AI, Topic 7: MCP

Builds a chatbot that:
1. Fetches tool schemas dynamically from Asta MCP at startup
2. Converts them to OpenAI function-calling format
3. Lets GPT-4o mini decide which tools to call
4. Executes tool calls via MCP and feeds results back to the model
5. Loops until the model produces a final text answer
"""

import requests
import json
import os
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

SYSTEM_PROMPT = """You are a research assistant with access to Semantic Scholar,
a database of 225+ million academic papers. You can search for papers, retrieve
paper metadata, find citations, look up authors, and search text snippets.

When answering questions, use the available tools to fetch real data. Always
cite specific paper titles, authors, and years in your responses. If a tool
returns limited results (the API sometimes returns one result per call),
acknowledge this and work with what you have.
"""

# ---------------------------------------------------------------------------
# SSE / MCP helpers (carried over from exercises A & B)
# ---------------------------------------------------------------------------
def parse_sse(text: str) -> dict:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if data_str:
                return json.loads(data_str)
    raise ValueError(f"No data line found in SSE response:\n{text[:500]}")


def mcp_post(payload: dict) -> dict:
    resp = requests.post(MCP_URL, headers=asta_headers, json=payload)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "text/event-stream" in content_type or not resp.text.strip().startswith("{"):
        return parse_sse(resp.text)
    return resp.json()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------
def get_asta_tools() -> list:
    """Fetch tool schemas from MCP and convert to OpenAI function-calling format.

    MCP format:  { name, description, inputSchema }
    OpenAI format: { type: "function", function: { name, description, parameters } }
    """
    payload = {"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}}
    result = mcp_post(payload)
    tools = result["result"]["tools"]

    openai_tools = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"]
            }
        })
    return openai_tools


def call_asta_tool(name: str, arguments: dict) -> str:
    """Execute a tools/call against Asta and return the result as a string
    suitable for feeding back to the model as a tool message.
    """
    print(f"  [tool call] {name}({json.dumps(arguments)})")

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments}
    }

    try:
        result = mcp_post(payload)
    except Exception as e:
        return f"Error calling tool '{name}': {e}"

    if "error" in result:
        return f"MCP error: {result['error']}"

    content = result["result"]["content"][0]

    if result["result"].get("isError"):
        return f"Tool error: {content['text']}"

    raw = content["text"].strip()
    if not raw:
        return "Tool returned empty result."

    # Parse and pretty-print JSON for the model; truncate if very long
    try:
        parsed = json.loads(raw)
        formatted = json.dumps(parsed, indent=2)
        # Truncate to ~3000 chars to keep token cost manageable
        if len(formatted) > 3000:
            formatted = formatted[:3000] + "\n... [truncated]"
        return formatted
    except json.JSONDecodeError:
        return raw[:3000]


def chat(user_message: str, messages: list, tools: list) -> tuple[str, list]:
    """Run one full turn of the chatbot loop, handling any number of tool calls.

    Returns (final_response_text, updated_messages).
    """
    messages = messages + [{"role": "user", "content": user_message}]

    while True:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        # Always append the assistant message (with or without tool_calls)
        messages.append(msg)

        # If no tool calls, we have the final answer
        if not msg.tool_calls:
            return msg.content, messages

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            result_text = call_asta_tool(name, arguments)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_text
            })


# ---------------------------------------------------------------------------
# Main — interactive chatbot loop
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Asta Research Chatbot (GPT-4o mini + Semantic Scholar)")
    print("Type 'quit' to exit")
    print("=" * 60)

    # Fetch tools once at startup
    print("\nFetching tools from Asta MCP server...")
    tools = get_asta_tools()
    print(f"Loaded {len(tools)} tools: {[t['function']['name'] for t in tools]}\n")

    # Conversation history (shared across turns)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Suggested test queries printed for convenience
    print("Suggested queries:")
    print("  1. Find recent papers about large language model agents")
    print("  2. Who wrote Attention is All You Need and what else have they published?")
    print("  3. What papers cite the original BERT paper?")
    print("  4. Tell me about the ReAct paper and its impact")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        print()
        response, messages = chat(user_input, messages, tools)
        print(f"\nAssistant: {response}\n")
        print("-" * 60)


if __name__ == "__main__":
    main()