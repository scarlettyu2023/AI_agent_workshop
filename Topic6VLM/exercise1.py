"""
Exercise 1: Vision-Language LangGraph Chat Agent
================================================
Compatible with Gradio 5.x (uses ChatMessage format).

Install dependencies:
    pip install ollama langgraph langchain-core gradio pillow

Start Ollama and pull the model:
    ollama pull llava
"""

import base64
import io
from typing import Annotated, Optional
from typing_extensions import TypedDict

import ollama
from PIL import Image

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

import gradio as gr
from gradio import ChatMessage


# ---------------------------------------------------------------------------
# State definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    image_b64: Optional[str]
    response: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image(pil_image: Image.Image, max_side: int = 1024) -> str:
    """Resize and base64-encode a PIL image as JPEG."""
    w, h = pil_image.size
    scale = min(max_side / w, max_side / h, 1.0)
    if scale < 1.0:
        pil_image = pil_image.resize(
            (int(w * scale), int(h * scale)), Image.LANCZOS
        )
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def call_llava(state: AgentState) -> AgentState:
    """Send conversation history + image to LLaVA, return its reply."""
    ollama_messages = []
    for i, msg in enumerate(state["messages"]):
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        entry = {"role": role, "content": msg.content}
        # Attach image only to the very first user message
        if i == 0 and role == "user" and state.get("image_b64"):
            entry["images"] = [state["image_b64"]]
        ollama_messages.append(entry)

    result = ollama.chat(model="llava", messages=ollama_messages)
    return {"response": result["message"]["content"]}


def append_assistant_reply(state: AgentState) -> AgentState:
    return {"messages": [AIMessage(content=state["response"])]}


# ---------------------------------------------------------------------------
# Build the LangGraph graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("llava", call_llava)
    graph.add_node("append_reply", append_assistant_reply)
    graph.set_entry_point("llava")
    graph.add_edge("llava", "append_reply")
    graph.add_edge("append_reply", END)
    return graph.compile()


APP = build_graph()


# ---------------------------------------------------------------------------
# Gradio chat function
# ---------------------------------------------------------------------------

def chat(user_message: str, uploaded_image, history: list, state: dict):
    if not user_message.strip():
        return history, "", state

    # Encode new image if provided
    if uploaded_image is not None:
        state["image_b64"] = encode_image(uploaded_image)
        state["history_pairs"] = []

    if not state.get("image_b64"):
        history = history + [
            ChatMessage(role="user", content=user_message),
            ChatMessage(role="assistant", content="⚠️ Please upload an image first."),
        ]
        return history, "", state

    # Rebuild LangChain message list from stored pairs
    lc_messages = []
    for u, a in (state.get("history_pairs") or []):
        lc_messages.append(HumanMessage(content=u))
        lc_messages.append(AIMessage(content=a))
    lc_messages.append(HumanMessage(content=user_message))

    agent_state: AgentState = {
        "messages": lc_messages,
        "image_b64": state["image_b64"],
        "response": "",
    }

    result = APP.invoke(agent_state)
    reply = result["response"]

    pairs = state.get("history_pairs") or []
    pairs.append((user_message, reply))
    state["history_pairs"] = pairs

    history = history + [
        ChatMessage(role="user", content=user_message),
        ChatMessage(role="assistant", content=reply),
    ]
    return history, "", state


def reset_conversation(state: dict):
    return [], {}, gr.update(value=None)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="VLM Chat Agent — Exercise 1") as demo:
    gr.Markdown(
        "## 🖼️ Vision-Language Chat Agent\n"
        "Upload an image, then ask questions about it in a multi-turn conversation."
    )

    with gr.Row():
        image_input = gr.Image(label="Upload Image", type="pil", scale=1)
        chatbot = gr.Chatbot(label="Conversation", scale=2, height=480)

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="Ask something about the image…",
            label="Your message",
            scale=4,
        )
        send_btn = gr.Button("Send", scale=1, variant="primary")

    reset_btn = gr.Button("🔄 Reset conversation")
    agent_state = gr.State({})

    send_btn.click(
        chat,
        inputs=[msg_box, image_input, chatbot, agent_state],
        outputs=[chatbot, msg_box, agent_state],
    )
    msg_box.submit(
        chat,
        inputs=[msg_box, image_input, chatbot, agent_state],
        outputs=[chatbot, msg_box, agent_state],
    )
    reset_btn.click(
        reset_conversation,
        inputs=[agent_state],
        outputs=[chatbot, agent_state, image_input],
    )

if __name__ == "__main__":
    demo.launch()