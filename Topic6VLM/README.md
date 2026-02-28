# CS6501 – Topic 6: Vision-Language Models

---

## Table of Contents

* [Exercise 1 – Vision-Language LangGraph Chat Agent](#exercise-1--vision-language-langgraph-chat-agent)
* [Exercise 2 – Video Surveillance Agent](#exercise-2--video-surveillance-agent)
* [Directory Structure](#directory-structure)
* [System Requirements](#system-requirements)
* [Setup Instructions](#setup-instructions)
* [Running Exercise 2](#running-exercise-2)
* [Logs](#logs)
* [Conclusion](#conclusion)

---

## Directory Structure

```
topic6/
│
├── exercise1_langgraph_chat.py
├── exercise2_surveillance.py
├── surveillance.mp4
├── frames_out/
├── results.json
│
└── submission/
    ├── README.md
    └── logs/
```

---

## System Requirements

* macOS or Linux
* Python 3.9+
* Ollama installed locally (for Exercise 2)
* LLaVA model pulled via Ollama
* Python packages:

  * opencv-python
  * requests
  * tqdm
  * torch
  * transformers
  * langgraph
  * gradio

---


---

# Exercise 1 – Vision-Language LangGraph Chat Agent

## Overview

This project implements a multi-turn vision-language chat agent using:

* HuggingFace Vision-Language Model (TinyLLaVA / LLaVA)
* LangGraph for structured state management
* Gradio for interactive image + chat interface
* PyTorch for model inference

The agent allows a user to:

1. Upload an image
2. Ask questions about the image
3. Continue a multi-turn conversation with preserved context

---

## Features

* Multi-turn image-based conversation
* Structured state management via LangGraph
* Explicit context tracking
* Image resolution optimization for speed
* GPU acceleration (when available)

---

## Architecture

### Model

Default model: `bczhou/tiny-llava-v1-hf`

The model receives:

* A formatted prompt including `<image>` token
* The uploaded image
* Conversation history

---

### LangGraph State Design

The agent uses a typed state object:

```python
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    user_input: str
    assistant_output: str
    image: Optional[Image.Image]
```

Each user turn executes:

```
START → call_vlm → END
```

---

### Prompt Formatting

TinyLLaVA does not provide a chat template, so prompts are manually structured:

```
SYSTEM: You are a helpful vision-language assistant.
USER: <image>
<user question>
ASSISTANT:
```

The `<image>` placeholder token is required for proper alignment of image features.

---

### Performance Optimization

To improve inference speed:

* Images are resized if too large
* `max_new_tokens` is limited
* GPU is used when available

---

# Exercise 2 – Video Surveillance Agent

## Overview

This project implements a simple video surveillance agent using a Vision-Language Model (LLaVA via Ollama).

The system:

1. Extracts frames from a video at fixed time intervals.
2. Uses a Vision-Language Model (LLaVA) to determine whether a person is visible in each frame.
3. Computes approximate time intervals during which a person is present.
4. Saves terminal session outputs as required by the assignment.


## Setup Instructions

### Install Ollama (Exercise 2)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
ollama pull llava
```

---

### Install Python Dependencies

```bash
python -m pip install opencv-python requests tqdm torch transformers langgraph gradio
```

---

## Running Exercise 2

```bash
python topic6/exercise2_surveillance.py --video surveillance.mp4 --every 2
```

---

## Logs

All terminal sessions were saved using:

```bash
command 2>&1 | tee submission/logs/filename.txt
```

---

## Conclusion

Exercise 1 demonstrates structured multi-turn multimodal interaction using LangGraph and a HuggingFace Vision-Language Model.

Exercise 2 demonstrates how a Vision-Language Model can be integrated into a simple surveillance pipeline to detect human presence over time.

Together, these exercises showcase multimodal reasoning, structured agent design, and practical deployment of vision-language systems.
