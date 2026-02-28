# Vision-Language LangGraph Chat Agent

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

This ensures:

* Clean separation of logic
* Explicit context passing
* No hidden global state

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

If the program runs slowly:

* Switch Colab runtime to GPU (T4 recommended)
* Reduce image resolution
* Lower generation token count

---


## Conclusion

This project demonstrates:

* Proper LangGraph-based agent structure
* Correct multimodal prompt construction
* Context-aware multi-turn conversation
* Efficient deployment in Colab

The system satisfies the requirements of Exercise 1.
