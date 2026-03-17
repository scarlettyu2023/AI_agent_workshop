# Topic6VLM

Vision-Language Model (VLM) exercises using Ollama + LLaVA.

## Table of Contents

| File | Description |
|------|-------------|
| [exercise1.py](#exercise-1-vision-language-langgraph-chat-agent) | Multi-turn chat agent for image Q&A |
| [exercise2_surveillance.py](#exercise-2-video-surveillance-agent) | Video surveillance agent using frame extraction |

---

## Exercise 1: Vision-Language LangGraph Chat Agent

**File:** `exercise1.py`

A multi-turn conversational agent that lets you upload an image and ask follow-up questions about it in a Gradio web interface. Built with LangGraph for state management and Ollama + LLaVA for vision-language inference.

### How it works

1. User uploads an image via the Gradio UI
2. The image is resized (max 1024px) and base64-encoded for efficient inference
3. Each user message is appended to a LangGraph `AgentState` containing the full conversation history
4. The graph passes the history to LLaVA via Ollama — the image is attached only to the first message; subsequent turns are text-only but the model retains visual context
5. The assistant reply is appended back to the state for the next turn

### LangGraph graph structure

```
[llava] → [append_reply] → END
```

### Dependencies

```bash
pip install ollama langgraph langchain-core gradio pillow
ollama pull llava
```

### Usage

```bash
python exercise1.py
# Open http://127.0.0.1:7860 in your browser
# Upload an image, then type questions in the chat box
# Press Ctrl+C in the terminal to stop the server
```

### Sample interaction

- **Image:** Three kittens on a wooden surface with green foliage background
- **User:** "What do you see in this image?"
- **LLaVA:** "The image shows three kittens: one is a solid white color, one has black and brown patches..."
- **User:** "How old do they look like?"
- **LLaVA:** "The cats in the image appear to be quite young, likely kittens..."

---

## Exercise 2: Video Surveillance Agent

**File:** `exercise2_surveillance.py`

A command-line agent that analyzes a video file for human presence by extracting frames every N seconds and querying LLaVA on each one. Reports the time intervals during which a person is visible.

### How it works

1. Frames are extracted from the video at a configurable interval (default: every 2 seconds) using OpenCV and saved as JPEG files
2. Each frame is base64-encoded and sent to LLaVA via the Ollama REST API with a yes/no presence prompt
3. Responses are parsed by a scoring-based `normalize_yes_no()` function that handles LLaVA's verbose outputs
4. Consecutive detections are collapsed into time intervals (entry/exit times)
5. Raw detections are saved to a JSON file for inspection

### Dependencies

```bash
pip install opencv-python requests tqdm
ollama pull llava
```

### Usage

```bash
python exercise2_surveillance.py --video myvideo.mp4

# Optional flags:
#   --outdir   frames_out     Directory to save extracted frames (default: frames_out)
#   --every    2.0            Seconds between sampled frames (default: 2.0)
#   --model    llava          Ollama model name (default: llava)
#   --save_json results.json  Where to save raw detection output (default: results.json)
```

### Sample output

Run on `surveillance.mp4` (~3 min video, MacBook, default 2-second interval):

```
Extracting frames: 100%|█████████████| 9554/9554 [00:10<00:00, 927.48it/s]
Extracted 96 frames into frames_out
Running LLaVA: 100%|████████████████████| 96/96 [09:27<00:00, 5.91s/it]
Saved raw detections to results.json

=== Person present intervals (approx) ===
Person present from ~16.0s to ~28.0s
Person present from ~34.0s to ~38.0s
Person present from ~40.0s to ~44.0s
Person present from ~46.0s to ~48.0s
Person present from ~50.0s to ~62.0s
Person present from ~70.0s to ~76.0s
Person present from ~80.0s to ~84.0s
Person present from ~90.0s to ~92.0s
Person present from ~96.0s to ~106.0s
Person present from ~108.0s to ~112.0s
Person present from ~118.0s to ~120.0s
Person present from ~126.0s to ~128.0s
Person present from ~132.0s to ~136.0s
Person present from ~140.0s to ~156.0s
Person present from ~164.0s to ~166.0s
Person present from ~168.0s to ~176.0s
Person present from ~180.0s to ~182.0s
Person present from ~186.0s to ~190.0s
```

### Notes

- Processing speed on a MacBook (no discrete GPU): ~5.9s per frame, ~9.5 minutes total for 96 frames
- Detections can be fragmented — LLaVA may miss frames where the person is partially visible, moving fast, or at an angle, creating short gaps between intervals
- Reduce `--every` for coarser (faster) analysis; increase it for finer detection
- Raw detections are saved to `results.json` so you can re-analyze without re-running inference
