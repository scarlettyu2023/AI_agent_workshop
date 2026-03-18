# AI Agent Workshop
**CS 6501 — Building AI Agents | University of Virginia | Spring 2026**

*Scarlett Yu · bce9ka@virginia.edu*

---

## About This Repository

This repo contains my coursework and final project for CS 6501, a hands-on graduate workshop on building autonomous AI agents at UVA, taught by Prof. Henry Kautz.

The course covers the full modern AI agent stack — from running small LLMs locally, to multi-agent orchestration, RAG pipelines, vision-language models, and fine-tuning. Each topic folder contains working code built during and after class sessions.

---

## Final Project — Meal & Health Agent

A conversational AI agent that builds a personalized weekly meal plan based on your health profile, nutritional bloodwork, and ongoing weekly feedback.

### How it works

The agent guides the user through a natural conversation — asking about their health goals, dietary restrictions, and optionally reading an uploaded bloodwork PDF. It identifies nutritional gaps, calculates daily targets, and generates a weekly meal plan, grocery list, and recipe suggestions tailored specifically to that person.

Each week, the agent checks in on progress and adapts the plan based on what worked, what didn't, and how the user's metrics are changing.

### Key agent behaviors

- **Conversational onboarding** — asks one question at a time, never a form
- **PDF parsing** — reads bloodwork reports and extracts deficiency data
- **Nutritional reasoning** — compares user data against dietary guidelines and explains its decisions
- **Persistent memory** — profile grows over multiple weekly sessions via JSON
- **Dynamic adaptation** — plan evolves based on weight changes, meal feedback, and energy levels

### Stack

Python · OpenAI API · pypdf · JSON file memory

### Run it

```bash
cd meal_health_agent
pip install -r requirements.txt
cp .env.example .env        # add your OpenAI API key
python main.py
```

---

## Course Topics

| Folder | Topic |
|---|---|
| `Topic_1_Running_an_LLM_.ipynb` | Running small open-source LLMs locally and on Google Colab |
| `Topic2Frameworks/` | Agent control flows — HuggingFace smolagents, LangChain, LangGraph |
| `Topic 3/` | Few-shot learning and in-context learning |
| `Topic4Exploring/` | Chain-of-thought reasoning and Self-Refine |
| `Topic5RAG/` | Retrieval-augmented generation with vector databases |
| `Topic6VLM/` | Vision-language models |
| `Topic7MCP/` | Model Context Protocol |

---

## Key Papers Implemented

- **ReAct** — Yao et al. 2023. Synergizing Reasoning and Acting in Language Models
- **Self-Refine** — Madaan et al. 2023. Iterative Refinement with Self-Feedback
- **RAG** — Komeili et al. 2021. Internet-Augmented Dialogue Generation
- **QLoRA** — Dettmers et al. 2023. Efficient Finetuning of Quantized LLMs
- **Generative Agents** — Park et al. 2023. Interactive Simulacra of Human Behavior

---

## Skills Demonstrated

`Python` `LLM APIs` `Multi-agent systems` `RAG` `LangGraph` `HuggingFace` `Prompt engineering` `PDF parsing` `Conversational AI` `Agent memory`
