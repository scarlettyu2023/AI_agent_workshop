<div align="center">

# 🤖 AI Agent Workshop

**CS 6501 — Building AI Agents · University of Virginia · Spring 2026**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=flat-square&logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

*Scarlett Yu · bce9ka@virginia.edu · University of Virginia*

</div>

---

## 👩‍💻 About This Repository

This repo contains my coursework and final project for **CS 6501**, a hands-on graduate workshop on building autonomous AI agents at UVA, taught by [Prof. Henry Kautz](mailto:henry.kautz@virginia.edu).

The course covers the full modern AI agent stack — from running small LLMs locally, to multi-agent orchestration, RAG pipelines, vision-language models, and fine-tuning. Each topic folder contains working code built during and after class sessions.

---

## 🥗 Final Project — Meal & Health Agent

> *A conversational AI agent that builds a personalized weekly meal plan based on your health profile, nutritional bloodwork, and ongoing weekly feedback — and proactively holds you accountable.*

### ✨ How It Works

```
User chats with agent
     ↓
LLM reasons: what do I need to do next?   ← ReAct loop (Yao et al., 2023)
     ↓
Calls a tool: parse PDF / generate plan / log workout / save profile
     ↓
Observes result → loops if needed
     ↓
Final reply to user
     ↓
Next session: proactive opening based on workout history & weight trend
```

### 🧠 Key Agent Behaviors

| Behavior | Description |
|---|---|
| 💬 **Conversational onboarding** | Asks one question at a time, never a form |
| 📄 **PDF parsing** | Reads bloodwork reports and extracts nutritional markers |
| 🔬 **Nutritional reasoning** | Bloodwork gaps directly influence meal choices |
| 🧠 **Persistent memory** | Profile grows across sessions in a local JSON file |
| 🔄 **Weekly adaptation** | Plan regenerates based on check-in feedback and weight trend |
| 💪 **Proactive coaching** | Opening tone scales from celebrating → roasting based on workout history |

### 🛠️ Tools the Agent Can Call

| Tool | What it does |
|---|---|
| `parse_bloodwork_pdf` | Extracts iron, B12, vitamin D, etc. from a PDF |
| `generate_meal_plan` | Builds a 7-day plan with cuisine preference + bloodwork gaps |
| `generate_grocery_list` | Categorized shopping list from the current plan |
| `save_profile` | Persists user data to JSON across sessions |
| `log_workout` | Records workout type, duration, and date |
| `log_weight` | Tracks weight over time for trend analysis |
| `add_weekly_checkin` | Appends meal feedback to history for plan adaptation |

### 🚀 Run It

```bash
cd meal_health_agent
cp .env.example .env        # add your OpenAI API key
pip install -r requirements.txt
python main.py              # chat mode
python main.py --coach      # instant coaching check-in
```

### 💬 Example Session

```
Agent:  Hi! What's your main health goal?
You:    I want to gain muscle

Agent:  Any dietary restrictions?
You:    Lactose intolerant

  [tool] save_profile(key='goal', value='gain muscle')
  [tool] save_profile(key='restrictions', value='lactose intolerant')
  [tool] generate_meal_plan(goal='gain muscle', cuisine_preference='Asian food')

Agent:  Here's your 7-day plan...

You:    I went for a 40 minute run today

  [tool] log_workout(workout_type='running', duration_min=40)

--- Next session, 3 days later ---

Agent:  Hey! It's been 3 days since your last run...
        the pavement is starting to miss you. How did the meals go?
```

### 📦 Stack

```
Python  ·  OpenAI API (function calling)  ·  pypdf  ·  JSON memory
```

No LangChain — the ReAct loop is implemented directly using OpenAI function calling, making the agent behavior explicit and inspectable from the terminal.

---

## 📚 Course Topics

| # | Folder | Topic |
|---|---|---|
| 1 | `Topic_1_Running_an_LLM_.ipynb` | 🖥️ Running small open-source LLMs locally & on Colab |
| 2 | `Topic2Frameworks/` | 🔗 Agent control flows — smolagents, LangChain, LangGraph |
| 3 | `Topic 3/` | 🎯 Few-shot learning and in-context learning |
| 4 | `Topic4Exploring/` | 🧩 Chain-of-thought reasoning and Self-Refine |
| 5 | `Topic5RAG/` | 📦 Retrieval-augmented generation with vector databases |
| 6 | `Topic6VLM/` | 👁️ Vision-language models |
| 7 | `Topic7MCP/` | 🔌 Model Context Protocol |
| 8 | `meal_health_agent/` | 🥗 **Final Project** — ReAct agent with persistent memory |

---

## 📄 Key Papers Implemented

| Paper | Authors | Year | Applied in |
|---|---|---|---|
| **ReAct: Synergizing Reasoning and Acting** | Yao et al. | 2023 | Final project — core agent loop |
| **Toolformer: Language Models Can Teach Themselves to Use Tools** | Schick et al. | 2023 | Final project — tool-calling design |
| **Self-Refine: Iterative Refinement with Self-Feedback** | Madaan et al. | 2023 | Topic 4 |
| **Internet-Augmented Dialogue Generation** | Komeili et al. | 2021 | Topic 5 (RAG) |
| **Language Models are Few-Shot Learners** | Brown et al. | 2020 | Topic 3 |
| **QLoRA: Efficient Finetuning of Quantized LLMs** | Dettmers et al. | 2023 | Topic 7 |
| **Generative Agents: Interactive Simulacra** | Park et al. | 2023 | Topic 2 |

---

## 🏷️ Skills Demonstrated

![ReAct](https://img.shields.io/badge/-ReAct_Pattern-8B5CF6?style=flat-square)
![Function Calling](https://img.shields.io/badge/-Function_Calling-412991?style=flat-square&logo=openai&logoColor=white)
![LangGraph](https://img.shields.io/badge/-LangGraph-1C3C3C?style=flat-square)
![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![RAG](https://img.shields.io/badge/-RAG-0EA5E9?style=flat-square)
![Persistent Memory](https://img.shields.io/badge/-Persistent_Memory-22C55E?style=flat-square)
![PDF Parsing](https://img.shields.io/badge/-PDF_Parsing-EF4444?style=flat-square)
![Prompt Engineering](https://img.shields.io/badge/-Prompt_Engineering-F97316?style=flat-square)

<div align="center">

*Built with ☕ and a lot of late nights at UVA*

</div>
