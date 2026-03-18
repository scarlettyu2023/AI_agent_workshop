<div align="center">

# 🤖 AI Agent Workshop

**CS 6501 — Building AI Agents · University of Virginia · Spring 2026**

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
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

> *A conversational AI agent that builds a personalized weekly meal plan based on your health profile, nutritional bloodwork, and ongoing weekly feedback.*

### ✨ How It Works

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   User chats with agent                                 │
│        ↓                                                │
│   Agent asks about goals, restrictions, uploads PDF     │
│        ↓                                                │
│   Agent reads bloodwork → finds nutritional gaps        │
│        ↓                                                │
│   Generates weekly meal plan + grocery list + recipes   │
│        ↓                                                │
│   Next week: checks in → adapts plan to your progress  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 🧠 Key Agent Behaviors

| Behavior | Description |
|---|---|
| 💬 **Conversational onboarding** | Asks one question at a time, never a form |
| 📄 **PDF parsing** | Reads bloodwork reports and extracts deficiency data |
| 🔬 **Nutritional reasoning** | Compares your data against dietary guidelines and explains decisions |
| 🧠 **Persistent memory** | Profile grows over multiple weekly sessions |
| 🔄 **Dynamic adaptation** | Plan evolves based on weight changes, meal feedback, and energy levels |

### 🛠️ Stack

```
Python  ·  OpenAI API  ·  pypdf  ·  JSON memory
```

### 🚀 Run It

```bash
cd meal_health_agent
pip install -r requirements.txt
cp .env.example .env        # add your OpenAI API key
python main.py
```

### 💬 Example Session

```
Agent:  Hi! What's your main health goal right now?

You:    I want to lose some weight

Agent:  Got it. How old are you, and what's your
        current weight and height?

You:    28, 60kg, 170cm

Agent:  Any dietary restrictions or foods you dislike?

You:    Lactose intolerant, hate fish

Agent:  Do you have a recent bloodwork report to upload?
        It really helps me spot nutritional gaps.

You:    [uploads PDF]

Agent:  I can see your iron and B12 are both low.
        Since you avoid fish and dairy, I'll focus on
        lentils, spinach, and fortified foods.
        Here's your plan for the week...
```

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

---

## 📄 Key Papers Implemented

| Paper | Authors | Year |
|---|---|---|
| **ReAct: Synergizing Reasoning and Acting** | Yao et al. | 2023 |
| **Self-Refine: Iterative Refinement with Self-Feedback** | Madaan et al. | 2023 |
| **Internet-Augmented Dialogue Generation** | Komeili et al. | 2021 |
| **QLoRA: Efficient Finetuning of Quantized LLMs** | Dettmers et al. | 2023 |
| **Generative Agents: Interactive Simulacra** | Park et al. | 2023 |
| **AgentFold: Long-Horizon Context Management** | Ye et al. | 2024 |

---

## 🏷️ Skills Demonstrated

![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LLM APIs](https://img.shields.io/badge/-LLM%20APIs-412991?style=flat-square&logo=openai&logoColor=white)
![LangGraph](https://img.shields.io/badge/-LangGraph-1C3C3C?style=flat-square)
![HuggingFace](https://img.shields.io/badge/-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![RAG](https://img.shields.io/badge/-RAG-0EA5E9?style=flat-square)
![Multi--Agent](https://img.shields.io/badge/-Multi--Agent%20Systems-8B5CF6?style=flat-square)
![Prompt Engineering](https://img.shields.io/badge/-Prompt%20Engineering-F97316?style=flat-square)
![PDF Parsing](https://img.shields.io/badge/-PDF%20Parsing-EF4444?style=flat-square)

<div align="center">

*Built with ☕ and a lot of late nights at UVA*

</div>
