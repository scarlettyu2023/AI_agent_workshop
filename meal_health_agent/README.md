<div align="center">

# 🥗 Meal & Health Agent

**CS 6501 Final Project — Workshop on Building AI Agents · UVA · Spring 2026**

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?style=flat-square&logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)

*Scarlett Yu · bce9ka@virginia.edu · University of Virginia*

</div>

---

## What It Does

A conversational AI agent that builds a personalized weekly meal plan based on your health profile and holds you accountable over time. Unlike a one-shot ChatGPT prompt, the agent **remembers you across sessions**, **calls real tools**, and **reasons explicitly** before making recommendations.

```
┌─────────────────────────────────────────────────────────┐
│  User chats with agent                                  │
│       ↓                                                 │
│  Agent asks about goals, restrictions, cuisine          │
│       ↓                                                 │
│  RAG retrieves relevant nutrition facts                 │
│       ↓                                                 │
│  Chain-of-Thought reasons through nutritional needs     │
│       ↓                                                 │
│  Generates meal plan → Self-Refine critiques & improves │
│       ↓                                                 │
│  Next session: proactive nudge based on workout history │
└─────────────────────────────────────────────────────────┘
```

---

## Agentic AI Techniques

This project implements four techniques from the course syllabus:

### 1. ReAct — Reason + Act
`agent.py` runs a ReAct loop: the LLM reasons about what to do next, calls a tool, observes the result, and repeats until it has a final answer. Based on [Yao et al., 2023](https://arxiv.org/abs/2210.03629).

The agent has 7 callable tools:
| Tool | What it does |
|---|---|
| `generate_meal_plan` | Triggers the RAG + CoT + Self-Refine pipeline |
| `generate_grocery_list` | Categorized shopping list from current plan |
| `parse_bloodwork_pdf` | Extracts nutritional markers from a PDF |
| `log_workout` | Records a workout to persistent memory |
| `log_weight` | Tracks weight over time |
| `save_profile` | Persists any user info to JSON |
| `add_weekly_checkin` | Appends weekly meal feedback |

### 2. RAG — Retrieval-Augmented Generation
`rag.py` embeds a 20-entry nutrition knowledge base (`nutrition_kb.json`) using `text-embedding-3-small` and retrieves the top-4 most relevant entries before meal planning. This grounds the LLM's recommendations in real nutritional facts instead of relying on parametric memory alone.

### 3. Chain-of-Thought Reasoning
Before generating a meal plan, the agent reasons step-by-step through the user's needs — nutritional deficiencies, restrictions, cuisine preferences, and specific foods to prioritize. This reasoning is then fed into the plan generation step. Based on [Wei et al., 2022](https://arxiv.org/abs/2201.11903).

### 4. Self-Refine
After generating an initial plan, a critic pass evaluates it against the user's profile (missed restrictions? nutritional gaps not addressed?). If issues are found, a refine pass corrects them. Based on [Madaan et al., 2023](https://arxiv.org/abs/2303.17651).

The full pipeline is visible in the terminal on every plan generation:
```
[rag]    Retrieving relevant nutrition knowledge...
[cot]    Reasoning through nutritional needs...
[plan]   Generating initial meal plan...
[critic] Critiquing the plan...
[refine] Refining based on critique...
```

### 5. Persistent Memory + Proactive Coach
`memory.py` stores the full user profile, workout log, weight log, and check-ins in `user_profile.json`. On returning sessions, `coach.py` reads the history and generates a tone-scaled opening message — from celebrating a great week to roasting you for skipping the gym.

---

## Project Structure

```
meal_health_agent/
├── main.py              ← entry point, CLI loop
├── agent.py             ← ReAct loop, OpenAI function calling, tool dispatcher
├── memory.py            ← persistent JSON profile (workouts, weight, check-ins)
├── meal_planner.py      ← RAG + CoT + Self-Refine pipeline
├── rag.py               ← embedding, cosine similarity retrieval
├── nutrition_kb.json    ← 20-entry nutrition knowledge base
├── pdf_parser.py        ← bloodwork PDF → structured markers via LLM
├── coach.py             ← proactive tone-scaled coaching messages
├── requirements.txt
└── .env.example
```

---

## Setup

```bash
git clone https://github.com/scarlettyu2023/AI_agent_workshop
cd meal_health_agent

cp .env.example .env
# Open .env and add your OpenAI API key: OPENAI_API_KEY=sk-...

pip install -r requirements.txt
python main.py
```

---

## Usage

```bash
python main.py           # normal chat session
python main.py --coach   # standalone proactive check-in (good for cron)
```

**Commands during chat:**

| Type | Does |
|---|---|
| `coach` | Instant tone-scaled check-in based on workout history |
| `reset` | Clears your profile and starts over |
| `quit` | Exit |
| `/path/to/file.pdf` | Upload a bloodwork PDF for nutritional analysis |

**Example session:**
```
Agent:  Hey! It's been 3 days since your last workout... the weights
        are getting dusty. How did the meals go this week?

You:    Great, loved the teriyaki bowl. I did yoga for 45 min today.

        [tool] log_workout(workout_type='yoga', duration_min=45)

Agent:  Nice! Since you're back on track, want me to update
        next week's meal plan based on your feedback?

You:    yes, reduce the portions a bit

        [rag]    Retrieving relevant nutrition knowledge...
        [cot]    Reasoning through nutritional needs...
        [plan]   Generating initial meal plan...
        [critic] Critiquing the plan...
        [refine] Refining based on critique...

Agent:  Here's your updated plan for the week...
```

---

## Key Papers

| Paper | Technique |
|---|---|
| Yao et al., [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629), 2023 | ReAct loop |
| Wei et al., [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903), 2022 | CoT reasoning |
| Madaan et al., [Self-Refine: Iterative Refinement](https://arxiv.org/abs/2303.17651), 2023 | Self-Refine |
| Brown et al., [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), 2020 | In-context learning |

---

## Limitations

- PDF parsing works on text-based bloodwork reports only — scanned/image PDFs are not supported
- Nutrition knowledge base is static (20 entries) — a production system would use a larger vector database
- Calorie estimates are LLM-generated approximations, not clinically verified
- Always consult a healthcare provider before making significant dietary changes
