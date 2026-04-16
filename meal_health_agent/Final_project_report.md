# Meal & Health Agent
## CS 6501 — Workshop on Building AI Agents · Final Project Report
**Scarlett Yu · bce9ka@virginia.edu · University of Virginia · Spring 2026**

---

## 1. Motivation

Meal planning is deceptively hard. It requires balancing a health goal (losing weight, gaining muscle), dietary restrictions (allergies, intolerances, preferences), and sometimes clinical data from bloodwork — all at once, every week. Most people either give up after a few days or fall back to the same five meals on repeat.

Existing tools fall into two categories. Static recipe apps suggest individual meals but do not adapt to the user over time. Conversational tools like ChatGPT can generate a one-shot meal plan, but forget everything the moment the session ends — there is no memory, no follow-up, and no accountability.

The gap I wanted to fill is an agent that:

1. **Builds a personalized plan** from the user's health goal, dietary restrictions, cuisine preferences, and optionally bloodwork data
2. **Remembers the user across sessions** so recommendations improve over time
3. **Holds the user accountable** proactively — checking in on workout consistency and meal adherence without being asked

This project is also an opportunity to implement, in a working system, the core techniques covered in this course: ReAct, Retrieval-Augmented Generation (RAG), Chain-of-Thought (CoT) reasoning, and Self-Refine.

---

## 2. Methods

### 2.1 System Overview

The Meal & Health Agent is a CLI-based conversational agent. It is built in Python using the OpenAI API with function calling. The full pipeline runs on every meal plan generation:

```
User message
     ↓
ReAct loop (agent.py)
     ↓
Tool call: generate_meal_plan
     ↓
  [1] RAG   — retrieve relevant nutrition facts
  [2] CoT   — reason step-by-step through user needs
  [3] Plan  — generate initial 7-day meal plan
  [4] Critic — evaluate plan against user profile
  [5] Refine — fix identified problems
     ↓
Reply to user
```

The project has seven Python files, each with a single responsibility:

| File | Role |
|---|---|
| `main.py` | CLI entry point, session control |
| `agent.py` | ReAct loop, OpenAI function calling, tool dispatcher |
| `meal_planner.py` | RAG + CoT + Self-Refine pipeline |
| `rag.py` | Embedding, cosine similarity, KB retrieval |
| `nutrition_kb.json` | 20-entry nutrition knowledge base |
| `memory.py` | Persistent JSON profile, workout and weight log |
| `coach.py` | Proactive tone-scaled coaching messages |
| `pdf_parser.py` | Bloodwork PDF → structured nutritional markers |

### 2.2 Technique 1: ReAct (Reason + Act)

The agent follows the ReAct pattern from Yao et al. (2023). Rather than generating a response in a single forward pass, it iterates: the LLM reasons about what to do next, calls a tool, observes the result, and loops until it has enough to produce a final reply.

This is implemented in `agent.py` as the `_run()` method:

```python
for _ in range(MAX_TOOL_ROUNDS):
    response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    msg = response.choices[0].message

    if not msg.tool_calls:
        return msg.content   # final reply

    # Execute each tool, feed results back
    for tc in msg.tool_calls:
        result = self._execute_tool(tc.function.name, args)
        messages.append({"role": "tool", "content": result})
```

The agent has 7 callable tools: `generate_meal_plan`, `generate_grocery_list`, `parse_bloodwork_pdf`, `log_workout`, `log_weight`, `save_profile`, and `add_weekly_checkin`. The LLM decides autonomously when to call each one — for example, it calls `log_workout` the moment a user mentions exercising, without being explicitly told to.

### 2.3 Technique 2: Retrieval-Augmented Generation (RAG)

Before generating any meal plan, the agent retrieves relevant entries from a local nutrition knowledge base. This grounds the LLM's recommendations in real nutritional facts rather than relying purely on parametric memory.

The knowledge base (`nutrition_kb.json`) contains 20 entries covering topics such as iron-rich foods, B12 sources, muscle gain nutrition, post-workout meals, cuisine-specific nutrients, and blood glucose management.

Retrieval uses cosine similarity over OpenAI `text-embedding-3-small` embeddings:

```python
def retrieve(query: str, top_k: int = 4) -> list[str]:
    query_embedding = embed([query])[0]
    scored = [(i, cosine_similarity(query_embedding, kb_embeddings[i]))
              for i in range(len(kb))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [kb[i]["text"] for i, _ in scored[:top_k]]
```

Embeddings are computed once and cached to `nutrition_kb_embeddings.json`, so retrieval on subsequent runs is fast.

### 2.4 Technique 3: Chain-of-Thought Reasoning

Based on Wei et al. (2022), the agent is prompted to reason step-by-step through the user's nutritional needs before generating the meal plan. The CoT prompt forces explicit reasoning through five steps:

1. What macronutrient balance does the health goal require?
2. What deficiencies appear in bloodwork, and which foods address them?
3. What dietary restrictions must be respected?
4. Given cuisine preferences, which specific dishes fit?
5. Which 3–5 key foods should appear frequently in the plan?

This reasoning is then passed as context to the plan generation step, so nutritional decisions are grounded in explicit logic rather than being implicit.

### 2.5 Technique 4: Self-Refine

Based on Madaan et al. (2023), after generating an initial meal plan, a critic pass evaluates it against the user's profile. The critic checks for:

- Dietary restrictions violated
- Nutritional gaps not addressed
- Meals that are unrealistic for a weeknight
- Missing cuisine preference
- Insufficient protein for the stated goal

If the critique identifies real issues, a refine pass takes the original plan plus the critique and produces an improved version. The critique and reasoning trace are stored in the profile under `_critique` and `_reasoning` keys for transparency.

### 2.6 Persistent Memory

All user data is stored in `user_profile.json` and loaded on every session. The profile includes:

```json
{
  "goal": "gain muscle",
  "cuisine_preference": "Asian",
  "restrictions": "none",
  "biometrics_raw": "24 years old, 55kg, 165cm",
  "workout_log": [
    {"date": "2026-04-03", "type": "yoga", "duration_min": 45}
  ],
  "weight_log": [
    {"date": "2026-04-03", "weight_kg": 60.0}
  ],
  "checkins": ["meals were great, loved the teriyaki bowl"],
  "current_plan": { ... }
}
```

The profile summary is injected into the system prompt on every turn, so the LLM always has full context about the user.

### 2.7 Proactive Coach

On returning sessions, `coach.py` reads the workout history and weight trend and generates a tone-scaled opening message. The tone is determined by a ladder:

| Situation | Tone |
|---|---|
| Hit workout goal this week | Celebrating |
| 1 workout short of goal | Nudging |
| 2+ workouts short | Firm |
| 4+ days with no workout | Berating |
| No workout goal set | Gentle prompt |

This is implemented as a separate LLM call with a structured prompt that includes the user's profile data and explicit instructions for each tone level.

---

## 3. Evaluation

### 3.1 Quantitative Comparison with NutriGen

The closest published work is NutriGen (Khamesian et al., 2025), which benchmarks LLM-generated meal plans against USDA reference calorie values. They report Mean Absolute Error (MAE) as a percentage of the target calorie count across 10 synthetic dietary profiles.

Since my system uses `gpt-4o-mini` and does not optimize for caloric precision, I ran a lightweight comparison using 5 sample profiles. For each profile I generated a plan with my system (RAG + CoT + Self-Refine) and with a baseline single-prompt approach (no RAG, no CoT, no refinement), then manually checked whether the plan correctly respected the dietary restrictions and addressed stated nutritional goals.

| Profile | Restriction | Goal | Baseline respected restrictions | This system respected restrictions |
|---|---|---|---|---|
| A | Lactose intolerant | Lose weight | No — included Greek yogurt | Yes |
| B | Hates fish | Gain muscle | Yes | Yes |
| C | Low iron (bloodwork) | General health | No — no iron-rich foods | Yes — lentils, spinach featured |
| D | Vegan | Gain muscle | Yes | Yes |
| E | Lactose + no fish | Lose weight | No — included salmon + cheese | Yes |

The baseline violated dietary restrictions in 3 of 5 profiles. This system respected all restrictions in all 5 profiles, primarily because the CoT reasoning step explicitly names forbidden foods before generating the plan.

This is a small-scale qualitative-style check rather than a rigorous benchmark, but it illustrates the practical value of the CoT step.

### 3.2 Qualitative User Evaluation

I asked 3 people to use the agent for a single session and rate it on four dimensions using a 1–5 scale.

| Dimension | User 1 | User 2 | User 3 | Average |
|---|---|---|---|---|
| Meal plan was relevant to my goal | 5 | 4 | 5 | **4.7** |
| Dietary restrictions were respected | 5 | 5 | 5 | **5.0** |
| Conversation felt natural | 4 | 3 | 4 | **3.7** |
| I would use this again | 4 | 4 | 3 | **3.7** |

Selected feedback:

- *"The meal plan is actually really good and the Asian food preference was clearly applied throughout — much better than what I'd get from just asking ChatGPT the same question."* (User 1)
- *"The agent asked too many questions before generating the plan. It should just ask for goal and restrictions and go."* (User 2)
- *"It's impressive that it remembered me the next day and mentioned my yoga session. That's the part that feels genuinely different."* (User 3)

### 3.3 Observed Limitations

- **PDF parsing**: The bloodwork parser works on text-based PDFs only. Scanned or image-based bloodwork reports return empty results.
- **Knowledge base size**: 20 entries is sufficient for demonstration but would need to be a full vector database (e.g., the USDA FoodData Central) in a real system. NutriGen uses USDA as their reference; this system does not.
- **Calorie estimation**: The system does not track or target calories explicitly. Meal plans are optimized for nutritional quality and goal alignment, not caloric precision. This is a deliberate scope decision for a course project but a real limitation for clinical use.
- **System prompt growth**: Over many sessions the profile summary injected into the system prompt grows longer, which increases cost and may eventually degrade response quality. AgentFold (Ye et al., 2024) from the course syllabus describes exactly the context management technique needed to address this.
- **Over-questioning**: Users noted the agent sometimes asks too many clarifying questions before generating a plan. This is a prompt engineering issue — the system prompt could be tightened to generate the plan with goal + restrictions alone and ask for more detail afterward.

---

## 4. Conclusions

### What the project demonstrates

The Meal & Health Agent is a working end-to-end agentic system. It is not a chatbot that generates meal plans — it is an agent that reasons over a knowledge base, executes real tool calls, persists memory across sessions, and adapts its behavior based on the user's history. These are meaningfully different.

The most visible difference from a simple LLM prompt is the session-to-session continuity. A returning user is greeted by name with a message that references their last workout and meal feedback. That behavior requires memory, not just a good model.

### What I learned from building it

**RAG**: Choosing what to retrieve is harder than implementing the retrieval. A vague query embedding returns irrelevant context that can mislead the planner. The query needs to be built deliberately from the user's profile, not just their most recent message.

**Self-Refine**: Writing the critic prompt was harder than writing the planner prompt. The planner just needs to produce a good plan. The critic needs to define, explicitly, what "good" means — which forces you to articulate your evaluation criteria in a way that a prompt-only system never does.

**ReAct and tool schemas**: Tool descriptions are contracts with the LLM. Early versions of the agent misfired tools — calling `generate_meal_plan` before collecting restrictions, saving `workout_goal` as a free-form string instead of a structured dict. Every misfiring traced back to an ambiguous tool description. Precise schemas prevent a large class of bugs.

**Persistent memory**: A user profile is both powerful and fragile. Powerful because the agent genuinely improves across sessions. Fragile because bad data — a misclassified field, a hallucinated value saved by the LLM — persists and affects all future interactions. Validation on write would be an important addition.

### Future work

- **Larger knowledge base**: Replace the 20-entry JSON with a proper vector database over the USDA FoodData Central for accurate nutritional grounding
- **Calorie tracking**: Add explicit caloric targets following the NutriGen approach
- **Context management**: Implement AgentFold-style memory folding to handle long-term users without bloating the system prompt
- **SMS reminders**: Proactive nudges pushed to the user rather than only shown at session start
- **Recipe detail**: Expand meal descriptions to include actual recipes with prep instructions

---

## References

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv preprint arXiv:2210.03629*.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Chi, E., Le, Q., & Zhou, D. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. *arXiv preprint arXiv:2201.11903*.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., ... & Clark, P. (2023). Self-Refine: Iterative Refinement with Self-Feedback. *arXiv preprint arXiv:2303.17651*.

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language Models are Few-Shot Learners. *arXiv preprint arXiv:2005.14165*.

Khamesian, S., Arefeen, A., Carpenter, S. M., & Ghasemzadeh, H. (2025). NutriGen: Personalized Meal Plan Generator Leveraging Large Language Models to Enhance Dietary and Nutritional Adherence. *arXiv preprint arXiv:2502.20601*.

Ye, R., Zhang, Z., Li, K., Yin, H., Tao, Z., Zhao, Y., ... & Jiang, Y. (2024). AgentFold: Long-Horizon Web Agents with Proactive Context Management. *arXiv preprint arXiv:2510.24699*.

Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023). Generative Agents: Interactive Simulacra of Human Behavior. *arXiv preprint arXiv:2304.03442*.

---

## Appendix: Example Input/Output

### Example 1 — First session onboarding

**Input:**
```
You: I want to gain muscle
You: no restrictions, Asian food
You: 24 years old, 55kg, 165cm
You: I did yoga for 45 min today
```

**Agent tool traces:**
```
[tool] save_profile(key='goal', value='gain muscle')
[tool] save_profile(key='restrictions', value='none')
[tool] save_profile(key='cuisine_preference', value='Asian food')
[tool] save_profile(key='biometrics_raw', value='24 years old, 55kg, 165cm')
[tool] log_workout(workout_type='yoga', duration_min=45)
[tool] generate_meal_plan(goal='gain muscle', cuisine_preference='Asian food')

  [rag]    Retrieving relevant nutrition knowledge...
  [cot]    Reasoning through nutritional needs...
  [plan]   Generating initial meal plan...
  [critic] Critiquing the plan...
  [refine] Refining based on critique...
```

**Sample output (Monday):**
```
Breakfast: Japanese tamagoyaki with miso soup (tofu)
Lunch:     Beef bulgogi with brown rice and steamed bok choy
Dinner:    Korean sundubu jjigae with a side of kimchi
Snack:     Edamame
```

---

### Example 2 — Returning session

**Agent opening (unprompted):**
```
Agent: Hey! It's been 1 day since your last workout —
       keep the yoga going! How did the meals go this week?
```

**Input:**
```
You: meals were great, portions a bit too big
```

**Agent tool traces:**
```
[tool] add_weekly_checkin(note='meals were great, portions a bit too big')
[tool] generate_meal_plan(goal='gain muscle', feedback='reduce portions', ...)

  [rag]    Retrieving relevant nutrition knowledge...
  [cot]    Reasoning through nutritional needs...
  [plan]   Generating initial meal plan...
  [critic] Critiquing the plan...
  [refine] Refining based on critique...
```

---

### Example 3 — Coach mode

**Input:** `coach`

**Output (4 days after last workout):**
```
Coach: Look, it's been 4 days since you last moved your body.
       Your goal is to gain muscle — that doesn't happen on the couch.
       Pick ONE thing today: a 20-minute walk, 3 sets of push-ups,
       anything. Do it before you eat dinner tonight.
```
