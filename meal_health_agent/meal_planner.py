"""
Meal & Health Agent — meal_planner.py

Implements three agentic AI techniques:

  1. RAG (Retrieval-Augmented Generation)
     Retrieves relevant nutrition facts from a local knowledge base before
     planning. Grounds recommendations in real data instead of hallucinations.

  2. Chain-of-Thought (CoT) reasoning
     Forces the LLM to reason step-by-step through nutritional gaps,
     restrictions, and food choices BEFORE generating the final plan.
     Based on: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning
     in Large Language Models", 2022.

  3. Self-Refine
     After generating an initial plan, a critic pass evaluates it against
     the user's profile, then a refine pass improves it.
     Based on: Madaan et al., "Self-Refine: Iterative Refinement with
     Self-Feedback", 2023.
"""

import os
import json
from openai import OpenAI
from rag import retrieve, format_context

# ── 1. CoT reasoning prompt ───────────────────────────────────────────────────

COT_PROMPT = """
You are an expert nutritionist. Before creating a meal plan, reason step-by-step
through the user's needs. Be specific — name actual foods and nutrients.

Think through these steps:
Step 1: What is the user's primary health goal, and what macronutrient balance does that require?
Step 2: What nutritional deficiencies or concerns appear in their bloodwork (if any)? What foods address each one?
Step 3: What dietary restrictions must be strictly respected? What common foods must be avoided?
Step 4: Given their cuisine preference, what specific dishes or ingredients fit their goal and restrictions?
Step 5: What are the 3-5 key foods or ingredients that should appear frequently in this plan?

User profile:
{profile_context}

Relevant nutrition knowledge (from knowledge base):
{rag_context}

Write your step-by-step reasoning now. Be concrete and specific.
"""

# ── 2. Plan generation prompt ─────────────────────────────────────────────────

PLANNER_PROMPT = """
You are an expert nutritionist and meal planner.
Using the reasoning below, generate a practical 7-day meal plan.

Rules:
- Directly apply the reasoning — use the specific foods identified
- Respect ALL dietary restrictions strictly
- Weekday meals: ≤ 30 min prep. Weekend: more elaborate is fine.
- Include one snack per day
- Prioritize whole foods

Nutritional reasoning:
{cot_reasoning}

Return ONLY valid JSON — no markdown, no code fences.

JSON shape:
{{
  "Monday":    {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
  "Tuesday":   {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
  "Wednesday": {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
  "Thursday":  {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
  "Friday":    {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
  "Saturday":  {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
  "Sunday":    {{"breakfast": "...", "lunch": "...", "dinner": "...", "snack": "..."}},
  "notes": "2-3 sentences explaining the key nutritional choices made."
}}
"""

# ── 3. Self-Refine: critic prompt ─────────────────────────────────────────────

CRITIC_PROMPT = """
You are a nutrition critic. Evaluate this meal plan against the user's profile.

User profile:
{profile_context}

Meal plan to evaluate:
{plan}

Identify specific problems only — do not praise what works.
Check for:
- Any dietary restrictions violated
- Nutritional gaps NOT addressed (e.g. still no iron-rich foods despite low iron)
- Meals that are unrealistic to cook on a weeknight (> 30 min)
- Missing cuisine preference
- Lack of variety (same protein 5+ days)
- Protein targets not met for the stated goal

If the plan is good, say "No significant issues found."
Otherwise list issues concisely, one per line.
"""

# ── 4. Self-Refine: refine prompt ─────────────────────────────────────────────

REFINE_PROMPT = """
You are an expert meal planner. Improve this meal plan based on the critique.

Original plan:
{plan}

Critique:
{critique}

Fix only the issues identified. Keep meals that are already good.
Return ONLY valid JSON in the same shape as the original plan — no markdown, no code fences.
"""


def generate_meal_plan(profile: dict) -> dict:
    """
    Generate a meal plan using RAG + CoT + Self-Refine.

    Pipeline:
      1. RAG: retrieve relevant nutrition facts
      2. CoT: reason through the user's needs step by step
      3. Generate initial plan from the reasoning
      4. Self-Refine: critique the plan, then refine it
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    profile_context = _build_profile_context(profile)

    # ── Step 1: RAG ───────────────────────────────────────────────────────────
    print("  [rag] Retrieving relevant nutrition knowledge...")
    rag_query = " ".join(filter(None, [
        profile.get("goal"),
        profile.get("restrictions"),
        profile.get("cuisine_preference"),
        " ".join(profile.get("bloodwork", {}).keys()) if profile.get("bloodwork") else "",
    ]))
    retrieved   = retrieve(rag_query, top_k=4)
    rag_context = format_context(retrieved)

    # ── Step 2: Chain-of-Thought reasoning ───────────────────────────────────
    print("  [cot] Reasoning through nutritional needs...")
    cot_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": COT_PROMPT.format(
                profile_context=profile_context,
                rag_context=rag_context,
            ),
        }],
        temperature=0.4,
        max_tokens=600,
    )
    cot_reasoning = cot_response.choices[0].message.content.strip()

    # ── Step 3: Generate initial plan ────────────────────────────────────────
    print("  [plan] Generating initial meal plan...")
    plan_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": PLANNER_PROMPT.format(cot_reasoning=cot_reasoning),
        }],
        temperature=0.5,
        max_tokens=1500,
    )
    plan_text = plan_response.choices[0].message.content.strip()

    try:
        plan = json.loads(plan_text)
    except json.JSONDecodeError:
        return {"error": "Could not parse initial meal plan.", "raw": plan_text}

    # ── Step 4: Self-Refine ───────────────────────────────────────────────────
    print("  [critic] Critiquing the plan...")
    critic_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": CRITIC_PROMPT.format(
                profile_context=profile_context,
                plan=json.dumps(plan, indent=2),
            ),
        }],
        temperature=0.3,
        max_tokens=300,
    )
    critique = critic_response.choices[0].message.content.strip()

    # Only refine if the critic found real issues
    if "no significant issues" not in critique.lower():
        print("  [refine] Refining based on critique...")
        refine_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": REFINE_PROMPT.format(
                    plan=json.dumps(plan, indent=2),
                    critique=critique,
                ),
            }],
            temperature=0.4,
            max_tokens=1500,
        )
        refined_text = refine_response.choices[0].message.content.strip()
        try:
            plan = json.loads(refined_text)
        except json.JSONDecodeError:
            pass  # keep original plan if refine fails to parse

    # Attach reasoning trace for transparency
    plan["_reasoning"] = cot_reasoning
    plan["_critique"]  = critique

    return plan


def _build_profile_context(profile: dict) -> str:
    lines = []
    if profile.get("goal"):
        lines.append(f"Health goal: {profile['goal']}")
    if profile.get("biometrics_raw"):
        lines.append(f"Biometrics: {profile['biometrics_raw']}")
    if profile.get("restrictions"):
        lines.append(f"Dietary restrictions: {profile['restrictions']}")
    if profile.get("cuisine_preference"):
        lines.append(f"Cuisine preference: {profile['cuisine_preference']}")
    if profile.get("bloodwork"):
        bw = {k: v for k, v in profile["bloodwork"].items() if k != "raw_extraction"}
        if bw:
            lines.append(f"Bloodwork: {', '.join(f'{k}: {v}' for k, v in bw.items())}")
    if profile.get("checkins"):
        lines.append(f"Last check-in: {profile['checkins'][-1]}")
    if profile.get("plan_feedback"):
        lines.append(f"Feedback on previous plan: {profile['plan_feedback']}")
    return "\n".join(lines) if lines else "No profile data yet."


# ── Grocery list ──────────────────────────────────────────────────────────────

GROCERY_SYSTEM_PROMPT = """
You are a meal prep assistant. Given a weekly meal plan, generate a concise
categorized grocery list. Group items into exactly these categories:
Produce, Protein, Grains & Legumes, Dairy & Alternatives, Pantry & Sauces, Other.

Consolidate duplicates across days. Include approximate quantities.
Return ONLY valid JSON — no markdown, no code fences.

JSON shape:
{
  "Produce":              ["item (qty)", ...],
  "Protein":              ["item (qty)", ...],
  "Grains & Legumes":     ["item (qty)", ...],
  "Dairy & Alternatives": ["item (qty)", ...],
  "Pantry & Sauces":      ["item (qty)", ...],
  "Other":                ["item (qty)", ...]
}
"""


def generate_grocery_list(meal_plan: dict) -> dict:
    """Generate a categorized grocery list from a meal plan dict."""
    if not meal_plan:
        return {"error": "No meal plan provided."}

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    meal_lines = [
        f"{day} {meal_type}: {desc}"
        for day, meals in meal_plan.items()
        if isinstance(meals, dict)
        for meal_type, desc in meals.items()
        if meal_type not in ("notes", "_reasoning", "_critique")
    ]
    plan_text = "\n".join(meal_lines) if meal_lines else str(meal_plan)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": GROCERY_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Meal plan:\n{plan_text}"},
        ],
        temperature=0,
        max_tokens=700,
    )

    content = response.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"raw": content}


# ── Quick standalone test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    sample = {
        "goal": "gain muscle",
        "biometrics_raw": "24 years old, 55kg, 165cm",
        "restrictions": "lactose intolerant",
        "cuisine_preference": "Asian",
        "bloodwork": {"Iron": "low (55 µg/dL)", "Vitamin B12": "low (180 pg/mL)"},
    }
    plan = generate_meal_plan(sample)
    # Print without the reasoning trace for readability
    display = {k: v for k, v in plan.items() if not k.startswith("_")}
    print(json.dumps(display, indent=2))
    print("\n--- CoT Reasoning ---")
    print(plan.get("_reasoning", ""))
    print("\n--- Critique ---")
    print(plan.get("_critique", ""))