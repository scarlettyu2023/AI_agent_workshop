"""
Meal & Health Agent — coach.py
Proactive health coach: analyses workout history and generates a
motivational message whose tone scales with how well (or badly)
the user is doing.

Tone ladder:
  - No workout goal set yet  → gentle prompt to set one
  - On track (≥ goal days)   → warm encouragement + meal tip
  - Slightly behind (1 miss) → friendly nudge
  - Behind (2+ missed)       → firm reminder
  - Very behind (4+ days w/o workout) → full berate mode
"""

import os
from openai import OpenAI
from memory import UserProfile


def get_coach_message(profile: UserProfile) -> str:
    """Return a proactive coaching message based on current profile state."""

    wg          = profile.get("workout_goal")
    goal_days   = wg.get("days_per_week") if isinstance(wg, dict) else None
    this_week   = profile.workouts_this_week()
    days_since  = profile.days_since_last_workout()
    weight_trend = profile.weight_trend()
    name        = profile.get("name", "")
    health_goal = profile.get("goal", "improve health")

    # ── Determine tone ────────────────────────────────────────────────────────
    if goal_days is None:
        tone = "gentle"
        situation = (
            "The user hasn't set a workout goal yet. "
            "Encourage them to set one and explain why it matters for their goal."
        )
    else:
        done      = len(this_week)
        days_left = max(0, goal_days - done)
        if days_since is None or days_since >= 4:
            tone = "berating"
            situation = (
                f"The user's workout goal is {goal_days} days/week. "
                f"They've done {done} workout(s) this week "
                f"and haven't worked out in {days_since or 'many'} day(s). "
                "They are seriously behind. Do NOT let them off easy."
            )
        elif days_left == 0:
            tone = "celebrating"
            situation = (
                f"The user hit their goal of {goal_days} workouts this week! "
                "Celebrate warmly. Suggest a good recovery meal or treat."
            )
        elif days_left == 1:
            tone = "nudging"
            situation = (
                f"The user needs just 1 more workout to hit their goal of "
                f"{goal_days} this week. They've done {done}. "
                "Give an encouraging push."
            )
        else:
            tone = "firm"
            situation = (
                f"The user needs {days_left} more workouts to hit {goal_days}/week. "
                f"They've only done {done} so far. Be direct and motivating."
            )

    # ── Weight context ────────────────────────────────────────────────────────
    weight_context = ""
    if weight_trend:
        delta = weight_trend.get("delta_kg", 0)
        current = weight_trend.get("current_kg")
        if delta < -0.5:
            weight_context = f"Their weight is trending down ({delta:+}kg) — great progress."
        elif delta > 0.5:
            weight_context = f"Their weight is trending up ({delta:+}kg) — worth mentioning."
        else:
            weight_context = f"Their weight is stable at {current}kg."

    # ── Build prompt ──────────────────────────────────────────────────────────
    name_clause = f"Their name is {name}. " if name else ""
    prompt = f"""
You are a tough-love health coach delivering a PROACTIVE daily check-in.
{name_clause}Their health goal: {health_goal}.

Situation: {situation}
{weight_context}

Tone: {tone.upper()}
Rules:
- Be direct and personal. Do NOT be generic.
- If tone is BERATING: be genuinely stern and disappointed, not just firm.
  Use specific language about the consequences of skipping workouts.
  A little guilt is appropriate. End with a concrete action they should do TODAY.
- If tone is CELEBRATING: be genuinely warm. Suggest one specific food reward.
- If tone is NUDGING or FIRM: give one specific workout suggestion for today.
- Keep it under 5 sentences.
- No disclaimers, no "as an AI", no excessive caveats.
""".strip()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.85,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()