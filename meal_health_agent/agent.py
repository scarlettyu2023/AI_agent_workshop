"""
Meal & Health Agent — agent.py
ReAct-style agent using OpenAI function calling.

Loop: Reason → Act (tool call) → Observe (tool result) → repeat until done.
The LLM decides WHEN to call tools — no hardcoded state machine.
"""

import os
import json
from openai import OpenAI
from memory import UserProfile
from pdf_parser import parse_bloodwork_pdf
from meal_planner import generate_meal_plan, generate_grocery_list

# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "parse_bloodwork_pdf",
            "description": (
                "Extract nutritional markers (iron, B12, vitamin D, etc.) from a "
                "bloodwork PDF. Call this when the user provides a .pdf file path."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the PDF file."},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_meal_plan",
            "description": (
                "Generate a personalized 7-day meal plan. Call once you know the "
                "user's goal and restrictions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal":               {"type": "string"},
                    "restrictions":       {"type": "string"},
                    "biometrics":         {"type": "string"},
                    "bloodwork":          {"type": "object"},
                    "feedback":           {"type": "string"},
                    "cuisine_preference": {"type": "string"},
                },
                "required": ["goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_grocery_list",
            "description": "Generate a categorized grocery list from the current meal plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "meal_plan": {"type": "object"},
                },
                "required": ["meal_plan"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_profile",
            "description": (
                "Save something important the user told you: name, goal, restrictions, "
                "biometrics, cuisine preference, plan feedback."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "enum": [
                            "name", "goal", "biometrics_raw", "restrictions",
                            "cuisine_preference", "bloodwork", "current_plan",
                            "plan_feedback", "workout_goal",
                        ],
                    },
                    "value": {},
                },
                "required": ["key", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_workout",
            "description": (
                "Record a workout the user completed. Call this when the user "
                "mentions they worked out, exercised, went to the gym, ran, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "workout_type":  {"type": "string",  "description": "e.g. running, lifting, yoga, cycling"},
                    "duration_min":  {"type": "integer", "description": "Duration in minutes."},
                    "notes":         {"type": "string",  "description": "Optional notes."},
                },
                "required": ["workout_type", "duration_min"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_weight",
            "description": "Record the user's current weight. Call when they mention their weight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number"},
                },
                "required": ["weight_kg"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_weekly_checkin",
            "description": "Append a weekly check-in note. Call when the user reports how last week's meals went.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                },
                "required": ["note"],
            },
        },
    },
]

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a warm but no-nonsense nutrition and fitness accountability coach.
You use tools to take real actions: parsing bloodwork, generating meal plans,
logging workouts, saving the user's profile, and building grocery lists.

Conversation style:
- Ask ONE question at a time.
- Be encouraging for good progress, gently firm for missed goals.
- Explain nutritional reasoning in plain language.
- Remind users to consult a healthcare provider for medical decisions.

Tool-calling rules:
- Call save_profile whenever you learn something new about the user.
- Call generate_meal_plan only after you know goal AND restrictions.
- Call log_workout the moment the user mentions completing any exercise.
- Call log_weight whenever the user mentions their current weight.
- Call generate_grocery_list when the user asks for a shopping list.
- Chain tool calls in one turn when it makes sense.

Today's date: {today}

Current user profile:
{profile}
"""

# ── Proactive opening prompt ──────────────────────────────────────────────────

PROACTIVE_PROMPT = """
You are opening a new session with a returning user. Based on their profile below,
craft ONE short, punchy opening message (2-4 sentences max).

Rules:
- If they haven't logged a workout in 3+ days AND have a workout goal → gently
  roast them. Be playful and specific, not mean. E.g. "I see it's been 4 days
  since you last hit the gym... the weights are starting to get dusty."
- If they worked out consistently this week → celebrate it enthusiastically.
- If their weight is trending toward their goal → call it out with a specific number.
- If it's been 5+ days since their last session → ask how the meal plan went.
- If the profile is thin (no workout data, no checkins) → just ask warmly how
  things are going and whether they stuck to their meals.
- Never ask more than one question. End with ONE question.

Today's date: {today}
User profile:
{profile}
"""

MAX_TOOL_ROUNDS = 6


class MealHealthAgent:

    def __init__(self, profile_path: str = "user_profile.json"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.profile = UserProfile(profile_path)
        self.history: list[dict] = []

    # ── Session control ───────────────────────────────────────────────────────

    def start_session(self):
        from datetime import date
        today = str(date.today())

        if self.profile.is_new_user():
            instruction = (
                "Greet the user warmly and ask what their main health goal is. "
                "One question only — do not ask anything else."
            )
            reply = self._run(instruction, is_system_nudge=True)
        else:
            # Proactive opening: LLM decides tone based on profile
            prompt = PROACTIVE_PROMPT.format(
                today=today,
                profile=self.profile.summary(),
            )
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=200,
            )
            reply = response.choices[0].message.content.strip()

        print(f"\nAgent: {reply}\n")

    def reset(self):
        self.profile.clear()
        self.history = []

    # ── Main chat entry ───────────────────────────────────────────────────────

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        return self._run(user_message)

    # ── ReAct loop ────────────────────────────────────────────────────────────

    def _run(self, user_message: str, is_system_nudge: bool = False) -> str:
        from datetime import date
        system = SYSTEM_PROMPT.format(
            today=str(date.today()),
            profile=self.profile.summary(),
        )
        messages = [{"role": "system", "content": system}]

        if is_system_nudge:
            messages.append({"role": "user", "content": user_message})
        else:
            messages += self.history

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=800,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                reply = msg.content.strip()
                if not is_system_nudge:
                    self.history.append({"role": "assistant", "content": reply})
                return reply

            # Serialize to plain dict — older SDK versions can't re-send Pydantic objects
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })
            print()

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                print(f"  [tool] {name}({_pretty_args(args)})")
                result = self._execute_tool(name, args)
                print(f"  [result] {_truncate(str(result))}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result) if isinstance(result, dict) else str(result),
                })

        return "Something went wrong — please try again."

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def _execute_tool(self, name: str, args: dict):
        if name == "parse_bloodwork_pdf":
            try:
                return parse_bloodwork_pdf(args.get("file_path", ""))
            except FileNotFoundError:
                return {"error": f"File not found: {args.get('file_path')}"}
            except Exception as e:
                return {"error": str(e)}

        elif name == "generate_meal_plan":
            merged = {
                "goal":               args.get("goal")               or self.profile.get("goal"),
                "restrictions":       args.get("restrictions")       or self.profile.get("restrictions"),
                "biometrics_raw":     args.get("biometrics")         or self.profile.get("biometrics_raw"),
                "bloodwork":          args.get("bloodwork")          or self.profile.get("bloodwork"),
                "plan_feedback":      args.get("feedback")           or self.profile.get("plan_feedback"),
                "cuisine_preference": args.get("cuisine_preference") or self.profile.get("cuisine_preference"),
            }
            plan = generate_meal_plan(merged)
            self.profile.set("current_plan", plan)
            return plan

        elif name == "generate_grocery_list":
            plan = args.get("meal_plan") or self.profile.get("current_plan", {})
            return generate_grocery_list(plan)

        elif name == "save_profile":
            key, value = args.get("key"), args.get("value")
            if key:
                self.profile.set(key, value)
                return {"saved": key}
            return {"error": "No key provided"}

        elif name == "log_workout":
            self.profile.log_workout(
                workout_type=args.get("workout_type", "unknown"),
                duration_min=args.get("duration_min", 0),
                notes=args.get("notes", ""),
            )
            return {
                "logged": True,
                "workouts_this_week": len(self.profile.workouts_this_week()),
            }

        elif name == "log_weight":
            w = args.get("weight_kg")
            if w:
                self.profile.log_weight(float(w))
                trend = self.profile.weight_trend()
                return {"logged": True, "trend": trend}
            return {"error": "No weight provided"}

        elif name == "add_weekly_checkin":
            self.profile.append_checkin(args.get("note", ""))
            return {"saved": "checkin"}

        else:
            return {"error": f"Unknown tool: {name}"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pretty_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, dict):
            parts.append(f"{k}={{...}}")
        elif isinstance(v, str) and len(v) > 40:
            parts.append(f"{k}='{v[:37]}...'")
        else:
            parts.append(f"{k}={repr(v)}")
    return ", ".join(parts)


def _truncate(s: str, n: int = 120) -> str:
    return s if len(s) <= n else s[:n] + "..."