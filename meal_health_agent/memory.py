"""
Meal & Health Agent — memory.py
Persistent user profile stored as JSON.

Schema:
{
  "name":              str,
  "goal":              str,
  "biometrics_raw":    str,
  "restrictions":      str,
  "cuisine_preference":str,
  "bloodwork":         dict,
  "current_plan":      dict,
  "plan_feedback":     str,
  "checkins":          [str, ...],
  "workout_goal": {
    "days_per_week":   int,
    "types":           [str, ...]   e.g. ["running", "lifting"]
  },
  "workout_log": [
    {"date": "YYYY-MM-DD", "type": str, "duration_min": int, "notes": str},
    ...
  ],
  "weight_log": [
    {"date": "YYYY-MM-DD", "weight_kg": float},
    ...
  ]
}
"""

import json
import os
from datetime import date, timedelta


class UserProfile:

    def __init__(self, path: str = "user_profile.json"):
        self.path = path
        self.data: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value):
        self.data[key] = value
        self._save()

    def clear(self):
        self.data = {}
        if os.path.exists(self.path):
            os.remove(self.path)

    def is_new_user(self) -> bool:
        return not bool(self.data)

    # ── Check-ins ─────────────────────────────────────────────────────────────

    def append_checkin(self, note: str):
        checkins = self.data.get("checkins", [])
        checkins.append(note)
        self.data["checkins"] = checkins
        self._save()

    # ── Workout log ───────────────────────────────────────────────────────────

    def log_workout(self, workout_type: str, duration_min: int, notes: str = ""):
        log = self.data.get("workout_log", [])
        log.append({
            "date": str(date.today()),
            "type": workout_type,
            "duration_min": duration_min,
            "notes": notes,
        })
        self.data["workout_log"] = log
        self._save()

    def workouts_this_week(self) -> list[dict]:
        today = date.today()
        week_ago = today - timedelta(days=7)
        return [
            w for w in self.data.get("workout_log", [])
            if date.fromisoformat(w["date"]) >= week_ago
        ]

    def days_since_last_workout(self) -> int | None:
        log = self.data.get("workout_log", [])
        if not log:
            return None
        last = max(date.fromisoformat(w["date"]) for w in log)
        return (date.today() - last).days

    # ── Weight log ────────────────────────────────────────────────────────────

    def log_weight(self, weight_kg: float):
        log = self.data.get("weight_log", [])
        log.append({"date": str(date.today()), "weight_kg": weight_kg})
        self.data["weight_log"] = log
        self._save()

    def weight_trend(self) -> dict:
        log = self.data.get("weight_log", [])
        if not log:
            return {}
        log_sorted = sorted(log, key=lambda x: x["date"])
        first, last = log_sorted[0], log_sorted[-1]
        delta = round(last["weight_kg"] - first["weight_kg"], 1)
        return {
            "current_kg": last["weight_kg"],
            "start_kg":   first["weight_kg"],
            "delta_kg":   delta,
            "entries":    len(log),
        }

    # ── Summary for system prompt ─────────────────────────────────────────────

    def summary(self) -> str:
        if not self.data:
            return "No profile yet — first session."
        lines = []
        for key in ("name", "goal", "biometrics_raw", "restrictions", "cuisine_preference"):
            if self.data.get(key):
                lines.append(f"{key}: {self.data[key]}")
        if self.data.get("bloodwork"):
            bw = ", ".join(f"{k}={v}" for k, v in self.data["bloodwork"].items())
            lines.append(f"bloodwork: {bw}")
        if self.data.get("checkins"):
            lines.append(f"last_checkin: {self.data['checkins'][-1]}")
        # Workout summary
        wg = self.data.get("workout_goal")
        if wg:
            if isinstance(wg, dict):
                lines.append(f"workout_goal: {wg.get('days_per_week')} days/week, types={wg.get('types')}")
            else:
                lines.append(f"workout_goal: {wg}")
        this_week = self.workouts_this_week()
        lines.append(f"workouts_this_week: {len(this_week)}")
        days_since = self.days_since_last_workout()
        if days_since is not None:
            lines.append(f"days_since_last_workout: {days_since}")
        wt = self.weight_trend()
        if wt:
            lines.append(f"weight: {wt['current_kg']}kg (delta {wt['delta_kg']:+}kg over {wt['entries']} entries)")
        return "\n".join(lines) if lines else "Profile partially filled."