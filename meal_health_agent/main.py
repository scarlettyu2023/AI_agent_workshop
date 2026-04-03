"""
Meal & Health Agent — main.py

Usage:
  python main.py           # normal conversation
  python main.py --coach   # proactive daily check-in (run via cron)
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from agent import MealHealthAgent
from coach import get_coach_message
from memory import UserProfile


def run_coach():
    """Proactive mode: analyse profile, print a coaching message, and exit."""
    profile = UserProfile()
    if profile.is_new_user():
        print("No profile found. Run 'python main.py' to get started.")
        return
    print("\n--- Daily check-in ---")
    msg = get_coach_message(profile)
    print(f"\nCoach: {msg}\n")


def run_chat():
    """Normal conversational mode."""
    print("\n🥗 Meal & Health Agent")
    print("=" * 42)
    print("Commands: 'quit'  'reset'  'coach' (instant check-in)\n")

    agent = MealHealthAgent()
    agent.start_session()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Agent: See you next time! Keep it up. 💪")
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("Agent: Profile cleared. Starting fresh!\n")
            agent.start_session()
            continue
        if user_input.lower() == "coach":
            msg = get_coach_message(agent.profile)
            print(f"\nCoach: {msg}\n")
            continue

        response = agent.chat(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    if "--coach" in sys.argv:
        run_coach()
    else:
        run_chat()
