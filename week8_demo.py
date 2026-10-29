"""
Week 8 Demo -- Selected Request -> Intent -> Message -> TTS -> Voice Output
==============================================================================
This is the roadmap's Week 8 diagram end-to-end, using a specific
user's personalization (Week 7) to pick their most natural message,
speaking it (Week 8), and logging the whole event.
"""

from intents import IntentName
from message_templates import generate_message
from user_history import UserHistoryStore
from tts_module import speak
from communication_log import log_communication, read_log


def handle_request(user_id: str, intent: IntentName, store: UserHistoryStore):
    """One full cycle: record the selection, generate a message, speak it, log it."""
    user = store.get_or_create(user_id)
    user.record(intent)  # Week 7: update this user's frequency history

    message = generate_message(intent)  # Week 6: intent -> natural sentence
    spoke_for_real = speak(message)  # Week 8: attempt real TTS, else fallback
    log_communication(user_id, intent.value, message, spoken=spoke_for_real)

    return message, spoke_for_real


def run_demo():
    print("=" * 70)
    print("WEEK 8 DEMO -- Selected Request -> Intent -> Message -> TTS -> Log")
    print("=" * 70)

    store = UserHistoryStore("user_history.json")

    requests = [
        ("User A", IntentName.WATER),
        ("User A", IntentName.HELP),
        ("User B", IntentName.PAIN),
    ]

    for user_id, intent in requests:
        print(f"\n--- {user_id} selects: {intent.value} ---")
        message, spoke_for_real = handle_request(user_id, intent, store)
        print(f"Message generated: \"{message}\"")
        print(f"Spoken for real:   {spoke_for_real}")

    store.save()

    print(f"\n{'-'*70}\nFull communication log so far:\n{'-'*70}")
    for entry in read_log():
        print(f"  [{entry['timestamp']}] {entry['user_id']:<8} "
              f"{entry['intent']:<12} \"{entry['message']}\"")


if __name__ == "__main__":
    run_demo()
