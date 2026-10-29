"""
Week 7 -- Personalization Demo
=================================
Reproduces the roadmap's own example (User A: WATER x25, FOOD x12,
CALL x8, HELP x5), adds a couple of lower-frequency intents for
realism, then shows:
  1. Most frequent requests, ranked
  2. A personalized display/scan order for the whole intent set
  3. That a second user's history stays completely independent
"""

from intents import IntentName
from user_history import UserHistoryStore


def simulate_usage(user, counts: dict[IntentName, int]):
    """Record `count` occurrences of each intent for this user."""
    for intent, count in counts.items():
        for _ in range(count):
            user.record(intent)


def run_demo():
    print("=" * 70)
    print("WEEK 7 DEMO -- Per-User Personalization")
    print("=" * 70)

    store = UserHistoryStore("user_history.json")

    # ---- User A: matches the roadmap's own example, plus a couple more ----
    user_a = store.get_or_create("User A")
    simulate_usage(user_a, {
        IntentName.WATER: 25,
        IntentName.FOOD: 12,
        IntentName.CALL: 8,
        IntentName.HELP: 5,
        IntentName.TOILET: 3,
        IntentName.ENTERTAINMENT: 1,
    })

    print("\nUser A -- most frequent requests:")
    for intent, count in user_a.most_frequent():
        print(f"  {intent:<16} {count} times")

    print("\nUser A -- personalized display/scan order (frequent-first):")
    print(" ", " -> ".join(user_a.personalized_order()))

    # ---- User B: a different usage pattern, to prove histories are independent ----
    user_b = store.get_or_create("User B")
    simulate_usage(user_b, {
        IntentName.PAIN: 15,
        IntentName.MEDICINE: 10,
        IntentName.HELP: 9,
        IntentName.EMERGENCY: 2,
    })

    print("\nUser B -- most frequent requests:")
    for intent, count in user_b.most_frequent():
        print(f"  {intent:<16} {count} times")

    print("\nUser B -- personalized display/scan order (frequent-first):")
    print(" ", " -> ".join(user_b.personalized_order()))

    store.save()
    print(f"\nSaved both users' histories -> {store.path}")

    print("\nNote: User A's top request (WATER) and User B's top request")
    print("(PAIN) are completely different -- confirms histories are")
    print("tracked per-user, not shared/mixed.")


if __name__ == "__main__":
    run_demo()
