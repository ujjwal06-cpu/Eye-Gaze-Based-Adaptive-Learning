"""
Week 1 -- End-to-end demo (so far)
====================================
Pipeline built so far:

    Simulated Gaze Selection -> Intent

Weeks 2-8 will extend this with a real phrase dataset, NLP cleaning,
ML/DistilBERT classification, message generation, personalization,
and text-to-speech. For now this proves the intent taxonomy and the
simulated selector work end-to-end -- this is what you can show your
professor on Wednesday.
"""

from intents import get_intent, list_intents, critical_intents
from simulate_gaze import simulate_gaze_selection


def run_demo():
    print("=" * 60)
    print("WEEK 1 DEMO -- Intent Taxonomy + Simulated Gaze Selection")
    print("=" * 60)

    print("\nAll defined intents:")
    for name in list_intents():
        info = get_intent(name)
        flag = "  [CRITICAL]" if info.urgency == "critical" else ""
        print(f"  - {name.value:<16} {info.description}{flag}")

    print("\nCritical (urgent) intents:", [n.value for n in critical_intents()])

    print("\nSimulating 3 gaze selections:")
    for _ in range(3):
        selected = simulate_gaze_selection()
        info = get_intent(selected)
        print(f'  Simulated selection -> {selected.value}  '
              f'(default message: "{info.default_message}")')


if __name__ == "__main__":
    run_demo()
