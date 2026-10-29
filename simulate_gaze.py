"""
Week 1 -- Simulated Gaze Selection
===================================
The roadmap is explicit: this module doesn't need real eye tracking.
It simulates the OUTPUT of a gaze/blink-scanning system -- i.e. a
single selected intent -- so the rest of the pipeline (Weeks 2-8) can
be built and demoed without waiting on hardware or a tracking model.

If a real gaze-tracking system (e.g. a blink-scanning AAC device) is
integrated later, it only needs to produce the same output type
(an IntentName) and everything downstream keeps working unchanged.
"""

import random
from intents import IntentName, list_intents


def simulate_gaze_selection(choice: str | None = None) -> IntentName:
    """Simulate what a gaze/blink-scanning system would output.

    Args:
        choice: Optionally force a specific intent by name (e.g. "WATER").
            If omitted, a random intent is returned -- useful for demos
            and for generating test traffic before a real UI exists.

    Returns:
        The selected IntentName.
    """
    if choice is not None:
        return IntentName(choice.upper())
    return random.choice(list_intents())


if __name__ == "__main__":
    # Quick manual check: run this file directly to see a few
    # simulated selections printed out.
    for _ in range(5):
        selected = simulate_gaze_selection()
        print(f"Simulated gaze selection -> {selected.value}")
