"""
Week 1 -- Communication Intent Definitions
===========================================
This module defines the fixed set of communication intents the AAC
(Augmentative and Alternative Communication) system recognizes.

Every later stage of the pipeline builds on this taxonomy:
  - Week 2's dataset labels examples with these intent names
  - Week 4/5's classifiers predict one of these labels
  - Week 6's message generator maps each label to a spoken sentence
  - Week 7's personalization counts how often each label is picked

Getting this list right now is the foundation for everything after it.
"""

from dataclasses import dataclass
from enum import Enum


class IntentName(str, Enum):
    """The fixed vocabulary of intents the system can recognize."""
    WATER = "WATER"
    FOOD = "FOOD"
    HELP = "HELP"
    CALL = "CALL"
    EMERGENCY = "EMERGENCY"
    MEDICINE = "MEDICINE"
    TOILET = "TOILET"
    PAIN = "PAIN"
    ENTERTAINMENT = "ENTERTAINMENT"
    GENERAL_MESSAGE = "GENERAL_MESSAGE"


@dataclass(frozen=True)
class Intent:
    """Metadata for a single intent.

    Attributes:
        name: The canonical label (matches IntentName).
        description: What this intent means -- keeps Week 2's dataset
            labeling consistent.
        urgency: "critical" intents (EMERGENCY, MEDICINE, PAIN) are
            flagged so a later stage can let them skip the normal
            queue; "routine" intents follow the standard flow.
        default_message: A simple placeholder phrase, used until
            Week 6 builds real message generation.
    """
    name: IntentName
    description: str
    urgency: str  # "critical" | "routine"
    default_message: str


# ---------------------------------------------------------------------------
# The intent registry -- the single source of truth for Week 1.
# ---------------------------------------------------------------------------
INTENTS: dict[IntentName, Intent] = {
    IntentName.WATER: Intent(
        IntentName.WATER, "Patient wants water / is thirsty.",
        "routine", "I need some water, please."),
    IntentName.FOOD: Intent(
        IntentName.FOOD, "Patient wants food / is hungry.",
        "routine", "I need some food, please."),
    IntentName.HELP: Intent(
        IntentName.HELP, "Patient needs general assistance.",
        "routine", "I need help, please come here."),
    IntentName.CALL: Intent(
        IntentName.CALL, "Patient wants a specific person contacted (e.g. caregiver).",
        "routine", "Please call my caregiver."),
    IntentName.EMERGENCY: Intent(
        IntentName.EMERGENCY, "Patient is signaling a medical or safety emergency.",
        "critical", "This is an emergency. I need immediate help."),
    IntentName.MEDICINE: Intent(
        IntentName.MEDICINE, "Patient needs medication.",
        "critical", "I need my medicine now."),
    IntentName.TOILET: Intent(
        IntentName.TOILET, "Patient needs to use the restroom.",
        "routine", "I need to use the restroom."),
    IntentName.PAIN: Intent(
        IntentName.PAIN, "Patient is reporting pain or discomfort.",
        "critical", "I am in pain. Please check on me."),
    IntentName.ENTERTAINMENT: Intent(
        IntentName.ENTERTAINMENT, "Patient wants entertainment (TV, music, etc.).",
        "routine", "Can you turn on some entertainment for me?"),
    IntentName.GENERAL_MESSAGE: Intent(
        IntentName.GENERAL_MESSAGE, "Catch-all for anything not covered above.",
        "routine", "I have something to say."),
}


def get_intent(name: IntentName) -> Intent:
    """Look up full intent metadata by name."""
    return INTENTS[name]


def list_intents() -> list[IntentName]:
    """Return all defined intent names, in definition order."""
    return list(INTENTS.keys())


def critical_intents() -> list[IntentName]:
    """Return only the intents flagged as critical/urgent."""
    return [n for n, i in INTENTS.items() if i.urgency == "critical"]
