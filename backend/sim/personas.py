"""Virtual patients for the ward simulation.

Each persona has
  routines   (label, local time, jitter, probability) daily habits
  episodes   needs that build up over minutes and (usually) show on the webcam first:
             distress before pain/help, fatigue before rest, confusion before help
  spurious   webcam changes that lead to NO request (so the webcam is never a perfect cue)
  noise      random requests with no warning at all
  drift      a routine change part-way through (e.g. new therapy timetable), to test adaptation

These are modelling assumptions for a simulation, chosen to be plausible, not clinical data.
"""
from __future__ import annotations

from dataclasses import dataclass, field


def hm(h: int, m: int = 0) -> int:
    return h * 60 + m


@dataclass
class Routine:
    label: str
    minute: int
    jitter: float = 20.0
    prob: float = 0.92
    weekdays: tuple | None = None


@dataclass
class Episode:
    state: str                 # distress | fatigue | confusion
    labels: dict               # resulting request label -> probability
    per_day: float
    hours: tuple = (8, 22)     # local hours when it can happen
    lead_min: tuple = (6, 15)  # signature builds up over this many minutes
    p_visible: float = 0.8     # chance the webcam shows it at all
    p_request: float = 1.0     # chance it ends in a request (else it fades)


@dataclass
class Persona:
    key: str
    name: str
    notes: str
    routines: list
    episodes: list = field(default_factory=list)
    spurious: dict = field(default_factory=dict)      # state -> per day
    noise_per_day: float = 0.8
    baseline: dict = field(default_factory=dict)      # personal webcam baseline overrides
    drift: tuple | None = None                        # (fraction of days, shift minutes)
    awake: tuple = (hm(7), hm(22, 30))


PERSONAS = [
    Persona(
        "post_op", "Kavya (post-op, hip)", "Post-surgical, pain-prone; gaze input after anaesthesia",
        routines=[
            Routine("bathroom", hm(7, 30)), Routine("food", hm(8, 30)), Routine("water", hm(9, 30), 30),
            Routine("water", hm(12), 30), Routine("food", hm(13)), Routine("bathroom", hm(16)),
            Routine("water", hm(15, 30), 30), Routine("water", hm(18), 30), Routine("food", hm(19, 30)),
        ],
        episodes=[
            Episode("distress", {"pain": 0.85, "help": 0.15}, 2.0, (8, 22), (8, 18), 0.8),
            Episode("fatigue", {"rest": 1.0}, 0.8, (13, 17), (8, 15), 0.75),
        ],
        spurious={"distress": 0.6},
        baseline={"head": 3.0, "stability": 0.34, "blink": 3.0},
    ),
    Persona(
        "als", "Ravi (ALS)", "ALS, communicates by gaze only; tires in the afternoon",
        routines=[
            Routine("food", hm(9)), Routine("water", hm(10)), Routine("bathroom", hm(11)),
            Routine("food", hm(13)), Routine("water", hm(14, 30)), Routine("bathroom", hm(17)),
            Routine("water", hm(18, 30)), Routine("food", hm(19)),
        ],
        episodes=[
            Episode("fatigue", {"rest": 1.0}, 1.6, (13, 21), (10, 20), 0.85),
            Episode("distress", {"bathroom": 0.6, "pain": 0.4}, 0.5, (9, 21), (5, 10), 0.7),
        ],
        spurious={"fatigue": 0.5},
        baseline={"head": 1.5, "stability": 0.36, "blink": 2.5, "fixation": 0.4},
    ),
    Persona(
        "dementia", "Meena (dementia)", "Early dementia; sundowning in the evening",
        routines=[
            Routine("food", hm(8)), Routine("water", hm(10, 30)), Routine("food", hm(12, 30)),
            Routine("rest", hm(14)), Routine("water", hm(15, 30)), Routine("food", hm(18)),
            Routine("rest", hm(21)),
        ],
        episodes=[
            Episode("confusion", {"help": 0.8, "bathroom": 0.2}, 1.5, (16, 21), (6, 14), 0.8),
            Episode("distress", {"pain": 0.7, "help": 0.3}, 0.3, (8, 21), (6, 12), 0.7),
        ],
        spurious={"confusion": 0.8},
        baseline={"head": 4.0, "stability": 0.31, "deviation": 5.0},
    ),
    Persona(
        "stroke", "Asha (post-stroke)", "Post-stroke; very regular, timetable changes halfway",
        routines=[
            Routine("bathroom", hm(8, 30)), Routine("water", hm(10)), Routine("food", hm(13)),
            Routine("rest", hm(15)), Routine("water", hm(17)), Routine("bathroom", hm(18)),
            Routine("food", hm(19, 30)),
        ],
        episodes=[Episode("distress", {"pain": 1.0}, 0.3, (9, 21), (6, 12), 0.75)],
        noise_per_day=0.5,
        drift=(0.55, 90),
        baseline={"head": 2.0, "stability": 0.35},
    ),
    Persona(
        "stepdown", "Arjun (ICU step-down)", "Recovering from ICU; irregular, frequent needs",
        routines=[
            Routine("water", hm(9), 45, 0.7), Routine("water", hm(12), 45, 0.7), Routine("water", hm(15), 45, 0.7),
            Routine("water", hm(18), 45, 0.7), Routine("food", hm(8, 30), 30, 0.8), Routine("food", hm(13), 30, 0.8),
            Routine("food", hm(19), 30, 0.8),
        ],
        episodes=[
            Episode("distress", {"help": 0.5, "pain": 0.5}, 1.5, (8, 22), (5, 12), 0.75),
            Episode("distress", {"bathroom": 1.0}, 1.0, (8, 22), (4, 8), 0.6),
            Episode("fatigue", {"rest": 1.0}, 1.0, (12, 21), (8, 15), 0.7),
        ],
        spurious={"distress": 0.8, "fatigue": 0.4},
        noise_per_day=1.5,
        baseline={"head": 5.0, "stability": 0.32, "blink": 3.5},
    ),
]

BY_KEY = {p.key: p for p in PERSONAS}
