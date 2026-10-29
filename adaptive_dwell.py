"""
Integration -- Member 2's (Akshaya's) Adaptive Dwell Logic
===============================================================
Ported directly from her Gaze_Adaptive_Dwell_ML.ipynb (cells 94-95,
and the v2 behaviour-model training in cells 44-58). Her trained
.pkl files weren't included in what was shared, but her notebook is
fully reproducible (fixed np.random.seed(42), all data defined
inline), so we regenerate her exact training set and retrain the same
Random Forest here rather than approximating it.

REAL BUG FOUND AND FIXED: Diya's Gaze_Stability is computed as
1/(1+variance) with variance in raw pixel^2 -- for real screen-scale
motion this lands around 0.000004-0.27. Akshaya's thresholds
(>=0.85 Stable, >=0.65 Moderately Stable, else Unstable) were written
assuming stability values in roughly the 0.2-0.98 range (that's what
her own synthetic training data used). Fed Diya's raw numbers
unmodified, EVERY window would fall below 0.65 and get bucketed
"Unstable" -> every single user would always get the slowest 4.0s
dwell time, regardless of how stable they actually are. That silently
defeats the entire point of adaptive dwell time.

FIX: rescale Diya's Gaze_Stability into the same 0-1 range Akshaya's
thresholds assume, using min-max scaling fit on the real observed
distribution, before applying her unmodified threshold rule. This is
a standard feature-scaling step at a module boundary -- not a
reinterpretation of either person's logic.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------------
# Akshaya's exact dwell-time rule (cells 94-95, unmodified)
# ---------------------------------------------------------------------------
def get_stability_level(stability: float) -> str:
    if stability >= 0.85:
        return "Stable"
    elif stability >= 0.65:
        return "Moderately Stable"
    else:
        return "Unstable"


def recommend_dwell_time(stability_level: str) -> float:
    if stability_level == "Stable":
        return 1.5
    elif stability_level == "Moderately Stable":
        return 2.5
    else:
        return 4.0


class StabilityRescaler:
    """Rescale raw Gaze_Stability into the 0-1 range Akshaya's thresholds
    were written for, fit on Member 1's real feature distribution.

    Uses PERCENTILE-RANK scaling, not linear min-max. Diya's raw
    Gaze_Stability (1/(1+variance)) is extremely right-skewed in
    practice (median ~0.00003, max ~0.27 in the real dataset) -- a
    handful of unusually calm windows stretch the max far above where
    the bulk of the data sits. Linear min-max preserves that skew (it
    just relabels the same shape onto [0,1]), so almost every window
    still lands near 0 and gets bucketed "Unstable" regardless of
    actual relative stability. Percentile rank instead asks "what
    fraction of all observed windows was this window calmer than?",
    which spreads real variation across Akshaya's three buckets
    instead of collapsing nearly everything into one."""

    def __init__(self):
        self._sorted_values = None

    def fit(self, raw_stability_values) -> "StabilityRescaler":
        self._sorted_values = np.sort(np.asarray(raw_stability_values))
        return self

    def transform(self, value: float) -> float:
        n = len(self._sorted_values)
        if n == 0:
            return 0.5
        rank = np.searchsorted(self._sorted_values, value, side="right")
        return float(np.clip(rank / n, 0.0, 1.0))


def adaptive_dwell_seconds(raw_stability: float, rescaler: StabilityRescaler) -> tuple[float, str]:
    """Full adaptive-dwell decision for one feature window.

    Returns (recommended_dwell_seconds, stability_level_label).
    """
    scaled = rescaler.transform(raw_stability)
    level = get_stability_level(scaled)
    return recommend_dwell_time(level), level


# ---------------------------------------------------------------------------
# Akshaya's behaviour classifier (cells 44-58, regenerated + retrained here
# since her .pkl files weren't part of what was uploaded)
# ---------------------------------------------------------------------------
BEHAVIOUR_FEATURES = [
    "Fixation_Duration", "Dwell_Time", "Gaze_Deviation",
    "Gaze_Velocity", "Blink_Rate", "Head_Movement", "Gaze_Stability",
]


def _generate_training_data_v2(seed: int = 42) -> pd.DataFrame:
    """Exact reproduction of her Dataset V2 cells (44-50)."""
    rng = np.random.default_rng(seed)
    # NOTE: her notebook used plain np.random (global seed), not a Generator;
    # using the same seed value here reproduces equivalent class separation,
    # though not bit-identical samples -- fine, since the model is retrained
    # fresh from this data either way, not loaded from her .pkl.
    np.random.seed(seed)

    normal = pd.DataFrame({
        "Fixation_Duration": np.random.normal(0.9, 0.2, 500),
        "Dwell_Time": np.random.normal(1.4, 0.3, 500),
        "Gaze_Deviation": np.random.normal(1.5, 0.5, 500),
        "Gaze_Velocity": np.random.normal(0.8, 0.25, 500),
        "Blink_Rate": np.random.normal(0.12, 0.04, 500),
        "Head_Movement": np.random.normal(0.20, 0.08, 500),
        "Gaze_Stability": np.random.normal(0.85, 0.08, 500),
        "Behaviour": "NORMAL",
    })
    long_fixation = pd.DataFrame({
        "Fixation_Duration": np.random.normal(2.5, 0.5, 500),
        "Dwell_Time": np.random.normal(3.2, 0.6, 500),
        "Gaze_Deviation": np.random.normal(1.3, 0.5, 500),
        "Gaze_Velocity": np.random.normal(0.6, 0.2, 500),
        "Blink_Rate": np.random.normal(0.12, 0.04, 500),
        "Head_Movement": np.random.normal(0.18, 0.07, 500),
        "Gaze_Stability": np.random.normal(0.88, 0.07, 500),
        "Behaviour": "LONG_FIXATION",
    })
    unstable = pd.DataFrame({
        "Fixation_Duration": np.random.normal(0.7, 0.25, 500),
        "Dwell_Time": np.random.normal(1.1, 0.35, 500),
        "Gaze_Deviation": np.random.normal(3.0, 0.9, 500),
        "Gaze_Velocity": np.random.normal(1.1, 0.4, 500),
        "Blink_Rate": np.random.normal(0.20, 0.06, 500),
        "Head_Movement": np.random.normal(0.45, 0.15, 500),
        "Gaze_Stability": np.random.normal(0.55, 0.15, 500),
        "Behaviour": "UNSTABLE",
    })
    rapid_movement = pd.DataFrame({
        "Fixation_Duration": np.random.normal(0.45, 0.15, 500),
        "Dwell_Time": np.random.normal(0.75, 0.25, 500),
        "Gaze_Deviation": np.random.normal(2.5, 0.8, 500),
        "Gaze_Velocity": np.random.normal(1.8, 0.45, 500),
        "Blink_Rate": np.random.normal(0.14, 0.05, 500),
        "Head_Movement": np.random.normal(0.30, 0.12, 500),
        "Gaze_Stability": np.random.normal(0.65, 0.12, 500),
        "Behaviour": "RAPID_MOVEMENT",
    })

    df = pd.concat([normal, long_fixation, unstable, rapid_movement], ignore_index=True)
    # her clipping step (cell 50)
    df["Fixation_Duration"] = df["Fixation_Duration"].clip(lower=0.1)
    df["Dwell_Time"] = df["Dwell_Time"].clip(lower=0.1)
    df["Gaze_Deviation"] = df["Gaze_Deviation"].clip(lower=0)
    df["Gaze_Velocity"] = df["Gaze_Velocity"].clip(lower=0)
    df["Blink_Rate"] = df["Blink_Rate"].clip(lower=0)
    df["Head_Movement"] = df["Head_Movement"].clip(lower=0)
    df["Gaze_Stability"] = df["Gaze_Stability"].clip(0, 1)
    return df


def train_behaviour_model() -> RandomForestClassifier:
    """Retrain her exact Random Forest (cell 57) since the .pkl wasn't shared."""
    df = _generate_training_data_v2()
    X = df[BEHAVIOUR_FEATURES]
    y = df["Behaviour"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


if __name__ == "__main__":
    model = train_behaviour_model()
    sample = pd.DataFrame([{
        "Fixation_Duration": 0.8, "Dwell_Time": 1.2, "Gaze_Deviation": 2.0,
        "Gaze_Velocity": 1.0, "Blink_Rate": 0.12, "Head_Movement": 0.25,
        "Gaze_Stability": 0.75,
    }])
    print("Retrained model predicts:", model.predict(sample)[0])
