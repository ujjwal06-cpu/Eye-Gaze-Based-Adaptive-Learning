"""
Integration -- Diya + Akshaya + Harshini, End to End
=========================================================
    Diya's processed gaze + dwell events (REAL data, REAL files)
        -> Akshaya's adaptive dwell-time rule (REAL logic, rescaled to fit)
        -> Grid mapping (NEW glue layer -- the piece nobody owned)
        -> Harshini's message / personalization / TTS / log pipeline (REAL, built earlier)

For every dwell event in Diya's dwell_events.csv:
  1. Find the matching feature window (nearest window_start_time, same
     user) in her gaze_features_dataset.csv to get that moment's
     Gaze_Stability.
  2. Run it through Akshaya's rescaled adaptive-dwell rule to get a
     personalized dwell threshold (instead of her fixed 0.5s).
  3. Compare: would FIXED vs ADAPTIVE thresholds have confirmed this
     selection? (Both Diya's and Akshaya's Week 8 evaluation ask for
     exactly this comparison -- this answers it with real shared data
     instead of each person's separate toy example.)
  4. If adaptively confirmed, map the AOI to an intent and run it
     through Harshini's full pipeline: generate a personalized
     message, attempt TTS, log the communication.
"""

import pandas as pd

from grid_mapping import aoi_to_intent, AOI_TO_INTENT
from adaptive_dwell import StabilityRescaler, adaptive_dwell_seconds, train_behaviour_model, BEHAVIOUR_FEATURES
from user_history import UserHistoryStore
from message_templates import generate_message
from tts_module import speak
from communication_log import log_communication, read_log

FIXED_THRESHOLD_S = 0.5  # Diya's original fixed threshold, kept for comparison


def nearest_feature_row(features_user: pd.DataFrame, t: float) -> pd.Series:
    """Find the feature window whose window_start_time is closest to t."""
    idx = (features_user["window_start_time"] - t).abs().idxmin()
    return features_user.loc[idx]


def run_integration(user_ids=None, verbose_events: int = 8):
    dwell = pd.read_csv("dwell_events.csv")
    features = pd.read_csv("gaze_features_dataset.csv")

    if user_ids is not None:
        dwell = dwell[dwell.User_ID.isin(user_ids)]
        features = features[features.User_ID.isin(user_ids)]

    # Fit the stability rescaler on the REAL observed distribution (the fix
    # for the scale mismatch described in adaptive_dwell.py).
    rescaler = StabilityRescaler().fit(features["Gaze_Stability"].to_numpy())
    raw_vals = features["Gaze_Stability"].to_numpy()
    print(f"Stability rescaler (percentile-rank) fit on real data: "
          f"raw range [{raw_vals.min():.6f}, {raw_vals.max():.6f}] -> [0, 1]")

    behaviour_model = train_behaviour_model()

    store = UserHistoryStore("user_history_integrated.json")

    results = []
    shown = 0

    for user_id, dwell_user in dwell.groupby("User_ID"):
        features_user = features[features.User_ID == user_id].reset_index(drop=True)
        user = store.get_or_create(user_id)

        for _, event in dwell_user.iterrows():
            feat_row = nearest_feature_row(features_user, event["start_time"])
            raw_stability = feat_row["Gaze_Stability"]
            adaptive_s, stability_level = adaptive_dwell_seconds(raw_stability, rescaler)

            selected_fixed = event["dwell_duration"] >= FIXED_THRESHOLD_S
            selected_adaptive = event["dwell_duration"] >= adaptive_s

            behaviour_sample = pd.DataFrame([feat_row[BEHAVIOUR_FEATURES]])
            predicted_behaviour = behaviour_model.predict(behaviour_sample)[0]

            row = {
                "User_ID": user_id,
                "AOI": event["AOI"],
                "dwell_duration": round(event["dwell_duration"], 3),
                "stability_level": stability_level,
                "adaptive_threshold_s": adaptive_s,
                "selected_fixed": selected_fixed,
                "selected_adaptive": selected_adaptive,
                "predicted_behaviour": predicted_behaviour,
                "intent": None,
                "message": None,
            }

            # Only ADAPTIVE confirmation reaches Harshini's pipeline --
            # that's the actual integration payoff: a personalized
            # threshold, not the fixed one, decides what gets spoken.
            if selected_adaptive:
                intent = aoi_to_intent(event["AOI"])
                if intent is not None:
                    user.record(intent)
                    message = generate_message(intent)
                    spoke_for_real = speak(message)
                    log_communication(str(user_id), intent.value, message,
                                       spoken=spoke_for_real, path="communication_log_integrated.jsonl")
                    row["intent"] = intent.value
                    row["message"] = message

                    if shown < verbose_events:
                        print(f"\n[{user_id}] dwell={row['dwell_duration']}s on '{event['AOI']}' "
                              f"(stability={stability_level}, adaptive threshold={adaptive_s}s, "
                              f"behaviour={predicted_behaviour})")
                        print(f"  -> CONFIRMED -> intent={intent.value} -> \"{message}\" "
                              f"(spoken={spoke_for_real})")
                        shown += 1

            results.append(row)

    store.save()
    return pd.DataFrame(results)


def print_fixed_vs_adaptive_summary(results: pd.DataFrame):
    print("\n" + "=" * 70)
    print("FIXED vs ADAPTIVE -- Full Comparison (Diya's + Akshaya's Week 8 ask)")
    print("=" * 70)
    n = len(results)
    fixed_count = results["selected_fixed"].sum()
    adaptive_count = results["selected_adaptive"].sum()
    agree = (results["selected_fixed"] == results["selected_adaptive"]).sum()

    print(f"Total dwell events evaluated: {n}")
    print(f"Selections under FIXED (0.5s):    {fixed_count} ({fixed_count/n:.1%})")
    print(f"Selections under ADAPTIVE:         {adaptive_count} ({adaptive_count/n:.1%})")
    print(f"Fixed and Adaptive agreed on:      {agree}/{n} events ({agree/n:.1%})")

    print("\nBy stability level:")
    by_level = results.groupby("stability_level").agg(
        n_events=("AOI", "count"),
        adaptive_threshold_s=("adaptive_threshold_s", "first"),
        pct_selected_adaptive=("selected_adaptive", "mean"),
        pct_selected_fixed=("selected_fixed", "mean"),
    )
    print(by_level.to_string())

    unmapped = results[results["selected_adaptive"] & results["intent"].isna()]
    if len(unmapped):
        print(f"\nNote: {len(unmapped)} adaptively-confirmed selections had no intent "
              f"mapping yet (AOIs not yet covered by the 5/10 grid mapping).")


if __name__ == "__main__":
    print("=" * 70)
    print("FULL INTEGRATION -- Diya (gaze) -> Akshaya (adaptive dwell) -> Harshini (comm.)")
    print("=" * 70)
    print(f"\nCurrent AOI->Intent coverage: {len(AOI_TO_INTENT)}/10 intents "
          f"(limited by Diya's 5-zone grid -- see grid_mapping.py)")

    results = run_integration()
    print_fixed_vs_adaptive_summary(results)

    results.to_csv("integration_results.csv", index=False)
    print(f"\nSaved full event-level results -> integration_results.csv")

    print(f"\n{'-'*70}\nFinal communication log (first 10 entries):\n{'-'*70}")
    for entry in read_log("communication_log_integrated.jsonl")[:10]:
        print(f"  [{entry['user_id']}] {entry['intent']:<16} \"{entry['message']}\"")
