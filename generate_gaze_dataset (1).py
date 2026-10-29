"""
Week 2 Deliverable: Synthetic Gaze Dataset Generator
------------------------------------------------------
Generates a synthetic gaze-tracking dataset covering all the pattern
categories specified in the Week 2 roadmap:
    - Normal gaze
    - Left / Right / Up / Down gaze
    - Long fixation
    - Short fixation
    - Random gaze
    - Blink events
    - Head movement

Columns match the template given in the roadmap document:
    User_ID, Time, Gaze_X, Gaze_Y, Blink, Head_X, Head_Y
plus an extra ground-truth column, Pattern_Label, which is NOT something
a real eye tracker would give you -- we add it here only because this is
SYNTHETIC data and we (the generator) know what pattern we told it to draw.
This label is what will let Week 4-6 (fixation detection / feature
extraction) be validated against a known ground truth before ever touching
real Tobii/webcam data.

Screen assumed: 1280x720 px (change SCREEN_W/SCREEN_H to match your setup).
Sample rate: 60 Hz (matches typical webcam / Tobii consumer tracker rate).
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixed seed -> reproducible dataset

SCREEN_W, SCREEN_H = 1280, 720
FS = 60.0          # sampling rate (Hz)
DT = 1.0 / FS       # seconds per sample

CENTER = (SCREEN_W / 2, SCREEN_H / 2)
ZONES = {
    "center": CENTER,
    "left":   (SCREEN_W * 0.10, SCREEN_H * 0.50),
    "right":  (SCREEN_W * 0.90, SCREEN_H * 0.50),
    "up":     (SCREEN_W * 0.50, SCREEN_H * 0.10),
    "down":   (SCREEN_W * 0.50, SCREEN_H * 0.90),
}

PATTERNS = [
    "normal_gaze", "left_gaze", "right_gaze", "up_gaze", "down_gaze",
    "long_fixation", "short_fixation", "random_gaze", "head_movement",
]


def clip(x, y):
    return np.clip(x, 0, SCREEN_W), np.clip(y, 0, SCREEN_H)


def gen_fixation(center, duration_s, dispersion_px, head_base):
    """A stable fixation: small Gaussian jitter around a fixed point."""
    n = max(1, int(duration_s * FS))
    x = center[0] + rng.normal(0, dispersion_px, n)
    y = center[1] + rng.normal(0, dispersion_px, n)
    x, y = clip(x, y)
    head_x = head_base[0] + rng.normal(0, 0.05, n)
    head_y = head_base[1] + rng.normal(0, 0.05, n)
    return x, y, head_x, head_y


def gen_saccade(start, end, duration_s, head_base):
    """Fast ballistic transition between two points (linear + small noise)."""
    n = max(1, int(duration_s * FS))
    t = np.linspace(0, 1, n)
    x = start[0] + (end[0] - start[0]) * t + rng.normal(0, 3, n)
    y = start[1] + (end[1] - start[1]) * t + rng.normal(0, 3, n)
    x, y = clip(x, y)
    head_x = head_base[0] + rng.normal(0, 0.05, n)
    head_y = head_base[1] + rng.normal(0, 0.05, n)
    return x, y, head_x, head_y


def gen_random_gaze(duration_s, head_base):
    """Unconstrained, high-variance gaze -- no clear target (e.g. distraction/noise)."""
    n = max(1, int(duration_s * FS))
    x = rng.uniform(0, SCREEN_W, n)
    y = rng.uniform(0, SCREEN_H, n)
    head_x = head_base[0] + rng.normal(0, 0.15, n)
    head_y = head_base[1] + rng.normal(0, 0.15, n)
    return x, y, head_x, head_y


def gen_head_movement(center, duration_s, head_base):
    """Deliberate head turn: gaze partially compensates, head offset grows then settles."""
    n = max(1, int(duration_s * FS))
    t = np.linspace(0, 1, n)
    swing = rng.choice([-1, 1]) * rng.uniform(3, 8)  # degrees-ish offset
    head_x = head_base[0] + swing * np.sin(np.pi * t) + rng.normal(0, 0.1, n)
    head_y = head_base[1] + (swing / 2) * np.sin(np.pi * t) + rng.normal(0, 0.1, n)
    # gaze drifts slightly opposite to head to simulate partial vestibulo-ocular compensation
    x = center[0] - swing * 4 * np.sin(np.pi * t) + rng.normal(0, 6, n)
    y = center[1] + rng.normal(0, 6, n)
    x, y = clip(x, y)
    return x, y, head_x, head_y


def inject_blinks(n_samples, blink_rate_per_min=17, mean_blink_ms=200):
    """Return boolean blink mask for a segment of n_samples using a Poisson process."""
    mask = np.zeros(n_samples, dtype=int)
    duration_s = n_samples / FS
    expected_blinks = blink_rate_per_min / 60.0 * duration_s
    n_blinks = rng.poisson(max(expected_blinks, 0))
    for _ in range(n_blinks):
        start = rng.integers(0, max(n_samples - 1, 1))
        blink_len = max(1, int(rng.normal(mean_blink_ms, 60) / 1000 * FS))
        mask[start:start + blink_len] = 1
    return mask


def generate_user_session(user_id, session_duration_s=80, head_base=None):
    if head_base is None:
        head_base = (rng.normal(0, 0.5), rng.normal(0, 0.5))  # per-user resting head pose

    # per-user individual variability (ties into later personalization work)
    fixation_dispersion = rng.uniform(3, 8)          # px, stability differs per person
    blink_rate = rng.uniform(12, 22)                 # blinks/min differs per person

    rows_x, rows_y, rows_head_x, rows_head_y, rows_label = [], [], [], [], []
    blink_all = []
    t_elapsed = 0.0
    current_pos = CENTER

    while t_elapsed < session_duration_s:
        pattern = rng.choice(PATTERNS)

        if pattern == "normal_gaze":
            target = ZONES["center"]
            sx, sy, shx, shy = gen_saccade(current_pos, target, 0.08, head_base)
            fx, fy, fhx, fhy = gen_fixation(target, rng.uniform(0.3, 0.6), fixation_dispersion, head_base)
            x = np.concatenate([sx, fx]); y = np.concatenate([sy, fy])
            hx = np.concatenate([shx, fhx]); hy = np.concatenate([shy, fhy])
            current_pos = target

        elif pattern in ("left_gaze", "right_gaze", "up_gaze", "down_gaze"):
            zone = pattern.split("_")[0]
            target = ZONES[zone]
            sx, sy, shx, shy = gen_saccade(current_pos, target, 0.08, head_base)
            fx, fy, fhx, fhy = gen_fixation(target, rng.uniform(0.5, 1.0), fixation_dispersion, head_base)
            x = np.concatenate([sx, fx]); y = np.concatenate([sy, fy])
            hx = np.concatenate([shx, fhx]); hy = np.concatenate([shy, fhy])
            current_pos = target

        elif pattern == "long_fixation":
            target = current_pos
            x, y, hx, hy = gen_fixation(target, rng.uniform(0.8, 1.6), fixation_dispersion * 0.7, head_base)

        elif pattern == "short_fixation":
            target = (rng.uniform(0.2, 0.8) * SCREEN_W, rng.uniform(0.2, 0.8) * SCREEN_H)
            sx, sy, shx, shy = gen_saccade(current_pos, target, 0.05, head_base)
            fx, fy, fhx, fhy = gen_fixation(target, rng.uniform(0.10, 0.25), fixation_dispersion, head_base)
            x = np.concatenate([sx, fx]); y = np.concatenate([sy, fy])
            hx = np.concatenate([shx, fhx]); hy = np.concatenate([shy, fhy])
            current_pos = target

        elif pattern == "random_gaze":
            x, y, hx, hy = gen_random_gaze(rng.uniform(0.5, 1.2), head_base)
            current_pos = (x[-1], y[-1])

        elif pattern == "head_movement":
            x, y, hx, hy = gen_head_movement(current_pos, rng.uniform(0.6, 1.2), head_base)
            current_pos = (x[-1], y[-1])

        n = len(x)
        blink_mask = inject_blinks(n, blink_rate_per_min=blink_rate)
        x_out = np.where(blink_mask == 1, np.nan, x)
        y_out = np.where(blink_mask == 1, np.nan, y)

        rows_x.append(x_out); rows_y.append(y_out)
        rows_head_x.append(hx); rows_head_y.append(hy)
        rows_label.extend([pattern] * n)
        blink_all.append(blink_mask)
        t_elapsed += n * DT

    gaze_x = np.concatenate(rows_x)
    gaze_y = np.concatenate(rows_y)
    head_x = np.concatenate(rows_head_x)
    head_y = np.concatenate(rows_head_y)
    blink = np.concatenate(blink_all)
    n_total = len(gaze_x)
    time = np.arange(n_total) * DT

    df = pd.DataFrame({
        "User_ID": user_id,
        "Time": np.round(time, 4),
        "Gaze_X": np.round(gaze_x, 2),
        "Gaze_Y": np.round(gaze_y, 2),
        "Blink": blink,
        "Head_X": np.round(head_x, 3),
        "Head_Y": np.round(head_y, 3),
        "Pattern_Label": rows_label[:n_total],
    })
    return df


if __name__ == "__main__":
    N_USERS = 12
    all_sessions = []
    for i in range(1, N_USERS + 1):
        uid = f"U{i:02d}"
        df_user = generate_user_session(uid, session_duration_s=80)
        all_sessions.append(df_user)

    full_df = pd.concat(all_sessions, ignore_index=True)
    out_path = "/mnt/user-data/outputs/synthetic_gaze_dataset_week2.csv"
    full_df.to_csv(out_path, index=False)

    print("Total records:", len(full_df))
    print("Users:", full_df.User_ID.nunique())
    print("\nRecords per pattern:\n", full_df.Pattern_Label.value_counts())
    print("\nOverall blink rate (fraction of samples flagged as blink):",
          full_df.Blink.mean())
    print("\nMissing (blink-caused NaN) Gaze_X fraction:", full_df.Gaze_X.isna().mean())
    print("\nSample rows:\n", full_df.head(8))
