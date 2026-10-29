"""Turn predicted needs + patient state into caregiver actions, with reasons and a priority."""
from __future__ import annotations

from .config import LABEL_TEXT, LABELS, fmt_duration
from . import webcam as wc

NEED_ACTIONS = {
    "water": ("Offer water", "Offer a drink, following the patient's swallowing and fluid plan."),
    "food": ("Offer food", "Offer a meal or snack in line with their diet orders."),
    "help": ("Check on the patient", "Go to the bedside; ask yes/no questions or bring up GazePad."),
    "pain": ("Assess pain", "Ask for a pain score with yes/no gaze answers and check the analgesia chart."),
    "bathroom": ("Offer toileting help", "Offer bathroom assistance or check continence care."),
    "rest": ("Help the patient rest", "Reposition, dim the lights and reduce noise."),
}
URGENT_LABELS = {"help", "pain"}
PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def build(needs, ws, confused, last_request, feedback_since_last, now, upcoming):
    """needs: [{label, prob, reasons}] sorted by prob. Returns (alert, actions)."""
    actions = []

    # 1. an urgent request nobody has marked done yet
    open_request = None
    if last_request and last_request["label"] in URGENT_LABELS:
        age = now - last_request["t"]
        if age <= 10 * 60 and not feedback_since_last:
            open_request = last_request
            lbl = last_request["label"]
            title, detail = NEED_ACTIONS[lbl]
            actions.append({
                "id": f"request:{lbl}", "kind": "request", "label": lbl, "priority": "high",
                "title": f"Respond: patient asked for {LABEL_TEXT[lbl]}",
                "detail": f"Requested {fmt_duration(age)} ago. {detail}",
                "prob": 1.0, "reasons": [f"{LABEL_TEXT[lbl]} selected on GazePad {fmt_duration(age)} ago"],
            })

    # 2. webcam states
    st = ws.get("states", {}) if ws.get("available") else {}
    if ws.get("available") and ws.get("emergency"):
        actions.append({
            "id": "state:emergency", "kind": "state", "priority": "high", "title": "Help tile used in the last 90 s",
            "detail": "The patient selected Help. Go to the bedside.", "prob": 1.0,
            "reasons": ["Emergency_Button > 0 in the latest webcam windows"],
        })
    if st.get("distress", 0) >= 0.6:
        actions.append({
            "id": "state:distress", "kind": "state", "priority": "high" if st["distress"] >= 0.8 else "medium",
            "title": "Check for discomfort",
            "detail": "Webcam signals suggest the patient may be uncomfortable or in pain.",
            "prob": st["distress"], "reasons": [wc.describe(ws, "distress")],
        })
    if confused or st.get("confusion", 0) >= 0.65:
        actions.append({
            "id": "state:confusion", "kind": "state", "priority": "medium", "title": "Reassure and reorient",
            "detail": "Speak calmly, explain where they are, and offer the GazePad choices one at a time.",
            "prob": max(st.get("confusion", 0), 0.7 if confused else 0),
            "reasons": ["Gaze is scattered across the tiles" if confused else wc.describe(ws, "confusion")],
        })
    if st.get("fatigue", 0) >= 0.65:
        actions.append({
            "id": "state:fatigue", "kind": "state", "priority": "low", "title": "Patient may be tired",
            "detail": "Keep interactions short and let them rest; recalibrate GazePad later if accuracy drops.",
            "prob": st["fatigue"], "reasons": [wc.describe(ws, "fatigue")],
        })
    age = ws.get("last_window_age_s")
    if not ws.get("available") and age is not None and 60 < age < 30 * 60:
        actions.append({
            "id": "device:camera", "kind": "device", "priority": "low", "title": "Check the camera",
            "detail": f"No webcam window for {fmt_duration(age)}: the face may be out of view or GazePad closed.",
            "prob": 0.0, "reasons": [],
        })

    # 3. predicted needs
    n_need = 0
    for n in needs:
        if n_need >= 3 or n["prob"] < 0.25:
            break
        if open_request and n["label"] == open_request["label"]:
            continue
        title, detail = NEED_ACTIONS[n["label"]]
        pri = "medium" if n["prob"] >= 0.45 else "low"
        if n["label"] in URGENT_LABELS and st.get("distress", 0) >= 0.6:
            pri = "high"
        actions.append({
            "id": f"need:{n['label']}", "kind": "need", "label": n["label"], "priority": pri,
            "title": title, "detail": detail, "prob": n["prob"],
            "reasons": n["reasons"][:3] or [f"{round(n['prob'] * 100)}% of the model's estimate"],
        })
        n_need += 1

    # 4. what is coming up (only if nothing more pressing)
    for u in upcoming[:2]:
        if any(a.get("label") == u["label"] for a in actions):
            continue
        title, detail = NEED_ACTIONS[u["label"]]
        actions.append({
            "id": f"upcoming:{u['label']}", "kind": "upcoming", "label": u["label"], "priority": "low",
            "title": f"Plan: {title.lower()} in ~{u['in_min']} min", "detail": detail,
            "prob": u["usual_fraction"], "reasons": [u["text"]],
        })

    actions.sort(key=lambda a: (PRIORITY_ORDER[a["priority"]], -a["prob"]))

    if any(a["priority"] == "high" for a in actions):
        top = next(a for a in actions if a["priority"] == "high")
        alert = {"level": "urgent", "text": top["title"]}
    elif any(a["priority"] == "medium" for a in actions):
        top = next(a for a in actions if a["priority"] == "medium")
        alert = {"level": "attention", "text": top["title"]}
    else:
        alert = {"level": "routine", "text": "No pressing needs predicted"}
    return alert, actions


__all__ = ["build", "NEED_ACTIONS", "LABELS"]
