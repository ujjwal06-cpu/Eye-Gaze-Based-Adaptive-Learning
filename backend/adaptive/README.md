# Adaptive engine

This is pure Python and numpy. `app/adaptive_service.py` connects it to the database, and `sim/ward.py`
uses it without a database.

## Evidence ("experts")

Each expert returns P(label | a request now), or stays silent when it has nothing to say.

| Expert | Looks at |
|---|---|
| routine | what this patient asks for at this time of day (circular kernel, σ 35 min, older requests decay over about 10 days, weekday vs weekend) |
| transition | what usually follows their last request |
| due | needs that are overdue compared with this patient's usual interval |
| webcam | distress, fatigue and confusion from the last 90 s of B1 windows, compared with the patient's own baseline (median and MAD of their last 720 windows) |
| gaze_focus | a tile they kept looking at in the last 30 s without selecting it |
| learned | a per-patient online log-linear pool. It learns how far to trust each expert and what *this patient's* webcam states lead to (for example, discomfort leads to Pain) |
| learned_global | the same pool shared by all patients, so new patients start from it |
| static_rf | the existing RandomForest from `model/`, when it has been trained |

## Combining and learning

- The experts are mixed with **Hedge weights for specialists**. Only experts that spoke are scored and
  re-weighted (log loss, η = 0.7). A **fixed share** of 2% keeps every expert re-learnable, so a routine
  change is picked up within days.
- Learning is **prequential**. When a request arrives at time t, the engine first predicts as of t − 15 s,
  stores that evaluation in `prediction_evals`, and only then updates. The accuracy shown on the dashboard
  is therefore always out-of-sample.
- **Caregiver feedback** applies after the mixture. "Done" multiplies that need by 0.4 for 30 min, and
  "Not needed" multiplies it by 0.35 for 45 min.

## Output (`GET /insights?patient_id=`)

`needs` (all 6 labels, each with reasons), `suggestions`, `confused` and `confusion_score` (the same shape
as `/predict`), `webcam` (states, z-scores, current values against the patient's usual values), `alert`,
`actions` (priority, title, detail, reasons), `upcoming` (routine needs in the next 2 h), `experts`
(weight, whether each spoke, its top label) and `model` (number learned, rolling accuracy, and what the
personal model has learned).

When a patient's model needs relearning from scratch, use `POST /model/retrain?patient_id=`, or leave out
the id to retrain everyone. Raising `ENGINE_VERSION` in `engine.py` does the same automatically.
