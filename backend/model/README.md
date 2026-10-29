# Model

Run from `backend/`:

```bash
python -m model.simulate              # 60 days, 3 patients -> model/data/{requests,gaze,episodes}.csv
python -m model.train                 # time-split eval, tunes confusion threshold, saves model/artifacts/model.joblib
python -m model.train --refit-all     # same, but the saved model is refit on all data
```

`from model.predict import predict` then gives `predict(patient_id, ts)` returning
`{suggestions: [{label, prob}] x3, confused, confusion_score}`.

## Design

- **Features** (`features.py`, shared by train and predict): hour, minute, day_of_week,
  minutes since the last request of each label (capped at 24h), one-hot last request label,
  zone switches / gaze entropy / mean dwell over the last 30 s.
- **Entropy** is dwell-weighted Shannon entropy over the 6 zones plus "no zone", log base 7,
  so it is 0-1. It is the `confusion_score`.
- **Classifier:** one RandomForest per patient (patient_id is not a feature, and routines
  differ), plus a global RandomForest for unknown patients. LightGBM is not required; swap
  `fit_rf` in `train.py` if you want it.
- **Confused** = entropy > tuned threshold and >= 5 gaze events in the window. Then the
  suggestions come from the patient's time-of-day prior (circular Gaussian kernel, sigma 30 min,
  over past request times) instead of the classifier.
- `ts` is treated as the patient's wall-clock time (timezone dropped, not converted).

## Simulated data

Patient 1: water ~10:00, food ~13:00, rest ~15:00. Patients 2 and 3 have longer routines, some
weekday-only or weekend-only. Each request has +-20 min jitter, 10% are replaced by a random
label at a random time, and 5% of days are "confused": 2 routine requests are not made and
3-8 min of scattered gaze (short dwells, many zone switches) appear instead.

`episodes.csv` is simulator ground truth (focus / idle / confused windows) used only to tune
and evaluate the threshold. It is not part of the contract.

## Plugging in the real database

`predict()` reads history through a data source. The default reads the simulated CSVs. In the
backend, call `model.predict.set_data_source(obj)` with an object that has
`requests(patient_id, until)` and `gaze(patient_id, since, until)`; see the docstring in
`predict.py` for the exact return types.
