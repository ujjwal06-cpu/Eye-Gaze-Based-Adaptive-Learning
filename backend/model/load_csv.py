import json
import pandas as pd
import requests

API = "http://localhost:8000"
ID_MAP = {}   # if 2c showed p1/p2/p3, set {"p1": 1, "p2": 2, "p3": 3}; otherwise leave {}

req = pd.read_csv("model/data/requests.csv")
if ID_MAP:
    req["patient_id"] = req["patient_id"].map(ID_MAP)
rows = json.loads(req.to_json(orient="records"))
print("patients:", sorted(req.patient_id.unique()), "| rows:", len(rows))

s = requests.Session()
for i, row in enumerate(rows):
    r = s.post(f"{API}/requests", json=row)
    if not r.ok:
        raise SystemExit(f"row {i} failed: {r.status_code} {r.text}\n{row}")
    if i % 200 == 0:
        print(i, "/", len(rows))
print("done")