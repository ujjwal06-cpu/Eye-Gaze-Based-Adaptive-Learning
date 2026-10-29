# b1-server

Node.js server (no dependencies, Node 18+) between GazePad and Team B's B1.

```
GazePad (browser)  --POST /window-->  b1-server  <--GET /latest every 3 s--  B1 script
```

## Run
```
node server.js            # port 8100
```
Testing GazePad in mock mode (mouse)? Mock windows are held back from B1 unless you run:
`B1_ALLOW_SIMULATED=1 node server.js`

## What B1 calls (no API key)
`GET http://<this-machine>:8100/latest` returns a JSON list: the latest completed 10-second window of each patient,
exactly the 10 fields of the B1 spec. `[]` until the first window. The same window stays until the next is ready; B1 skips
windows it already processed. `GET /latest/USER001` returns one patient as a plain object.
Team B must be able to reach this machine's IP/port (same network, or a tunnel/hosting), so give them `http://<your-ip>:8100/latest`.

## Other endpoints (for us)
`POST /window` (GazePad), `GET /status` (counts, last poll time), `GET /logs?n=50` (accepted / rejected / held windows), `GET /health`.
Logs are in `data/events.jsonl`, the latest windows in `data/latest.json` (survive restarts).

## Settings
`PORT` (8100), `B1_ALLOW_SIMULATED` (0), `B1_DATA_DIR` (./data), `B1_CORS_ORIGINS`, `B1_STALE_S` (0 = keep serving the last window; set e.g. 30 to stop serving old windows).

## Tests
`npm test`
