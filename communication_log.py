"""
Week 8 -- Communication Log
==============================
Every completed request (intent -> message -> spoken) gets appended
here as one JSON-lines record. This is what turns individual TTS
calls into a real usage history -- useful for caregivers reviewing
what was communicated, and it's also exactly the raw data Week 7's
personalization counts are built from.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = "communication_log.jsonl"


def log_communication(
    user_id: str,
    intent: str,
    message: str,
    spoken: bool,
    path: str = DEFAULT_LOG_PATH,
) -> None:
    """Append one communication event to the log file."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "intent": intent,
        "message": message,
        "spoken": spoken,
    }
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_log(path: str = DEFAULT_LOG_PATH) -> list[dict]:
    """Read all records back, oldest first."""
    log_path = Path(path)
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    log_communication("demo_user", "WATER", "I need some water, please.", spoken=False)
    print("Logged one entry. Current log:")
    for entry in read_log():
        print(" ", entry)
