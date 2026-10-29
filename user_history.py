"""
Week 7 -- Personalization: Per-User Request History
=======================================================
Roadmap example:
    User A
    WATER -> 25 times
    FOOD  -> 12 times
    CALL  -> 8 times
    HELP  -> 5 times
    Then identify frequently used requests.

This tracks how often EACH user selects each intent, persisted to disk
(JSON) so history survives across sessions -- not just kept in memory
for one run. The practical payoff: once you know a user's most
frequent intents, you can surface those FIRST in the UI/scan order
instead of a fixed layout -- the same "put likely things first" idea
as predictive scanning, just applied to intent frequency instead of
letter frequency.
"""

import json
from pathlib import Path

from intents import IntentName, list_intents

DEFAULT_STORE_PATH = "user_history.json"


class UserHistory:
    """Tracks intent-selection counts for a single user."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.counts: dict[str, int] = {intent.value: 0 for intent in list_intents()}

    def record(self, intent: IntentName) -> None:
        """Log one occurrence of the user selecting this intent."""
        self.counts[intent.value] += 1

    def most_frequent(self, n: int | None = None) -> list[tuple[str, int]]:
        """Return (intent, count) pairs sorted by frequency, descending.

        Ties broken alphabetically for stable, predictable ordering.
        Intents never used are excluded.
        """
        ranked = sorted(
            ((intent, c) for intent, c in self.counts.items() if c > 0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:n] if n is not None else ranked

    def personalized_order(self) -> list[str]:
        """Full intent list, frequent-first, then unused intents alphabetically.

        This is what a real UI would use to decide display/scan order --
        the direct personalization payoff of tracking history at all.
        """
        used = [intent for intent, _ in self.most_frequent()]
        unused = sorted(i for i in self.counts if i not in used)
        return used + unused

    def to_dict(self) -> dict:
        return {"user_id": self.user_id, "counts": self.counts}

    @classmethod
    def from_dict(cls, data: dict) -> "UserHistory":
        history = cls(data["user_id"])
        history.counts.update(data["counts"])
        return history


class UserHistoryStore:
    """Holds multiple users' histories, persisted to a single JSON file."""

    def __init__(self, path: str = DEFAULT_STORE_PATH):
        self.path = Path(path)
        self.users: dict[str, UserHistory] = {}
        if self.path.exists():
            self.load()

    def get_or_create(self, user_id: str) -> UserHistory:
        if user_id not in self.users:
            self.users[user_id] = UserHistory(user_id)
        return self.users[user_id]

    def save(self) -> None:
        data = {uid: h.to_dict() for uid, h in self.users.items()}
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.users = {uid: UserHistory.from_dict(d) for uid, d in data.items()}


if __name__ == "__main__":
    store = UserHistoryStore("demo_history.json")
    user = store.get_or_create("demo_user")
    for _ in range(5):
        user.record(IntentName.WATER)
    for _ in range(2):
        user.record(IntentName.FOOD)
    store.save()
    print("Most frequent:", user.most_frequent())
