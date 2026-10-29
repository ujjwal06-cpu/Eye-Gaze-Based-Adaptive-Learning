"""Online log-linear pool: the per-patient "learned" expert.

    score(label) = sum_i beta_i * centred log p_i(label)        how much to trust each expert
                 + sum_s gamma[s, label] * state_s              what THIS patient's webcam states mean
                 + bias(label)
    P(label) = softmax(score)

It is a stacker on top of the other experts. It learns, one request at a time, things like
"when this patient looks distressed they ask for pain relief, otherwise follow the routine",
which a fixed mixture cannot express. AdaGrad steps + L2; state is JSON-serialisable.
A ward-wide pool trained on every patient gives new patients a sensible starting point.
"""
from __future__ import annotations

import numpy as np

from .config import LABELS, N

POOL_EXPERTS = ["routine", "transition", "due", "gaze_focus", "static_rf"]
STATES = ["distress", "fatigue", "confusion", "emergency"]
BETA0 = {"routine": 0.8, "transition": 0.2, "due": 0.2, "gaze_focus": 0.5, "static_rf": 0.2}


def pool_inputs(outs: dict, ws: dict):
    """Centred log-probabilities of the input experts (0 when asleep) and the webcam states."""
    L = np.zeros((len(POOL_EXPERTS), N))
    for i, k in enumerate(POOL_EXPERTS):
        p = outs.get(k, (None, None))[0]
        if p is not None:
            lp = np.log(0.92 * np.asarray(p) + 0.08 / N)
            L[i] = lp - lp.mean()
    s = np.zeros(len(STATES))
    if ws.get("available"):
        st = ws["states"]
        s[:3] = [st["distress"], st["fatigue"], st["confusion"]]
        s[3] = 1.0 if ws.get("emergency") else 0.0
    return L, s


class LogPool:
    def __init__(self, state: dict | None = None, init: dict | None = None, lr: float = 0.6, l2: float = 2e-3):
        self.lr, self.l2 = lr, l2
        self._beta0 = np.array([BETA0[k] for k in POOL_EXPERTS])
        own = state if state and state.get("kind") == "logpool" else None
        src = own or (init if init and init.get("kind") == "logpool" else None)  # warm start from the ward model
        if src:
            self.beta = np.asarray(src["beta"], dtype=float)
            self.gamma = np.asarray(src["gamma"], dtype=float)
            self.bias = np.asarray(src["bias"], dtype=float)
        else:
            self.beta = self._beta0.copy()
            self.gamma = np.zeros((len(STATES), N))
            self.bias = np.zeros(N)
        self.n = int(own.get("n", 0)) if own else 0
        if own and own.get("G"):
            self.G = [np.maximum(np.asarray(x, dtype=float), 1e-6) for x in own["G"]]
        else:
            self.G = [np.full_like(self.beta, 1e-6), np.full_like(self.gamma, 1e-6), np.full_like(self.bias, 1e-6)]

    def proba(self, L: np.ndarray, s: np.ndarray) -> np.ndarray:
        z = self.beta @ L + s @ self.gamma + self.bias
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def update(self, L: np.ndarray, s: np.ndarray, y: int) -> None:
        q = self.proba(L, s)
        q[y] -= 1.0                                   # d logloss / d score
        grads = [L @ q + self.l2 * (self.beta - self._beta0),
                 np.outer(s, q) + self.l2 * self.gamma,
                 q + self.l2 * self.bias]
        params = [self.beta, self.gamma, self.bias]
        for p, g, G in zip(params, grads, self.G):
            G += g * g
            p -= self.lr * g / np.sqrt(G)
        np.clip(self.beta, 0.0, 3.0, out=self.beta)
        self.n += 1

    def state(self) -> dict:
        r = lambda a: np.round(a, 5).tolist()
        return {"kind": "logpool", "beta": r(self.beta), "gamma": r(self.gamma), "bias": r(self.bias),
                "G": [g.tolist() for g in self.G], "n": self.n}

    def explain(self) -> dict:
        return {"trust": {k: round(float(b), 3) for k, b in zip(POOL_EXPERTS, self.beta)},
                "state_effects": {s: {l: round(float(v), 3) for l, v in zip(LABELS, self.gamma[i])}
                                  for i, s in enumerate(STATES)}}
