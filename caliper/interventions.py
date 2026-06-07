"""Evidence-graded intervention -> metric map.

Loads data/interventions.json and renders it honestly: every entry carries an
Attia-style grade (A best .. D none), what it measurably moves, and a time-to-effect.
Grade-D entries (e.g. mewing) are shown precisely so users do NOT attribute noise to them.
"""
from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "data" / "interventions.json"
_GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


def load(path: Path | None = None) -> dict:
    with open(path or _PATH) as f:
        return json.load(f)


def render(data: dict | None = None) -> str:
    data = data or load()
    items = sorted(data["interventions"].items(),
                   key=lambda kv: (_GRADE_ORDER.get(kv[1]["grade"], 9), kv[0]))
    L = ["INTERVENTION -> METRIC MAP  (evidence-graded; A best .. D none)", ""]
    for _, v in items:
        affects = ", ".join(v["affects"]) if v["affects"] else "(nothing measurable)"
        when = ""
        if "time_weeks" in v:
            a, b = v["time_weeks"]
            when = f"  ~{a}-{b} wk" if a != b else f"  ~{a} wk"
        L.append(f"  [{v['grade']}] {v['label']}{when}")
        L.append(f"       moves: {affects}")
        L.append(f"       {v['note']}")
        L.append("")
    L.append("Caliper only attributes a metric change to an intervention when the change")
    L.append("exceeds the measurement-noise band (MDC95). Grade-D inputs are never credited.")
    return "\n".join(L)
