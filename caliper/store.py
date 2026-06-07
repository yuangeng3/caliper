"""Local-first longitudinal store: on-device SQLite + an EXIF-stripped image vault,
with minimal-detectable-change (MDC95) gating so a trend reflects real change, not
capture noise. Nothing here ever touches the network.

MDC95 = SEM x 1.96 x sqrt(2). A session-to-session delta only counts as "real"
(95% confidence) when it exceeds MDC95 — the single most important rigor move, and
the reason Caliper refuses to draw a trend through measurement noise.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DIR = Path.home() / ".caliper"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    profile TEXT NOT NULL,
    ts TEXT NOT NULL,
    ancestry TEXT, sex TEXT, age INTEGER,
    image_path TEXT, meta TEXT
);
CREATE TABLE IF NOT EXISTS measurements (
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    metric TEXT NOT NULL, value REAL NOT NULL, unit TEXT
);
CREATE INDEX IF NOT EXISTS ix_meas ON measurements(session_id, metric);
"""


# --- minimal detectable change -------------------------------------------
def sem_from_repeats(values: list[float]) -> float:
    """Standard error of measurement from same-session repeated captures."""
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(var)


def mdc95(sem: float) -> float:
    return sem * 1.96 * math.sqrt(2.0)


def is_real_change(delta: float, mdc: float) -> bool:
    return abs(delta) > mdc


def fallback_mdc(value: float, error_pct: float) -> float:
    """When no test-retest exists, treat the calibration error as the SEM band."""
    return mdc95(abs(value) * error_pct / 100.0)


# --- EXIF / GPS stripping on ingest --------------------------------------
def strip_and_store_image(src: str, dst: str) -> list[str]:
    """Re-encode the photo without any metadata. Returns the tag names removed,
    so the user gets a visible receipt of what was stripped."""
    from PIL import Image
    from PIL.ExifTags import GPSTAGS, TAGS

    img = Image.open(src)
    removed: list[str] = []
    exif = img.getexif()
    for tag_id in exif:
        name = TAGS.get(tag_id, str(tag_id))
        removed.append(name)
        if name == "GPSInfo":
            removed.extend(GPSTAGS.get(k, str(k)) for k in (exif.get_ifd(tag_id) or {}))

    img = img.convert("RGB")
    clean = Image.frombytes(img.mode, img.size, img.tobytes())  # raw pixels only
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    clean.save(dst)
    return sorted(set(removed))


# --- store ----------------------------------------------------------------
@dataclass
class Session:
    id: int
    profile: str
    ts: str
    ancestry: str | None
    sex: str | None
    age: int | None


class Store:
    def __init__(self, path: str | Path | None = None):
        base = Path(path) if path else (DEFAULT_DIR / "caliper.db")
        base.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(base))
        self.conn.executescript(_SCHEMA)

    def add_session(self, profile: str, ts: str, metrics: dict[str, tuple[float, str]],
                    *, ancestry=None, sex=None, age=None, image_path=None, meta=None) -> int:
        cur = self.conn.execute(
            "INSERT INTO sessions(profile, ts, ancestry, sex, age, image_path, meta) "
            "VALUES (?,?,?,?,?,?,?)",
            (profile, ts, ancestry, sex, age, image_path, json.dumps(meta or {})))
        sid = cur.lastrowid
        self.conn.executemany(
            "INSERT INTO measurements(session_id, metric, value, unit) VALUES (?,?,?,?)",
            [(sid, m, v, u) for m, (v, u) in metrics.items()])
        self.conn.commit()
        return sid

    def sessions(self, profile: str) -> list[Session]:
        rows = self.conn.execute(
            "SELECT id, profile, ts, ancestry, sex, age FROM sessions "
            "WHERE profile=? ORDER BY ts", (profile,)).fetchall()
        return [Session(*r) for r in rows]

    def series(self, profile: str, metric: str) -> list[tuple[str, float]]:
        return self.conn.execute(
            "SELECT s.ts, m.value FROM measurements m JOIN sessions s ON s.id=m.session_id "
            "WHERE s.profile=? AND m.metric=? ORDER BY s.ts", (profile, metric)).fetchall()

    def close(self) -> None:
        self.conn.close()
