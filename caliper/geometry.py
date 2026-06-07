"""Deterministic frontal facial geometry from MediaPipe landmarks.

Pure NumPy — no MediaPipe/OpenCV import — so it is unit-testable on synthetic
landmarks. Linear distances are Euclidean (rotation-invariant); axis-dependent
metrics (canthal tilt, facial thirds/fifths, FWHR) are computed after levelling
head roll via the inter-iris line. Everything here is descriptive measurement;
interpretation against a cohort lives in norms.py / report.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import constants as C

LM = C.IDX


@dataclass
class Metric:
    id: str
    label: str
    value: float
    unit: str            # mm | deg | ratio | index | %
    grade: str           # measurement confidence A/B/C/D (this is NOT a norm percentile)
    approximate: bool = False
    note: str = ""


def level_roll(P: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate every point so the inter-iris line is horizontal. Returns (P', roll_deg).

    The two iris-center indices are not guaranteed to be ordered left->right in image
    x, so the raw angle can come out near ±180° (which would flip the face top-to-bottom
    and corrupt every axis-dependent metric). We fold it to the nearest horizontal so we
    level the tilt without ever flipping an upright face.
    """
    r = P[C.RIGHT_IRIS_CENTER]
    lf = P[C.LEFT_IRIS_CENTER]
    ang = math.atan2(lf[1] - r[1], lf[0] - r[0])
    if ang > math.pi / 2:
        ang -= math.pi
    elif ang < -math.pi / 2:
        ang += math.pi
    c, s = math.cos(-ang), math.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    ctr = (r + lf) / 2.0
    return (P - ctr) @ rot.T + ctr, math.degrees(ang)


def _d(P: np.ndarray, a: str, b: str) -> float:
    return float(np.hypot(*(P[LM[a]] - P[LM[b]])))


def _tilt(inner: np.ndarray, outer: np.ndarray) -> float:
    # positive = outer corner higher than inner (image y grows downward)
    return math.degrees(math.atan2(inner[1] - outer[1], abs(outer[0] - inner[0])))


def compute(P_px: np.ndarray, mm_per_px: float) -> dict[str, Metric]:
    """Return all frontal metrics keyed by id. P_px: (>=478, 2) pixel coords."""
    P, _ = level_roll(P_px)
    mm = mm_per_px
    out: dict[str, Metric] = {}

    def add(m: Metric) -> None:
        out[m.id] = m

    # --- linear distances (Euclidean, mm) ---------------------------------
    add(Metric("intercanthal_width", "Intercanthal width (en-en)",
               _d(P, "l_canthus_in", "r_canthus_in") * mm, "mm", "A"))
    nasal_w = _d(P, "alare_l", "alare_r") * mm
    add(Metric("nasal_width", "Nasal width (al-al)", nasal_w, "mm", "B",
               note="alar points — confirm with --annotate"))
    nasal_h = _d(P, "nasion", "subnasale") * mm
    add(Metric("nasal_height", "Nasal height (n-sn)", nasal_h, "mm", "B"))
    bizyg = _d(P, "zygo_l", "zygo_r") * mm
    add(Metric("bizygomatic_width", "Face width (zy-zy, approx.)", bizyg, "mm", "C",
               approximate=True, note="cheek-contour stand-in for true bizygomatic width"))
    facial_h = _d(P, "nasion", "menton") * mm
    add(Metric("facial_height", "Morphological face height (n-gn)", facial_h, "mm", "B"))
    efl = ((_d(P, "r_canthus_out", "r_canthus_in") +
            _d(P, "l_canthus_out", "l_canthus_in")) / 2.0) * mm
    add(Metric("eye_fissure_length", "Eye fissure length (ex-en)", efl, "mm", "A"))
    add(Metric("mouth_width", "Mouth width (ch-ch)",
               _d(P, "cheilion_l", "cheilion_r") * mm, "mm", "A"))
    ipd = float(np.hypot(*(P[C.LEFT_IRIS_CENTER] - P[C.RIGHT_IRIS_CENTER]))) * mm
    add(Metric("ipd", "Interpupillary distance", ipd, "mm", "A"))

    # --- indices (scale-free, so calibration-independent) -----------------
    add(Metric("nasal_index", "Nasal index (al-al / n-sn x100)",
               nasal_w / nasal_h * 100.0, "index", "B"))
    add(Metric("facial_index", "Facial index (n-gn / zy-zy x100)",
               facial_h / bizyg * 100.0, "index", "C", approximate=True))

    # --- angles / axis-dependent (computed after roll levelling) ----------
    tilt = (_tilt(P[LM["r_canthus_in"]], P[LM["r_canthus_out"]]) +
            _tilt(P[LM["l_canthus_in"]], P[LM["l_canthus_out"]])) / 2.0
    add(Metric("canthal_tilt", "Canthal tilt", tilt, "deg", "A"))

    up = P[LM["glabella"]][1] - P[LM["forehead_top"]][1]
    mid = P[LM["subnasale"]][1] - P[LM["glabella"]][1]
    low = P[LM["menton"]][1] - P[LM["subnasale"]][1]
    tot = up + mid + low
    add(Metric("third_upper", "Upper facial third", 100.0 * up / tot, "%", "C",
               approximate=True, note="mesh tops at forehead, not hairline"))
    add(Metric("third_middle", "Middle facial third", 100.0 * mid / tot, "%", "B"))
    add(Metric("third_lower", "Lower facial third", 100.0 * low / tot, "%", "B"))

    face_w_px = abs(P[LM["zygo_l"]][0] - P[LM["zygo_r"]][0])
    unit = face_w_px / 5.0
    add(Metric("fifth_intercanthal", "Central fifth (intercanthal)",
               abs(P[LM["l_canthus_in"]][0] - P[LM["r_canthus_in"]][0]) / unit, "ratio", "B",
               note="~1.00 = one eye-width (rule of fifths)"))

    height = P[LM["lip_top"]][1] - min(P[LM["brow_r"]][1], P[LM["brow_l"]][1])
    add(Metric("fwhr", "Facial width-to-height ratio", face_w_px / height, "ratio", "C",
               approximate=True))
    return out
