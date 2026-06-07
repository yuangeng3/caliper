"""Profile (side-view) soft-tissue analysis.

Two hard honesty rules from the literature:

1. **Soft-tissue only.** No skeletal angle (SNA/SNB, true gonial, mandibular plane)
   can be recovered from a photo — those need a radiograph. We compute only
   soft-tissue surrogates (E-line, nasolabial, facial convexity) and say so.
2. **Norms differ by ancestry.** The Caucasian E-line over-retracts East Asian and
   African faces; we score within the user's cohort (norms in
   ``data/norms/profile_cephalometric.json``, same schema as the frontal table, so
   ``norms.evaluate`` is reused).

The angle math here is pure geometry on a supplied set of 2D profile soft-tissue
points. v0.1 does NOT wire this to the frontal MediaPipe detector — MediaPipe is
frontal-optimized and would fabricate profile numbers. A profile-capable backend
(3DDFA_V2, opt-in) is the planned extraction front-end.
"""
from __future__ import annotations

import math

import numpy as np

from .geometry import Metric

# the soft-tissue profile points this module expects (image x,y)
POINTS = ("glabella", "nasion", "pronasale", "subnasale",
          "labrale_superius", "labrale_inferius", "pogonion", "menton")


def _angle(center, a, b) -> float:
    v1 = np.asarray(a, float) - center
    v2 = np.asarray(b, float) - center
    cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _eline_offset(pron, pog, lip, facing: int) -> float:
    """Signed horizontal offset (px) of a lip point from the nose-tip->chin E-line.
    Positive = anterior (in front of the line); the Ricketts convention is negative
    (lips sit behind). `facing` = +1 if the profile faces +x, else -1."""
    pron, pog, lip = map(lambda p: np.asarray(p, float), (pron, pog, lip))
    if abs(pog[1] - pron[1]) < 1e-9:
        return 0.0
    x_on_line = pron[0] + (pog[0] - pron[0]) * (lip[1] - pron[1]) / (pog[1] - pron[1])
    return facing * float(lip[0] - x_on_line)


def compute(pts: dict[str, tuple[float, float]], mm_per_px: float = 1.0,
            facing: int = 1) -> dict[str, Metric]:
    """pts: the POINTS above as (x,y). Returns soft-tissue profile metrics."""
    P = {k: np.asarray(v, float) for k, v in pts.items()}
    out: dict[str, Metric] = {}

    ul = _eline_offset(P["pronasale"], P["pogonion"], P["labrale_superius"], facing) * mm_per_px
    ll = _eline_offset(P["pronasale"], P["pogonion"], P["labrale_inferius"], facing) * mm_per_px
    out["e_line_ul"] = Metric("e_line_ul", "Upper lip to E-line", ul, "mm", "A",
                              note="negative = behind the nose-tip->chin line (soft tissue)")
    out["e_line_ll"] = Metric("e_line_ll", "Lower lip to E-line", ll, "mm", "A")

    out["nasolabial_angle"] = Metric(
        "nasolabial_angle", "Nasolabial angle (Cm-Sn-Ls)",
        _angle(P["subnasale"], P["pronasale"], P["labrale_superius"]), "deg", "A")
    # convexity is conventionally the deviation from straight (180 - included angle):
    # ~12 deg European, ~7.7 deg Korean; lower = straighter profile.
    out["facial_convexity"] = Metric(
        "facial_convexity", "Facial convexity (G-Sn-Pog')",
        180.0 - _angle(P["subnasale"], P["glabella"], P["pogonion"]), "deg", "A")
    out["nasofrontal_angle"] = Metric(
        "nasofrontal_angle", "Nasofrontal angle (G-N-dorsum)",
        _angle(P["nasion"], P["glabella"], P["pronasale"]), "deg", "C",
        approximate=True, note="reference-point dependent (glabella vs nasion)")
    return out
