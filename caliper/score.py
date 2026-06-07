"""The "score", demystified — decomposed into a few mechanism-named components,
never a single number.

Research is unambiguous: ~half of attractiveness judgment is private taste
(Honekopp 2006), and even "diverse" beauty datasets fail cross-ethnicity parity in
>90% of comparisons. So Caliper ships NO overall score. Instead it surfaces the few
things research finds are cross-culturally and mechanistically meaningful, each with
its own caveat:

  - tone evenness   — the most evidence-backed AND modifiable signal (worth up to
                      ~20 years of perceived age); ancestry-neutral.
  - typicality      — closeness to your OWN cohort's average (averageness is the
                      single most robust cross-cultural cue; conditioned, not universal).
  - symmetry        — reported but explicitly DOWN-RANKED (weak-to-null once
                      averageness is controlled).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import constants as C
from .geometry import level_roll

# left/right correspondences for the symmetry estimate
_PAIRS = [
    ("r_canthus_out", "l_canthus_out"), ("r_canthus_in", "l_canthus_in"),
    ("alare_r", "alare_l"), ("zygo_r", "zygo_l"),
    ("cheilion_r", "cheilion_l"), ("brow_r", "brow_l"),
]
_CENTRAL = ["nasion", "subnasale", "menton", "glabella", "nose_tip"]


def symmetry_index(P_px: np.ndarray) -> float:
    """Mean left/right deviation as a % of interpupillary distance (0 = perfectly
    symmetric). Mirror each right point across the facial midline, compare to its
    left partner. Down-ranked by design — see module docstring."""
    P, _ = level_roll(P_px)
    mid = float(np.mean([P[C.IDX[c]][0] for c in _CENTRAL]))
    ipd = float(np.hypot(*(P[C.LEFT_IRIS_CENTER] - P[C.RIGHT_IRIS_CENTER])))
    devs = []
    for ri, li in _PAIRS:
        xr, yr = P[C.IDX[ri]]
        xl, yl = P[C.IDX[li]]
        devs.append(math.hypot((2 * mid - xr) - xl, yr - yl))
    return float(np.mean(devs) / ipd * 100.0)


def typicality(norm_results: dict) -> tuple[float | None, int]:
    """RMS of available within-cohort z-scores. Lower = closer to your cohort's
    average ('more typical'); averageness is the most robust cross-cultural cue."""
    zs = [nr.z for nr in norm_results.values()
          if nr is not None and nr.status in ("ok", "outside_band") and nr.z is not None]
    if not zs:
        return None, 0
    return float(math.sqrt(sum(z * z for z in zs) / len(zs))), len(zs)


@dataclass
class Components:
    symmetry_pct: float
    typicality_rms_z: float | None
    typicality_n: int
    evenness_sd: float | None


def compute(norm_results: dict, P_px: np.ndarray, skin=None) -> Components:
    rms, n = typicality(norm_results)
    return Components(
        symmetry_pct=symmetry_index(P_px),
        typicality_rms_z=rms,
        typicality_n=n,
        evenness_sd=(skin.evenness_sd if skin is not None else None),
    )


def render(c: Components) -> str:
    L = ["COMPONENTS  (decomposed, not a score — there is deliberately no overall number)"]
    if c.evenness_sd is not None:
        L.append(f"  tone evenness   ITA SD {c.evenness_sd:.1f}  — lower is more even; the most")
        L.append("                  evidence-backed & modifiable signal (sun, sleep, retinoids).")
    if c.typicality_rms_z is not None:
        L.append(f"  typicality      RMS z {c.typicality_rms_z:.2f} over {c.typicality_n} metrics — "
                 "closeness to YOUR")
        L.append("                  cohort's average (lower = more typical). Conditioned, not universal.")
    L.append(f"  symmetry        {c.symmetry_pct:.1f}% L/R deviation — reported but DOWN-RANKED:")
    L.append("                  symmetry is weak-to-null for attractiveness once averageness is held.")
    L.append("  (No component is combined into a single rank. ~half of 'attractiveness' is private")
    L.append("   taste, so one number would be statistically dishonest.)")
    return "\n".join(L)
