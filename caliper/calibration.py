"""Pixel-to-millimetre calibration via the iris (an ~11.7 mm physical ruler).

The iris is the only object of near-constant real-world size in a face photo, so
it lets a single uncalibrated selfie produce absolute millimetres. We cross-check
the implied interpupillary distance for plausibility and always carry the expected
measurement error as a confidence band rather than reporting false precision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import constants as C


@dataclass
class Calibration:
    mm_per_px: float
    iris_diameter_px: float
    implied_ipd_mm: float
    plausible: bool
    error_pct: float
    source: str          # "iris" or "user_ipd"
    note: str


def _iris_diameter_px(P: np.ndarray, center: int, ring: tuple[int, ...]) -> float:
    c = P[center]
    radii = [float(np.hypot(*(P[i] - c))) for i in ring]
    return 2.0 * float(np.mean(radii))   # diameter = 2 * mean ring radius (roll-invariant)


def calibrate(P: np.ndarray, user_ipd_mm: float | None = None) -> Calibration:
    """P: (>=478, 2) array of landmark pixel coordinates."""
    d_left = _iris_diameter_px(P, C.LEFT_IRIS_CENTER, C.LEFT_IRIS_RING)
    d_right = _iris_diameter_px(P, C.RIGHT_IRIS_CENTER, C.RIGHT_IRIS_RING)
    iris_px = (d_left + d_right) / 2.0
    ipd_px = float(np.hypot(*(P[C.LEFT_IRIS_CENTER] - P[C.RIGHT_IRIS_CENTER])))

    if user_ipd_mm is not None:
        mm_per_px = user_ipd_mm / ipd_px
        source = "user_ipd"
        iris_check_mm = mm_per_px * iris_px
        off = abs(iris_check_mm - C.IRIS_DIAMETER_MM) / C.IRIS_DIAMETER_MM * 100.0
        note = (f"calibrated from your measured IPD ({user_ipd_mm:.1f} mm); "
                f"iris cross-check reads {iris_check_mm:.1f} mm vs 11.7 mm ({off:.0f}% off)")
    else:
        mm_per_px = C.IRIS_DIAMETER_MM / iris_px
        source = "iris"
        note = "calibrated from iris diameter (11.7 mm); pass --ipd-mm for a personalized scale"

    implied_ipd_mm = ipd_px * mm_per_px
    lo, hi = C.PLAUSIBLE_IPD_MM
    plausible = lo <= implied_ipd_mm <= hi
    return Calibration(
        mm_per_px=mm_per_px,
        iris_diameter_px=iris_px,
        implied_ipd_mm=implied_ipd_mm,
        plausible=plausible,
        error_pct=C.IRIS_CALIBRATION_ERROR_PCT,
        source=source,
        note=note,
    )
