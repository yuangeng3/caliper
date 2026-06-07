"""Core math tests on a synthetic 478-point face — no MediaPipe/OpenCV needed.

The synthetic face is built so every metric has a known closed-form value, which
lets us pin sign conventions, the iris->mm scale, roll-invariance, and the
cohort-norm logic. If a real photo's numbers look wrong, the bug is almost always
a landmark index in constants.py, not the math here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from caliper import calibration, constants as C, geometry, norms

MM_PER_PX = C.IRIS_DIAMETER_MM / 20.0  # iris diameter is 20 px in the fixture -> 0.585


def make_synthetic_face() -> np.ndarray:
    P = np.zeros((478, 2), dtype=float)

    # iris: centers 108 px apart, ring radius 10 px (=> diameter 20 px), both level at y=200
    P[C.RIGHT_IRIS_CENTER] = (146, 200)
    for idx, off in zip(C.RIGHT_IRIS_RING, [(10, 0), (0, 10), (-10, 0), (0, -10)]):
        P[idx] = (146 + off[0], 200 + off[1])
    P[C.LEFT_IRIS_CENTER] = (254, 200)
    for idx, off in zip(C.LEFT_IRIS_RING, [(10, 0), (0, 10), (-10, 0), (0, -10)]):
        P[idx] = (254 + off[0], 200 + off[1])

    g = C.IDX
    P[g["r_canthus_out"]] = (120, 195)
    P[g["r_canthus_in"]] = (172, 205)
    P[g["l_canthus_in"]] = (228, 205)
    P[g["l_canthus_out"]] = (280, 195)
    P[g["alare_r"]] = (168, 250)
    P[g["alare_l"]] = (232, 250)
    P[g["nasion"]] = (200, 180)
    P[g["subnasale"]] = (200, 265)
    P[g["nose_tip"]] = (200, 240)
    P[g["menton"]] = (200, 360)
    P[g["forehead_top"]] = (200, 120)
    P[g["glabella"]] = (200, 180)
    P[g["zygo_r"]] = (100, 210)
    P[g["zygo_l"]] = (300, 210)
    P[g["cheilion_r"]] = (160, 300)
    P[g["cheilion_l"]] = (240, 300)
    P[g["brow_r"]] = (150, 170)
    P[g["brow_l"]] = (250, 170)
    P[g["lip_top"]] = (200, 295)
    return P


def rotate(P: np.ndarray, deg: float, cx: float, cy: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return (P - (cx, cy)) @ R.T + (cx, cy)


@pytest.fixture
def face() -> np.ndarray:
    return make_synthetic_face()


# --- calibration ----------------------------------------------------------
def test_iris_calibration(face):
    cal = calibration.calibrate(face)
    assert cal.iris_diameter_px == pytest.approx(20.0, abs=1e-6)
    assert cal.mm_per_px == pytest.approx(MM_PER_PX, abs=1e-6)
    assert cal.implied_ipd_mm == pytest.approx(108 * MM_PER_PX, abs=1e-6)  # ~63.2 mm
    assert cal.plausible is True


def test_user_ipd_overrides_scale(face):
    cal = calibration.calibrate(face, user_ipd_mm=63.0)
    assert cal.source == "user_ipd"
    assert cal.mm_per_px == pytest.approx(63.0 / 108.0, abs=1e-6)


# --- geometry -------------------------------------------------------------
def test_metric_values(face):
    m = geometry.compute(face, MM_PER_PX)
    assert m["intercanthal_width"].value == pytest.approx(32.76, abs=1e-2)
    assert m["nasal_width"].value == pytest.approx(37.44, abs=1e-2)
    assert m["nasal_height"].value == pytest.approx(49.725, abs=1e-2)
    assert m["nasal_index"].value == pytest.approx(75.29, abs=1e-1)
    assert m["bizygomatic_width"].value == pytest.approx(117.0, abs=1e-2)
    assert m["facial_height"].value == pytest.approx(105.3, abs=1e-2)
    assert m["facial_index"].value == pytest.approx(90.0, abs=1e-1)
    assert m["eye_fissure_length"].value == pytest.approx(30.977, abs=1e-2)
    assert m["mouth_width"].value == pytest.approx(46.8, abs=1e-2)
    assert m["ipd"].value == pytest.approx(63.18, abs=1e-2)


def test_canthal_tilt_sign_and_value(face):
    m = geometry.compute(face, MM_PER_PX)
    assert m["canthal_tilt"].value == pytest.approx(math.degrees(math.atan2(10, 52)), abs=1e-3)
    assert m["canthal_tilt"].value > 0  # outer corner higher than inner = positive


def test_thirds_sum_to_100(face):
    m = geometry.compute(face, MM_PER_PX)
    total = m["third_upper"].value + m["third_middle"].value + m["third_lower"].value
    assert total == pytest.approx(100.0, abs=1e-6)
    assert m["third_upper"].value == pytest.approx(25.0, abs=1e-3)


def test_fifths_and_fwhr(face):
    m = geometry.compute(face, MM_PER_PX)
    assert m["fifth_intercanthal"].value == pytest.approx(1.4, abs=1e-3)
    assert m["fwhr"].value == pytest.approx(1.6, abs=1e-3)


def test_iris_index_order_does_not_flip_face(face):
    """Regression: on real faces the two iris indices can be reversed in image-x,
    which used to make level_roll rotate the whole face 180° (flipping tilt/thirds)."""
    swapped = face.copy()
    for a, b in zip([C.LEFT_IRIS_CENTER, *C.LEFT_IRIS_RING],
                    [C.RIGHT_IRIS_CENTER, *C.RIGHT_IRIS_RING]):
        swapped[a], swapped[b] = face[b].copy(), face[a].copy()
    base = geometry.compute(face, MM_PER_PX)
    rev = geometry.compute(swapped, MM_PER_PX)
    assert rev["canthal_tilt"].value == pytest.approx(base["canthal_tilt"].value, abs=1e-6)
    assert rev["canthal_tilt"].value > 0          # stays upright, not flipped negative
    assert rev["third_upper"].value == pytest.approx(base["third_upper"].value, abs=1e-6)


def test_roll_invariance(face):
    """Tilting the head must not change measurements (level_roll re-levels first)."""
    tilted = rotate(face, 15.0, 200, 250)
    base = geometry.compute(face, MM_PER_PX)
    rolled = geometry.compute(tilted, MM_PER_PX)
    assert rolled["canthal_tilt"].value == pytest.approx(base["canthal_tilt"].value, abs=1e-6)
    assert rolled["intercanthal_width"].value == pytest.approx(
        base["intercanthal_width"].value, abs=1e-6)
    assert rolled["third_middle"].value == pytest.approx(base["third_middle"].value, abs=1e-6)


# --- norms (the "no universal ideal" logic) -------------------------------
def test_norm_percentile_within_cohort():
    data = norms.load()
    r = norms.evaluate("intercanthal_width", 32.76, "european", "male", 30, data)
    assert r.status == "ok"
    assert r.z == pytest.approx((32.76 - 33.3) / 2.7, abs=1e-3)
    assert r.percentile == pytest.approx(42.07, abs=0.5)
    assert "North American White" in r.population_label


def test_norm_same_value_different_cohort_differs():
    data = norms.load()
    eu = norms.evaluate("nasal_width", 37.44, "european", "male", 30, data)
    af = norms.evaluate("nasal_width", 37.44, "african", "male", 30, data)
    # 37.4 mm is wide for the European cohort but narrow for the Kenyan cohort
    assert eu.percentile > 80
    assert af.percentile < 20


def test_norm_missing_sd_is_honest():
    data = norms.load()
    r = norms.evaluate("bizygomatic_width", 130.0, "european", "male", 30, data)
    assert r.status == "no_sd"
    assert r.percentile is None
    assert r.mean is not None


def test_norm_missing_metric_and_population():
    data = norms.load()
    assert norms.evaluate("facial_height", 100.0, "european", "male", 30, data).status == "no_metric"
    assert norms.evaluate("nasal_width", 35.0, "atlantean", "male", 30, data).status == "no_population"


def test_norm_outside_age_band_flagged():
    data = norms.load()
    r = norms.evaluate("intercanthal_width", 33.0, "european", "male", 67, data)
    assert r.status == "outside_band"
    assert r.percentile is not None  # still computed, but flagged
