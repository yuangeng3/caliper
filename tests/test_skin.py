"""Skin colour-math tests — pure NumPy, no MediaPipe/image needed for the core."""
from __future__ import annotations

import math

import numpy as np
import pytest

from caliper import constants as C, skin


def test_srgb_to_lab_anchors():
    L, a, b = skin.srgb_to_lab((255, 255, 255))
    assert L == pytest.approx(100.0, abs=1e-3)
    assert (a, b) == pytest.approx((0.0, 0.0), abs=5e-2)   # tiny chroma from matrix rounding
    L0, a0, b0 = skin.srgb_to_lab((0, 0, 0))
    assert (L0, a0, b0) == pytest.approx((0.0, 0.0, 0.0), abs=1e-3)


def test_ita_canonical_formula():
    assert skin.ita([60.0, 0.0, 20.0]) == pytest.approx(math.degrees(math.atan2(10, 20)), abs=1e-6)
    assert skin.ita([70.0, 0.0, 10.0]) == pytest.approx(math.degrees(math.atan2(20, 10)), abs=1e-6)


def test_ita_categories_del_bino():
    assert skin.ita_category(56) == "very light"
    assert skin.ita_category(50) == "light"
    assert skin.ita_category(30) == "intermediate"
    assert skin.ita_category(15) == "tan"
    assert skin.ita_category(0) == "brown"
    assert skin.ita_category(-40) == "dark"


def test_ita_to_monk_monotonic_and_bounded():
    assert skin.ita_to_monk(60) == 1
    assert skin.ita_to_monk(-60) == 10
    seq = [skin.ita_to_monk(x) for x in (55, 30, 0, -30, -55)]
    assert seq == sorted(seq)          # darker (lower ITA) -> higher Monk
    assert all(1 <= m <= 10 for m in seq)


def test_ita_to_monk_published_boundaries():
    # P4: bin boundaries align with Del Bino category edges (not a naive linear interp)
    assert skin.ita_to_monk(35) == 4   # > 34
    assert skin.ita_to_monk(30) == 5   # > 28 (intermediate)
    assert skin.ita_to_monk(20) == 6   # > 18
    assert skin.ita_to_monk(12) == 7   # > 10 (tan)


def test_analyze_uniform_skin_is_even():
    rgb = (210, 160, 140)
    img = np.full((220, 220, 3), rgb, dtype=np.uint8)
    P = np.zeros((478, 2))
    # place every skin region anchor at a distinct in-bounds point
    for k, idx in C.SKIN_REGIONS.items():
        P[idx] = (50 + 20 * idx % 150 + 30, 60 + (idx % 7) * 15 + 30)
    rep = skin.analyze(img, P, calibrated=False)
    expected_ita = skin.ita(skin.srgb_to_lab(rgb))
    assert rep.overall_ita == pytest.approx(expected_ita, abs=1e-6)
    assert rep.evenness_sd == pytest.approx(0.0, abs=1e-6)   # uniform => perfectly even
    assert rep.calibrated is False and rep.normalized is False  # no eye-white in a blank image
    assert rep.tone_category == skin.ita_category(expected_ita)
    assert len(rep.priority_concerns) >= 2


def test_priority_concerns_flip_by_tone():
    # construct reports at a deep vs fair tone via monkeypatch-free direct call
    deep = skin.analyze(np.full((60, 60, 3), (110, 80, 65), np.uint8),
                        _anchored(), calibrated=False)
    fair = skin.analyze(np.full((60, 60, 3), (240, 215, 205), np.uint8),
                        _anchored(), calibrated=False)
    assert any("PIH" in c or "pigmentation" in c.lower() for c in deep.priority_concerns)
    assert any("erythema" in c.lower() or "redness" in c.lower() for c in fair.priority_concerns)


def _anchored() -> np.ndarray:
    P = np.zeros((478, 2))
    for idx in C.SKIN_REGIONS.values():
        P[idx] = (30, 30)
    return P


# distinct, well-separated anchors for the regions the malar metric uses
_REGION_XY = {10: (150, 40), 9: (150, 70), 50: (90, 160),
              280: (210, 160), 195: (150, 120), 200: (150, 250)}


def _placed() -> np.ndarray:
    P = np.zeros((478, 2))
    for idx, xy in _REGION_XY.items():
        P[idx] = xy
    return P


def test_malar_signals_zero_on_uniform_skin():
    # P2: uniform face => no cheek asymmetry and no malar elevation
    img = np.full((300, 300, 3), (210, 170, 150), np.uint8)
    rep = skin.analyze(img, _placed(), ancestry="east_asian", calibrated=False)
    assert rep.cheek_asymmetry == pytest.approx(0.0, abs=1e-6)
    assert rep.malar_delta == pytest.approx(0.0, abs=1e-6)


def test_malar_signals_detect_cheek_pigment():
    # P2: the global evenness SD is blind to malar melasma; these signals are not.
    base = (210, 170, 150)
    dark = (150, 110, 95)   # lower red channel -> higher melanin index

    # one cheek darker -> left/right asymmetry
    img = np.full((300, 300, 3), base, np.uint8)
    x, y = _REGION_XY[50]                       # right_cheek
    img[y - 12:y + 13, x - 12:x + 13] = dark
    rep = skin.analyze(img, _placed(), ancestry="east_asian", calibrated=False)
    assert rep.cheek_asymmetry > 1.0

    # both cheeks darker than forehead, equally -> symmetric malar elevation (melasma pattern)
    img2 = np.full((300, 300, 3), base, np.uint8)
    for idx in (50, 280):
        x, y = _REGION_XY[idx]
        img2[y - 12:y + 13, x - 12:x + 13] = dark
    rep2 = skin.analyze(img2, _placed(), ancestry="east_asian", calibrated=False)
    assert rep2.malar_delta > 1.0
    assert rep2.cheek_asymmetry == pytest.approx(0.0, abs=1e-6)


def test_pigment_guidance_has_depigmenting_agents_and_ancestry_notes():
    # P1/P3/P5: pigment-first users get the real first-line agents + East-Asian-specific
    # barrier/aging notes + a PIH-prevention line + the malar read.
    from caliper import interpret

    rep_skin = skin.SkinReport(
        regions=[], overall_ita=20.0, tone_category="tan", monk=6, evenness_sd=3.0,
        mean_erythema=1.0, priority_concerns=["melasma"], calibrated=False, confidence="x",
        bucket="deep", malar_delta=5.0, cheek_asymmetry=1.0)
    out = interpret.interpret({}, {}, ancestry="east_asian", sex="female", age=34, skin=rep_skin)

    moves = " ".join(out.skin_moves).lower()
    assert "azelaic" in moves                              # P1: first-line pigment agent present
    assert ("tranexamic" in moves) or ("niacinamide" in moves)
    extra = " ".join(out.skin_extra).lower()
    assert "decade" in extra                               # P3: wrinkles-lag reassurance
    assert "barrier" in extra                              # P3: drier/reactive barrier note
    assert ("pih" in extra) or ("pick" in extra)           # P5: PIH prevention
    assert "malar" in extra                                # P2: malar read surfaced
