"""Demystified-component tests: symmetry + typicality (no overall score exists)."""
from __future__ import annotations

import math

import pytest

from caliper import norms, score
from tests.test_core import make_synthetic_face


def test_symmetric_face_is_near_zero():
    P = make_synthetic_face()           # built perfectly symmetric about x=200
    assert score.symmetry_index(P) == pytest.approx(0.0, abs=1e-6)


def test_asymmetry_is_detected():
    P = make_synthetic_face()
    P[score.C.IDX["cheilion_l"]] = (255, 300)   # nudge one mouth corner outward 15px
    assert score.symmetry_index(P) > 1.0        # clearly above the symmetric baseline (0)


def test_typicality_rms_z():
    nr = {
        "a": norms.NormResult("a", "ok", z=1.0),
        "b": norms.NormResult("b", "ok", z=-1.0),
        "c": norms.NormResult("c", "ok", z=2.0),
        "d": norms.NormResult("d", "no_sd"),     # ignored (no z)
    }
    rms, n = score.typicality(nr)
    assert n == 3
    assert rms == pytest.approx(math.sqrt((1 + 1 + 4) / 3), abs=1e-9)


def test_typicality_empty_is_none():
    rms, n = score.typicality({"x": norms.NormResult("x", "no_population")})
    assert rms is None and n == 0
