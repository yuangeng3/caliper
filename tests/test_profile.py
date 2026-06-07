"""Profile soft-tissue angle math + cohort norms (reusing norms.evaluate)."""
from __future__ import annotations

from pathlib import Path

import pytest

from caliper import norms, profile

# a synthetic right-facing profile (anterior = +x)
PTS = {
    "glabella": (100, 50),
    "nasion": (95, 70),
    "pronasale": (130, 100),
    "subnasale": (105, 110),
    "labrale_superius": (110, 125),
    "labrale_inferius": (108, 145),
    "pogonion": (115, 180),
    "menton": (110, 195),
}
_PROFILE_JSON = Path("data/norms/profile_cephalometric.json")


def test_eline_lips_behind_line_are_negative():
    m = profile.compute(PTS, mm_per_px=1.0, facing=1)
    assert m["e_line_ul"].value == pytest.approx(-15.31, abs=0.1)
    assert m["e_line_ul"].value < 0 and m["e_line_ll"].value < 0


def test_nasolabial_and_convexity_in_clinical_ranges():
    m = profile.compute(PTS, mm_per_px=1.0, facing=1)
    assert 80 < m["nasolabial_angle"].value < 110     # Cm-Sn-Ls convention
    assert 0 <= m["facial_convexity"].value < 20       # deviation-from-straight convention
    assert m["nasofrontal_angle"].value == pytest.approx(116.6, abs=1.0)


def test_profile_norms_are_ancestry_conditioned():
    data = norms.load(_PROFILE_JSON)
    # a slightly protrusive upper lip (-1mm): typical for East Asian, protrusive for European
    ea = norms.evaluate("e_line_ul", -1.0, "east_asian", "female", 30, data)
    eu = norms.evaluate("e_line_ul", -1.0, "european", "female", 30, data)
    assert ea.status == "ok" and eu.status == "ok"
    assert eu.percentile > ea.percentile          # -1mm is more anterior vs the -4mm European norm
    # cohorts we have no profile data for are honest about it
    assert norms.evaluate("e_line_ul", -1.0, "african", "female", 30, data).status == "no_population"
