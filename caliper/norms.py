"""Ancestry-, sex-, age-conditioned reference distributions.

CORE PRINCIPLE: there is no single ideal face. Each metric is scored as a
percentile WITHIN the user's own (ancestry, sex, age) cohort, never against a
universal template. Where reference data for a cohort is missing, or exists but
lacks a standard deviation, we say so rather than fabricate a percentile.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

_NORMS_PATH = (Path(__file__).resolve().parents[1] /
               "data" / "norms" / "frontal_anthropometry.json")
ADULT_BAND = "18-45"


@dataclass
class NormResult:
    metric_id: str
    status: str           # ok | no_sd | no_metric | no_population | outside_band
    population_label: str = ""
    source: str = ""
    mean: float | None = None
    sd: float | None = None
    z: float | None = None
    percentile: float | None = None   # 0-100, within cohort
    message: str = ""
    age_adjusted: bool = False         # True when the expected mean was shifted for age
    adjusted_mean: float | None = None  # age-shifted mean the percentile was computed against
    age_grade: str = ""                # evidence grade of the aging trend used
    age_source: str = ""               # citation for that trend


def load(path: Path | None = None) -> dict:
    with open(path or _NORMS_PATH) as f:
        return json.load(f)


def _percentile(z: float) -> float:
    return 50.0 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _age_shift(metric_id: str, age: int, sex: str, data: dict):
    """Evidence-graded mean shift (mm) for a metric at a given age, or None if no
    aging trend is recorded. Adds a post-menopause acceleration term for women.
    Returns (shift_mm, effect_dict)."""
    ae = data.get("age_effects")
    if not ae:
        return None
    eff = ae.get("metrics", {}).get(metric_id)
    if not eff:
        return None
    ref = ae.get("_ref_age", 31)
    slope = eff["per_year_mm"]
    shift = slope * (age - ref)
    meno = ae.get("_menopause", {})
    if meno and sex == meno.get("sex") and age > meno.get("onset_age", float("inf")):
        # extra change accrues only for the years past menopause onset
        shift += slope * (meno.get("accel_factor", 1.0) - 1.0) * (age - meno["onset_age"])
    return shift, eff


def evaluate(metric_id: str, value: float, ancestry: str, sex: str,
             age: int | None, data: dict) -> NormResult:
    amap = data.get("ancestry_map", {})
    if ancestry not in amap:
        return NormResult(metric_id, "no_population",
                          message=f"no reference cohort for ancestry '{ancestry}'")
    pop = amap[ancestry]["population"]
    pop_meta = data["populations"][pop]
    pop_label = pop_meta["label"]
    src_id = pop_meta.get("source_id", "")
    source = data.get("sources", {}).get(src_id, {}).get("citation", src_id)

    cell = data["norms"].get(pop, {}).get(sex, {}).get(ADULT_BAND, {}).get(metric_id)
    if cell is None:
        return NormResult(metric_id, "no_metric", pop_label, source,
                          message=f"no {pop_label} ({sex}) reference for this metric")

    mean = cell.get("mean")
    sd = cell.get("sd")
    if sd is None:
        return NormResult(metric_id, "no_sd", pop_label, source, mean=mean,
                          message=(f"{pop_label} mean ~ {mean} (SD unavailable — "
                                   "percentile not computable)"))

    z = (value - mean) / sd
    res = NormResult(metric_id, "ok", pop_label, source, mean, sd, z, _percentile(z))
    if age is not None and not (18 <= age <= 45):
        shifted = _age_shift(metric_id, age, sex, data)
        if shifted is not None:
            shift, eff = shifted
            adj_mean = mean + shift
            z = (value - adj_mean) / sd
            res = NormResult(metric_id, "ok", pop_label, source, mean, sd, z, _percentile(z))
            res.age_adjusted = True
            res.adjusted_mean = adj_mean
            res.age_grade = eff.get("grade", "")
            res.age_source = eff.get("source", "")
            res.message = (f"expected mean age-adjusted {shift:+.1f} mm to ~{adj_mean:.1f} mm for "
                           f"age {age} (directional cross-sectional trend, grade {eff.get('grade', '?')})")
        else:
            res.status = "outside_band"
            res.message = ("compared against adult (18-45) norms; no evidence-based aging "
                           "adjustment exists for this metric yet")
    return res
