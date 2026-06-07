"""Skin analysis — objective tone (ITA / Monk), evenness, redness, and a
tone-conditioned read of what actually matters for this skin.

Two honesty rules run through everything here:

1. ITA and the Del Bino bins use the canonical ``atan2((L*-50), b*)`` formula —
   the widely-copied variant breaks exactly at the very dark and very light skin
   you most need to serve.
2. Uncalibrated phone RGB cannot be trusted as absolute colour (perceptual
   white-balance does NOT guarantee clinical validity). So without an in-frame
   colour chart we mark every skin-colour metric "directional only" — a relative
   read and a longitudinal trend, never a diagnosis.

The colour math is pure NumPy and unit-tested; only region sampling needs an image.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import constants as C

# --- sRGB -> CIE L*a*b* (D65) ---------------------------------------------
_D65 = np.array([95.047, 100.0, 108.883])
_M = np.array([
    [0.4124, 0.3576, 0.1805],
    [0.2126, 0.7152, 0.0722],
    [0.0193, 0.1192, 0.9505],
])


def srgb_to_lab(rgb) -> np.ndarray:
    """rgb: (...,3) in 0-255. Returns (...,3) L*,a*,b*."""
    c = np.asarray(rgb, dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T * 100.0
    t = xyz / _D65

    def f(x):
        return np.where(x > 0.008856, np.cbrt(x), 7.787 * x + 16.0 / 116.0)

    fx, fy, fz = f(t[..., 0]), f(t[..., 1]), f(t[..., 2])
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def ita(lab) -> float:
    """Individual Typology Angle in degrees (canonical atan2 form)."""
    lab = np.asarray(lab, dtype=float)
    return float(np.degrees(np.arctan2(lab[..., 0] - 50.0, lab[..., 2])))


# Del Bino / Chardon six tone categories (descriptive, never ranked).
_ITA_BINS = [(55, "very light"), (41, "light"), (28, "intermediate"),
             (10, "tan"), (-30, "brown")]


def ita_category(ita_deg: float) -> str:
    for thr, name in _ITA_BINS:
        if ita_deg > thr:
            return name
    return "dark"


# ITA->Monk(1-10) crosswalk via published Del Bino/Monk bin boundaries rather than a
# naive linear interp — aligned with the npj Digital Medicine 2025 method that mapped
# ITA->Monk at 89-92% on clinical images. Boundaries coincide with the Del Bino category
# edges (>55 very light, 41 light, 28 intermediate, 10 tan, -30 brown). Monotonic.
_ITA_MONK_BINS = [(55, 1), (48, 2), (41, 3), (34, 4), (28, 5),
                  (18, 6), (10, 7), (-10, 8), (-30, 9)]


def ita_to_monk(ita_deg: float) -> int:
    """ITA->Monk(1-10) crosswalk (1 = lightest), via published bin boundaries."""
    for thr, monk in _ITA_MONK_BINS:
        if ita_deg > thr:
            return monk
    return 10


def erythema_index(rgb) -> float:
    """Melanin-aware redness index (Dawson-style) for a single RGB triple:
    emphasises haemoglobin while correcting for melanin, so redness isn't
    under-read on darker skin. Relative units."""
    r, g, _ = (np.asarray(rgb, dtype=float).ravel()[:3] + 1.0)
    ei = 100.0 * (np.log10(255.0 / g) - np.log10(255.0 / r))   # = 100*log10(r/g)
    melanin = 100.0 * np.log10(255.0 / r)                       # Dawson correction
    return float(ei + 0.04 * melanin)


def melanin_index(rgb) -> float:
    """Relative melanin index from red-channel optical density (single RGB triple)."""
    r = float(np.asarray(rgb, dtype=float).ravel()[0]) + 1.0
    return float(100.0 * np.log10(255.0 / r))


# tone-conditioned priority concerns (the Chang lens)
_DEEP = {"intermediate", "tan", "brown", "dark"}
_PRIORITY = {
    "deep": ["post-inflammatory hyperpigmentation (PIH)", "melasma / uneven tone",
             "tone evenness"],
    "fair": ["erythema / redness", "telangiectasia", "photoaging & fine wrinkling",
             "actinic (sun) damage"],
}
# Skin behaviour (pigment-prone vs photoaging-prone) tracks ancestry/phototype more
# reliably than a single noisy selfie ITA, so the GUIDANCE bucket is ancestry-led.
_PIGMENT_FIRST = {"east_asian", "south_asian", "african", "african_american", "middle_eastern"}
# Rough upper ITA bound per ancestry; if the detected tone reads well above this the
# photo is over-bright and absolute tone is not trustworthy.
_ANCESTRY_ITA_CEIL = {"african": 25, "african_american": 30, "south_asian": 45,
                      "east_asian": 50, "middle_eastern": 50, "european": 66}
_ANCESTRY_PHOTOTYPE = {"east_asian": "III–IV (light-medium)", "south_asian": "IV–V",
                       "african": "V–VI", "african_american": "IV–VI",
                       "middle_eastern": "III–V", "european": "I–III (varies widely)"}


@dataclass
class RegionSkin:
    region: str
    rgb: tuple[float, float, float]
    ita: float
    erythema: float
    melanin: float


@dataclass
class SkinReport:
    regions: list[RegionSkin]
    overall_ita: float
    tone_category: str
    monk: int
    evenness_sd: float           # SD of ITA across regions (lower = more even)
    mean_erythema: float
    priority_concerns: list[str]
    calibrated: bool
    confidence: str
    bucket: str = "fair"         # guidance bucket: "deep" (pigment-first) | "fair" (photoaging-first)
    wb_method: str = "none"      # how lighting was normalized
    normalized: bool = False
    n_sclera: int = 0
    tone_reliable: bool = True   # False when the photo's brightness defeats absolute tone
    expected_phototype: str = ""  # ancestry-based fallback when tone is unreliable
    # Melasma-/PIH-sensitive pigment signals. The global evenness SD above is dominated by region
    # anatomy (nose vs chin) and barely moves with malar melasma — the #1 East Asian concern — so
    # these target the actual pattern. Both relative, within-photo, MDC95-trackable.
    cheek_asymmetry: float = 0.0  # |L-R| cheek melanin: asymmetry -> sun spots / unilateral PIH
    malar_delta: float = 0.0      # mean cheek - forehead melanin: symmetric elevation -> melasma
    notes: list[str] = field(default_factory=list)


def _sample_patch(img_rgb: np.ndarray, xy, half: int = 9) -> np.ndarray:
    """Robust skin sample: trim specular highlights and deep shadows, then median."""
    h, w = img_rgb.shape[:2]
    x, y = int(round(xy[0])), int(round(xy[1]))
    patch = img_rgb[max(0, y - half):y + half + 1,
                    max(0, x - half):x + half + 1].reshape(-1, 3).astype(float)
    if len(patch) < 4:
        return np.zeros(3)
    lum = patch.mean(axis=1)
    lo, hi = np.percentile(lum, [20, 70])
    keep = patch[(lum >= lo) & (lum <= hi)]
    src = keep if len(keep) >= 3 else patch
    return np.median(src, axis=0)


def _sclera_from_eye(img: np.ndarray, P, outer: int, inner: int, iris: int):
    """Low-saturation pixels between the eye corners — candidate eye-white (the bright
    end of these is the actual sclera; iris/lash contaminate the dark end)."""
    h, w = img.shape[:2]
    x0, x1 = sorted([int(P[outer][0]), int(P[inner][0])])
    if x1 - x0 < 6:
        return None
    cy = int(P[iris][1])
    half = max(3, int(0.30 * (x1 - x0)))
    box = img[max(0, cy - half):min(h, cy + half),
              max(0, x0):min(w, x1)].reshape(-1, 3).astype(float)
    if len(box) < 20:
        return None
    mx, mn = box.max(axis=1), box.min(axis=1)
    sat = (mx - mn) / (mx + 1e-6)
    cand = box[sat < 0.22]                       # neutral pixels only (sclera + neutral skin)
    return cand if len(cand) >= 12 else None


def estimate_illuminant(img_rgb: np.ndarray, P_px: np.ndarray):
    """Estimate the scene white point from the sclera — the one near-neutral reference
    in a bare face. Uses the BRIGHT end of the eye-white candidates (the true white;
    the dark end is iris/lash). Returns (illuminant_rgb, n_pixels) or (None, 0)."""
    parts = []
    for outer, inner, iris in (
        (C.IDX["r_canthus_out"], C.IDX["r_canthus_in"], C.RIGHT_IRIS_CENTER),
        (C.IDX["l_canthus_in"], C.IDX["l_canthus_out"], C.LEFT_IRIS_CENTER),
    ):
        s = _sclera_from_eye(img_rgb, P_px, outer, inner, iris)
        if s is not None:
            parts.append(s)
    if not parts:
        return None, 0
    allpix = np.vstack(parts)
    lum = allpix.mean(axis=1)
    bright = allpix[lum >= np.percentile(lum, 75)]   # the actual white of the eye
    return np.median(bright, axis=0), len(bright)


def analyze(img_rgb: np.ndarray, P_px: np.ndarray, *, ancestry: str | None = None,
            calibrated: bool = False) -> SkinReport:
    """img_rgb: HxWx3 (0-255, RGB). P_px: (>=478,2) landmark pixels.

    Lighting colour-cast is corrected using the sclera (white of the eye) as an
    internal neutral reference. Absolute tone from an uncalibrated selfie is still
    only an estimate, so we reconcile the detected tone against the ancestry prior and
    flag it when the photo's brightness makes it untrustworthy. The robust, trackable
    signals are evenness and redness (relative, within-photo)."""
    illum, n_scl = estimate_illuminant(img_rgb, P_px)
    notes: list[str] = []
    if illum is not None:
        gain = np.clip(illum.mean() / np.maximum(illum, 1.0), 0.5, 2.0)  # cast only, brightness-preserving
        wb_method = "sclera (eye-white) colour-cast correction"
        confidence = "estimated (cast-corrected from your eye-whites)"
        normalized = True
    else:
        gain = np.ones(3)
        wb_method = "none — no clear eye-white found"
        confidence = "low (uncalibrated lighting)"
        normalized = False

    regions: list[RegionSkin] = []
    for name, idx in C.SKIN_REGIONS.items():
        corr = np.clip(_sample_patch(img_rgb, P_px[idx]) * gain, 0, 255)
        lab = srgb_to_lab(corr)
        regions.append(RegionSkin(name, tuple(float(v) for v in corr),
                                   ita(lab), erythema_index(corr), melanin_index(corr)))

    by_name = {r.region: r for r in regions}
    cheeks = [by_name[k].ita for k in ("right_cheek", "left_cheek") if k in by_name]
    itas = np.array([r.ita for r in regions])
    overall = float(np.median(cheeks)) if cheeks else float(np.median(itas))
    cat = ita_category(overall)

    # melasma-/PIH-sensitive pigment signals (melanin index; see SkinReport)
    rc, lc, fh = by_name.get("right_cheek"), by_name.get("left_cheek"), by_name.get("forehead")
    cheek_asym = abs(rc.melanin - lc.melanin) if rc is not None and lc is not None else 0.0
    cheek_mels = [r.melanin for r in (rc, lc) if r is not None]
    malar_delta = float(np.mean(cheek_mels)) - fh.melanin if cheek_mels and fh is not None else 0.0

    # guidance bucket follows ancestry (robust); detected tone refines the displayed value
    if ancestry in _PIGMENT_FIRST:
        bucket = "deep"
    elif ancestry == "european":
        bucket = "fair"
    else:
        bucket = "deep" if cat in _DEEP else "fair"

    tone_reliable = True
    expected = _ANCESTRY_PHOTOTYPE.get(ancestry, "")
    ceil = _ANCESTRY_ITA_CEIL.get(ancestry)
    if ceil is not None and overall > ceil + 6:
        tone_reliable = False
        confidence = "tone uncertain — photo is bright; reconciled with ancestry"
        notes.append(f"Detected tone reads lighter than typical for {ancestry}, so the photo is "
                     "bright and absolute tone is unreliable here. Evenness & redness below are "
                     "robust; for tone, use the ancestry-based estimate.")
    elif normalized:
        notes.append("Lighting cast-corrected from your eye-whites — good for tracking change; "
                     "add a grey card in-frame for lab-grade absolute tone.")
    else:
        notes.append("No clear eye-white found to correct lighting, so absolute tone is a rough "
                     "guess; guidance leans on your stated ancestry.")

    return SkinReport(
        regions=regions, overall_ita=overall, tone_category=cat, monk=ita_to_monk(overall),
        evenness_sd=float(np.std(itas)), mean_erythema=float(np.mean([r.erythema for r in regions])),
        priority_concerns=_PRIORITY[bucket], calibrated=calibrated, confidence=confidence,
        bucket=bucket, wb_method=wb_method, normalized=normalized, n_sclera=n_scl,
        tone_reliable=tone_reliable, expected_phototype=expected,
        cheek_asymmetry=float(cheek_asym), malar_delta=float(malar_delta), notes=notes,
    )
