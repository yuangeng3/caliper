"""Turn raw measurements into plain language a non-specialist can act on.

The job: answer "what does this mean for me?" Every metric becomes a plain name, a
plain standing (closer-set / typical / broad…), and an honest meaning. Facial
geometry is framed as a descriptive baseline (normal variation, not a to-do list);
skin is framed as where you actually have leverage, with guidance driven by your
self-reported ancestry (reliable) rather than a noisy uncalibrated colour reading.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import interventions as iv_mod

_GEN_PATH = Path(__file__).resolve().parents[1] / "data" / "general_reference.json"

# order to present, grouped how a person reads a face
_ORDER = ["canthal_tilt", "intercanthal_width", "fifth_intercanthal", "eye_fissure_length",
          "nasal_width", "nasal_height", "nasal_index", "mouth_width",
          "bizygomatic_width", "facial_height", "facial_index",
          "third_middle", "third_lower", "third_upper", "fwhr", "ipd"]


def load_general(path: Path | None = None) -> dict:
    with open(path or _GEN_PATH) as f:
        return json.load(f)


@dataclass
class Item:
    metric_id: str
    plain_name: str
    value_str: str
    standing: str          # plain-language position
    meaning: str
    notable: bool = False


@dataclass
class Interpretation:
    summary: list[str]
    items: list[Item]
    skin_summary: str
    skin_watch: list[str]
    skin_moves: list[str]
    skin_note: str
    tone_line: str = ""
    skin_extra: list[str] = field(default_factory=list)  # malar read, ancestry notes, PIH prevention


def _band_label(value: float, bands: list[dict]) -> str:
    for b in bands:
        if value <= b["max"]:
            return b["label"]
    return bands[-1]["label"]


def _interpret_metric(m, nr, info: dict) -> Item:
    val = f"{m.value:.1f}{info.get('unit', '')}".replace(".0", ".0")
    meaning = info.get("meaning", "")
    notable = False

    if info.get("unreliable"):
        standing = "not reliable from a photo"
    elif nr is not None and nr.status in ("ok", "outside_band"):
        pct = nr.percentile
        typical = 15 <= pct <= 85
        edge = "" if typical else " — toward the edge, still normal variation"
        standing = f"{_ord(pct)} percentile for {nr.population_label}{edge}"
        notable = not typical
    elif "bands" in info:
        standing = _band_label(m.value, info["bands"]) + " (general range)"
    elif nr is not None and nr.status == "no_sd":
        standing = f"near the {nr.population_label} average"
    else:
        standing = "your baseline — for tracking change over time"

    return Item(m.id, info.get("plain_name", m.label), val, standing, meaning, notable)


def _ord(p: float) -> str:
    n = int(round(p))
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _summary(items: list[Item], cohort_label: str) -> list[str]:
    notable = [it for it in items if it.notable]
    out = []
    if not notable:
        out.append(f"Your facial proportions all sit within the typical range for {cohort_label}. "
                   "Nothing here is unusual or needs attention.")
    else:
        names = ", ".join(it.plain_name.lower() for it in notable)
        out.append(f"Almost everything is typical for {cohort_label}. A few sit toward the edges of "
                   f"the normal range (still normal variation): {names}.")
    out.append("Facial structure is essentially fixed in adults, so read this as your personal "
               "baseline for tracking change over time — not a list of things to 'fix'.")
    return out


def _format_move(iv: dict) -> str:
    effect = iv["affects"][0] if iv.get("affects") else "general skin health"
    return f"{iv['label']} (evidence grade {iv['grade']}) — helps with {effect}."


def _skin_guidance(ancestry: str, skin, general: dict, iv_data: dict):
    sk = general["skin_by_ancestry"]
    # bucket from the detected report when available (it is ancestry-led + reconciled)
    detected_bucket = getattr(skin, "bucket", None)
    if detected_bucket == "deep":
        key = "pigment_first"
    elif detected_bucket == "fair":
        key = "photoaging_first"
    else:
        key = "pigment_first" if ancestry in sk["_pigment_first"] else "photoaging_first"
    block = sk[key]
    ivs = iv_data["interventions"]
    moves = [_format_move(ivs[k]) for k in block["interventions"] if k in ivs]

    tone_line, note = "", ""
    if skin is not None:
        if getattr(skin, "tone_reliable", True):
            tone_line = f"Detected skin tone: {skin.tone_category} (lighting cast-corrected from your eye-whites)."
        else:
            tone_line = (f"Skin tone: uncertain from this photo (it's bright) — based on your "
                         f"ancestry, roughly Fitzpatrick {skin.expected_phototype}.")
        note = (f"Tone evenness is {skin.evenness_sd:.1f} (lower = more even) — this relative number "
                "is robust and the right thing to track over time.")

    extra: list[str] = []
    # Malar/asymmetry pigment read — the signal that actually tracks melasma & spotting, where the
    # global evenness SD is blind. Only meaningful for pigment-first skin.
    if skin is not None and key == "pigment_first":
        md, ca = getattr(skin, "malar_delta", 0.0), getattr(skin, "cheek_asymmetry", 0.0)
        if md or ca:
            extra.append(
                f"Pigment pattern: malar delta (cheek vs forehead) {md:+.1f}, left/right cheek "
                f"asymmetry {ca:.1f}. A higher symmetric malar delta points to melasma; higher "
                "asymmetry points to sun spots or one-sided marks. Track these, not just overall evenness.")
    # Ancestry-resolved notes (e.g. East Asian: drier/reactive barrier; wrinkles lag ~a decade).
    extra.extend(sk.get("ancestry_notes", {}).get(ancestry, []))
    # PIH prevention (reference-only; no model).
    if block.get("prevent"):
        extra.append(block["prevent"])
    return block["summary"], list(block["watch"]), moves, note, tone_line, extra


def interpret(measured: dict, norm_results: dict, *, ancestry: str, sex: str,
              age: int | None, skin=None, general: dict | None = None,
              iv_data: dict | None = None) -> Interpretation:
    general = general or load_general()
    iv_data = iv_data or iv_mod.load()
    info_all = general["metrics"]

    items: list[Item] = []
    for mid in _ORDER:
        m = measured.get(mid)
        if m is None or mid not in info_all:
            continue
        items.append(_interpret_metric(m, norm_results.get(mid), info_all[mid]))

    cohort_label = next((nr.population_label for nr in norm_results.values()
                         if nr and nr.population_label), f"{ancestry} {sex}")
    s_sum, s_watch, s_moves, s_note, tone_line, s_extra = _skin_guidance(
        ancestry, skin, general, iv_data)
    return Interpretation(_summary(items, cohort_label), items, s_sum, s_watch, s_moves,
                          s_note, tone_line, s_extra)
