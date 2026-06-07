"""Render a plain-language readout: the gist, what's worth your attention (skin),
every measurement explained in normal words, and what this is NOT.

Built on interpret.py so the output leads with meaning instead of raw numbers — the
fix for "I see data, I don't understand it." Pure stdlib + the interpret layer.
"""
from __future__ import annotations

import textwrap

from .calibration import Calibration
from .geometry import Metric
from .norms import NormResult
from .quality import Quality


def _wrap(text: str, indent: str = "  ", width: int = 84) -> list[str]:
    return textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent) or [indent.rstrip()]


def render(measured: dict[str, Metric],
           norm_results: dict[str, NormResult],
           cal: Calibration,
           qual: Quality,
           *,
           ancestry: str, sex: str, age: int | None,
           skin=None, components=None, interpretation=None) -> str:
    from . import interpret as interp_mod
    rep = interpretation or interp_mod.interpret(
        measured, norm_results, ancestry=ancestry, sex=sex, age=age, skin=skin)

    L: list[str] = []
    L.append("CALIPER  ·  what your measurements actually mean")
    L.append("Runs on your device. Nothing left your computer.")
    L.append("")

    L.append("THE GIST")
    for s in rep.summary:
        L += _wrap(s)
    L.append("")

    L.append("WORTH YOUR ATTENTION — YOUR SKIN")
    L.append("  (facial bone structure is fixed in adults; skin is where habits actually move things)")
    L += _wrap(rep.skin_summary)
    if rep.tone_line:
        L += _wrap(rep.tone_line)
    if rep.skin_note:
        L += _wrap(rep.skin_note)
    if rep.skin_watch:
        L += _wrap("watch first: " + "; ".join(rep.skin_watch))
    if rep.skin_moves:
        L.append("  highest-yield, evidence-graded moves:")
        for m in rep.skin_moves:
            L += _wrap("- " + m, indent="    ")
    for ex in rep.skin_extra:
        L += _wrap(ex)
    L.append("")

    L.append("YOUR FACE, EXPLAINED  (all descriptive — normal variation, nothing to 'fix')")
    for it in rep.items:
        line = f"{it.plain_name}: {it.value_str} — {it.standing}"
        if it.meaning:
            line += f"  ({it.meaning})"
        L += _wrap(line)
    L.append("")

    L.append("WHAT THIS IS — AND ISN'T")
    L += _wrap("A descriptive baseline plus skin guidance. NOT a beauty score, NOT a list of "
               "things to fix, NOT a medical diagnosis. Its real power is tracking change over "
               "time — re-shoot in similar light and watch the numbers, not any single reading.")
    L.append("")

    qa = "; ".join(qual.issues) if qual.issues else "looks good"
    L.append(f"photo quality: {qa}.   ruler: iris -> {cal.mm_per_px:.3f} mm/px (+/-{cal.error_pct:.0f}%).")
    return "\n".join(L)
