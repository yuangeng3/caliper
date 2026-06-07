"""Caliper command-line interface.

    caliper analyze selfie.jpg --ancestry east_asian --sex female --age 34
    caliper analyze selfie.jpg --ancestry european --sex male --annotate check.png
    caliper analyze selfie.jpg --ancestry african --sex female --json

The first run downloads the MediaPipe model (~3 MB) once; everything else is
fully on-device. Run with --annotate the first time and open the overlay to
confirm each labeled landmark lands where it should — tune constants.py if not.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__, constants as C


def _analyze(args: argparse.Namespace) -> int:
    # heavy imports happen here so `caliper --help` stays instant
    import numpy as np

    from . import (calibration, geometry, landmarks, norms, quality, report,
                   score as scoremod, skin as skinmod)

    if args.ancestry not in C.ANCESTRY_CHOICES:
        sys.exit(f"--ancestry must be one of: {', '.join(C.ANCESTRY_CHOICES)}")
    if args.sex not in C.SEX_CHOICES:
        sys.exit(f"--sex must be one of: {', '.join(C.SEX_CHOICES)}")

    try:
        res = landmarks.detect(args.image)
    except (ValueError, RuntimeError) as e:
        sys.exit(str(e))

    if args.annotate:
        landmarks.annotate(res, args.annotate)
        print(f"wrote landmark overlay -> {args.annotate}", file=sys.stderr)

    cal = calibration.calibrate(res.points_px, user_ipd_mm=args.ipd_mm)
    qual = quality.assess(res.points_px, res.transform_matrix, res.image_bgr)
    measured = geometry.compute(res.points_px, cal.mm_per_px)

    data = norms.load()
    norm_results = {
        mid: norms.evaluate(mid, m.value, args.ancestry, args.sex, args.age, data)
        for mid, m in measured.items()
    }

    skin_rep = None
    if not args.no_skin:
        rgb = np.asarray(res.image_bgr)[..., ::-1]   # BGR -> RGB without importing cv2 again
        skin_rep = skinmod.analyze(rgb, res.points_px, ancestry=args.ancestry,
                                   calibrated=args.color_chart)

    components = scoremod.compute(norm_results, res.points_px, skin=skin_rep)

    if args.json:
        print(json.dumps(_to_dict(measured, norm_results, cal, qual, skin_rep, args), indent=2))
    else:
        print(report.render(measured, norm_results, cal, qual,
                            ancestry=args.ancestry, sex=args.sex, age=args.age,
                            skin=skin_rep, components=components))

    if args.save:
        _save_session(args, measured, skin_rep)
    return 0


def _save_session(args, measured, skin_rep) -> None:
    from datetime import datetime
    from pathlib import Path

    from . import store

    db = store.Store()
    vault = store.DEFAULT_DIR / "vault" / args.profile
    ts = datetime.now().isoformat(timespec="seconds")
    stored_img = vault / f"{ts.replace(':', '-')}{Path(args.image).suffix}"
    removed = store.strip_and_store_image(args.image, str(stored_img))

    metrics = {mid: (m.value, m.unit) for mid, m in measured.items()}
    if skin_rep is not None:
        metrics["skin_ita"] = (skin_rep.overall_ita, "deg")
        metrics["skin_evenness_sd"] = (skin_rep.evenness_sd, "ita")
        metrics["skin_malar_delta"] = (skin_rep.malar_delta, "rel")
        metrics["skin_cheek_asym"] = (skin_rep.cheek_asymmetry, "rel")
        metrics["skin_erythema"] = (skin_rep.mean_erythema, "rel")
    db.add_session(args.profile, ts, metrics, ancestry=args.ancestry, sex=args.sex,
                   age=args.age, image_path=str(stored_img))
    db.close()
    print(f"\nsaved session '{args.profile}' @ {ts}", file=sys.stderr)
    print(f"  photo copied to local vault with metadata stripped: "
          f"{', '.join(removed) if removed else 'none found'}", file=sys.stderr)


def _trend(args: argparse.Namespace) -> int:
    from . import constants as C, store

    db = store.Store()
    series = db.series(args.profile, args.metric)
    db.close()
    if len(series) < 2:
        print(f"need >=2 saved sessions for '{args.profile}' / {args.metric} "
              f"(have {len(series)}). Run `analyze --save --profile {args.profile}` over time.")
        return 0

    print(f"TREND  {args.metric}  ({args.profile})\n")
    base_ts, base_val = series[0]
    prev_val = base_val
    for ts, val in series:
        mdc = store.fallback_mdc(val, C.IRIS_CALIBRATION_ERROR_PCT)
        d_prev = val - prev_val
        verdict = ("real change" if store.is_real_change(d_prev, mdc)
                   else "within noise") if val != base_val else "baseline"
        print(f"  {ts:<20} {val:8.2f}   d-prev {d_prev:+6.2f}   "
              f"(MDC95 +/-{mdc:.2f})  {verdict}")
        prev_val = val
    print(f"\n  vs baseline: {prev_val - base_val:+.2f}. A change only counts when it "
          "exceeds the MDC95 band (capture noise).")
    return 0


def _interventions(args: argparse.Namespace) -> int:
    from . import interventions
    print(interventions.render())
    return 0


def _to_dict(measured, norm_results, cal, qual, skin_rep, args) -> dict:
    d = {
        "cohort": {"ancestry": args.ancestry, "sex": args.sex, "age": args.age},
        "calibration": {
            "mm_per_px": cal.mm_per_px, "source": cal.source,
            "implied_ipd_mm": cal.implied_ipd_mm, "plausible": cal.plausible,
            "error_pct": cal.error_pct,
        },
        "quality": {
            "yaw": qual.yaw, "pitch": qual.pitch, "roll": qual.roll,
            "passed": qual.passed, "issues": qual.issues,
        },
        "metrics": {
            mid: {
                "value": m.value, "unit": m.unit, "grade": m.grade,
                "approximate": m.approximate,
                "norm": _norm_dict(norm_results.get(mid)),
            }
            for mid, m in measured.items()
        },
    }
    if skin_rep is not None:
        d["skin"] = {
            "overall_ita": skin_rep.overall_ita, "tone_category": skin_rep.tone_category,
            "monk": skin_rep.monk, "evenness_sd": skin_rep.evenness_sd,
            "cheek_asymmetry": skin_rep.cheek_asymmetry, "malar_delta": skin_rep.malar_delta,
            "mean_erythema": skin_rep.mean_erythema, "calibrated": skin_rep.calibrated,
            "confidence": skin_rep.confidence, "priority_concerns": skin_rep.priority_concerns,
            "regions": [{"region": r.region, "ita": r.ita, "erythema": r.erythema,
                         "melanin": r.melanin} for r in skin_rep.regions],
        }
    return d


def _norm_dict(nr) -> dict | None:
    if nr is None:
        return None
    return {
        "status": nr.status, "population": nr.population_label,
        "mean": nr.mean, "sd": nr.sd, "z": nr.z, "percentile": nr.percentile,
        "source": nr.source, "message": nr.message,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="caliper",
        description="Honest, ancestry/age/sex-aware, fully-local facial geometry analysis.")
    p.add_argument("--version", action="version", version=f"caliper {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="analyze a frontal face photo")
    a.add_argument("image")
    a.add_argument("--ancestry", required=True, choices=C.ANCESTRY_CHOICES,
                   help="self-reported ancestry (descriptive; selects the reference cohort)")
    a.add_argument("--sex", required=True, choices=C.SEX_CHOICES)
    a.add_argument("--age", type=int, default=None, help="age in years (norms cover 18-45)")
    a.add_argument("--ipd-mm", type=float, default=None, dest="ipd_mm",
                   help="your ruler-measured interpupillary distance (mm) for a personalized scale")
    a.add_argument("--annotate", metavar="OUT.png", help="also write a labeled landmark overlay")
    a.add_argument("--no-skin", action="store_true", dest="no_skin",
                   help="skip skin analysis (geometry only)")
    a.add_argument("--color-chart", action="store_true", dest="color_chart",
                   help="declare a colour reference is in frame (enables absolute skin tone)")
    a.add_argument("--save", action="store_true",
                   help="save this session to the local store (photo metadata-stripped)")
    a.add_argument("--profile", default="me", help="profile name for saved/longitudinal data")
    a.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    a.set_defaults(func=_analyze)

    t = sub.add_parser("trend", help="show a saved metric over time with MDC95 gating")
    t.add_argument("--profile", default="me")
    t.add_argument("--metric", required=True, help="e.g. nasal_width, canthal_tilt, skin_evenness_sd")
    t.set_defaults(func=_trend)

    iv = sub.add_parser("interventions", help="print the evidence-graded intervention map")
    iv.set_defaults(func=_interventions)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
