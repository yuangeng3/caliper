"""Longitudinal store, MDC95 gating, EXIF stripping, and the intervention map."""
from __future__ import annotations

import math

import pytest
from PIL import Image

from caliper import interventions, store


# --- minimal detectable change -------------------------------------------
def test_mdc95_formula_and_gating():
    assert store.sem_from_repeats([10.0, 10.0, 10.0]) == pytest.approx(0.0)
    sem = store.sem_from_repeats([9.0, 10.0, 11.0])          # sample SD = 1.0
    assert sem == pytest.approx(1.0, abs=1e-9)
    mdc = store.mdc95(sem)
    assert mdc == pytest.approx(1.96 * math.sqrt(2.0), abs=1e-9)  # ~2.77
    assert store.is_real_change(3.0, mdc) is True
    assert store.is_real_change(2.0, mdc) is False


def test_fallback_mdc_from_error_band():
    # a 33 mm metric at 4.3% error -> SEM band ~1.42 mm -> MDC ~3.93 mm
    mdc = store.fallback_mdc(33.0, 4.3)
    assert mdc == pytest.approx(33.0 * 0.043 * 1.96 * math.sqrt(2.0), abs=1e-9)


# --- SQLite store ---------------------------------------------------------
def test_store_roundtrip(tmp_path):
    db = store.Store(tmp_path / "t.db")
    db.add_session("yuan", "2026-06-01T09:00", {"nasal_width": (37.4, "mm")},
                   ancestry="east_asian", sex="female", age=40)
    db.add_session("yuan", "2026-06-08T09:00", {"nasal_width": (37.6, "mm")})
    series = db.series("yuan", "nasal_width")
    assert [v for _, v in series] == [pytest.approx(37.4), pytest.approx(37.6)]
    assert len(db.sessions("yuan")) == 2
    assert db.sessions("other") == []
    db.close()


# --- EXIF / GPS stripping -------------------------------------------------
def test_exif_stripped_on_ingest(tmp_path):
    src, dst = tmp_path / "in.jpg", tmp_path / "vault" / "out.jpg"
    img = Image.new("RGB", (16, 16), (120, 120, 120))
    exif = img.getexif()
    exif[271] = "SomeMake"    # 271 = Make
    exif[272] = "SomeModel"   # 272 = Model
    img.save(src, exif=exif)

    removed = store.strip_and_store_image(str(src), str(dst))
    assert "Make" in removed and "Model" in removed
    # the stored copy carries no EXIF
    assert len(Image.open(dst).getexif()) == 0


# --- intervention map -----------------------------------------------------
def test_intervention_map_grades_and_honesty():
    data = interventions.load()
    iv = data["interventions"]
    assert iv["tretinoin"]["grade"] == "A"
    assert iv["mewing"]["grade"] == "D" and iv["mewing"]["affects"] == []
    text = interventions.render(data)
    assert "MDC95" in text
    assert "[A] Topical tretinoin" in text
    # grade A must be listed before grade D
    assert text.index("[A]") < text.index("[D]")
