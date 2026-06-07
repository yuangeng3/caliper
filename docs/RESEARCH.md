# Caliper — research & design record

Every non-obvious design choice in Caliper traces to primary literature. This is
the cited backbone (8 research areas, ~200 sources reviewed). It doubles as the
**license/dataset manifest** for a freely-redistributable tool.

> House voice: rigorous and mechanism-driven (Attia-style evidence grading) crossed
> with a skin-of-colour dermatology lens (Chang-style). Grade the evidence; never
> overclaim; condition everything on the person in front of you.

---

## 0. The thesis: there is no ideal face

The Renaissance neoclassical canons (equal facial thirds, "eye-width = intercanthal",
rule of fifths) **fail in the large majority of every population measured — including
the Caucasians they were derived from.**

- Two best-performing canons hold in only ~40% / ~37% of young Caucasians; the rest in 16–37% (Farkas).
- Facial three-section canon and orbital canon: **0%** of Southern Chinese; orbital canon 33%/27% of Kenyan men/women. ([PLOS One](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0052593))

**Consequence for the code:** no universal template. Score each metric as a percentile
*within* the user's (ancestry, sex, age) cohort; surface "insufficient data" rather than
fabricate. This is implemented in `norms.py` + `data/norms/frontal_anthropometry.json`.

Primary: Farkas, Katic & Forrest (2005), *International Anthropometric Study of Facial
Morphology in Various Ethnic Groups/Races*, J Craniofac Surg 16:615–646 ([PubMed](https://pubmed.ncbi.nlm.nih.gov/16077306/)).

---

## 1. Landmark engine

**Decision: MediaPipe FaceLandmarker (Apache-2.0) everywhere.** It is the only dense
landmarker that runs truly client-side in-browser (via `@mediapipe/tasks-vision` WASM/WebGPU)
*and* has a clean Python path. 478 landmarks (468 surface + 10 iris) + 52 blendshapes +
a 4×4 facial-transformation matrix (free head-pose).

- Frontal-optimized; degrades past ~45° yaw → route profile shots elsewhere (§6).
- License-clean end to end (model + code Apache-2.0). InsightFace code is MIT but its
  pretrained weights are **non-commercial**; 3DDFA_V2/SynergyNet code is MIT but trained on
  research-only 300W-LP — a redistribution caveat. ([MediaPipe docs](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/web_js))

## 2. Calibration — the iris as a ruler

Horizontal visible iris diameter is **~11.7 mm, near-constant across the population**
(no meaningful sex/age effect), so it converts pixels → millimetres from a single photo.

- Iris-scaled selfie error: **~2.9% horizontal / 4.3% vertical** MAPE (pitch foreshortens vertical).
- Calibrated MediaPipe 3D-mesh distance error ~2.4% vs ruler; uncalibrated 2D ~28% — **never use raw 2D pixel ratios.**
- Cross-check the implied IPD for plausibility (54–72 mm adult); reject off-axis/partial-iris shots.

Implemented in `calibration.py` (`IRIS_DIAMETER_MM = 11.7`, error carried as a band).
Sources: [MediaPipe Iris (Google Research)](https://research.google/blog/mediapipe-iris-real-time-iris-tracking-depth-estimation/), [iris-scaling DL paper PMC10447546](https://pmc.ncbi.nlm.nih.gov/articles/PMC10447546/), [dermatology landmark POC PMC12936401](https://pmc.ncbi.nlm.nih.gov/articles/PMC12936401/).

## 3. Ancestry/age/sex norms — the differentiator

Concrete, cited reference values (mm, mean ± SD) span European (NAW), African American,
Kenyan, Hong Kong/Han Chinese, Iranian, and Gujarati cohorts. Highlights:

- **Nasal width (al-al)** is the single most ancestry-discriminating metric: NAW male 34.9 mm vs African American 44.1 mm vs Kenyan 43.2 mm.
- **Intercanthal width (en-en)**: East Asian largest (HK Chinese male 40.6 mm) — the rule of fifths systematically fails there.
- **Bizygomatic width** is the *least* ancestry-variable → used as a normalizer.
- **Aging** is quantifiable (orbital aperture widens, nasal width +~4 mm, glabellar/maxillary angles flatten 2–5°, lips thin), with a sharp **post-menopause** acceleration in women (Windhager 2019). **Shipped (Jun 2026):** `age_effects` in `frontal_anthropometry.json` shifts the expected *mean* (SD untouched — no honest basis to rescale dispersion) for ages outside the 18–45 band, per metric and evidence-graded, with a female post-menopause acceleration term; metrics without a recorded trend stay flagged `outside_band` rather than guess. A 67yo's nasal width now reads ~median for his age instead of a spurious 90th percentile.
- **Ancestry is self-reported**, descriptive, never auto-classified as fact (FairFace race accuracy ~81.5%).

Sources: [Kenyan/AfAm/NAW PMC6384287](https://pmc.ncbi.nlm.nih.gov/articles/PMC6384287/),
[HK Chinese PMC3730197](https://pmc.ncbi.nlm.nih.gov/articles/PMC3730197/),
[ethnic-variability review PMC3074358](https://pmc.ncbi.nlm.nih.gov/articles/PMC3074358/),
[FaceBase 3D Facial Norms](https://www.facebase.org/facial_norms/),
[Windhager 2019](https://onlinelibrary.wiley.com/doi/full/10.1002/ajpa.23878).

## 4. Skin — ITA°, Monk, and the Chang lens *(in progress)*

- **Objective tone = ITA°**: `ITA = atan2(L*−50, b*) · 180/π` in CIE Lab (D65). Del Bino bins:
  >55 very light, 41–55 light, 28–41 intermediate, 10–28 tan, −30–10 brown, <−30 dark.
  Use `atan2` — the widely-copied altered formula breaks at the extremes you most need to serve.
- **Use Monk (10-tone), not Fitzpatrick** — Fitzpatrick is unreliable for darker skin (npj Digital Medicine 2025).
- **Separate melanin from haemoglobin** (Tsumura ICA in log-RGB, or Beer-Lambert/NMF) so **PIH (pigment)**
  and **erythema (redness)** are disentangled. Raw a* underreads redness in dark skin → use a melanin-corrected (Dawson) index.
- **Priority concern flips by tone**: dark/Asian/Latino → PIH, melasma, evenness; fair/MC1R-red-hair → erythema, telangiectasia, photoaging, fine wrinkling.
- **Honesty gate**: no in-frame colour chart (CIEDE2000 < 2) → skin-colour metrics are "directional only, not trackable" (color-clinical decoupling, 2025). Never diagnose.

Sources: [OpenOximetry ITA](https://openoximetry.org/skin-color-quantification/),
[Monk vs Fitzpatrick, npj Digital Medicine 2025](https://www.nature.com/articles/s41746-025-02245-2),
[pigment decomposition arXiv 2404.00552](https://arxiv.org/html/2404.00552),
[color-clinical decoupling arXiv 2512.21988](https://arxiv.org/html/2512.21988),
[PIH in skin of color (JCAD)](https://jcadonline.com/postinflammatory-hyperpigmentation-a-review-of-the-epidemiology-clinical-features-and-treatment-options-in-skin-of-color/).

## 4a. The Asian-skin module — pigment-first, not wrinkle-first

> **Built (Jun 2026):** azelaic acid / tranexamic acid / niacinamide added to the
> evidence map and wired into the pigment-first move list; malar-delta + cheek-asymmetry
> signals in `skin.py` (melasma/PIH-sensitive, MDC95-trackable via `trend`); ancestry-resolved
> East-Asian notes (drier/reactive barrier; wrinkles lag ~a decade); a PIH-prevention line;
> and a published Del Bino/Monk ITA→Monk crosswalk. Roadmap below = age-conditioned norms.

The Western skin model (and most commercial analyzers) treats wrinkles as the aging
signal. **For East Asian skin that is the wrong axis.** Perception of facial aging in
East Asian women is mediated by **hyperpigmentation and tone unevenness**, not rhytids;
wrinkling appears roughly **a decade later** than in Caucasian skin. A tool that flags
"wrinkles" first systematically mis-serves Asian users and misses what they actually track.

**Consequence for the code:** for the East Asian cohort, the aging-conditioned norms must
**up-weight evenness / PIH / melasma and down-rank the wrinkle index.** Priority concern by
ancestry is a first-class config, not a footnote.

- **Pigment is the primary aging + concern axis.** Melasma alone affects **8.8–40%** of Asian
  women — the single highest-prevalence concern in this cohort. Track it as a longitudinal
  **evenness/melanin-distribution** metric (MDC95-gated), never as a diagnosis.
- **PIH susceptibility is predictable** (the ATBP neural net: VISIA-derived, 1,953 patients /
  93,477 labels). A "PIH-risk" component is evidence-backed — **but VISIA-trained = proprietary,
  so reference-only, never bundled.** Same rule as §0: cite the finding, don't ship the weights.
- **ITA → Monk is the validated tone backend.** Automated ITA from CIELAB (DensePose/OpenFace)
  maps to Monk at **89–92% accuracy** on clinical images, with high 3D-scan agreement and
  *poor* Fitzpatrick alignment — independent confirmation of the §4 "Monk not Fitzpatrick" call.
  This is the methods citation for Caliper's ITA pipeline.
- **Asian-skin physiology tempers the flags.** Weakest barrier under mechanical challenge,
  *lower* NMF (drier than assumed), oilier/higher sebum, **smallest pore area of any group**,
  and higher reactivity (eccrine density). → Do **not** raise a "large pores" flag for Asian
  users against a pan-population threshold; pores are normatively smaller. Drier-than-expected
  barrier argues for a hydration/barrier note over a sebum-control nudge.
- **Honesty gate unchanged.** No in-frame colour chart (CIEDE2000 < 2) → pigment/evenness
  metrics are "directional only, not trackable." Melasma/PIH severity is *surfaced and tracked*,
  never *diagnosed*.

Sources: [East Asian aging = pigmentation, JAAD](https://www.jaad.org/article/S0190-9622(06)03046-5/abstract),
[aging differences in ethnic skin (JCAD)](https://jcadonline.com/aging-differences-in-ethnic-skin/),
[Asian skin structure/function — Rawlings 2006](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-2494.2006.00302.x),
[clinical parameters of Asian skin-tone perception — Wang 2025](https://onlinelibrary.wiley.com/doi/10.1111/jocd.70542),
[PIH susceptibility neural net (ATBP)](https://www.sciencedirect.com/science/article/abs/pii/S1350453322001321),
[melasma severity DL 2025, PMC12049110](https://pmc.ncbi.nlm.nih.gov/articles/PMC12049110/),
[cGAN pigmented-skin analysis](https://ieeexplore.ieee.org/document/10478489/),
[automated ITA→Monk, npj Digital Medicine 2025](https://www.nature.com/articles/s41746-025-01770-4)
([PMC12179258](https://pmc.ncbi.nlm.nih.gov/articles/PMC12179258/)),
[Monk scale dermatology should not overlook, JAAD 2025](https://www.jaad.org/article/S0190-9622(25)02226-1/fulltext).

## 5. The "score", demystified — why there isn't one

- The benchmark beauty dataset SCUT-FBP5500 was rated by **60 Chinese students aged 18–27**; models reproduce *their* taste (in-group bias is in the labels).
- A 2025 audit: even "diverse" MEBeauty satisfies cross-ethnicity distributional parity in **<10%** of comparisons.
- **~50% of attractiveness judgment is private taste** (Honekopp 2006: inter-rater r 0.31–0.50). A single number is statistically indefensible.
- What *is* cross-culturally robust: **averageness** (d ≈ 1.45) and **skin homogeneity** (worth up to ~20 years of perceived age, and modifiable); **symmetry is weak-to-null** once averageness is controlled.

**Decision:** no single score. Decompose into mechanism-named components (skin-tone evenness, averageness distance, a down-ranked symmetry index), each with a CI and rater provenance; ranking percentile is an opt-in, never the landing screen.

Sources: [bias audit arXiv 2509.24138](https://arxiv.org/pdf/2509.24138),
[SCUT-FBP5500 arXiv 1801.06345](https://arxiv.org/pdf/1801.06345),
[Kleisner 2024 cross-cultural](https://dspace.stir.ac.uk/bitstream/1893/35636/1/Kleisner%20et%20al%20EvolHumBehav_final%20accepted%20version.pdf),
[Honekopp summary](https://datepsychology.com/how-accurate-are-facial-attractiveness-ratings/),
[skin homogeneity Fink & Matts](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-3083.2011.04316.x).

## 6. Profile mode — soft-tissue only *(in progress)*

Profile metrics (E-line, nasolabial, nasofrontal, facial convexity, Holdaway H-angle) have
well-documented norms **that differ by ancestry**: the Caucasian E-line (UL −4, LL −2 mm)
over-retracts East Asian and African faces (whose lips are normatively more protrusive);
soft-tissue convexity ~7.7° Korean vs 12° Euro vs 14° Turkish.

- Extraction needs a full-pose landmarker (**3DDFA_V2**, MIT code) — MediaPipe/dlib silently degrade at profile.
- **No skeletal angles from a photo, ever** (SNA/SNB, true gonial, mandibular-plane need a radiograph). Soft-tissue ≠ bone, and soft-tissue thickness itself varies by ancestry.

Sources: [Legan-Burstone norms PMC3606791](https://pmc.ncbi.nlm.nih.gov/articles/PMC3606791/),
[E-line by ancestry (IIUM)](https://journals.iium.edu.my/ktn/index.php/ijohs/article/view/164),
[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2).

## 7. Longitudinal rigor *(in progress)*

- **Report change as MDC95**, not false precision: `MDC95 = SEM · 1.96 · √2` from a test-retest; a change is "real" only if it beats the band.
- **Evidence-graded intervention → metric map**: tretinoin **A** (texture/wrinkles), sunscreen **A** (photoaging prevention), weight **A** (jaw/adiposity; 2.38 kg/m² BMI just-noticeable threshold), vitamin C **B**, sleep (**perceived A / measured B–C**, high acute variance), facial exercise **C**, **mewing D — no measurable effect in adults; we say so** (AAO), fillers/tox **A** (clinical).
- **Local-first**: SQLite + EXIF/GPS stripped on ingest; zero network egress asserted in CI.

Sources: [Hughes 2013 sunscreen RCT](https://www.acpjournals.org/doi/10.7326/0003-4819-158-11-201306040-00002),
[tretinoin meta-analysis 2024](https://dpcj.org/index.php/dpc/article/view/5172),
[Re & Rule BMI JND](https://rule.psych.utoronto.ca/pubs/2016/Re&Rule(2016_SPPS).pdf),
[Holding 2019 sleep RCT](https://onlinelibrary.wiley.com/doi/abs/10.1111/jsr.12860).

## 8. Why it's built to be noticed

The wedge nobody occupies: **honest + ancestry/age/sex-aware + fully local + open-source + evidence-graded.**
Positioned explicitly against looksmaxxing apps (documented teen-dysmorphia harm) and the Qoves cloud funnel.
Distribution lever = a one-click GitHub Pages WASM demo with a verifiable "nothing uploaded" claim.

Sources: [looksmaxxing harm (Yahoo/psychologists)](https://finance.yahoo.com/news/looksmaxxing-apps-rate-teen-boys-163942148.html),
[Qoves](https://en.wikipedia.org/wiki/Qoves),
[OpenRAIL licensing](https://huggingface.co/blog/open_rail),
[BIPA](https://en.wikipedia.org/wiki/Biometric_Information_Privacy_Act).

---

## License / dataset manifest

| Component | License | Shippable? |
|---|---|---|
| MediaPipe FaceLandmarker (model + code) | Apache-2.0 | ✅ default engine |
| 3DDFA_V2 (code) | MIT (weights trained on research-only 300W-LP) | ⚠️ opt-in profile download w/ notice |
| SMIRK + FLAME 2023 **Open** | MIT + CC-BY-4.0 | ✅ v2 opt-in (only commercially-clean FLAME) |
| DECA / EMOCA / MICA / Pixel3DMM | non-commercial | ❌ reference only |
| Basel Face Model (BFM) | paid commercial | ❌ |
| Fitzpatrick17k / PAD-UFES-20 | CC-BY / CC-BY-4.0 | ✅ redistributable |
| SCIN | custom open | ⚠️ non-commercial calibration |
| HAM10000 / DDI | CC-BY-NC / non-commercial | ❌ local validation only |
| SCUT-FBP5500 / MEBeauty | non-commercial research | ❌ never bundle (and biased) |

**Rule:** "MIT code" ≠ usable weights. Always check the model/dataset license separately.
