<h1 align="center">Caliper</h1>

<p align="center">
  <b>Honest, ancestry-aware, fully-local facial geometry &amp; skin analysis.</b><br>
  No universal "ideal" face. No cloud. No single attractiveness score. Your face never leaves your device.
</p>

<p align="center">
  <code>Apache-2.0</code> · runs on-device (Python CLI + in-browser WASM) · <i>Qoves, but honest, private, and open.</i>
</p>

---

## Why this exists

Face-analysis tools fail in one of three ways:

- **Looksmaxxing apps** (Umax, LooksMax AI) sell teenagers a black-box "you're a 4/10" against a single Eurocentric ideal — a pattern now flagged by a 2025 *Lancet Child & Adolescent Health* review as a driver of body dysmorphia.
- **Cosmetic-surgery funnels** (Qoves) are more rigorous but cloud-based, opaque, and oriented toward "fixing" deviations from a norm. (Even Qoves now disavows the golden ratio as "a stylised Eurocentric aesthetic.")
- **Beauty-brand skin scanners** (Haut.AI, Perfect Corp) are dermatologically real but closed, cloud-bound, and exist to sell you product.

**None is simultaneously honest, ancestry/age/sex-aware, fully local, open-source, and evidence-graded.** That four-way intersection is Caliper.

The foundational fact the whole project is built on: **the neoclassical "ideal" facial canons fail in 60–100% of every population studied — including the Caucasians they were derived from** (the facial-thirds canon holds in 0% of Southern Chinese and only ~37% of young Caucasians; Farkas et al.). So there is no single ideal. Caliper measures you against *your own* (ancestry, sex, age) reference distribution — like a lab reference range — and tells you how confident it is.

## What it does

| | |
|---|---|
| **Geometry** | 478-point landmarks (MediaPipe), calibrated to real millimetres using the iris as an 11.7 mm ruler. Canthal tilt, facial thirds/fifths, intercanthal & nasal width, indices — each as a **percentile within your cohort**, never vs. an ideal. |
| **Ancestry/age/sex norms** | Reference tables with a **cited source per cell**. Where data is thin, it says *"no reference for your cohort"* instead of fabricating a number. |
| **Skin** *(in progress)* | ITA° tone (not the unreliable Fitzpatrick), Monk 10-tone scale, melanin/haemoglobin separation so pigment (PIH) and redness (erythema) are disentangled — with the priority concern **conditioned on your skin tone**. |
| **Longitudinal** *(in progress)* | Track change over interventions with **MDC95 gating** — a change only counts as real if it beats measurement noise. Evidence-graded intervention map (tretinoin **A**, sunscreen **A**, mewing **D — no effect, we say so**). |
| **No score** | There is no overall number, by design. ~half of attractiveness judgment is private taste (Honekopp 2006); a single score would be statistically dishonest. |

## Privacy is the architecture, not a setting

- Everything runs **on-device** (Python via MediaPipe pip; web via MediaPipe WASM — nothing is uploaded, verifiable in your browser's Network tab).
- EXIF/GPS is stripped on ingest; the local store is the only copy.
- The repo's `.gitignore` makes it impossible to accidentally commit a photo or your profile data.

## Quickstart

> Requires **Python 3.10+** (verified through 3.13 with MediaPipe 0.10.35). Apple Silicon & Intel macOS, Linux, Windows.

```bash
git clone https://github.com/<you>/caliper && cd caliper
python -m venv .venv && source .venv/bin/activate
pip install -e .

# analyze a frontal, neutral-expression, evenly-lit photo
caliper analyze selfie.jpg --ancestry east_asian --sex female --age 34

# first time: verify the landmarks land correctly on your face
caliper analyze selfie.jpg --ancestry european --sex male --annotate check.png
```

`--ancestry` is one of: `european · african · african_american · east_asian · south_asian · middle_eastern`
(self-reported, descriptive, and only used to pick the reference cohort).

For the most accurate millimetres, measure your interpupillary distance once with a ruler and pass `--ipd-mm 63`.

```bash
# track a metric over time (only counts a change once it beats the MDC95 noise band)
caliper analyze selfie.jpg --ancestry east_asian --sex female --age 40 --save --profile me
caliper trend --profile me --metric skin_evenness_sd

# the evidence-graded intervention -> metric map (tretinoin A ... mewing D)
caliper interventions
```

### Try it in your browser (zero install)

```bash
cd web && python -m http.server 8000   # then open http://localhost:8000
```

Drop a selfie — landmarks, calibration, cohort percentiles, and skin tone compute entirely in your
browser via MediaPipe WASM. Open the **Network** tab to confirm nothing is uploaded. To publish the
public demo, point **GitHub Pages** at the `web/` directory.

## Why no beauty score?

Because it would be a lie dressed as a number. The best-known beauty dataset (SCUT-FBP5500) was rated by **60 Chinese university students aged 18–27**; a model trained on it predicts *their* taste, not yours, and a 2025 audit showed even "diverse" datasets satisfy distributional parity across ethnicities in under 10% of comparisons. Caliper instead surfaces the few things research finds are *cross-culturally* and *mechanistically* meaningful (e.g. skin-tone evenness — worth up to ~20 years of perceived age, and actually modifiable) and shows each with a confidence interval and its rater provenance. See [`docs/RESEARCH.md`](docs/RESEARCH.md).

## Status

v0.1 is built and tested (30 passing tests, ruff-clean):

- ✅ **Frontal geometry** — 16 metrics, iris→mm calibration, pose/quality gates, ancestry/age/sex norms
- ✅ **Skin** — ITA° tone, Monk scale, evenness, melanin-aware erythema, tone-conditioned concerns
- ✅ **Demystified components** — tone evenness, typicality, down-ranked symmetry; **no overall score**
- ✅ **Longitudinal** — local SQLite, EXIF-stripped image vault, MDC95 change-gating, intervention map
- ✅ **Profile** — soft-tissue E-line / nasolabial / convexity math + cited cohort norms (library; the
  profile-capable extraction backend, 3DDFA_V2, is the planned opt-in front-end)
- ✅ **Web** — fully in-browser WASM demo, nothing uploaded

Next: more norm cohorts (the tables are the moat — PRs with cited data welcome), aging-conditioned
norms, the colour-chart calibration path, and the 3DDFA_V2 profile backend.

## Responsible use

Read [`ETHICS.md`](ETHICS.md). Short version: this is self-analysis and education, **not** a medical, orthodontic, or diagnostic tool, **not** an attractiveness ranking, and **not** for scoring other people or minors.

## License

Code: **Apache-2.0**. Any future trained weights will ship under OpenRAIL-M (no surveillance, no biometric identification, no discriminatory ranking). MediaPipe models are Apache-2.0. See [`docs/RESEARCH.md`](docs/RESEARCH.md) for the full dependency/dataset license manifest.
