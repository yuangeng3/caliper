# Responsible use

Caliper measures faces. Tools that measure faces have hurt people — by selling a
single "ideal," by ranking strangers, by scoring children's insecurity. This
document is part of the software, not a disclaimer bolted on after.

## The principles the code enforces

1. **There is no ideal face.** Every metric is reported as a position within the
   user's *own* (ancestry, sex, age) distribution — a descriptive reference range,
   like a lab panel. The code has no universal template to deviate from, and where
   a cohort lacks data it says *"no reference for your cohort"* rather than borrow
   another group's. Scoring a face against a single standard is the specific harm
   we refuse.

2. **No overall score.** There is no attractiveness/"looks"/PSL number, by design.
   Roughly half of attractiveness judgment is private, idiosyncratic taste
   (Honekopp 2006: between-rater agreement r ≈ 0.3–0.5), so a single number would be
   statistically dishonest. We surface decomposed, mechanism-named components with
   confidence intervals instead.

3. **Honesty about confidence.** Iris-scaled measurement carries ~3% error; we
   report it as a band and treat sub-band change as noise. Skin-colour metrics
   without a physical colour reference are labelled *directional only*. We grade the
   evidence (A–D) for every metric and intervention and **down-weight confidence on
   darker skin tones**, where the field's training data is thin — an honest
   disclosure rather than a hidden failure.

4. **Privacy is architectural.** Nothing leaves your device. There is no telemetry,
   no upload, no account. EXIF/GPS is stripped on ingest. The web demo runs entirely
   in your browser and you can verify zero network egress in the Network tab.

## What Caliper is **not**

- **Not** a medical, dermatological, or orthodontic diagnosis. Photo soft-tissue
  metrics are not skeletal cephalometrics; no skeletal angle (SNA/SNB, true gonial)
  is ever computed from a photo, and no skin condition is diagnosed.
- **Not** an attractiveness ranking or a "glow-up" plan.
- **Not** for analyzing other people. Use it on your own face, with your own
  consent. Do not use it to score, rate, or surveil anyone else.
- **Not** for minors. The documented mental-health harms of appearance-scoring fall
  hardest on adolescents.

## If appearance is causing you distress

If using this — or any appearance tool — is fueling anxiety, compulsive checking, or
distress about how you look, that is worth taking seriously. Consider talking to a
clinician about body dysmorphic disorder. In the US: 988 Suicide & Crisis Lifeline
(call/text 988). Internationally: <https://findahelpline.com>.

## Legal note

Even fully local, facial-geometry extraction is "biometric" under laws like Illinois
BIPA and GDPR Article 9. Caliper's local-only, zero-retention design is the
mitigation; any hosted demo must gate on explicit consent ("I am analyzing my own
face; results stay on my device") and retain nothing.
