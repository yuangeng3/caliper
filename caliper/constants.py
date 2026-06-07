"""Canonical constants for Caliper.

Landmark indices follow the MediaPipe Face Mesh / FaceLandmarker 478-point
topology (468 surface points + 10 iris points). Verify any index on your own
photo with ``caliper analyze <img> --annotate out.png`` and tune it here if one
lands in the wrong place — the nose and jaw points especially are worth a look.
"""
from __future__ import annotations

# --- physical constants ---------------------------------------------------
IRIS_DIAMETER_MM = 11.7           # Horizontal visible iris diameter, near-constant across
                                  # the population (11.7 +/- 0.5 mm). Source: MediaPipe Iris,
                                  # Google Research 2020. This is the ruler for px->mm scaling.
IRIS_CALIBRATION_ERROR_PCT = 4.3  # Mean relative error of iris-scaled measurement. Carried
                                  # through as a confidence band — we never hide it.
PLAUSIBLE_IPD_MM = (54.0, 72.0)   # Adult interpupillary-distance sanity range.

# --- iris landmarks (center + 4 ring points per eye) ----------------------
LEFT_IRIS_CENTER = 468
LEFT_IRIS_RING = (469, 470, 471, 472)
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_RING = (474, 475, 476, 477)

# --- key facial landmarks --------------------------------------------------
# "right" = the subject's right side = the left side of the image.
IDX = {
    "menton": 152,         # lowest point of the chin (gnathion)
    "forehead_top": 10,    # superior forehead — note: upper-third bound is APPROXIMATE
    "glabella": 9,         # between the brows
    "nasion": 168,         # nasal root / sellion
    "subnasale": 2,        # base of the nose, top of the philtrum
    "nose_tip": 1,         # pronasale
    "r_canthus_out": 33,   # right eye, outer corner (ex)
    "r_canthus_in": 133,   # right eye, inner corner (en)
    "l_canthus_in": 362,   # left eye, inner corner (en)
    "l_canthus_out": 263,  # left eye, outer corner (ex)
    "alare_r": 48,         # right nostril wing (al) -- verify with --annotate
    "alare_l": 278,        # left nostril wing (al)
    "zygo_r": 234,         # right face-width point (bizygomatic stand-in)
    "zygo_l": 454,         # left face-width point
    "cheilion_r": 61,      # right mouth corner (ch)
    "cheilion_l": 291,     # left mouth corner (ch)
    "brow_r": 105,         # right brow top (FWHR vertical bound)
    "brow_l": 334,         # left brow top
    "lip_top": 0,          # upper lip, outer top center (labrale superius)
}

# Self-reported ancestry options. Descriptive only, never ranked. Each maps in
# norms.json to the nearest population for which we have a cited reference table.
ANCESTRY_CHOICES = (
    "european",
    "african",            # nearest reference: Kenyan
    "african_american",
    "east_asian",         # nearest reference: Hong Kong / Han Chinese
    "south_asian",
    "middle_eastern",
)
SEX_CHOICES = ("female", "male")

# Skin-sampling regions: a landmark anchor per region; we sample a patch around
# each (median, to reject specular highlights and pores). Tune with --annotate.
SKIN_REGIONS = {
    "forehead": 10,
    "glabella": 9,
    "right_cheek": 50,
    "left_cheek": 280,
    "nose": 195,
    "chin": 200,
}
