"""Capture-quality gating. A measurement is only as trustworthy as the photo.

Head pose (yaw/pitch/roll), focus, and exposure are checked before we report any
number. Pitch matters most — it foreshortens vertical distances and is the main
reason uncalibrated selfies drift session to session. Pose math is pure NumPy;
OpenCV is imported lazily only for the image-sharpness checks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import constants as C

POSE_TOL_DEG = 8.0     # reject if the face is turned/tilted past this off square-to-camera
BLUR_MIN = 10.0        # variance-of-Laplacian floor; phone/Photo Booth selfies are soft
                       # by this measure, so only flag genuinely blurry shots
CLIP_MAX_PCT = 5.0     # max fraction of blown-out / crushed pixels before we warn


@dataclass
class Quality:
    yaw: float | None
    pitch: float | None
    roll: float
    blur_var: float | None
    exposure_clip_pct: float | None
    passed: bool
    issues: list[str] = field(default_factory=list)


def decompose_pose(matrix4x4) -> tuple[float, float, float]:
    """Yaw, pitch, roll (degrees) from a 4x4 facial-transformation matrix."""
    R = np.asarray(matrix4x4, dtype=float)[:3, :3]
    scale = np.linalg.norm(R, axis=0)
    scale[scale == 0] = 1.0
    Rn = R / scale
    sy = math.hypot(Rn[0, 0], Rn[1, 0])
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(Rn[2, 1], Rn[2, 2]))
        yaw = math.degrees(math.atan2(-Rn[2, 0], sy))
        roll = math.degrees(math.atan2(Rn[1, 0], Rn[0, 0]))
    else:  # gimbal lock
        pitch = math.degrees(math.atan2(-Rn[1, 2], Rn[1, 1]))
        yaw = math.degrees(math.atan2(-Rn[2, 0], sy))
        roll = 0.0
    return yaw, pitch, roll


def roll_from_iris(P: np.ndarray) -> float:
    r, lf = P[C.RIGHT_IRIS_CENTER], P[C.LEFT_IRIS_CENTER]
    return math.degrees(math.atan2(lf[1] - r[1], lf[0] - r[0]))


def assess(P: np.ndarray, transform_matrix=None, image_bgr=None) -> Quality:
    issues: list[str] = []
    if transform_matrix is not None:
        yaw, pitch, roll = decompose_pose(transform_matrix)
    else:
        yaw = pitch = None
        roll = roll_from_iris(P)

    for name, val in (("yaw", yaw), ("pitch", pitch), ("roll", roll)):
        if val is not None and abs(val) > POSE_TOL_DEG:
            issues.append(f"{name} {val:+.0f}deg exceeds +/-{POSE_TOL_DEG:.0f}deg "
                          "(face not square to camera)")

    blur_var = clip = None
    if image_bgr is not None:
        blur_var, clip = _image_quality(image_bgr)
        if blur_var < BLUR_MIN:
            issues.append(f"image is soft / out of focus (sharpness {blur_var:.0f})")
        if clip > CLIP_MAX_PCT:
            issues.append(f"{clip:.0f}% of pixels blown out or crushed (uneven lighting)")

    return Quality(yaw, pitch, roll, blur_var, clip, passed=not issues, issues=issues)


def _image_quality(bgr) -> tuple[float, float]:
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    clip = float(((gray <= 2) | (gray >= 253)).mean() * 100.0)
    return blur, clip
