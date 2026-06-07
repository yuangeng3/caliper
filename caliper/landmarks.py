"""MediaPipe FaceLandmarker wrapper (478 points + iris + pose matrix).

The only module that touches MediaPipe/OpenCV — imported lazily so the rest of the
package (and the test suite) loads without them. The model is downloaded once to
``models/`` on first run (Apache-2.0 weights) and then everything is on-device.
"""
from __future__ import annotations

import shutil
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


@dataclass
class LandmarkResult:
    points_px: np.ndarray              # (478, 2) pixel coordinates
    points_norm: np.ndarray            # (478, 3) normalized coordinates
    transform_matrix: np.ndarray | None  # (4, 4) facial transformation, or None
    blendshapes: dict[str, float]
    image_bgr: object                  # the loaded BGR image (for quality + annotate)
    width: int
    height: int


def _ssl_context() -> ssl.SSLContext:
    # macOS Python builds often lack a usable CA store; prefer certifi's bundle
    # (pulled in by mediapipe) so the one-time model download just works.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "caliper"})
        tmp = MODEL_PATH.with_suffix(".part")
        with urllib.request.urlopen(req, context=_ssl_context()) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.replace(MODEL_PATH)  # atomic: never leave a half-downloaded model
    return MODEL_PATH


def detect(image_path: str) -> LandmarkResult:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise ValueError(f"could not read image: {image_path}")
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    opts = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_model())),
        num_faces=1,
        output_facial_transformation_matrixes=True,
        output_face_blendshapes=True,
    )
    with vision.FaceLandmarker.create_from_options(opts) as lm:
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not res.face_landmarks:
        raise RuntimeError("no face detected — try a clearer, front-facing photo")
    lms = res.face_landmarks[0]
    if len(lms) < 478:
        raise RuntimeError(f"expected 478 landmarks, got {len(lms)} (iris refinement missing)")

    norm = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
    px = norm[:, :2] * np.array([w, h], dtype=float)

    tm = None
    if getattr(res, "facial_transformation_matrixes", None):
        tm = np.array(res.facial_transformation_matrixes[0], dtype=float).reshape(4, 4)
    bs: dict[str, float] = {}
    if getattr(res, "face_blendshapes", None):
        bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}

    return LandmarkResult(px, norm, tm, bs, bgr, w, h)


def annotate(result: LandmarkResult, out_path: str) -> None:
    """Write a labeled overlay so you can verify each key landmark lands correctly."""
    import cv2

    from . import constants as C
    img = result.image_bgr.copy()
    for (x, y) in result.points_px.astype(int):
        cv2.circle(img, (int(x), int(y)), 1, (160, 160, 160), -1)
    for name, i in C.IDX.items():
        x, y = result.points_px[i].astype(int)
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.putText(img, str(i), (int(x) + 4, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)
