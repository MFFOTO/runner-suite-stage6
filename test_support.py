# -*- coding: utf-8 -*-
"""Shared helpers for the engine's self-tests (test_*.py in this folder).

The tests drive the engine's real per-photo logic with a fake detector, so they
need no YOLO weights, GPU or real photos. ultralytics and tqdm are only stubbed
when they are not installed (the engine refuses to import without them).
"""

from __future__ import annotations

import copy
import sys
import tempfile
import threading
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _name, _attr in (("ultralytics", "YOLO"), ("tqdm", "tqdm")):
    try:
        __import__(_name)
    except Exception:
        _m = types.ModuleType(_name)
        setattr(_m, _attr, object if _name == "ultralytics" else None)
        sys.modules[_name] = _m

import cv2                              # noqa: E402
import numpy as np                      # noqa: E402
import runner_suite_core as core        # noqa: E402

fails: list = []


def chk(label, ok, detail=None):
    print(("PASS  " if ok else "FAIL  ") + label + (f"   [{detail}]" if detail is not None and not ok else ""))
    if not ok:
        fails.append(label)


def finish():
    print("\nFAILED:" if fails else "\nALL PASS", fails or "")
    sys.exit(1 if fails else 0)


def make_engine(root: Path, overrides=()):
    """A HighResRunnerSuite without loading a model. overrides: (section, key, value)."""
    s = core.HighResRunnerSuite.__new__(core.HighResRunnerSuite)
    s.cfg = copy.deepcopy(core.DEFAULT_CONFIG)
    for section, key, value in overrides:
        s.cfg.setdefault(section, {})[key] = value
    s.cfg["image_quality"]["enhance_by_class"] = {"premium": "none", "good": "none", "review": "none"}
    s.cfg["debug"]["write_csv"] = False
    s.stats = {}
    s.stats_lock = threading.Lock()
    s.report_rows = []
    s.report_lock = threading.Lock()
    s.sr_lock = threading.Lock()
    s._face_restorer = None
    s.output_folder = root
    s.cfg["paths"]["output_folder"] = str(root)
    s.rejects_folder = root / "_rejects"
    return s


class _T:                               # stand-in for a torch tensor
    def __init__(self, a): self.a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self.a


class _Boxes:
    def __init__(self, boxes, confs): self.xyxy, self.conf, self._n = _T(boxes), _T(confs), len(boxes)
    def __len__(self): return self._n


def detection(boxes, confs, kps):
    """What YOLO-pose returns for one photo: boxes, confidences, 17 keypoints each."""
    return types.SimpleNamespace(boxes=_Boxes(boxes, confs), keypoints=types.SimpleNamespace(data=_T(kps)))


def crisp_image(w=2400, h=1600, seed=3):
    """A fine-grained photo: sharp at the small scale the subject-sharpness measure looks at."""
    return cv2.GaussianBlur((np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8), (0, 0), 0.8)


_LAYOUT = [(.5, .07), (.46, .06), (.54, .06), (.42, .07), (.58, .07), (.35, .20), (.65, .20), (.30, .35), (.70, .35),
           (.30, .48), (.70, .48), (.40, .50), (.60, .50), (.40, .72), (.60, .72), (.40, .95), (.60, .95)]


def keypoints(box, upper=True):
    """17 anatomical keypoints inside a box. upper=False hides head, shoulders, elbows and wrists."""
    x1, y1, x2, y2 = box
    return [[x1 + fx * (x2 - x1), y1 + fy * (y2 - y1), 0.95 if (upper or i > 10) else 0.05]
            for i, (fx, fy) in enumerate(_LAYOUT)]


def run_photo(engine, img, boxes, confs, kps):
    """Judge one photo with the engine's real loop; returns the crops it wrote (relative paths)."""
    src = engine.output_folder / "photo.jpg"
    cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    engine._process_detection_results(img, src, [detection(boxes, confs, kps)])
    return sorted(str(p.relative_to(engine.output_folder)) for p in engine.output_folder.rglob("*.jpg")
                  if p.name != "photo.jpg")


def fresh_root() -> Path:
    return Path(tempfile.mkdtemp())
