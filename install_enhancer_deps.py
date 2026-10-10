# -*- coding: utf-8 -*-
"""
Installs the optional AI-upscaling stack (basicsr, realesrgan, gfpgan, facexlib)
and keeps exactly one OpenCV build in the venv.

Why this exists:
- basicsr 1.4.2's setup.py reads its version via exec()/locals(), which no longer
  works on Python 3.13+ ("KeyError: '__version__'"), so `pip install basicsr`
  fails there. We download the official sdist from PyPI, verify its SHA-256 against
  PyPI's published digest, patch that one function and install from the patched copy.
- basicsr/realesrgan/gfpgan/facexlib all depend on `opencv-python`, which pip then
  installs next to `opencv-contrib-python`. Both write the same `cv2` package and
  clobber each other (version mix-ups, missing cv2.dnn_superres). After installing we
  remove the plain wheels and force-reinstall the contrib build the suite needs.

Run by setup_runner_suite.bat and, if the libraries are missing, by the suite itself:
    python install_enhancer_deps.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASICSR_VERSION = "1.4.2"
OPENCV_SPEC = "opencv-contrib-python>=4.10,<4.12"
# Runtime deps of basicsr/realesrgan/gfpgan/facexlib EXCEPT opencv-python. The four
# packages are installed with --no-deps so pip never pulls in a second OpenCV
# (which would also fail to replace cv2 while the suite has it loaded on Windows).
EXTRA_DEPS = ["numpy", "Pillow", "tqdm", "pyyaml", "requests", "addict", "future", "lmdb",
              "scipy", "scikit-image", "tb-nightly", "yapf", "filterpy", "numba"]
OLD_VERSION_FN = """    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']"""
NEW_VERSION_FN = """    ns = {}
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'), ns)
    return ns['__version__']"""


def pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])


def installed(name: str) -> bool:
    try:
        md.version(name)
        return True
    except md.PackageNotFoundError:
        return False


def alias_functional_tensor() -> None:
    """torchvision >= 0.17 removed transforms.functional_tensor, which basicsr imports."""
    import torchvision.transforms.functional as tvf  # type: ignore
    sys.modules.setdefault("torchvision.transforms.functional_tensor", tvf)


def stack_importable() -> bool:
    try:
        alias_functional_tensor()
        from basicsr.archs.rrdbnet_arch import RRDBNet  # noqa: F401
        from realesrgan import RealESRGANer  # noqa: F401
        from gfpgan import GFPGANer  # noqa: F401
        return True
    except Exception:
        return False


def install_patched_basicsr() -> None:
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/basicsr/{BASICSR_VERSION}/json", timeout=60))
    sdist = next(u for u in meta["urls"] if u["packagetype"] == "sdist")
    with tempfile.TemporaryDirectory(prefix="basicsr_build_") as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "basicsr.tar.gz"
        urllib.request.urlretrieve(sdist["url"], archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != sdist["digests"]["sha256"]:
            raise RuntimeError(f"basicsr sdist SHA-256 mismatch (got {digest}) -- refusing to install")
        with tarfile.open(archive) as tar:
            tar.extractall(tmp_path, filter="data") if sys.version_info >= (3, 12) else tar.extractall(tmp_path)
        src = tmp_path / f"basicsr-{BASICSR_VERSION}"
        setup_py = src / "setup.py"
        text = setup_py.read_text(encoding="utf-8")
        if OLD_VERSION_FN in text:
            setup_py.write_text(text.replace(OLD_VERSION_FN, NEW_VERSION_FN), encoding="utf-8")
        pip("install", "--no-deps", str(src))


def ensure_single_opencv() -> None:
    plain = [p for p in ("opencv-python", "opencv-python-headless") if installed(p)]
    if plain:
        pip("uninstall", "-y", *plain)
    # (Re)install contrib even if only the plain wheel was present, and after an
    # uninstall: the plain wheels shared cv2/'s files with it.
    if plain or not installed("opencv-contrib-python"):
        pip("install", "--force-reinstall", "--no-deps", OPENCV_SPEC)


def main() -> int:
    if not stack_importable():
        print("[ENHANCER DEPS] Installing basicsr (patched), realesrgan, gfpgan, facexlib ...")
        if not installed("basicsr"):
            install_patched_basicsr()
        pip("install", *EXTRA_DEPS)
        pip("install", "--upgrade", "--no-deps", "realesrgan", "gfpgan", "facexlib")
    else:
        print("[ENHANCER DEPS] basicsr / realesrgan / gfpgan already importable.")
    try:
        ensure_single_opencv()
    except Exception as exc:
        print(f"[ENHANCER DEPS] Could not clean up OpenCV ({exc}). Close the suite and run "
              "setup_runner_suite.bat again.")
    ok = stack_importable()
    try:
        import cv2  # type: ignore
        print(f"[ENHANCER DEPS] OpenCV {cv2.__version__}, dnn_superres: {hasattr(cv2, 'dnn_superres')}")
    except Exception as exc:
        print(f"[ENHANCER DEPS] OpenCV import failed: {exc}")
        ok = False
    print("[ENHANCER DEPS] " + ("OK" if ok else "FAILED -- see messages above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
