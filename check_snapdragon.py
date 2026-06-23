# -*- coding: utf-8 -*-
"""
Snapdragon / Windows-on-ARM backend probe.

Reports which inference backends are actually available on this device so we
can pick the right acceleration path for the runner suite:
  - PyTorch ARM64 (does it import AND compute?)
  - ONNX Runtime + which execution providers (QNN = Hexagon NPU,
    DirectML = Adreno GPU, CPU = fallback)
  - ultralytics / opencv / numpy / psutil presence

Run on the Snapdragon device inside the project venv:
    .venv\\Scripts\\python check_snapdragon.py
and paste the full output back.
"""
import platform
import sys


def section(title: str) -> None:
    print("\n=== " + title + " ===")


section("System")
print("python       :", sys.version.split()[0])
print("executable   :", sys.executable)
print("machine      :", platform.machine())
print("platform     :", platform.platform())
print("processor    :", platform.processor())

section("PyTorch")
try:
    import torch  # type: ignore
    print("torch        :", torch.__version__)
    print("cuda avail   :", torch.cuda.is_available())
    # ARM wheels sometimes import but fail on the first real op -- verify compute.
    try:
        x = torch.randn(64, 64)
        _ = (x @ x.t()).sum().item()
        print("cpu compute  : OK")
    except Exception as exc:
        print("cpu compute  : FAILED ->", exc)
except Exception as exc:
    print("torch        : NOT AVAILABLE ->", exc)

section("ONNX Runtime")
try:
    import onnxruntime as ort  # type: ignore
    print("onnxruntime  :", ort.__version__)
    providers = ort.get_available_providers()
    print("providers    :", providers)
    for want in ("QNNExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
        print(f"  {want:26s}:", "YES" if want in providers else "no")
except Exception as exc:
    print("onnxruntime  : NOT AVAILABLE ->", exc)

section("Other modules")
for mod in ("ultralytics", "cv2", "numpy", "psutil"):
    try:
        m = __import__(mod)
        print(f"{mod:12s}:", getattr(m, "__version__", "OK"))
    except Exception as exc:
        print(f"{mod:12s}: NOT AVAILABLE ->", exc)

section("CPU / RAM")
try:
    import psutil  # type: ignore
    print("logical cores :", psutil.cpu_count(logical=True))
    print("physical cores:", psutil.cpu_count(logical=False))
    print("RAM GB        :", round(psutil.virtual_memory().total / 1024 ** 3, 1))
except Exception as exc:
    print("psutil       : NOT AVAILABLE ->", exc)

print("\nDone. Paste this whole output back.")
