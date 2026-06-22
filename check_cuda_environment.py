# -*- coding: utf-8 -*-
import sys

print("=" * 60)
print("CUDA / PyTorch Check")
print("=" * 60)
try:
    import torch
    print("Python:", sys.version.replace("\n", " "))
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            print(f"VRAM frei: {free_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB")
        except Exception as exc:
            print("VRAM info nicht verfügbar:", exc)
    else:
        print("WARN: CUDA ist nicht aktiv. Die Suite läuft dann deutlich langsamer.")
except Exception as exc:
    print("FEHLER beim PyTorch/CUDA-Check:", exc)

print("=" * 60)
try:
    import cv2, numpy, ultralytics, piexif, tqdm
    print("OpenCV:", cv2.__version__)
    print("cv2.dnn_superres:", hasattr(cv2, "dnn_superres"))
    print("numpy:", numpy.__version__)
    print("ultralytics:", ultralytics.__version__)
    print("piexif: OK")
    print("tqdm: OK")
except Exception as exc:
    print("FEHLER beim Bibliothekscheck:", exc)
print("=" * 60)
