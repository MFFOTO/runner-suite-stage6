# -*- coding: utf-8 -*-
"""
HighRes Runner Suite - Stage 6 HighRes Plus

Goals of this version:
- Fast and productive again, like the simple HighRes suite
- Only a few hard rejects
- 4000px output
- A_Premium / B_Good / C_Review sorting
- Full-frame runner mode for runners who fill nearly the entire frame
- Fast fence/grid check
- Optional CSV reporting

Deliberately NOT active by default:
- Tiling
- ROI second pass
- FSRCNN upscaling

Required files in the same folder:
- runner_suite_core.py
- settings.json
- yolov8m-pose.pt
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import queue
import subprocess
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_MISSING_MODULES: List[str] = []

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore
    _MISSING_MODULES.append("opencv-contrib-python")

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore
    _MISSING_MODULES.append("numpy")

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore
    _MISSING_MODULES.append("ultralytics")

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore
    _MISSING_MODULES.append("tqdm")

try:
    import piexif  # type: ignore
except Exception:
    piexif = None  # type: ignore
    _MISSING_MODULES.append("piexif")

# Optional: used by the hardware auto-tuner to read RAM and physical-core
# counts. If absent, auto-tuning falls back to os.cpu_count() and skips the
# RAM-based caps -- it is not required to run the suite.
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore

# Optional: face-restoration model for the C_Review AI-enhancement feature.
# Not required to run the suite -- if missing, enhancement falls back to a
# simpler upscale/contrast/sharpen pipeline using only OpenCV.
try:
    from gfpgan import GFPGANer  # type: ignore
except Exception:
    GFPGANer = None  # type: ignore

if _MISSING_MODULES:
    print("\n[ERROR] Missing Python libraries:")
    for module_name in sorted(set(_MISSING_MODULES)):
        print(f"  - {module_name}")
    print("\nPlease run setup_runner_suite.bat first.")
    sys.exit(1)

Box = Tuple[float, float, float, float]

DEFAULT_CONFIG: Dict[str, Any] = {
    # When true, any config value left as the string "auto" is filled in at
    # startup from the detected hardware (GPU/VRAM, CPU arch/cores, RAM).
    # Explicit values always win. Set false to use the built-in defaults below.
    "auto_hardware": True,
    "paths": {
        "input_folder": "D:/20401/20401_originals/CCJU2LS1",
        "output_folder": "D:/RUN_OUT_PREMIUM",
        "recursive": False,
    },
    "hardware": {
        "model_path": "yolov8m-pose.pt",
        "prefer_gpu": True,
        "use_openvino_cpu": False,
        # Run inference on a DirectX-12 GPU via onnxruntime-directml when an
        # exported .onnx is present. "auto" = on for ARM (Snapdragon/Adreno);
        # set true to also use it on x86 (e.g. an Intel Arc iGPU).
        "prefer_directml": "auto",
        "model_thread_lock": False,
        "show_cuda_check": True,
    },
    "detector": {
        "imgsz": 1024,
        "iou": 0.55,
        "max_det": 120,
    },
    "selection_filters": {
        "enable_safe_zone": True,
        "safe_zone_percent": 12,
        "min_height_ratio": 0.15,
        "min_box_height_px": 220,
        "conf_threshold": 0.33,
        "keypoint_conf": 0.30,
        "min_keypoints": 4,
        "require_frontal_face": False,
        "min_sharpness_threshold": 80,
        "hard_reject_fence": True,
        "hard_reject_extreme_blur": True,
    },
    "full_frame_runner": {
        "enabled": True,
        "trigger_box_height_ratio": 0.70,
        "trigger_box_area_ratio": 0.34,
        "bypass_safe_zone": True,
        "bypass_min_keypoints": True,
        "crop_to_aspect_ratio": True,
    },
    "crop": {
        "aspect_ratio": 0.666,
        "pad_x": 0.38,
        "pad_top": 0.18,
        "pad_bottom": 0.28,
        "small_person_threshold_px": 700,
        "small_person_extra_pad_multiplier": 1.15,
        # When resize_final has to trim height to hit the target aspect ratio,
        # this is the fraction of the excess removed from the TOP (the rest from
        # the bottom). 0.25 keeps the head in the upper third instead of clipping
        # it; 0.5 would trim symmetrically (the old behaviour).
        "vertical_trim_top_fraction": 0.25,
        # Framing method:
        #   "pad"     -> pad the detection box by fixed fractions, then fill the
        #                aspect ratio (legacy; can push the runner up + add floor).
        #   "anatomy" -> lock the vertical extent to head-top .. foot-bottom with
        #                fixed margins and DERIVE the width from the aspect ratio.
        #                Frames the runner head-to-toe at any vantage (low/center/
        #                high) with no arbitrary floor. The pad_* values are unused
        #                in this mode.
        "mode": "pad",
        "anatomy_headroom_ratio": 0.08,     # sky kept above the crown (x person height)
        "anatomy_footroom_ratio": 0.06,     # ground kept below the feet
        "anatomy_crown_allowance": 0.06,    # crown height above the face keypoints
        "anatomy_side_margin": 0.12,        # extra width beyond the runner when width-limited
    },
    "completeness_guard": {
        # Keeps partial / cut-off athletes out of A_Premium / B_Good: a crop
        # missing the head or whole upper body, or truncated by the frame top,
        # is SOFT-routed to C_Review by default (nothing is discarded). Aimed at
        # dense / occluded events (MTB start corrals, tight courses) where the
        # pose model returns a partial box (legs-only, or head clipped).
        "enabled": True,
        "require_upper_body": True,     # need a head OR both shoulders for A/B
        "truncation_guard": True,       # detect a head cut off by the frame top
        "edge_margin_px": 6,            # px from a border that counts as "touching"
        "review_on_partial": True,      # True = demote to Review; False = reject
        "head_extend": True,            # grow the crop up to include an occluded head
        "head_extend_ratio": 0.7,       # head room above shoulders, x torso length
        "min_headroom_ratio": 0.12,     # margin kept above the crown when the head IS visible
        "head_estimate_ratio": 0.22,    # min head allowance (x person height) when the head is weak
    },
    "fence_detection": {
        "enabled": True,
        "bright_threshold": 210,
        "vertical_kernel_height_ratio": 0.11,
        "min_component_height_ratio": 0.32,
        "max_component_width_ratio": 0.11,
        "min_component_aspect_ratio": 3.0,
        "central_band_width_ratio": 0.58,
        "reject_vertical_line_count": 5,
        "reject_person_occlusion_coverage": 0.13,
        "reject_central_occlusion_score": 0.22,
        "reject_vertical_line_coverage": 0.15,
    },
    "whole_image_fallback": {
        "enabled": False,
        "min_original_sharpness": 95,
        "reject_if_fence": True,
        "save_class": "review",
    },
    "review_enhancement": {
        # AI enhancement pass for the C_Review folder, offered interactively
        # once ALL input images have been cropped/sorted. Uses GFPGAN
        # (face/body restoration); the library + model weights are installed
        # automatically on first use if missing (requires internet access).
        "enabled": False,
        "prompt_after_run": True,
        "auto_install": True,
        "gfpgan_model_path": "GFPGANv1.4.pth",
        "gfpgan_model_url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        # SHA-256 of GFPGANv1.4.pth -- an auto-downloaded file that does not
        # match is discarded rather than handed to GFPGAN. Set to "" to skip
        # the hash check (a size + HTML-page sanity check still runs).
        "gfpgan_model_sha256": "e2cd4703ab14f4d01fd1383a8a8b266f9a5833dacee8e6a79d3bf21a1b6be5ad",
        "upscale": 1,
        "output_subfolder": "_enhanced",
        # Marks review-class crops that are likely to benefit from AI
        # restoration (moderately soft / under- or over-exposed, but not bad
        # enough to be discarded) by routing them into a clearly named
        # subfolder inside C_Review for quick visual triage.
        "flag_candidates": True,
        "candidates_subfolder": "_promising_for_enhancement",
        "candidate_max_sharpness": 160,
        "candidate_min_brightness": 60,
        "candidate_max_brightness": 195,
    },
    "image_quality": {
        "target_height": 4000,
        "enable_fsrcnn": False,
        "ai_upscaling_limit": 1200,
        "fsrcnn_model_path": "FSRCNN_x3.pb",
        # Auto-downloaded on first use if the .pb is missing (mirrors the GFPGAN
        # auto-install). Set auto_download_fsrcnn=False to require a manual file.
        "auto_download_fsrcnn": True,
        "fsrcnn_model_url": "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x3.pb",
        # Expected SHA-256 of FSRCNN_x3.pb -- an auto-downloaded file that does
        # not match (partial download, 404 HTML page, tampered mirror) is
        # discarded instead of being saved. Set to "" to skip the hash check
        # (a structural sanity check still runs).
        "fsrcnn_model_sha256": "efd38655a815908c6c8954db6052f128e76a735f1de657894c477d0dc0b64481",
        # Upscaler for small crops: "fsrcnn" (fast, CPU) or "realesrgan"
        # (much higher quality, GPU-only -- used by the ENHANCER preset). If
        # realesrgan is requested but no CUDA GPU / library / model is
        # available, it falls back to FSRCNN, then plain Lanczos.
        "upscaler": "fsrcnn",
        "realesrgan_model_path": "RealESRGAN_x4plus.pth",
        "realesrgan_model_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "realesrgan_model_sha256": "",
        "realesrgan_tile": 256,
        "realesrgan_auto_install": True,
        # Per-quality-class enhancement. Modes: "none" (Lanczos only),
        # "upscale" (Real-ESRGAN/FSRCNN), "upscale+face" (Real-ESRGAN + GFPGAN
        # face restoration). Default: same "upscale" for every class (the fast
        # suite's behaviour); the ENHANCER preset overrides per class.
        "enhance_by_class": {"premium": "upscale", "good": "upscale", "review": "upscale"},
        "sharpening": "strong",
        "denoise": False,
        # Premium crops are selected for already being sharp, so the global
        # denoise + sharpen passes are skipped for them by default (avoids
        # over-sharpening your best frames). Set True to post-process Premium
        # like Good/Review.
        "post_process_premium": False,
        "jpeg_quality": 95,
        "preserve_exif": True,
    },
    "quality_scoring": {
        "save_quality_subfolders": True,
        "save_review": False,
        "hard_reject_below_score": 62,
        "classes": {
            "premium": 73,
            "good": 65,
            "review": 45,
        },
        "excellent_sharpness": 300,
        "ideal_runner_height_px": 700,
        "ideal_confidence": 0.65,
        "weights": {
            "sharpness": 0.30,
            "center": 0.12,
            "size": 0.16,
            "keypoints": 0.12,
            "isolation": 0.08,
            "confidence": 0.10,
            "fence_free": 0.12,
        },
    },
    "performance": {
        "workers": 20,
        "batch_size": 1,
        "prefetch_maxsize": 4,
    },
    "debug": {
        "write_csv": True,
        "save_rejects": False,
        "rejects_folder": "_rejects_stage6",
    },
    "pre_check": {
        "enabled": True,
        "min_image_sharpness": 50.0,
        "min_image_brightness": 30.0,
        "max_image_brightness": 230.0,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class HighResRunnerSuite:
    def __init__(self, config_path: str = "settings.json"):
        self.base_dir = Path(__file__).resolve().parent
        self._silence_ultralytics_deprecation()
        self.config_path = self._resolve_path(config_path)
        self.cfg = self._load_config(self.config_path)
        self._auto_tune()
        self.output_folder = self._resolve_path(str(self.cfg["paths"]["output_folder"]))
        self.rejects_folder = self.output_folder / str(self.cfg["debug"].get("rejects_folder", "_rejects_stage6"))

        self.stats: Dict[str, int] = {
            "total_images": 0,
            "pre_check_rejected": 0,
            "images_with_person": 0,
            "processed_crops": 0,
            "quality_premium": 0,
            "quality_good": 0,
            "quality_review": 0,
            "no_person": 0,
            "too_small": 0,
            "unsafe_edge": 0,
            "bad_pose": 0,
            "blurred": 0,
            "fence_rejected": 0,
            "low_quality_score": 0,
            "crop_failed": 0,
            "partial_body": 0,
            "truncated_frame": 0,
            "demoted_partial": 0,
            "fallback_saved": 0,
            "write_failed": 0,
            "errors": 0,
        }
        self.stats_lock = threading.Lock()
        self.report_rows: List[Dict[str, Any]] = []
        self.report_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.sr_lock = threading.Lock()  # guards the shared GFPGAN restorer only

        # FSRCNN upscalers are created per-thread (cv2's DnnSuperResImpl is not
        # safe to share), so super-resolution runs in parallel across CPU cores
        # instead of being serialized behind one lock.
        self._sr_tls = threading.local()
        self._sr_ready = False
        self._sr_model_path: Optional[str] = None
        self._sr_scale = 3

        # Real-ESRGAN upscaler (ENHANCER preset). Shared GPU model, guarded by a
        # lock; None unless successfully loaded on a CUDA machine.
        self._realesrgan = None
        self._realesrgan_lock = threading.Lock()
        self._realesrgan_scale = 4

        # GFPGAN face restorer for the "upscale+face" enhance mode (shares the
        # Real-ESRGAN lock so GPU work stays serialized). None unless loaded.
        self._face_restorer = None

        self.device_mode = "cpu"
        self.backend = "pytorch"
        self.model = self._setup_hardware_and_yolo()
        self._setup_upscaler()
        self._setup_realesrgan()
        self._setup_pipeline_face_restorer()
        self._warmup_model()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _resolve_path(self, path_value: str) -> Path:
        p = Path(path_value)
        return p if p.is_absolute() else self.base_dir / p

    def _load_config(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            print(f"[WARN] settings.json not found: {path}")
            print("       Using default configuration.")
            return DEFAULT_CONFIG
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        return deep_merge(DEFAULT_CONFIG, user_cfg)

    @staticmethod
    def _silence_ultralytics_deprecation() -> None:
        """Keep FP16 (half=True) for speed without the per-predict console spam:
        newer ultralytics logs "'half' is deprecated" on every call. Drop just
        those messages from its logger (and the warnings channel)."""
        try:
            import logging

            class _DropDeprecated(logging.Filter):
                def filter(self, record: logging.LogRecord) -> bool:
                    try:
                        return "deprecated" not in record.getMessage().lower()
                    except Exception:
                        return True

            logging.getLogger("ultralytics").addFilter(_DropDeprecated())
        except Exception:
            pass
        try:
            import warnings
            warnings.filterwarnings("ignore", message=r".*deprecated.*", module=r".*ultralytics.*")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Hardware auto-tuning
    # ------------------------------------------------------------------
    def _detect_hardware(self) -> Dict[str, Any]:
        """Best-effort hardware probe. Every lookup is guarded so a missing
        dependency (psutil) or a failed GPU query never aborts startup."""
        info: Dict[str, Any] = {
            "has_cuda": False,
            "gpu_name": "",
            "vram_gb": 0.0,
            "arch": platform.machine().lower(),
            "logical_cores": os.cpu_count() or 4,
            "physical_cores": 0,
            "ram_gb": 0.0,
            "has_openvino": False,
        }
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info["has_cuda"] = True
                info["gpu_name"] = props.name
                info["vram_gb"] = props.total_memory / (1024 ** 3)
        except Exception:
            pass
        try:
            import importlib.util
            info["has_openvino"] = importlib.util.find_spec("openvino") is not None
        except Exception:
            pass
        info["onnx_providers"] = []
        try:
            import onnxruntime as ort  # type: ignore
            info["onnx_providers"] = list(ort.get_available_providers())
        except Exception:
            pass
        if psutil is not None:
            try:
                info["physical_cores"] = psutil.cpu_count(logical=False) or 0
                info["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
            except Exception:
                pass
        if not info["physical_cores"]:
            info["physical_cores"] = max(1, int(info["logical_cores"]) // 2)
        info["gpu_names"] = self._detect_gpu_names()
        return info

    @staticmethod
    def _detect_gpu_names() -> List[str]:
        """Names of the display adapters (Windows), so the tuner can recognise a
        capable non-CUDA GPU (Intel Arc / AMD discrete) and enable DirectML on
        its own. Best-effort; returns [] on failure or non-Windows."""
        if platform.system() != "Windows":
            return []
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name"],
                stderr=subprocess.DEVNULL, timeout=15,
            )
            return [ln.strip() for ln in out.decode(errors="ignore").splitlines() if ln.strip()]
        except Exception:
            return []

    @staticmethod
    def _cpu_name() -> str:
        """Best-effort friendly CPU brand string (e.g. '13th Gen Intel(R)
        Core(TM) i9-13900HX'). Falls back to the platform identifier."""
        try:
            if platform.system() == "Windows":
                import winreg  # type: ignore
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                )
                try:
                    name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                finally:
                    winreg.CloseKey(key)
                if name:
                    return str(name).strip()
        except Exception:
            pass
        for fn in (platform.processor, platform.machine):
            try:
                value = fn()
                if value:
                    return str(value).strip()
            except Exception:
                pass
        return "unknown CPU"

    def _prompt_workers(self, cpu_name: str, cores: int, suggested: int) -> int:
        """Show the detected CPU and a suggested worker count, then let the
        user accept it (Y) or enter their own (N). Non-interactive input
        (EOF/Ctrl-C) falls back to the suggestion."""
        print(f"\nCPU detected:  {cpu_name}")
        print(f"Logical cores: {cores}")
        try:
            answer = input(f"Use the suggested {suggested} worker threads? [Y/n, or type a number]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"(no input -- using suggested {suggested})")
            return suggested
        if answer in ("", "y", "yes", "j", "ja"):
            return suggested
        # Allow typing the desired count directly at this prompt.
        if answer.isdigit() and int(answer) >= 1:
            return int(answer)
        while True:
            try:
                raw = input("Enter the number of worker threads to use: ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"(no input -- using suggested {suggested})")
                return suggested
            try:
                chosen = int(raw)
            except ValueError:
                print("Please enter a whole number, e.g. 8.")
                continue
            if chosen >= 1:
                return chosen
            print("Please enter a positive integer (1 or more).")

    def _auto_tune(self) -> None:
        """Fill any config value left as the string "auto" with a setting
        derived from the detected hardware. Explicit values in settings.json
        are always kept. With "auto_hardware": false the "auto" sentinels
        resolve to the built-in defaults instead of hardware-based picks."""
        auto = bool(self.cfg.get("auto_hardware", True))
        hw = self._detect_hardware()
        arch = str(hw["arch"])
        is_x86 = any(tag in arch for tag in ("amd64", "x86_64", "x64", "i386", "i686"))
        is_arm = ("arm" in arch) or ("aarch64" in arch)
        cuda = bool(hw["has_cuda"])
        has_openvino = bool(hw.get("has_openvino", False))
        onnx_providers = hw.get("onnx_providers", [])
        has_dml = "DmlExecutionProvider" in onnx_providers
        cores = int(hw["logical_cores"])
        ram = float(hw["ram_gb"])
        vram = float(hw["vram_gb"])
        cpu_name = self._cpu_name()

        def recommended_batch() -> int:
            if not cuda:
                return 1
            if vram >= 14:
                return 8
            if vram >= 10:
                return 6
            if vram >= 7:
                return 4
            if vram >= 5:
                return 2
            return 1

        # workers: many on GPU (post-processing parallelism), few on a CPU
        # backend (let the inference engine own the cores instead of
        # oversubscribing). Capped by RAM (~0.3 GB per in-flight worker).
        if cuda:
            workers = cores
        else:
            # CPU backend: inference is serialized (model_thread_lock), so extra
            # workers mainly speed up the parallel post-processing. Use about
            # half the logical cores, leaving headroom for the inference engine.
            workers = max(2, min(cores // 2, 16))
        if ram > 0:
            workers = min(workers, max(2, int(ram * 0.6 / 0.3)))
        # Cap to keep thread/oversubscription overhead sane on big servers.
        workers = max(1, min(workers, 32))

        # Worker count: suggest from the detected CPU, then let an interactive
        # user accept (Y) or type their own (N). Pinning a number in
        # settings.json skips the prompt; a non-interactive run (no TTY) uses
        # the suggestion silently.
        cur_workers = self.cfg.get("performance", {}).get("workers", "auto")
        if isinstance(cur_workers, str) and cur_workers.strip().lower() == "auto":
            if not auto:
                workers = int(DEFAULT_CONFIG["performance"]["workers"])
            elif sys.stdin is not None and sys.stdin.isatty():
                workers = self._prompt_workers(cpu_name, cores, workers)
            # else: non-interactive auto run -> keep the computed suggestion
            self.cfg["performance"]["workers"] = workers

        # Model tier: keep the accurate medium model wherever there's a capable
        # GPU; step down to small on CPU-only or tiny-VRAM GPUs, and nano on ARM
        # where compute is scarcest. Any unlisted/unknown hardware falls through
        # to the small model as a safe middle ground. Smaller weights are
        # auto-fetched by ultralytics on first use if not present locally.
        if cuda:
            base_model = "yolov8m-pose.pt" if (vram >= 4 or vram == 0) else "yolov8s-pose.pt"
        elif is_arm:
            base_model = "yolov8n-pose.pt"
        else:
            base_model = "yolov8s-pose.pt"

        # Recognise a capable non-CUDA GPU by name (Intel Arc iGPU/dGPU, AMD
        # discrete Radeon RX/Pro). Weak display adapters (Intel UHD/Iris) are
        # deliberately excluded -- DirectML on those is slower than the CPU.
        gpu_names = [str(n).lower() for n in hw.get("gpu_names", [])]
        has_capable_dml_gpu = any(
            ("arc" in n) or ("radeon" in n and ("rx" in n or "pro" in n))
            for n in gpu_names
        )

        # DirectML GPU path: run the ONNX form of that model on a DirectX-12 GPU
        # (Adreno on Snapdragon, Intel Arc on x86). "auto" enables it on ARM or
        # whenever a capable Arc/discrete GPU is detected; true forces it on.
        prefer_dml = str(self.cfg["hardware"].get("prefer_directml", "auto")).strip().lower()
        want_dml = (not cuda) and has_dml and (
            prefer_dml in ("1", "true", "yes", "on")
            or (prefer_dml == "auto" and (is_arm or has_capable_dml_gpu))
        )
        onnx_name = base_model.replace(".pt", ".onnx")
        if want_dml and self._resolve_path(onnx_name).exists():
            model_choice = onnx_name
        else:
            model_choice = base_model

        picks: Dict[Tuple[str, str], Any] = {
            ("hardware", "model_path"): model_choice,
            ("hardware", "prefer_gpu"): cuda,
            ("hardware", "use_openvino_cpu"): (not cuda) and is_x86 and has_openvino,
            # 1280 on GPU improves keypoint recall on hunched/occluded riders
            # (fewer partial/cut-off detections); CPU/ARM stay lower for speed.
            ("detector", "imgsz"): 1280 if cuda else (768 if is_x86 else 640),
            # Keep the validated single-image path by default; batching is an
            # opt-in lever (the profile below prints the VRAM-based suggestion).
            ("performance", "batch_size"): 1,
            ("performance", "prefetch_maxsize"): 4 if ram >= 24 else 2,
            ("image_quality", "enable_fsrcnn"): (cores >= 8) and not is_arm,
            ("image_quality", "ai_upscaling_limit"): 1200 if cores >= 16 else (800 if cores >= 8 else 500),
        }

        for (section, key), auto_value in picks.items():
            cur = self.cfg.get(section, {}).get(key, "auto")
            if isinstance(cur, str) and cur.strip().lower() == "auto":
                self.cfg[section][key] = auto_value if auto else DEFAULT_CONFIG[section][key]

        model_is_onnx = str(self.cfg["hardware"]["model_path"]).lower().endswith(".onnx")
        if cuda and bool(self.cfg["hardware"]["prefer_gpu"]):
            backend = f"cuda ({hw['gpu_name']})"
        elif model_is_onnx and has_dml:
            backend = "onnxruntime / DirectML (GPU)"
        elif bool(self.cfg["hardware"]["use_openvino_cpu"]):
            backend = "openvino-cpu"
        else:
            backend = "pytorch-cpu"

        print("\n--- [AUTO-TUNE] ---")
        print(f"Mode:     {'hardware-adaptive' if auto else 'defaults (auto_hardware off)'}")
        cpu_line = f"CPU:      {cpu_name} [{arch}], {cores} logical cores"
        if ram > 0:
            cpu_line += f", {ram:.1f} GB RAM"
        print(cpu_line)
        if cuda:
            print(f"GPU:      {hw['gpu_name']} ({vram:.1f} GB)")
        elif hw.get("gpu_names"):
            print(f"GPU:      {', '.join(hw['gpu_names'])}"
                  + ("  [DirectML-capable]" if has_capable_dml_gpu else ""))
        else:
            print("GPU:      none")
        print(f"Backend:  {backend}")
        print(f"Model:    {self.cfg['hardware']['model_path']}")
        print(
            f"Tuned:    imgsz={self.cfg['detector']['imgsz']}  workers={self.cfg['performance']['workers']}  "
            f"batch={self.cfg['performance']['batch_size']}  prefetch={self.cfg['performance']['prefetch_maxsize']}  "
            f"fsrcnn={bool(self.cfg['image_quality']['enable_fsrcnn'])}(limit {self.cfg['image_quality']['ai_upscaling_limit']})"
        )
        if psutil is None:
            print("Note:     psutil not installed -- RAM/physical-core tuning skipped (using cpu_count).")
        if (not cuda) and is_x86 and not has_openvino and not model_is_onnx:
            print("Tip:      'pip install openvino' enables a faster x86 CPU backend "
                  "(currently using PyTorch CPU).")
        if want_dml and not str(self.cfg["hardware"]["model_path"]).lower().endswith(".onnx"):
            imgsz_now = int(self.cfg["detector"]["imgsz"])
            print(f"Tip:      DirectML GPU available. Export '{onnx_name}' "
                  f"(yolo export model={base_model} format=onnx imgsz={imgsz_now}) into")
            print("          this folder to run inference on the GPU instead of the CPU.")
        elif (not cuda) and has_capable_dml_gpu and not has_dml:
            print("Tip:      A DirectML-capable GPU (e.g. Intel Arc) was detected but "
                  "onnxruntime-directml isn't installed.")
            print("          'pip install onnxruntime-directml' + export the .onnx to run on it.")
        if cuda and recommended_batch() > 1 and int(self.cfg["performance"]["batch_size"]) == 1:
            print(f"Tip:      this GPU can likely handle batch_size={recommended_batch()} "
                  f"-- set it explicitly to enable the batched path.")
        print("-------------------\n")

    def _show_cuda_info(self) -> None:
        if not bool(self.cfg["hardware"].get("show_cuda_check", True)):
            return
        print("\n--- [CUDA CHECK] ---")
        try:
            import torch  # type: ignore
            print(f"Torch:       {torch.__version__}")
            print(f"CUDA avail.: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA device: {torch.cuda.get_device_name(0)}")
                try:
                    free_mem, total_mem = torch.cuda.mem_get_info(0)
                    print(f"VRAM free:   {free_mem / 1024**3:.2f} GB / {total_mem / 1024**3:.2f} GB")
                except Exception:
                    pass
            else:
                print("CUDA device: CPU fallback")
        except Exception as exc:
            print(f"CUDA check failed: {exc}")
        print("--------------------\n")

    def _warmup_model(self) -> None:
        """Run one tiny inference on the main thread so ultralytics builds its
        predictor / AutoBackend and moves the model onto the GPU exactly once,
        before any worker threads start. Without this, the first wave of worker
        threads can each trigger model setup concurrently and race on the CUDA
        context -- which surfaces as 'CUDA error: misaligned address' inside
        model.to(device). Failure here is non-fatal: the per-image error
        handling still applies during the real run."""
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._predict(dummy)
            print("--- [WARMUP] Model initialised and ready ---")
        except Exception as exc:
            print(f"[WARN] Model warm-up failed: {exc}")

    @staticmethod
    def _apply_fuse_guard(model):
        """Work around ultralytics builds that crash while (re-)fusing the model
        at load time -- "AttributeError: 'Conv' object has no attribute 'bn'",
        raised from AutoBackend.load_model -> model.fuse(). Reporting the model
        as already fused makes ultralytics skip that (buggy) Conv+BN fuse pass.
        The model then runs un-fused: identical detections, only a small speed
        cost. No-op on backends (e.g. OpenVINO) without a torch is_fused()."""
        try:
            underlying = getattr(model, "model", None)
            if underlying is not None and hasattr(underlying, "is_fused"):
                underlying.is_fused = lambda *args, **kwargs: True
        except Exception:
            pass
        return model

    @staticmethod
    def _install_onnx_provider_patch() -> str:
        """Make ultralytics' ONNX inference run on the best available execution
        provider. ultralytics' ONNX backend would otherwise default to CPU; we
        wrap onnxruntime.InferenceSession to inject DirectML (Adreno GPU on
        Snapdragon) ahead of CPU, so only the heavy conv work moves to the GPU
        while ultralytics keeps doing pre/post-processing. Returns the provider
        name that will be used."""
        # Stop ultralytics' auto-updater from pip-installing plain onnx/onnxruntime
        # when it loads the .onnx: that overwrites onnxruntime-directml (same
        # module), the install fails on the locked DLL, and inference silently
        # drops back to CPU. We manage the runtime, so disable that check.
        os.environ["YOLO_AUTOINSTALL"] = "false"
        for _mod in ("ultralytics.nn.autobackend", "ultralytics.utils.checks"):
            try:
                import importlib
                m = importlib.import_module(_mod)
                if hasattr(m, "check_requirements"):
                    m.check_requirements = lambda *args, **kwargs: True
            except Exception:
                pass
        import onnxruntime as ort  # type: ignore
        available = ort.get_available_providers()
        # DirectML first (broad GPU support, no model changes). QNN/NPU needs an
        # INT8 model + provider options, so it's left to a dedicated future path.
        accel = "DmlExecutionProvider" if "DmlExecutionProvider" in available else ""
        if not accel:
            return "CPUExecutionProvider"
        providers = [accel, "CPUExecutionProvider"]
        if not getattr(ort, "_runner_suite_patched", False):
            _orig_session = ort.InferenceSession

            def _patched_session(*args, **kwargs):
                kwargs["providers"] = providers
                return _orig_session(*args, **kwargs)

            ort.InferenceSession = _patched_session
            ort._runner_suite_patched = True
        return accel

    def _setup_hardware_and_yolo(self):
        model_value = str(self.cfg["hardware"].get("model_path", "yolov8m-pose.pt"))
        model_path = self._resolve_path(model_value)
        if model_path.exists():
            model_arg = str(model_path)
        elif Path(model_value).name == model_value and not Path(model_value).is_absolute():
            # A bare ultralytics model name (e.g. an auto-tiered yolov8s-pose.pt)
            # that isn't on disk yet -- let ultralytics fetch it from its model
            # hub. A missing *path* (with directories) is treated as an error.
            print(f"[INFO] Model '{model_value}' not present locally -- ultralytics will download it on first use.")
            model_arg = model_value
        else:
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self._show_cuda_info()
        prefer_gpu = bool(self.cfg["hardware"].get("prefer_gpu", True))
        use_openvino_cpu = bool(self.cfg["hardware"].get("use_openvino_cpu", False))

        # ONNX model -> run through ultralytics' ONNX backend, but force the
        # session onto an accelerated provider (DirectML GPU / QNN NPU) when one
        # is available. ultralytics still does all pre/post-processing.
        if model_arg.lower().endswith(".onnx"):
            try:
                provider = self._install_onnx_provider_patch()
                self.device_mode = "cpu"
                self.backend = "onnx"
                print(f"--- [HARDWARE] ONNX Runtime active (provider: {provider}) ---")
                return self._apply_fuse_guard(YOLO(model_arg))
            except Exception as exc:
                print(f"[WARN] ONNX backend setup failed ({exc}); falling back to PyTorch CPU.")

        if prefer_gpu:
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    self.device_mode = "cuda:0"
                    self.backend = "pytorch"
                    torch.backends.cudnn.benchmark = True
                    print("--- [HARDWARE] NVIDIA GPU/CUDA active ---")
                    return self._apply_fuse_guard(YOLO(model_arg))
            except Exception as exc:
                print(f"[WARN] CUDA check failed, using fallback: {exc}")

        if use_openvino_cpu:
            try:
                default_ov = f"{Path(model_value).stem}_openvino_model"
                ov_dir = self._resolve_path(str(self.cfg["hardware"].get("openvino_model_dir", default_ov)))
                if not ov_dir.exists():
                    print("--- [HARDWARE] Generating OpenVINO model for CPU ... ---")
                    YOLO(model_arg).export(format="openvino", imgsz=int(self.cfg["detector"].get("imgsz", 1024)))
                self.device_mode = "cpu"
                self.backend = "openvino"
                print("--- [HARDWARE] CPU/OpenVINO active ---")
                return YOLO(str(ov_dir), task="pose")
            except Exception as exc:
                # openvino package missing or export failed -- don't crash the
                # whole run; fall through to the plain PyTorch CPU backend.
                print(f"[WARN] OpenVINO setup failed ({exc}); falling back to PyTorch CPU.")

        self.device_mode = "cpu"
        self.backend = "pytorch"
        print("--- [HARDWARE] PyTorch CPU active ---")
        return self._apply_fuse_guard(YOLO(model_arg))

    def _setup_upscaler(self) -> None:
        iq = self.cfg["image_quality"]
        if not bool(iq.get("enable_fsrcnn", False)):
            return
        if not hasattr(cv2, "dnn_superres"):
            print("[WARN] cv2.dnn_superres not available. FSRCNN disabled.")
            return
        model_path = self._resolve_path(str(iq.get("fsrcnn_model_path", "FSRCNN_x3.pb")))
        if not model_path.exists() and not self._download_fsrcnn_model(model_path):
            print("[WARN] FSRCNN model unavailable. FSRCNN disabled.")
            return
        # Validate the model loads once here (fail fast / clear message); the
        # actual instances used for upscaling are created lazily per worker
        # thread in _get_sr().
        try:
            probe = cv2.dnn_superres.DnnSuperResImpl_create()
            probe.readModel(str(model_path))
            probe.setModel("fsrcnn", self._sr_scale)
        except Exception as exc:
            print(f"[WARN] FSRCNN could not be loaded: {exc}")
            return
        self._sr_model_path = str(model_path)
        self._sr_ready = True
        print("--- [AI PIXELS] FSRCNN loaded (per-thread, parallel) ---")

    def _get_sr(self):
        """Return this thread's FSRCNN upscaler, creating it on first use.
        Per-thread instances let super-resolution run concurrently across CPU
        cores; cv2's DnnSuperResImpl is not safe to share between threads."""
        if not self._sr_ready:
            return None
        sr = getattr(self._sr_tls, "sr", None)
        if sr is None:
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(self._sr_model_path)
                sr.setModel("fsrcnn", self._sr_scale)
                self._sr_tls.sr = sr
            except Exception:
                return None
        return sr

    def _setup_realesrgan(self) -> None:
        """Load Real-ESRGAN as the upscaler when image_quality.upscaler is
        'realesrgan' (the ENHANCER preset). GPU-only: it needs CUDA, the
        realesrgan/basicsr libraries, and the model weights. Any of those
        missing -> leave self._realesrgan = None and the pipeline falls back to
        FSRCNN, then Lanczos. Never fatal."""
        iq = self.cfg["image_quality"]
        if str(iq.get("upscaler", "fsrcnn")).lower() != "realesrgan":
            return
        try:
            import torch  # type: ignore
            cuda = torch.cuda.is_available()
        except Exception:
            cuda = False
        if not cuda:
            print("[WARN] Real-ESRGAN requested but no CUDA GPU -- using FSRCNN/Lanczos instead.")
            return

        # basicsr imports torchvision.transforms.functional_tensor, which newer
        # torchvision (>= 0.17, i.e. torch 2.x) removed. Alias it to the current
        # module so the import succeeds.
        try:
            import torchvision.transforms.functional as _tvf  # type: ignore
            sys.modules.setdefault("torchvision.transforms.functional_tensor", _tvf)
        except Exception:
            pass

        def _import_realesrgan():
            from realesrgan import RealESRGANer  # type: ignore
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            return RealESRGANer, RRDBNet

        try:
            RealESRGANer, RRDBNet = _import_realesrgan()
        except Exception as exc:
            if not bool(iq.get("realesrgan_auto_install", True)):
                print(f"[WARN] realesrgan/basicsr not available ({exc}); using FSRCNN/Lanczos.")
                return
            print("[AI PIXELS] Installing realesrgan + basicsr (one-time, can take a few minutes) ...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "realesrgan", "basicsr"])
                RealESRGANer, RRDBNet = _import_realesrgan()
            except Exception as exc2:
                print(f"[WARN] Could not install/import realesrgan ({exc2}); using FSRCNN/Lanczos.")
                return
        model_path = self._resolve_path(str(iq.get("realesrgan_model_path", "RealESRGAN_x4plus.pth")))
        if not model_path.exists() and not self._download_realesrgan_model(model_path):
            print("[WARN] Real-ESRGAN model unavailable; using FSRCNN/Lanczos.")
            return
        try:
            net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self._realesrgan = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=net,
                tile=int(iq.get("realesrgan_tile", 0)),
                tile_pad=10,
                pre_pad=0,
                half=True,
                device="cuda",
            )
            self._realesrgan_scale = 4
            print("--- [AI PIXELS] Real-ESRGAN x4 loaded (GPU) ---")
        except Exception as exc:
            print(f"[WARN] Real-ESRGAN could not be loaded: {exc}; using FSRCNN/Lanczos.")
            self._realesrgan = None

    def _download_realesrgan_model(self, model_path: Path) -> bool:
        """Auto-download the Real-ESRGAN weights (mirrors the FSRCNN/GFPGAN
        downloaders: temp file -> validate -> promote). Returns True on success."""
        iq = self.cfg["image_quality"]
        url = str(iq.get("realesrgan_model_url", "")).strip()
        if not url:
            print(f"[WARN] Real-ESRGAN model missing at {model_path} and no URL configured.")
            return False
        print(f"[AI PIXELS] Real-ESRGAN model not found -- downloading weights to {model_path} ...")
        tmp_path = model_path.with_name(model_path.name + ".part")
        try:
            import urllib.request
            ensure_dir(model_path.parent)
            urllib.request.urlretrieve(url, str(tmp_path))
        except Exception as exc:
            self._unlink_quietly(tmp_path)
            print(f"[WARN] Could not download Real-ESRGAN model: {exc}")
            return False
        expected_sha = str(iq.get("realesrgan_model_sha256", "")).strip().lower()
        ok, reason = self._validate_model_download(tmp_path, expected_sha, min_bytes=1_000_000)
        if not ok:
            self._unlink_quietly(tmp_path)
            print(f"[WARN] Downloaded Real-ESRGAN model failed validation ({reason}). Discarding it.")
            return False
        tmp_path.replace(model_path)
        print("[AI PIXELS] Real-ESRGAN model download complete (validated).")
        return True

    def _apply_upscaler(self, crop: Any) -> Any:
        """Super-resolve a small crop. Prefers Real-ESRGAN (GPU, ENHANCER) when
        loaded; otherwise thread-local FSRCNN; otherwise returns the crop
        unchanged (the caller's final Lanczos resize still runs)."""
        if self._realesrgan is not None:
            try:
                with self._realesrgan_lock:
                    # Real-ESRGAN prints a line per tile ("Tile 9/24"); swallow
                    # that so it doesn't bury the progress bar.
                    with contextlib.redirect_stdout(io.StringIO()):
                        out, _ = self._realesrgan.enhance(crop, outscale=self._realesrgan_scale)
                return out
            except Exception as exc:
                print(f"[WARN] Real-ESRGAN upscaling skipped: {exc}")
                return crop
        sr = self._get_sr()
        if sr is not None:
            try:
                return sr.upsample(crop)
            except Exception as exc:
                print(f"[WARN] FSRCNN upscaling skipped: {exc}")
        return crop

    def _setup_pipeline_face_restorer(self) -> None:
        """Load GFPGAN (with Real-ESRGAN as the background upsampler) for the
        'upscale+face' enhance mode. Only set up when some quality class asks
        for faces AND Real-ESRGAN is available (GPU). Otherwise 'upscale+face'
        degrades gracefully to plain upscaling."""
        modes = self.cfg["image_quality"].get("enhance_by_class", {})
        if not any("face" in str(m).lower() for m in modes.values()):
            return
        if self._realesrgan is None:
            print("[WARN] Face restoration needs the Real-ESRGAN GPU pipeline; 'upscale+face' -> upscaling.")
            return
        if not self._ensure_gfpgan_ready() or GFPGANer is None:
            print("[WARN] GFPGAN not available; 'upscale+face' -> upscaling.")
            return
        re_cfg = self.cfg.get("review_enhancement", {})
        model_path = self._resolve_path(str(re_cfg.get("gfpgan_model_path", "GFPGANv1.4.pth")))
        try:
            self._face_restorer = GFPGANer(
                model_path=str(model_path),
                upscale=self._realesrgan_scale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=self._realesrgan,
            )
            print("--- [AI PIXELS] GFPGAN face restoration active (Real-ESRGAN background) ---")
        except Exception as exc:
            print(f"[WARN] Could not load GFPGAN ({exc}); 'upscale+face' -> upscaling.")
            self._face_restorer = None

    def _apply_face_restore(self, crop: Any) -> Any:
        """GFPGAN face restoration + Real-ESRGAN background in one pass.
        Serialized on the Real-ESRGAN lock (GPU); falls back to plain upscaling
        on any error."""
        if self._face_restorer is None:
            return self._apply_upscaler(crop)
        try:
            with self._realesrgan_lock:
                with contextlib.redirect_stdout(io.StringIO()):
                    _, _, out = self._face_restorer.enhance(
                        crop, has_aligned=False, only_center_face=False, paste_back=True
                    )
            return out if out is not None else crop
        except Exception as exc:
            print(f"[WARN] Face restoration skipped ({exc}); upscaling instead.")
            return self._apply_upscaler(crop)

    def _download_fsrcnn_model(self, model_path: Path) -> bool:
        """Download the FSRCNN super-resolution weights automatically if they
        are missing and 'auto_download_fsrcnn' is enabled (mirrors the GFPGAN
        auto-download). Returns True if the model file is present afterwards."""
        iq = self.cfg["image_quality"]
        if not bool(iq.get("auto_download_fsrcnn", True)):
            print(f"[INFO] FSRCNN model not found at {model_path} and auto-download is disabled.")
            return False
        url = str(iq.get("fsrcnn_model_url", "")).strip()
        if not url:
            print(f"[WARN] FSRCNN model missing at {model_path} and no download URL is configured.")
            return False
        print(f"[AI PIXELS] FSRCNN model not found -- downloading weights to {model_path} ...")
        # Download to a temp file first; only promote it to the real path once it
        # passes validation, so a partial/corrupt fetch never lands as the .pb.
        tmp_path = model_path.with_name(model_path.name + ".part")
        try:
            import urllib.request
            ensure_dir(model_path.parent)
            urllib.request.urlretrieve(url, str(tmp_path))
        except Exception as exc:
            self._unlink_quietly(tmp_path)
            print(f"[WARN] Could not download FSRCNN model automatically: {exc}")
            print(f"       Please download it manually from {url}")
            print(f"       and place it at: {model_path}")
            return False

        expected_sha = str(iq.get("fsrcnn_model_sha256", "")).strip().lower()
        ok, reason = self._validate_model_download(tmp_path, expected_sha)
        if not ok:
            self._unlink_quietly(tmp_path)
            print(f"[WARN] Downloaded FSRCNN model failed validation ({reason}). Discarding it.")
            print(f"       Please download it manually from {url}")
            print(f"       and place it at: {model_path}")
            return False

        tmp_path.replace(model_path)
        print("[AI PIXELS] FSRCNN model download complete (validated).")
        return True

    @staticmethod
    def _unlink_quietly(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    @staticmethod
    def _validate_model_download(path: Path, expected_sha256: str = "", min_bytes: int = 1024) -> Tuple[bool, str]:
        """Sanity-check a freshly downloaded model file. Catches the common
        failure modes -- truncated downloads and HTML error pages saved under a
        model name -- and, when an expected SHA-256 is given, verifies the
        content byte-for-byte. Returns (ok, reason)."""
        try:
            size = path.stat().st_size
        except OSError as exc:
            return False, f"unreadable: {exc}"
        if size < min_bytes:
            return False, f"too small ({size} bytes)"
        with open(path, "rb") as f:
            head = f.read(64).lstrip()
        if head[:1] == b"<" or head[:5].lower() == b"<!doc":
            return False, "looks like an HTML/error page, not a model"
        if expected_sha256:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            actual = h.hexdigest()
            if actual != expected_sha256:
                return False, f"sha256 mismatch (got {actual})"
        return True, "ok"

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------
    def _inc(self, key: str, amount: int = 1) -> None:
        with self.stats_lock:
            self.stats[key] = self.stats.get(key, 0) + amount

    def _report(self, row: Dict[str, Any]) -> None:
        if not bool(self.cfg["debug"].get("write_csv", True)):
            return
        with self.report_lock:
            self.report_rows.append(row)

    def _predict(self, img: Any, batch: bool = False):
        det = self.cfg["detector"]
        sf = self.cfg["selection_filters"]
        kwargs: Dict[str, Any] = {
            "imgsz": int(det.get("imgsz", 1024)),
            "conf": float(sf.get("conf_threshold", 0.33)),
            "iou": float(det.get("iou", 0.55)),
            "max_det": int(det.get("max_det", 120)),
            "verbose": False,
        }
        if self.backend == "pytorch":
            kwargs["device"] = self.device_mode
            # FP16 for single-image CUDA inference (roughly 2x faster than FP32
            # on the GPU). Batched calls stay FP32 -- some ultralytics versions
            # re-fuse the model on every predict() with half=True. The per-call
            # "half is deprecated" log is silenced in __init__.
            if str(self.device_mode).startswith("cuda") and not batch:
                kwargs["half"] = True
        else:
            kwargs["device"] = "cpu"

        if bool(self.cfg["hardware"].get("model_thread_lock", False)):
            with self.model_lock:
                return self.model.predict(img, **kwargs)
        return self.model.predict(img, **kwargs)

    def _extract_roi(self, img: Any, box: Box) -> Any:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box
        x1 = int(clamp(x1, 0, w - 1))
        y1 = int(clamp(y1, 0, h - 1))
        x2 = int(clamp(x2, x1 + 1, w))
        y2 = int(clamp(y2, y1 + 1, h))
        return img[y1:y2, x1:x2]

    def get_sharpness_score(self, image: Any) -> float:
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def get_brightness_score(self, image: Any) -> float:
        """Mean grayscale brightness (0-255)."""
        if image is None or image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _pre_check_image(self, img: Any) -> bool:
        """Fast full-image quality gate run before YOLO inference.
        Downscales to 256 px and checks sharpness and brightness.
        Returns False if the image is clearly too blurry or badly exposed
        to ever yield a premium or good crop — skipping GPU inference entirely."""
        pre_cfg = self.cfg.get("pre_check", {})
        if not bool(pre_cfg.get("enabled", True)):
            return True
        small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < float(pre_cfg.get("min_image_sharpness", 50.0)):
            return False
        brightness = float(np.mean(gray))
        if brightness < float(pre_cfg.get("min_image_brightness", 30.0)):
            return False
        if brightness > float(pre_cfg.get("max_image_brightness", 230.0)):
            return False
        return True

    def _is_promising_for_enhancement(self, crop: Any, sharpness: float) -> Tuple[bool, str]:
        """Cheap heuristic, evaluated ONLY for review-class crops (so it never
        slows down the main accept/reject hot path): flags images that are
        moderately soft or under-/over-exposed -- the kind of imperfection AI
        face/body restoration (GFPGAN) is good at fixing -- without being bad
        enough to have been discarded outright."""
        re_cfg = self.cfg.get("review_enhancement", {})
        if not bool(re_cfg.get("flag_candidates", True)):
            return False, ""
        if sharpness <= float(re_cfg.get("candidate_max_sharpness", 160)):
            return True, "soft_focus"
        brightness = self.get_brightness_score(crop)
        if brightness < float(re_cfg.get("candidate_min_brightness", 60)):
            return True, "underexposed"
        if brightness > float(re_cfg.get("candidate_max_brightness", 195)):
            return True, "overexposed"
        return False, ""

    def sharpen_image(self, image: Any) -> Any:
        level = str(self.cfg["image_quality"].get("sharpening", "strong")).lower()
        amount_map = {"none": 0.0, "light": 0.35, "medium": 0.60, "strong": 0.90}
        amount = float(amount_map.get(level, 0.60))
        if amount <= 0:
            return image
        blur = cv2.GaussianBlur(image, (0, 0), 1.0)
        return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)

    def denoise_image(self, image: Any) -> Any:
        if not bool(self.cfg["image_quality"].get("denoise", False)):
            return image
        return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)

    # ------------------------------------------------------------------
    # Crop logic
    # ------------------------------------------------------------------
    def _is_full_frame_runner(self, box: Box, img_shape: Tuple[int, int]) -> bool:
        if not bool(self.cfg["full_frame_runner"].get("enabled", True)):
            return False
        h_img, w_img = img_shape
        x1, y1, x2, y2 = box
        box_h_ratio = (y2 - y1) / max(1.0, h_img)
        box_area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, w_img * h_img)
        return (
            box_h_ratio >= float(self.cfg["full_frame_runner"].get("trigger_box_height_ratio", 0.70))
            or box_area_ratio >= float(self.cfg["full_frame_runner"].get("trigger_box_area_ratio", 0.34))
        )

    def _crop_to_aspect_around_center(self, img: Any, center_x: float, center_y: float) -> Tuple[Any, Box]:
        h, w = img.shape[:2]
        ratio = float(self.cfg["crop"].get("aspect_ratio", 0.666))
        current_ratio = w / max(1.0, h)
        if abs(current_ratio - ratio) < 0.01:
            return img, (0.0, 0.0, float(w), float(h))
        if current_ratio > ratio:
            crop_h = float(h)
            crop_w = crop_h * ratio
            x1 = center_x - crop_w / 2.0
            y1 = 0.0
        else:
            crop_w = float(w)
            crop_h = crop_w / ratio
            x1 = 0.0
            y1 = center_y - crop_h / 2.0
        x1 = clamp(x1, 0.0, max(0.0, w - crop_w))
        y1 = clamp(y1, 0.0, max(0.0, h - crop_h))
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        x1i, y1i = int(round(x1)), int(round(y1))
        x2i, y2i = int(round(x2)), int(round(y2))
        return img[y1i:y2i, x1i:x2i], (float(x1i), float(y1i), float(x2i), float(y2i))

    def get_full_frame_crop(self, img: Any, person_box: Box) -> Tuple[Any, Box]:
        x1, y1, x2, y2 = person_box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        if bool(self.cfg["full_frame_runner"].get("crop_to_aspect_ratio", True)):
            return self._crop_to_aspect_around_center(img, center_x, center_y)
        h, w = img.shape[:2]
        return img, (0.0, 0.0, float(w), float(h))

    def _assess_completeness(self, kp: Any, person_box: Box, img_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Judge whether a detection covers a whole athlete or a partial/cut-off
        one, using which COCO-17 keypoint groups are confidently present and
        whether the person box is jammed against a frame edge. Used to keep
        partial crops out of Premium/Good rather than to reject outright."""
        cg = self.cfg.get("completeness_guard", {})
        h_img, w_img = img_shape
        kc = float(self.cfg["selection_filters"].get("keypoint_conf", 0.30))

        def conf(i: int) -> bool:
            return kp is not None and i < len(kp) and float(kp[i][2]) > kc

        has_head = any(conf(i) for i in (0, 1, 2, 3, 4))
        both_shoulders = conf(5) and conf(6)
        any_shoulder = conf(5) or conf(6)
        has_hips = conf(11) or conf(12)
        has_legs = any(conf(i) for i in (13, 14, 15, 16))
        has_upper_body = has_head or both_shoulders
        partial_lower_body = (has_hips or has_legs) and not has_upper_body

        zones = [has_head, any_shoulder, has_hips, conf(13) or conf(14), conf(15) or conf(16)]
        completeness = sum(1 for z in zones if z) / float(len(zones))

        margin = float(cg.get("edge_margin_px", 6))
        y1, y2 = float(person_box[1]), float(person_box[3])
        truncated_top = bool(cg.get("truncation_guard", True)) and (y1 <= margin) and not has_head
        truncated_bottom = (y2 >= h_img - margin) and not has_legs  # informational (feet cut is OK)

        demote = (bool(cg.get("require_upper_body", True)) and not has_upper_body) or truncated_top
        reason = None
        if partial_lower_body:
            reason = "partial_lower_body"
        elif truncated_top:
            reason = "truncated_head"
        elif not has_upper_body:
            reason = "no_upper_body"
        return {
            "has_head": has_head, "has_upper_body": has_upper_body,
            "partial_lower_body": partial_lower_body,
            "truncated_top": truncated_top, "truncated_bottom": truncated_bottom,
            "completeness": round(completeness, 3),
            "ok_for_premium": not demote, "reason": reason,
        }

    def _anatomy_crop_box(self, kp: Any, box: Box, keypoint_conf: float, cp: Dict[str, Any]) -> Box:
        """Anatomy-anchored framing (vantage-invariant). Lock the vertical extent
        to head-top .. foot-bottom with small fixed margins and DERIVE the width
        from the aspect ratio, so the runner is framed head-to-toe whether shot
        from low, level, or high -- without the arbitrary 'floor' the pad method
        adds when it fills the aspect ratio vertically. Returns (x1, y1, x2, y2)."""
        bx1, by1, bx2, by2 = box
        ph = max(1.0, by2 - by1)
        ratio = float(cp.get("aspect_ratio", 0.666))
        cg = self.cfg.get("completeness_guard", {})

        def _conf(i: int) -> bool:
            return kp is not None and i < len(kp) and float(kp[i][2]) > keypoint_conf

        head_ys = [float(kp[i][1]) for i in (0, 1, 2, 3, 4) if _conf(i)]
        shoulder_ys = [float(kp[i][1]) for i in (5, 6) if _conf(i)]
        ankle_ys = [float(kp[i][1]) for i in (15, 16) if _conf(i)]

        # Head-top: the crown sits above the face keypoints; when the head is weak
        # (high vantage) estimate it above the shoulders. Fall back to the box top.
        if head_ys:
            head_top = min(by1, min(head_ys) - ph * float(cp.get("anatomy_crown_allowance", 0.06)))
        elif shoulder_ys:
            head_top = min(by1, (sum(shoulder_ys) / len(shoulder_ys)) - ph * float(cg.get("head_estimate_ratio", 0.22)))
        else:
            head_top = by1
        foot_bottom = max([by2] + ankle_ys)

        person_h = max(1.0, foot_bottom - head_top)
        top = head_top - person_h * float(cp.get("anatomy_headroom_ratio", 0.08))
        bottom = foot_bottom + person_h * float(cp.get("anatomy_footroom_ratio", 0.06))

        crop_w = (bottom - top) * ratio
        runner_w = (bx2 - bx1) * (1.0 + float(cp.get("anatomy_side_margin", 0.12)))
        if crop_w < runner_w:                     # arms out / wide stance -> grow downward
            crop_w = runner_w
            bottom = top + crop_w / ratio
        cx = (bx1 + bx2) / 2.0
        return cx - crop_w / 2.0, top, cx + crop_w / 2.0, bottom

    def get_smart_crop(self, img: Any, kp: Any, person_box: Box) -> Optional[Tuple[Any, Box]]:
        h_img, w_img = img.shape[:2]
        cp = self.cfg["crop"]
        sf = self.cfg["selection_filters"]

        x1, y1, x2, y2 = [float(v) for v in person_box]
        keypoint_conf = float(sf.get("keypoint_conf", 0.30))
        valid = kp[kp[:, 2] > keypoint_conf] if kp is not None else []
        if len(valid) >= int(sf.get("min_keypoints", 4)):
            x1 = min(x1, float(np.min(valid[:, 0])))
            y1 = min(y1, float(np.min(valid[:, 1])))
            x2 = max(x2, float(np.max(valid[:, 0])))
            y2 = max(y2, float(np.max(valid[:, 1])))

        # Anatomy-anchored framing (opt-in via crop.mode="anatomy"). Uses a plain
        # clip (not the shift-to-fit below) so a head near the source top is not
        # pushed downward into a floor of empty ground; resize_final does any
        # final aspect correction, bottom-biased and head-safe.
        if str(cp.get("mode", "pad")).lower() == "anatomy":
            ax1, ay1, ax2, ay2 = self._anatomy_crop_box(kp, (x1, y1, x2, y2), keypoint_conf, cp)
            x1i = int(clamp(ax1, 0, w_img - 1))
            y1i = int(clamp(ay1, 0, h_img - 1))
            x2i = int(clamp(ax2, x1i + 1, w_img))
            y2i = int(clamp(ay2, y1i + 1, h_img))
            crop = img[y1i:y2i, x1i:x2i]
            if crop is None or crop.size == 0:
                return None
            return crop, (float(x1i), float(y1i), float(x2i), float(y2i))

        # Head-safe framing. Two failure modes this guards against:
        #  * hunched cyclists (head low/forward/occluded), and
        #  * runners shot from a high vantage point -- looking down at the top of
        #    the head means the face keypoints go low-confidence, so the box
        #    stops at the neck and the head gets clipped.
        # When the head keypoints ARE visible, keep a margin above the crown;
        # when they're weak, estimate the head top above the shoulders (with a
        # person-height floor so the estimate survives torso foreshortening).
        cg = self.cfg.get("completeness_guard", {})
        if bool(cg.get("head_extend", True)) and kp is not None:
            def _conf(i: int) -> bool:
                return i < len(kp) and float(kp[i][2]) > keypoint_conf
            person_h = max(1.0, y2 - y1)
            head_ys = [float(kp[i][1]) for i in (0, 1, 2, 3, 4) if _conf(i)]
            shoulder_ys = [float(kp[i][1]) for i in (5, 6) if _conf(i)]
            hip_ys = [float(kp[i][1]) for i in (11, 12) if _conf(i)]
            if head_ys:                        # head visible -> margin above the crown
                y1 = min(y1, min(head_ys) - person_h * float(cg.get("min_headroom_ratio", 0.12)))
            elif shoulder_ys:                  # head weak -> estimate its top from shoulders
                sh_y = sum(shoulder_ys) / len(shoulder_ys)
                torso = (min(hip_ys) - sh_y) if hip_ys and min(hip_ys) > sh_y else 0.0
                head_room = max(torso * float(cg.get("head_extend_ratio", 0.7)),
                                person_h * float(cg.get("head_estimate_ratio", 0.22)))
                y1 = min(y1, sh_y - head_room)

        runner_w = max(1.0, x2 - x1)
        runner_h = max(1.0, y2 - y1)
        pad_mult = 1.0
        if runner_h < float(cp.get("small_person_threshold_px", 700)):
            pad_mult = float(cp.get("small_person_extra_pad_multiplier", 1.15))

        x1 -= runner_w * float(cp.get("pad_x", 0.38)) * pad_mult
        x2 += runner_w * float(cp.get("pad_x", 0.38)) * pad_mult
        y1 -= runner_h * float(cp.get("pad_top", 0.18)) * pad_mult
        y2 += runner_h * float(cp.get("pad_bottom", 0.28)) * pad_mult

        target_ratio = float(cp.get("aspect_ratio", 0.666))
        crop_w = max(1.0, x2 - x1)
        crop_h = max(1.0, y2 - y1)
        if crop_w / crop_h < target_ratio:
            extra = crop_h * target_ratio - crop_w
            x1 -= extra / 2.0
            x2 += extra / 2.0
        else:
            extra = crop_w / target_ratio - crop_h
            y1 -= extra * 0.45
            y2 += extra * 0.55

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w >= w_img:
            x1, x2 = 0.0, float(w_img)
        else:
            if x1 < 0:
                x2 -= x1
                x1 = 0.0
            if x2 > w_img:
                x1 -= (x2 - w_img)
                x2 = float(w_img)
        if crop_h >= h_img:
            y1, y2 = 0.0, float(h_img)
        else:
            if y1 < 0:
                y2 -= y1
                y1 = 0.0
            if y2 > h_img:
                y1 -= (y2 - h_img)
                y2 = float(h_img)

        x1i = int(clamp(x1, 0, w_img - 1))
        y1i = int(clamp(y1, 0, h_img - 1))
        x2i = int(clamp(x2, x1i + 1, w_img))
        y2i = int(clamp(y2, y1i + 1, h_img))
        crop = img[y1i:y2i, x1i:x2i]
        if crop is None or crop.size == 0:
            return None
        return crop, (float(x1i), float(y1i), float(x2i), float(y2i))

    # ------------------------------------------------------------------
    # Fast fence/occlusion detection
    # ------------------------------------------------------------------
    def _relative_box(self, abs_box: Box, parent_box: Box, max_w: int, max_h: int) -> Box:
        x1 = clamp(abs_box[0] - parent_box[0], 0, max_w - 1)
        y1 = clamp(abs_box[1] - parent_box[1], 0, max_h - 1)
        x2 = clamp(abs_box[2] - parent_box[0], x1 + 1, max_w)
        y2 = clamp(abs_box[3] - parent_box[1], y1 + 1, max_h)
        return (x1, y1, x2, y2)

    def detect_fence(self, crop: Any, crop_box: Box, person_box: Box) -> Dict[str, Any]:
        cfg = self.cfg.get("fence_detection", {})
        if not bool(cfg.get("enabled", True)):
            return {
                "fence_detected": False,
                "fence_score": 0.0,
                "vertical_line_count": 0,
                "person_occlusion_coverage": 0.0,
                "central_occlusion_score": 0.0,
                "vertical_line_coverage": 0.0,
            }
        h, w = crop.shape[:2]
        if h < 50 or w < 50:
            return {
                "fence_detected": False,
                "fence_score": 0.0,
                "vertical_line_count": 0,
                "person_occlusion_coverage": 0.0,
                "central_occlusion_score": 0.0,
                "vertical_line_coverage": 0.0,
            }
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, bright = cv2.threshold(gray, int(cfg.get("bright_threshold", 210)), 255, cv2.THRESH_BINARY)
        k_h = max(10, int(round(h * float(cfg.get("vertical_kernel_height_ratio", 0.11)))))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, k_h))
        vertical = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vertical, connectivity=8)

        component_mask = np.zeros_like(vertical)
        vertical_count = 0
        min_comp_h = h * float(cfg.get("min_component_height_ratio", 0.32))
        max_comp_w = max(3.0, w * float(cfg.get("max_component_width_ratio", 0.11)))
        min_aspect = float(cfg.get("min_component_aspect_ratio", 3.0))
        for label in range(1, num_labels):
            cw = stats[label, cv2.CC_STAT_WIDTH]
            ch = stats[label, cv2.CC_STAT_HEIGHT]
            if ch < min_comp_h:
                continue
            if cw > max_comp_w:
                continue
            if ch / max(1.0, cw) < min_aspect:
                continue
            component_mask[labels == label] = 255
            vertical_count += 1

        vertical_line_coverage = float(np.count_nonzero(component_mask)) / float(max(1, w * h))
        rel_person = self._relative_box(person_box, crop_box, w, h)
        px1, py1, px2, py2 = [int(v) for v in rel_person]
        person_mask = component_mask[py1:py2, px1:px2]
        person_area = float(max(1, (px2 - px1) * (py2 - py1)))
        person_occlusion = float(np.count_nonzero(person_mask)) / person_area if person_mask.size else 0.0

        band_ratio = float(cfg.get("central_band_width_ratio", 0.58))
        band_w = max(1, int(round(w * band_ratio)))
        bx1 = max(0, (w - band_w) // 2)
        bx2 = min(w, bx1 + band_w)
        central = component_mask[:, bx1:bx2]
        central_score = float(np.count_nonzero(central)) / float(max(1, central.size))

        fence_score = max(person_occlusion, central_score, vertical_line_coverage * 1.3)
        fence_detected = (
            vertical_count >= int(cfg.get("reject_vertical_line_count", 5))
            and (
                person_occlusion >= float(cfg.get("reject_person_occlusion_coverage", 0.13))
                or central_score >= float(cfg.get("reject_central_occlusion_score", 0.22))
                or vertical_line_coverage >= float(cfg.get("reject_vertical_line_coverage", 0.15))
            )
        )
        return {
            "fence_detected": bool(fence_detected),
            "fence_score": round(float(fence_score), 4),
            "vertical_line_count": int(vertical_count),
            "person_occlusion_coverage": round(float(person_occlusion), 4),
            "central_occlusion_score": round(float(central_score), 4),
            "vertical_line_coverage": round(float(vertical_line_coverage), 4),
        }

    # ------------------------------------------------------------------
    # Quality / score
    # ------------------------------------------------------------------
    def _center_offsets(self, crop_box: Box, person_box: Box) -> Tuple[float, float, float]:
        cx1, cy1, cx2, cy2 = crop_box
        px1, py1, px2, py2 = person_box
        crop_w = max(1.0, cx2 - cx1)
        crop_h = max(1.0, cy2 - cy1)
        crop_cx = (cx1 + cx2) / 2.0
        crop_cy = (cy1 + cy2) / 2.0
        person_cx = (px1 + px2) / 2.0
        person_cy = (py1 + py2) / 2.0
        off_x = abs(person_cx - crop_cx) / crop_w
        off_y = abs(person_cy - crop_cy) / crop_h
        return off_x, off_y, max(off_x, off_y)

    def _isolation_metrics(self, crop_box: Box, boxes: List[Box], current_idx: int) -> Tuple[float, float]:
        cx1, cy1, cx2, cy2 = crop_box
        crop_area = max(1.0, (cx2 - cx1) * (cy2 - cy1))
        max_crop_overlap = 0.0
        max_visible_fraction = 0.0
        for j, b in enumerate(boxes):
            if j == current_idx:
                continue
            x1, y1, x2, y2 = b
            other_area = max(1.0, (x2 - x1) * (y2 - y1))
            ix1, iy1 = max(cx1, x1), max(cy1, y1)
            ix2, iy2 = min(cx2, x2), min(cy2, y2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            max_crop_overlap = max(max_crop_overlap, inter / crop_area)
            max_visible_fraction = max(max_visible_fraction, inter / other_area)
        return max_crop_overlap, max_visible_fraction

    def _score_range(self, value: float, low: float, high: float) -> float:
        if high <= low:
            return 1.0 if value >= high else 0.0
        return clamp((value - low) / (high - low), 0.0, 1.0)

    def _quality_class(self, score: float) -> str:
        qs = self.cfg["quality_scoring"]
        classes = qs.get("classes", {})
        if score >= float(classes.get("premium", 73)):
            return "premium"
        if score >= float(classes.get("good", 65)):
            return "good"
        if score >= float(classes.get("review", 45)):
            return "review"
        return "reject"

    def _quality_folder_name(self, quality_class: str) -> str:
        return {
            "premium": "A_Premium",
            "good": "B_Good",
            "review": "C_Review",
            "reject": "Rejected",
        }.get(quality_class, "C_Review")

    def calculate_quality_score(
        self,
        img: Any,
        crop_box: Box,
        person_box: Box,
        kp: Any,
        boxes: List[Box],
        current_idx: int,
        sharpness: float,
        conf: float,
        fence_info: Dict[str, Any],
        is_full_frame: bool,
    ) -> Tuple[float, str, Dict[str, Any]]:
        qs = self.cfg["quality_scoring"]
        sf = self.cfg["selection_filters"]
        weights = qs.get("weights", {})

        min_sharp = float(sf.get("min_sharpness_threshold", 80))
        excellent_sharp = float(qs.get("excellent_sharpness", 300))
        sharp_component = self._score_range(sharpness, min_sharp, excellent_sharp)

        off_x, off_y, off_max = self._center_offsets(crop_box, person_box)
        # Stage 6 only sorts, so don't penalize hard.
        center_component = clamp(1.0 - min(1.0, off_max / 0.28), 0.0, 1.0)

        h_img, w_img = img.shape[:2]
        person_h = max(1.0, person_box[3] - person_box[1])
        min_h = max(float(sf.get("min_box_height_px", 220)), h_img * float(sf.get("min_height_ratio", 0.15)))
        ideal_h = float(qs.get("ideal_runner_height_px", 700))
        size_component = 1.0 if is_full_frame else self._score_range(person_h, min_h, ideal_h)

        keypoint_conf = float(sf.get("keypoint_conf", 0.30))
        visible_kps = int((kp[:, 2] > keypoint_conf).sum()) if kp is not None else 0
        if is_full_frame and bool(self.cfg["full_frame_runner"].get("bypass_min_keypoints", True)):
            keypoint_component = max(0.65, clamp(visible_kps / 17.0, 0.0, 1.0))
        else:
            keypoint_component = clamp(visible_kps / 17.0, 0.0, 1.0)

        crop_overlap, visible_fraction = self._isolation_metrics(crop_box, boxes, current_idx)
        # No hard isolation rejection like Stage 5; just a score penalty.
        isolation_component = clamp(1.0 - max(crop_overlap / 0.25, visible_fraction / 0.75), 0.0, 1.0)

        conf_component = self._score_range(conf, float(sf.get("conf_threshold", 0.33)), float(qs.get("ideal_confidence", 0.65)))
        fence_score = float(fence_info.get("fence_score", 0.0))
        fence_component = clamp(1.0 - min(1.0, fence_score / 0.25), 0.0, 1.0)

        components = {
            "sharpness": sharp_component,
            "center": center_component,
            "size": size_component,
            "keypoints": keypoint_component,
            "isolation": isolation_component,
            "confidence": conf_component,
            "fence_free": fence_component,
        }
        weighted_sum = 0.0
        weight_sum = 0.0
        for name, component in components.items():
            weight = float(weights.get(name, 0.0))
            weighted_sum += component * weight
            weight_sum += weight
        score = round((weighted_sum / max(0.001, weight_sum)) * 100.0, 2)
        quality_class = self._quality_class(score)

        details = {
            "quality_score": score,
            "quality_class": quality_class,
            "is_full_frame": bool(is_full_frame),
            "q_sharpness": round(sharp_component * 100.0, 1),
            "q_center": round(center_component * 100.0, 1),
            "q_size": round(size_component * 100.0, 1),
            "q_keypoints": round(keypoint_component * 100.0, 1),
            "q_isolation": round(isolation_component * 100.0, 1),
            "q_confidence": round(conf_component * 100.0, 1),
            "q_fence_free": round(fence_component * 100.0, 1),
            "center_offset_x": round(off_x, 4),
            "center_offset_y": round(off_y, 4),
            "center_offset_max": round(off_max, 4),
            "visible_keypoints": visible_kps,
            "other_runner_crop_overlap": round(crop_overlap, 4),
            "other_runner_visible_fraction": round(visible_fraction, 4),
            "person_height_px": round(person_h, 1),
            **fence_info,
        }
        return score, quality_class, details

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def resize_final(self, crop: Any, enhance_mode: str = "upscale") -> Any:
        iq = self.cfg["image_quality"]
        cp = self.cfg["crop"]
        target_h = int(iq.get("target_height", 4000))
        ratio = float(cp.get("aspect_ratio", 0.666))

        # Enhance small crops before the final resize, per the requested mode:
        #   "none"         -> no SR, Lanczos only (e.g. Premium -- already sharp)
        #   "upscale"      -> Real-ESRGAN / FSRCNN super-resolution (e.g. Good)
        #   "upscale+face" -> Real-ESRGAN + GFPGAN face restoration (e.g. Review)
        if enhance_mode != "none" and crop.shape[0] < int(iq.get("ai_upscaling_limit", 1200)):
            if enhance_mode == "upscale+face" and self._face_restorer is not None:
                crop = self._apply_face_restore(crop)
            else:
                crop = self._apply_upscaler(crop)

        # Center-crop to the exact target aspect ratio BEFORE scaling so the
        # output is never stretched. get_smart_crop already aims for this ratio
        # but clamps to the image bounds near the edges, so a crop can arrive a
        # little off-ratio; trim the excess (mostly padding) rather than
        # distorting the runner.
        h, w = crop.shape[:2]
        current_ratio = w / max(1.0, h)
        if current_ratio > ratio:
            new_w = max(1, int(round(h * ratio)))
            x0 = max(0, (w - new_w) // 2)
            crop = crop[:, x0:x0 + new_w]
        elif current_ratio < ratio:
            new_h = max(1, int(round(w / ratio)))
            excess = h - new_h
            # Bias the vertical trim toward the bottom so the head (upper third)
            # is preserved rather than symmetrically clipped near frame edges.
            top_frac = float(cp.get("vertical_trim_top_fraction", 0.25))
            y0 = int(clamp(round(excess * top_frac), 0, max(0, excess)))
            crop = crop[y0:y0 + new_h, :]

        final_w = max(1, int(round(target_h * ratio)))
        interpolation = cv2.INTER_AREA if crop.shape[0] > target_h else cv2.INTER_LANCZOS4
        return cv2.resize(crop, (final_w, target_h), interpolation=interpolation)

    def _copy_exif_without_orientation(self, src: Path, dst: Path) -> None:
        if piexif is None or not bool(self.cfg["image_quality"].get("preserve_exif", True)):
            return
        try:
            exif_dict = piexif.load(str(src))
            if "0th" in exif_dict:
                exif_dict["0th"].pop(piexif.ImageIFD.Orientation, None)
            piexif.insert(piexif.dump(exif_dict), str(dst))
        except Exception:
            pass

    def write_output(
        self,
        src_path: Path,
        candidate_idx: int,
        final: Any,
        sharpness: float,
        conf: float,
        quality_score: float,
        quality_class: str,
        suffix: str = "runner",
        subfolder: Optional[str] = None,
    ) -> Optional[Path]:
        iq = self.cfg["image_quality"]
        qs = self.cfg["quality_scoring"]
        if bool(qs.get("save_quality_subfolders", True)):
            out_folder = self.output_folder / self._quality_folder_name(quality_class)
        else:
            out_folder = self.output_folder
        if subfolder:
            out_folder = out_folder / subfolder
        ensure_dir(out_folder)
        out_name = (
            f"{src_path.stem}_{suffix}{candidate_idx:02d}"
            f"_q{int(round(quality_score))}"
            f"_s{int(round(sharpness))}"
            f"_c{int(round(conf * 100))}.jpg"
        )
        out_path = out_folder / out_name
        ok = cv2.imwrite(str(out_path), final, [cv2.IMWRITE_JPEG_QUALITY, int(iq.get("jpeg_quality", 95))])
        if not ok:
            return None
        self._copy_exif_without_orientation(src_path, out_path)
        return out_path

    def _save_reject_crop(self, img: Any, crop_box: Optional[Box], src_path: Path, idx: int, reason: str) -> str:
        if not bool(self.cfg["debug"].get("save_rejects", False)) or crop_box is None:
            return ""
        try:
            ensure_dir(self.rejects_folder)
            crop = self._extract_roi(img, crop_box)
            safe_reason = "".join(c if c.isalnum() or c in "._-" else "_" for c in reason)[:60]
            out = self.rejects_folder / f"{src_path.stem}_{idx}_{safe_reason}.jpg"
            cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return str(out)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Main per-image logic
    # ------------------------------------------------------------------
    def _should_save_class(self, quality_class: str, score: float) -> bool:
        qs = self.cfg["quality_scoring"]
        if quality_class == "reject":
            return score >= float(qs.get("hard_reject_below_score", 62)) and bool(qs.get("save_review", False))
        if quality_class == "review" and not bool(qs.get("save_review", False)):
            return False
        return True

    def process_image(self, img_path: Path) -> None:
        """Single-image path: read, run inference, then hand off to the shared
        result-processing logic. Used when batch_size <= 1."""
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                self._inc("errors")
                self._report({"image": str(img_path), "decision": "reject", "reason": "cv2_read_failed"})
                return
            if not self._pre_check_image(img):
                self._inc("pre_check_rejected")
                self._report({"image": str(img_path), "decision": "reject", "reason": "pre_check_failed"})
                return
            results = self._predict(img)
        except Exception as exc:
            self._inc("errors")
            print(f"\n[ERROR] {img_path}: {exc}")
            traceback.print_exc()
            self._report({"image": str(img_path), "decision": "reject", "reason": f"exception: {exc}"})
            return
        self._process_detection_results(img, img_path, results)

    def _read_images(self, img_paths: List[Path], read_pool: Optional[ThreadPoolExecutor] = None) -> Tuple[List[Path], List[Any]]:
        """Decode a list of image files (CPU/I/O-bound). Optionally parallelized
        via read_pool so disk reads and JPEG decoding overlap across files."""
        def _read_one(p: Path) -> Tuple[Path, Any]:
            return p, cv2.imread(str(p))

        if read_pool is not None:
            pairs = list(read_pool.map(_read_one, img_paths))
        else:
            pairs = [_read_one(p) for p in img_paths]

        valid_paths: List[Path] = []
        images: List[Any] = []
        for p, img in pairs:
            if img is None:
                self._inc("errors")
                self._report({"image": str(p), "decision": "reject", "reason": "cv2_read_failed"})
                continue
            if not self._pre_check_image(img):
                self._inc("pre_check_rejected")
                self._report({"image": str(p), "decision": "reject", "reason": "pre_check_failed"})
                continue
            valid_paths.append(p)
            images.append(img)
        return valid_paths, images

    def _run_batch_inference(
        self,
        valid_paths: List[Path],
        images: List[Any],
        post_pool: Optional[ThreadPoolExecutor] = None,
    ) -> None:
        """Run one batched YOLO call, then fan the per-image post-processing
        (crop/score/write) out to post_pool so CPU-bound work still runs in
        parallel with the next batch's GPU inference.

        IMPORTANT: we hand YOLO the file *paths* (not our own pre-decoded BGR
        arrays). Ultralytics' path-based loader is the well-tested route for
        batched inference — it letterboxes/preprocesses/stacks images of
        differing sizes internally. Passing a Python list of raw, differently
        sized numpy arrays (combined with half=True on CUDA) was silently
        producing zero detections for every image. We still decode the images
        ourselves (in _read_images) for cropping/scoring/writing, so this just
        means YOLO reads the file a second time -- a small price for correct,
        reliable detections."""
        if not images:
            return
        try:
            results = self._predict([str(p) for p in valid_paths], batch=True)
        except Exception as exc:
            for p in valid_paths:
                self._inc("errors")
                print(f"\n[ERROR] {p}: {exc}")
                self._report({"image": str(p), "decision": "reject", "reason": f"batch_predict_exception: {exc}"})
            return

        single_results = [[r] for r in results]
        if post_pool is not None:
            list(post_pool.map(self._process_detection_results, images, valid_paths, single_results))
        else:
            for img, p, r in zip(images, valid_paths, single_results):
                self._process_detection_results(img, p, r)

    def _prefetch_batches(
        self,
        batches: List[List[Path]],
        read_pool: ThreadPoolExecutor,
        out_queue: "queue.Queue",
    ) -> None:
        """Producer: decode batches of images on background threads and push
        them onto a bounded queue, so the GPU never has to wait idle for the
        next batch's images to be read off disk and decoded."""
        try:
            for batch in batches:
                valid_paths, images = self._read_images(batch, read_pool=read_pool)
                if images:
                    out_queue.put((valid_paths, images))
        finally:
            out_queue.put(None)  # sentinel: no more batches

    def _process_detection_results(self, img: Any, img_path: Path, results: Any) -> None:
        accepted_count = 0
        try:
            h_img, w_img = img.shape[:2]
            if not results or len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                self._inc("no_person")
                self._try_whole_image_fallback(img, img_path, reason="no_person")
                return

            r = results[0]
            boxes_np = r.boxes.xyxy.cpu().numpy()
            confs_np = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(boxes_np))
            boxes: List[Box] = [tuple(map(float, row[:4])) for row in boxes_np]
            kpts = r.keypoints.data.cpu().numpy() if (r.keypoints is not None and r.keypoints.data is not None) else None
            self._inc("images_with_person")

            sf = self.cfg["selection_filters"]
            safe_zone_enabled = bool(sf.get("enable_safe_zone", True))
            safe_zone = float(sf.get("safe_zone_percent", 12)) / 100.0
            min_height_ratio = float(sf.get("min_height_ratio", 0.15))
            min_box_height_px = int(sf.get("min_box_height_px", 220))
            keypoint_conf = float(sf.get("keypoint_conf", 0.30))
            min_keypoints = int(sf.get("min_keypoints", 4))
            min_sharp = float(sf.get("min_sharpness_threshold", 80))

            for i, person_box in enumerate(boxes):
                x1, y1, x2, y2 = person_box
                box_h = y2 - y1
                box_cx = (x1 + x2) / 2.0
                conf = float(confs_np[i]) if i < len(confs_np) else 1.0
                kp = kpts[i] if kpts is not None and i < len(kpts) else None
                is_full_frame = self._is_full_frame_runner(person_box, (h_img, w_img))
                base_row = {
                    "image": str(img_path),
                    "candidate_idx": i,
                    "confidence": round(conf, 4),
                    "box_x1": round(x1, 1),
                    "box_y1": round(y1, 1),
                    "box_x2": round(x2, 1),
                    "box_y2": round(y2, 1),
                    "box_height": round(box_h, 1),
                    "is_full_frame": bool(is_full_frame),
                }

                if safe_zone_enabled and not (is_full_frame and bool(self.cfg["full_frame_runner"].get("bypass_safe_zone", True))):
                    if box_cx < (w_img * safe_zone) or box_cx > (w_img * (1.0 - safe_zone)):
                        self._inc("unsafe_edge")
                        self._report({**base_row, "decision": "reject", "reason": "unsafe_edge"})
                        continue

                if not is_full_frame and box_h < max(h_img * min_height_ratio, min_box_height_px):
                    self._inc("too_small")
                    self._report({**base_row, "decision": "reject", "reason": "too_small"})
                    continue

                visible_kps = int((kp[:, 2] > keypoint_conf).sum()) if kp is not None else 0
                if not (is_full_frame and bool(self.cfg["full_frame_runner"].get("bypass_min_keypoints", True))):
                    if visible_kps < min_keypoints:
                        self._inc("bad_pose")
                        self._report({**base_row, "decision": "reject", "reason": "too_few_keypoints", "visible_keypoints": visible_kps})
                        continue
                    if bool(sf.get("require_frontal_face", False)) and kp is not None:
                        if not (kp[0][2] > 0.45 and (kp[1][2] > 0.35 or kp[2][2] > 0.35)):
                            self._inc("bad_pose")
                            self._report({**base_row, "decision": "reject", "reason": "no_frontal_face"})
                            continue

                if is_full_frame:
                    crop, crop_box = self.get_full_frame_crop(img, person_box)
                else:
                    crop_result = self.get_smart_crop(img, kp, person_box)
                    if crop_result is None:
                        self._inc("crop_failed")
                        self._report({**base_row, "decision": "reject", "reason": "crop_failed"})
                        continue
                    crop, crop_box = crop_result

                # Sharpness first — cheaper than fence detection (no morphological ops),
                # so blurry crops are rejected before the more expensive fence check runs.
                sharp_roi = crop if is_full_frame else self._extract_roi(img, person_box)
                sharpness = self.get_sharpness_score(sharp_roi)
                if bool(sf.get("hard_reject_extreme_blur", True)) and sharpness < min_sharp:
                    self._inc("blurred")
                    reject_path = self._save_reject_crop(img, crop_box, img_path, i, "blurred")
                    self._report({**base_row, "decision": "reject", "reason": "blurred", "sharpness": round(sharpness, 2), "reject_path": reject_path})
                    continue

                fence_info = self.detect_fence(crop, crop_box, person_box)
                if bool(sf.get("hard_reject_fence", True)) and bool(fence_info.get("fence_detected", False)):
                    self._inc("fence_rejected")
                    reject_path = self._save_reject_crop(img, crop_box, img_path, i, "fence_detected")
                    self._report({**base_row, "decision": "reject", "reason": "fence_detected", **fence_info, "reject_path": reject_path})
                    continue

                score, quality_class, details = self.calculate_quality_score(
                    img=img,
                    crop_box=crop_box,
                    person_box=person_box,
                    kp=kp,
                    boxes=boxes,
                    current_idx=i,
                    sharpness=sharpness,
                    conf=conf,
                    fence_info=fence_info,
                    is_full_frame=is_full_frame,
                )

                # Completeness guard: keep partial / head-truncated athletes out
                # of Premium/Good. Soft by default (demote to Review); nothing is
                # discarded unless review_on_partial is turned off.
                cg = self.cfg.get("completeness_guard", {})
                if bool(cg.get("enabled", True)) and not is_full_frame:
                    comp = self._assess_completeness(kp, person_box, (h_img, w_img))
                    details.update({
                        "completeness": comp["completeness"],
                        "has_upper_body": comp["has_upper_body"],
                        "truncated_top": comp["truncated_top"],
                    })
                    if not comp["ok_for_premium"]:
                        if bool(cg.get("review_on_partial", True)):
                            if quality_class in ("premium", "good"):
                                quality_class = "review"
                                details["quality_class"] = "review"
                                details["demoted_reason"] = comp["reason"]
                                self._inc("demoted_partial")
                        else:
                            self._inc("truncated_frame" if comp["truncated_top"] else "partial_body")
                            reject_path = self._save_reject_crop(img, crop_box, img_path, i, comp["reason"] or "partial_body")
                            self._report({**base_row, "decision": "reject", "reason": comp["reason"] or "partial_body", **details, "reject_path": reject_path})
                            continue

                if not self._should_save_class(quality_class, score):
                    self._inc("low_quality_score")
                    reject_path = self._save_reject_crop(img, crop_box, img_path, i, "low_quality_score")
                    self._report({**base_row, "decision": "reject", "reason": "low_quality_score", "sharpness": round(sharpness, 2), **details, "reject_path": reject_path})
                    continue

                # If the score is below 'review' but still above hard_reject_below_score, save to C_Review.
                if quality_class == "reject":
                    quality_class = "review"
                    details["quality_class"] = "review"

                # For review-class crops only: flag the ones that look like
                # they'd actually benefit from AI restoration (moderately
                # soft focus, or under-/over-exposed -- but not bad enough to
                # have been discarded already) and route them into a clearly
                # named subfolder for quick visual triage before enhancing.
                enhancement_subfolder = None
                is_enhancement_candidate = False
                if quality_class == "review":
                    is_enhancement_candidate, candidate_reason = self._is_promising_for_enhancement(crop, sharpness)
                    if is_enhancement_candidate:
                        re_cfg = self.cfg.get("review_enhancement", {})
                        if bool(re_cfg.get("flag_candidates", True)):
                            enhancement_subfolder = str(re_cfg.get("candidates_subfolder", "_promising_for_enhancement"))
                        details["enhancement_candidate_reason"] = candidate_reason
                details["enhancement_candidate"] = bool(is_enhancement_candidate)

                # Per-class enhancement (ENHANCER): e.g. Premium -> Lanczos only,
                # Good -> Real-ESRGAN, Review -> Real-ESRGAN + face restoration.
                enhance_mode = str(self.cfg["image_quality"].get("enhance_by_class", {}).get(quality_class, "upscale"))
                details["enhance_mode"] = enhance_mode

                final = self.resize_final(crop, enhance_mode=enhance_mode)
                # Premium crops are already sharp; skip the global denoise/sharpen
                # so the best frames aren't over-processed (override with
                # image_quality.post_process_premium).
                if quality_class != "premium" or bool(self.cfg["image_quality"].get("post_process_premium", False)):
                    final = self.denoise_image(final)
                    final = self.sharpen_image(final)
                out_path = self.write_output(img_path, i, final, sharpness, conf, score, quality_class, subfolder=enhancement_subfolder)
                if out_path is None:
                    self._inc("write_failed")
                    self._report({**base_row, "decision": "reject", "reason": "write_failed", **details})
                    continue

                accepted_count += 1
                self._inc("processed_crops")
                self._inc(f"quality_{quality_class}")
                self._report({
                    **base_row,
                    "decision": "accept",
                    "reason": "ok",
                    "sharpness": round(sharpness, 2),
                    **details,
                    "crop_x1": round(crop_box[0], 1),
                    "crop_y1": round(crop_box[1], 1),
                    "crop_x2": round(crop_box[2], 1),
                    "crop_y2": round(crop_box[3], 1),
                    "output_path": str(out_path),
                })

            if accepted_count == 0:
                self._try_whole_image_fallback(img, img_path, reason="no_accepted_candidates")

        except Exception as exc:
            self._inc("errors")
            print(f"\n[ERROR] {img_path}: {exc}")
            traceback.print_exc()
            self._report({"image": str(img_path), "decision": "reject", "reason": f"exception: {exc}"})

    def _try_whole_image_fallback(self, img: Any, img_path: Path, reason: str) -> None:
        cfg = self.cfg.get("whole_image_fallback", {})
        if not bool(cfg.get("enabled", False)):
            return
        crop, crop_box = self._crop_to_aspect_around_center(img, img.shape[1] / 2.0, img.shape[0] / 2.0)
        # Whole-image fallback has no person box; use the crop box as an approximation.
        pseudo_person_box = crop_box
        sharpness = self.get_sharpness_score(crop)
        if sharpness < float(cfg.get("min_original_sharpness", 95)):
            self._report({
                "image": str(img_path),
                "candidate_idx": "fallback",
                "decision": "reject",
                "reason": f"fallback_too_soft_after_{reason}",
                "sharpness": round(sharpness, 2),
            })
            return
        fence_info = self.detect_fence(crop, crop_box, pseudo_person_box)
        if bool(cfg.get("reject_if_fence", True)) and bool(fence_info.get("fence_detected", False)):
            self._inc("fence_rejected")
            self._report({
                "image": str(img_path),
                "candidate_idx": "fallback",
                "decision": "reject",
                "reason": f"fallback_fence_after_{reason}",
                "sharpness": round(sharpness, 2),
                **fence_info,
            })
            return
        quality_class = str(cfg.get("save_class", "review"))
        score = 50.0 if quality_class == "review" else 65.0
        final = self.resize_final(crop)
        final = self.denoise_image(final)
        final = self.sharpen_image(final)
        out_path = self.write_output(img_path, 0, final, sharpness, 0.0, score, quality_class, suffix="fullframe")
        if out_path is None:
            self._inc("write_failed")
            return
        self._inc("processed_crops")
        self._inc("fallback_saved")
        self._inc(f"quality_{quality_class}")
        self._report({
            "image": str(img_path),
            "candidate_idx": "fallback",
            "decision": "accept",
            "reason": f"whole_image_fallback_after_{reason}",
            "quality_score": score,
            "quality_class": quality_class,
            "is_full_frame": True,
            "sharpness": round(sharpness, 2),
            **fence_info,
            "output_path": str(out_path),
        })

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def _list_images(self, folder: Path) -> List[Path]:
        valid_ext = {".jpg", ".jpeg", ".JPG", ".JPEG"}
        recursive = bool(self.cfg["paths"].get("recursive", False))
        if recursive:
            imgs = [p for p in folder.rglob("*") if p.suffix in valid_ext and p.is_file()]
        else:
            imgs = [p for p in folder.iterdir() if p.suffix in valid_ext and p.is_file()]
        return sorted(imgs)

    def _write_report_csv(self) -> Optional[Path]:
        if not bool(self.cfg["debug"].get("write_csv", True)):
            return None
        ensure_dir(self.output_folder)
        report_path = self.output_folder / "runner_crop_report.csv"
        if not self.report_rows:
            return report_path
        fieldnames: List[str] = []
        for row in self.report_rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.report_rows)
        return report_path

    # ------------------------------------------------------------------
    # Optional AI enhancement for the C_Review folder
    # ------------------------------------------------------------------
    def _ensure_gfpgan_ready(self) -> bool:
        """Installs the gfpgan library (and its dependencies) and downloads
        the GFPGAN model weights automatically if either is missing. Runs
        only once the user has confirmed they want AI enhancement -- never
        during normal cropping -- and only if 'auto_install' is enabled.
        Returns True if GFPGAN is ready to use, False otherwise (caller
        should fall back to the basic OpenCV enhancement)."""
        global GFPGANer
        re_cfg = self.cfg.get("review_enhancement", {})
        auto_install = bool(re_cfg.get("auto_install", True))

        if GFPGANer is None:
            if not auto_install:
                print("[INFO] gfpgan not installed and auto_install is disabled -- using basic enhancement.")
                print("       To enable AI restoration, run: pip install gfpgan realesrgan basicsr facexlib")
                return False
            print("[AI ENHANCE] gfpgan not found -- installing automatically (this can take a few minutes) ...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    "gfpgan", "realesrgan", "basicsr", "facexlib",
                ])
            except Exception as exc:
                print(f"[WARN] Automatic installation of gfpgan failed: {exc}")
                print("       You can install it manually with: pip install gfpgan realesrgan basicsr facexlib")
                return False
            try:
                import importlib
                module = importlib.import_module("gfpgan")
                GFPGANer = getattr(module, "GFPGANer")
            except Exception as exc:
                print(f"[WARN] gfpgan installed but could not be imported: {exc}")
                return False
            print("[AI ENHANCE] gfpgan installed successfully.")

        model_path = self._resolve_path(str(re_cfg.get("gfpgan_model_path", "GFPGANv1.4.pth")))
        if not model_path.exists():
            if not auto_install:
                print(f"[INFO] GFPGAN model not found at {model_path} and auto_install is disabled.")
                return False
            url = str(re_cfg.get("gfpgan_model_url", "")).strip()
            if not url:
                print(f"[WARN] GFPGAN model missing at {model_path} and no download URL is configured.")
                return False
            print(f"[AI ENHANCE] Downloading face-restoration model weights to {model_path} ...")
            # Download to a temp file and validate before promoting it, so a
            # partial/corrupt fetch (or an HTML error page) is never saved as
            # the .pth and then handed to GFPGAN.
            tmp_path = model_path.with_name(model_path.name + ".part")
            try:
                import urllib.request
                ensure_dir(model_path.parent)
                urllib.request.urlretrieve(url, str(tmp_path))
            except Exception as exc:
                self._unlink_quietly(tmp_path)
                print(f"[WARN] Could not download GFPGAN model automatically: {exc}")
                print(f"       Please download it manually from {url}")
                print(f"       and place it at: {model_path}")
                return False

            expected_sha = str(re_cfg.get("gfpgan_model_sha256", "")).strip().lower()
            # The .pth is a few hundred MB; a 1 MB floor catches truncated
            # fetches / error pages without risking a false reject.
            ok, reason = self._validate_model_download(tmp_path, expected_sha, min_bytes=1_000_000)
            if not ok:
                self._unlink_quietly(tmp_path)
                print(f"[WARN] Downloaded GFPGAN model failed validation ({reason}). Discarding it.")
                print(f"       Please download it manually from {url}")
                print(f"       and place it at: {model_path}")
                return False

            tmp_path.replace(model_path)
            print("[AI ENHANCE] Model download complete (validated).")

        return True

    def _setup_face_restorer(self):
        """Creates a GFPGAN face restorer, installing the library and model
        weights automatically first if needed. Returns None (and the caller
        falls back to basic OpenCV enhancement) if GFPGAN can't be made
        ready for any reason."""
        if not self._ensure_gfpgan_ready() or GFPGANer is None:
            return None
        re_cfg = self.cfg.get("review_enhancement", {})
        model_path = self._resolve_path(str(re_cfg.get("gfpgan_model_path", "GFPGANv1.4.pth")))
        try:
            restorer = GFPGANer(
                model_path=str(model_path),
                upscale=int(re_cfg.get("upscale", 1)),
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
            print("--- [AI ENHANCE] GFPGAN face/body restorer loaded ---")
            return restorer
        except Exception as exc:
            print(f"[WARN] Could not load GFPGAN ({exc}). Using basic enhancement instead.")
            return None

    def _basic_enhance(self, image: Any) -> Any:
        """Fallback enhancement when GFPGAN isn't available: mild denoise,
        CLAHE contrast correction (helps faces/bodies in flat or harsh
        lighting), optional FSRCNN upscaling, and a sharpening pass."""
        enhanced = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        sr = self._get_sr()
        if sr is not None:
            try:
                # Keep the FSRCNN-upscaled result (more real detail) instead of
                # shrinking it back to the input size — the larger, sharper
                # image is the whole point of running SR here.
                enhanced = sr.upsample(enhanced)
            except Exception:
                pass
        return self.sharpen_image(enhanced)

    def enhance_review_images(self) -> None:
        """Runs an AI enhancement pass over every image in the C_Review
        folder: tries GFPGAN face/body restoration first, falls back to a
        basic OpenCV enhancement pipeline otherwise. Enhanced copies are
        written to a subfolder so originals are preserved."""
        re_cfg = self.cfg.get("review_enhancement", {})
        review_folder = self.output_folder / self._quality_folder_name("review")
        if not review_folder.exists():
            print(f"[INFO] No review folder found at {review_folder}; nothing to enhance.")
            return

        candidates_name = str(re_cfg.get("candidates_subfolder", "_promising_for_enhancement"))
        candidates_folder = review_folder / candidates_name
        candidate_images = []
        if bool(re_cfg.get("flag_candidates", True)) and candidates_folder.exists():
            candidate_images = [p for p in sorted(candidates_folder.iterdir()) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]

        if candidate_images:
            images = candidate_images
            print(f"[AI ENHANCE] Found {len(images)} image(s) flagged as promising for AI enhancement -- enhancing those.")
        else:
            images = [p for p in sorted(review_folder.iterdir()) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}]
            if images:
                print("[AI ENHANCE] No flagged candidates found -- enhancing all C_Review images instead.")

        if not images:
            print(f"[INFO] No images found in {review_folder}; nothing to enhance.")
            return

        out_folder = review_folder / str(re_cfg.get("output_subfolder", "_enhanced"))
        ensure_dir(out_folder)

        restorer = self._setup_face_restorer()
        jpeg_quality = int(self.cfg["image_quality"].get("jpeg_quality", 95))

        print(f"\n--- [AI ENHANCE] Enhancing {len(images)} review image(s) -> {out_folder} ---")
        iterator = tqdm(images, desc="Enhancing", total=len(images)) if tqdm is not None else images
        done, failed = 0, 0
        for img_path in iterator:
            try:
                image = cv2.imread(str(img_path))
                if image is None:
                    failed += 1
                    continue
                if restorer is not None:
                    with self.sr_lock:
                        _, _, enhanced = restorer.enhance(
                            image, has_aligned=False, only_center_face=False, paste_back=True
                        )
                    if enhanced is None:
                        enhanced = self._basic_enhance(image)
                else:
                    enhanced = self._basic_enhance(image)

                out_path = out_folder / img_path.name
                ok = cv2.imwrite(str(out_path), enhanced, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                if ok:
                    self._copy_exif_without_orientation(img_path, out_path)
                    done += 1
                else:
                    failed += 1
            except Exception as exc:
                failed += 1
                print(f"\n[WARN] Enhancement failed for {img_path.name}: {exc}")

        print(f"--- [AI ENHANCE] Done: {done} enhanced, {failed} failed. Output: {out_folder} ---\n")

    def _maybe_offer_review_enhancement(self) -> None:
        re_cfg = self.cfg.get("review_enhancement", {})
        if not bool(re_cfg.get("enabled", True)) or not bool(re_cfg.get("prompt_after_run", True)):
            return
        review_folder = self.output_folder / self._quality_folder_name("review")
        if not review_folder.exists() or not any(
            p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"} for p in review_folder.iterdir()
        ):
            return
        try:
            answer = input(
                "\nEnhance images in C_Review with AI (face/body restoration & cleanup)? [y/N]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if answer in ("y", "yes", "j", "ja"):
            self.enhance_review_images()
        else:
            print("Skipping AI enhancement of C_Review.")

    def run(self) -> None:
        input_folder = self._resolve_path(str(self.cfg["paths"]["input_folder"]))
        if not input_folder.exists():
            print(f"\n[ERROR] Input folder not found: {input_folder}")
            return
        imgs = self._list_images(input_folder)
        self.stats["total_images"] = len(imgs)
        if not imgs:
            print(f"\n[NOTE] No JPG/JPEG images found in: {input_folder}")
            return
        ensure_dir(self.output_folder)
        if bool(self.cfg["debug"].get("save_rejects", False)):
            ensure_dir(self.rejects_folder)

        print("\n" + "=" * 72)
        print("HighRes Runner Suite - Stage 6 HighRes Plus")
        print("=" * 72)
        print(f"Input:   {input_folder}")
        print(f"Output:  {self.output_folder}")
        print(f"Images:  {len(imgs)}")
        print(f"Device:  {self.device_mode}")
        print(f"imgsz:   {self.cfg['detector'].get('imgsz', 1024)}")
        print(f"Workers: {self.cfg['performance'].get('workers', 20)}")
        print(f"Batch:   {self.cfg['performance'].get('batch_size', 1)}")
        print("=" * 72)

        workers = max(1, int(self.cfg["performance"].get("workers", 20)))
        batch_size = max(1, int(self.cfg["performance"].get("batch_size", 1)))

        # With many Python worker threads, let each OpenCV call (decode, resize,
        # FSRCNN, sharpen, JPEG encode) run single-threaded and get parallelism
        # from the workers instead. Otherwise 32 workers x OpenCV's own thread
        # pool massively oversubscribe the CPU and stall on context switching.
        if workers > 1:
            try:
                cv2.setNumThreads(1)
            except Exception:
                pass

        # OpenVINO models are exported with a fixed batch size of 1 at export
        # time. Feeding a larger batch causes an input-shape mismatch error
        # inside the OpenVINO runtime.  Force single-image processing so every
        # call to _predict() sends exactly one image.
        if self.backend == "openvino" and batch_size > 1:
            print("[INFO] OpenVINO backend detected -- forcing batch_size=1 (static model shape).")
            batch_size = 1

        if batch_size <= 1:
            # Classic per-image path: one YOLO call per image, parallelized via threads.
            if workers == 1:
                iterator = tqdm(imgs, total=len(imgs), desc="Runner-Crops") if tqdm is not None else imgs
                for img_path in iterator:
                    self.process_image(img_path)
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    iterator = ex.map(self.process_image, imgs)
                    if tqdm is not None:
                        list(tqdm(iterator, total=len(imgs), desc="Runner-Crops"))
                    else:
                        list(iterator)
        else:
            # Batched + prefetching path: a background producer thread reads and
            # decodes upcoming batches (CPU/I/O-bound) into a bounded queue while
            # the main thread keeps the GPU fed with batched YOLO calls. This
            # overlaps image I/O with inference instead of letting the GPU idle
            # between bursts. Per-image post-processing (crop/score/write) is
            # fanned out to post_pool so it overlaps with the *next* batch's
            # inference too.
            batches = [imgs[i:i + batch_size] for i in range(0, len(imgs), batch_size)]
            progress = tqdm(total=len(imgs), desc="Runner-Crops") if tqdm is not None else None
            post_pool: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
            read_pool = ThreadPoolExecutor(max_workers=workers)
            # Prefetch depth keeps batches pre-decoded so the GPU is less likely
            # to stall waiting for I/O; the auto-tuner lowers it on low-RAM
            # machines so high-res batches don't blow the memory budget.
            prefetch_maxsize = max(1, int(self.cfg["performance"].get("prefetch_maxsize", 4)))
            batch_queue: "queue.Queue" = queue.Queue(maxsize=prefetch_maxsize)
            producer = threading.Thread(
                target=self._prefetch_batches,
                args=(batches, read_pool, batch_queue),
                daemon=True,
            )
            try:
                producer.start()
                while True:
                    item = batch_queue.get()
                    if item is None:
                        break
                    valid_paths, images = item
                    self._run_batch_inference(valid_paths, images, post_pool=post_pool)
                    if progress is not None:
                        progress.update(len(valid_paths))
            finally:
                producer.join()
                read_pool.shutdown(wait=True)
                if post_pool is not None:
                    post_pool.shutdown(wait=True)
                if progress is not None:
                    progress.close()

        report_path = self._write_report_csv()
        print("\n" + "=" * 72)
        print("REPORT")
        print("=" * 72)
        for key in [
            "total_images", "pre_check_rejected", "images_with_person", "processed_crops",
            "quality_premium", "quality_good", "quality_review", "fallback_saved",
            "no_person", "too_small", "unsafe_edge", "bad_pose", "blurred",
            "fence_rejected", "low_quality_score", "crop_failed",
            "demoted_partial", "partial_body", "truncated_frame", "write_failed", "errors"
        ]:
            print(f"{key:24s}: {self.stats.get(key, 0)}")
        if report_path is not None:
            print(f"CSV-Report: {report_path}")
        print("=" * 72)

        self._maybe_offer_review_enhancement()


if __name__ == "__main__":
    # Optional config path arg lets a launcher pick a preset, e.g. the ENHANCER
    # variant:  python runner_suite_core.py settings_enhancer.json
    config_file = sys.argv[1] if len(sys.argv) > 1 else "settings.json"
    HighResRunnerSuite(config_file).run()
