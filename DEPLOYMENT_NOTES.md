# Deployment & GitHub Guide — HighRes Runner Suite

Private repo: **https://github.com/MFFOTO/runner-suite-stage6**

The suite auto-detects the hardware at startup (`[AUTO-TUNE]` block) and picks the
backend, model, worker count, and image size on its own. You only ever set the
input/output paths per machine.

---

## 1. Set up a NEW machine from GitHub

**a) Install git** (once per machine):
- Normal Windows (Intel/AMD): `winget install --id Git.Git -e`
- Snapdragon / Windows-on-ARM: same command — Git for Windows ships an ARM64 build.
- (Or download from https://git-scm.com/download/win)

**b) Clone the repo** (the first clone opens a browser login, because the repo is private):
```
cd D:\
git clone https://github.com/MFFOTO/runner-suite-stage6.git
cd runner-suite-stage6
```

**c) Run setup:**
```
setup_runner_suite.bat
```
This creates `settings.json` from `settings.example.json`, builds the `.venv`,
and installs PyTorch + the suite libraries.

**d) Set your paths** — open `settings.json`, set `input_folder` and
`output_folder`. **Always use forward slashes** (`D:/22742/...`), never
backslashes — backslashes break JSON.

**e) Run:**
```
run_elite_suite.bat
```

---

## 2. Update ANY machine (the whole point)

```
cd D:\runner-suite-stage6
git pull
```
Only code updates land. Your `settings.json` and downloaded models are never
touched (they are per-machine / git-ignored).

---

## 3. Convert an EXISTING (non-git) folder into a clone

If a machine already has a working project folder and you want `git pull` to work
there without re-downloading everything:

```
cd D:\runner_suite_stage6_highres_plus      (your existing folder)
git init
git remote add origin https://github.com/MFFOTO/runner-suite-stage6.git
git fetch origin
del yolov8m-pose.pt yolov8n-pose.onnx 2>nul
git reset --hard origin/master
```
- `settings.json`, `.venv`, and the extra model weights (`.pb`/`.pth`) are
  git-ignored, so they stay as-is.
- The `del` line removes the two tracked model files so the reset can restore
  the repo's copies without an "untracked file would be overwritten" error
  (they come back from the local fetch — no re-download).

**Simpler alternative (recommended if unsure):** just clone fresh into a new
folder (Section 1), copy your old `settings.json` in, and delete the old folder
once it works.

---

## 4. How settings are handled

- `settings.json` — **per machine**, git-ignored. Your paths live here.
- `settings.example.json` — the tracked template. `setup_runner_suite.bat`
  copies it to `settings.json` on first run.
- Because `settings.json` is ignored, `git pull` never conflicts on it.

---

## 5. Hardware notes / gotchas learned the hard way

**Windows paths in `settings.json`** — use `/`, e.g. `"D:/events/originals"`.
A single `\` is an invalid JSON escape and crashes on load.

**Microsoft Store Python** (common on Copilot+ / Snapdragon laptops):
- A bare `python` command may point at the sandboxed Store Python, not the venv.
  `run_elite_suite.bat` already calls `.venv\Scripts\python.exe` explicitly.
- Store-Python venvs leak the per-user site-packages into `sys.path`, which can
  shadow venv packages (e.g. a plain `onnxruntime` hiding `onnxruntime-directml`).
  `run_elite_suite.bat` sets `PYTHONNOUSERSITE=1` to block that.

**Snapdragon / ARM GPU (DirectML):**
- The suite runs inference on the Adreno GPU when `yolov8n-pose.onnx` is present
  and `onnxruntime-directml` is installed in the venv
  (`pip install onnxruntime-directml`).
- **`yolov8n-pose.onnx` is included in the repo**, so a clone/pull already has it —
  no need to export it. (Exporting on-device is often blocked by Smart App
  Control on Copilot+ PCs; shipping the file sidesteps that entirely.)
- Confirm the GPU path in the `[AUTO-TUNE]` block:
  `Backend: onnxruntime / DirectML (GPU)`.

**Models:**
- `yolov8m-pose.pt` (GPU tier) and `yolov8n-pose.onnx` (ARM/GPU) are in the repo.
- `yolov8s-pose.pt` / `yolov8n-pose.pt` auto-download from ultralytics when needed.
- `FSRCNN_x3.pb` and `GFPGANv1.4.pth` auto-download (hash-validated) on first use.

**CPU-only speed:** `pip install openvino` gives a faster x86 CPU backend; the
suite switches to it automatically. On weak/CPU machines, lowering
`image_quality.target_height` (e.g. 4000 → 3000) is the biggest speed lever.
