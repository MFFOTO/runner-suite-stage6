@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Setup - HighRes Runner Suite Stage 6

echo =======================================================
echo   HighRes Runner Suite Stage 6 - Setup
echo =======================================================
echo.

if not exist "settings.json" (
    if exist "settings.example.json" (
        echo [0/5] Erstelle settings.json aus settings.example.json ...
        copy "settings.example.json" "settings.json" >nul
        echo        Bitte input_folder und output_folder in settings.json anpassen.
        echo.
    )
)

where python >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] Python wurde nicht gefunden.
    echo Bitte Python 3.10 oder 3.11 installieren und danach erneut starten.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Erzeuge lokale Python-Umgebung .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [FEHLER] Virtuelle Umgebung konnte nicht erstellt werden.
        echo.
        echo Meist ist kein echtes Python installiert - nur der Microsoft-Store-Platzhalter.
        echo   1. Echtes Python installieren:  winget install -e --id Python.Python.3.12
        echo      oder von https://www.python.org/downloads/ mit "Add python.exe to PATH".
        echo   2. Store-Aliase deaktivieren: Einstellungen, Apps, Erweiterte App-Einstellungen,
        echo      App-Ausfuehrungsaliase, dann python.exe und python3.exe ausschalten.
        echo   3. Neues Terminal oeffnen und setup_runner_suite.bat erneut starten.
        pause
        exit /b 1
    )
) else (
    echo [1/5] Lokale Python-Umgebung .venv ist vorhanden.
)

call ".venv\Scripts\activate.bat"

echo [2/5] Aktualisiere pip, setuptools und wheel ...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [FEHLER] pip konnte nicht aktualisiert werden.
    pause
    exit /b 1
)

echo [3/5] Installiere PyTorch ...
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo Keine NVIDIA-GPU erkannt. Installiere PyTorch CPU-Version.
    python -m pip install --upgrade torch torchvision torchaudio
) else (
    echo NVIDIA-GPU erkannt. Installiere PyTorch CUDA 12.4-Version.
    python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)
if errorlevel 1 (
    echo [FEHLER] PyTorch-Installation fehlgeschlagen.
    pause
    exit /b 1
)

echo [4/5] Installiere Runner-Suite-Bibliotheken ...
python -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
    echo [FEHLER] Installation aus requirements.txt fehlgeschlagen.
    pause
    exit /b 1
)

echo [5/5] Pruefe CUDA und Bibliotheken ...
python check_cuda_environment.py
if errorlevel 1 (
    echo [FEHLER] CUDA-/Bibliothekspruefung fehlgeschlagen.
    pause
    exit /b 1
)

if not exist "yolov8m-pose.pt" (
    echo.
    echo [WARN] yolov8m-pose.pt liegt nicht im aktuellen Ordner.
    echo Bitte die Datei in diesen Ordner kopieren oder settings.json anpassen.
)

echo.
echo =======================================================
echo   SETUP FERTIG
echo =======================================================
echo Starte danach: run_elite_suite.bat
echo.
pause
