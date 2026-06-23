@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title HighRes Runner Suite Stage 6 - Run

echo =======================================================
echo   HighRes Runner Suite Stage 6 - Run
echo =======================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [FEHLER] .venv wurde nicht gefunden.
    echo Bitte zuerst setup_runner_suite.bat ausfuehren.
    pause
    exit /b 1
)

if not exist "runner_suite_core.py" (
    echo [FEHLER] runner_suite_core.py wurde nicht gefunden.
    pause
    exit /b 1
)

if not exist "settings.json" (
    echo [FEHLER] settings.json wurde nicht gefunden.
    pause
    exit /b 1
)

if not exist "yolov8m-pose.pt" (
    echo [FEHLER] yolov8m-pose.pt wurde nicht gefunden.
    echo Bitte die Modell-Datei in diesen Ordner legen oder settings.json anpassen.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
rem Use the venv interpreter explicitly so a Store/global "python" on PATH
rem can never shadow it (that mismatch hides venv-only packages like
rem onnxruntime-directml).
".venv\Scripts\python.exe" check_cuda_environment.py
".venv\Scripts\python.exe" runner_suite_core.py

echo.
echo Lauf beendet.
pause
