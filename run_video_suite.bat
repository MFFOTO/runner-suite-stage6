@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title HighRes Runner Suite Stage 6 - VIDEO (Burst Fusion)

echo =======================================================
echo   HighRes Runner Suite Stage 6 - VIDEO
echo   (automatic multi-frame burst fusion from raw video)
echo =======================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [FEHLER] .venv wurde nicht gefunden.
    echo Bitte zuerst setup_runner_suite.bat ausfuehren.
    pause
    exit /b 1
)

if not exist "video_suite.py" (
    echo [FEHLER] video_suite.py wurde nicht gefunden.
    pause
    exit /b 1
)

if not exist "runner_suite_core.py" (
    echo [FEHLER] runner_suite_core.py wurde nicht gefunden.
    pause
    exit /b 1
)

if not exist "settings_video.json" (
    if exist "settings_video.example.json" (
        echo Erstelle settings_video.json aus settings_video.example.json ...
        copy "settings_video.example.json" "settings_video.json" >nul
        echo Bitte video.input_video und paths.output_folder in settings_video.json anpassen.
        echo.
    ) else (
        echo [FEHLER] settings_video.example.json fehlt.
        pause
        exit /b 1
    )
)

if not exist "yolov8m-pose.pt" (
    echo [FEHLER] yolov8m-pose.pt wurde nicht gefunden.
    echo Bitte die Modell-Datei in diesen Ordner legen oder settings_video.json anpassen.
    pause
    exit /b 1
)

rem Explicit venv interpreter + ignore Store-Python user-site (see run_elite_suite.bat).
set "PYTHONNOUSERSITE=1"
".venv\Scripts\python.exe" check_cuda_environment.py
".venv\Scripts\python.exe" video_suite.py settings_video.json

echo.
echo Lauf beendet.
pause
