@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title HighRes Runner Suite Stage 6 - ENHANCER (Real-ESRGAN)

echo =======================================================
echo   HighRes Runner Suite Stage 6 - ENHANCER
echo   (Real-ESRGAN upscaling, medium sharpen, strict reject)
echo =======================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [FEHLER] .venv wurde nicht gefunden.
    echo Bitte zuerst setup_runner_suite.bat ausfuehren.
    pause
    exit /b 1
)

if not exist "settings_enhancer.json" (
    if exist "settings_enhancer.example.json" (
        echo Erstelle settings_enhancer.json aus settings_enhancer.example.json ...
        copy "settings_enhancer.example.json" "settings_enhancer.json" >nul
        echo Bitte input_folder und output_folder in settings_enhancer.json anpassen.
        echo.
    ) else (
        echo [FEHLER] settings_enhancer.example.json fehlt.
        pause
        exit /b 1
    )
)

rem Explicit venv interpreter + ignore Store-Python user-site (see run_elite_suite.bat).
set "PYTHONNOUSERSITE=1"
".venv\Scripts\python.exe" check_cuda_environment.py
".venv\Scripts\python.exe" runner_suite_core.py settings_enhancer.json

echo.
echo Lauf beendet.
pause
