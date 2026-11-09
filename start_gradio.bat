@echo off
title PackVision AI - Gradio ML Research Workbench
color 0B
echo =======================================================================
echo        PackVision AI - Gradio ML & Packaging Research Suite
echo =======================================================================
echo.
cd /d "%~dp0"

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
)

echo [*] Starting Gradio Packaging Studio Workbench...
echo [*] Environment Python: %PYTHON_CMD%
echo.

start "" "http://localhost:7860"
%PYTHON_CMD% ui/app_gradio.py
pause
