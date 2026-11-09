@echo off
title PackVision AI - Luxury Packaging Design Studio
color 0A
echo =======================================================================
echo          PackVision AI - Neural Packaging Synthesis Studio
echo =======================================================================
echo.
cd /d "%~dp0"

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
)

echo [*] Starting PackVision AI Backend & Web Workbench...
echo [*] Environment Python: %PYTHON_CMD%
echo.

start "" "http://localhost:8000"
%PYTHON_CMD% -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
pause
