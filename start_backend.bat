@echo off
title PackVision AI - Backend API
color 0B
cd /d "%~dp0"

echo =======================================================================
echo          PackVision AI - FastAPI Backend API Server
echo =======================================================================
echo.

set "PYTHONPATH=%~dp0;%~dp0backend"

if exist "backend\venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0backend\venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [1/2] Using Python: %PYTHON_EXE%
echo [2/2] Starting Uvicorn on http://127.0.0.1:8000 ...
echo       API Docs available at http://127.0.0.1:8000/docs
echo.

"%PYTHON_EXE%" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Backend failed to start.
    pause
)
