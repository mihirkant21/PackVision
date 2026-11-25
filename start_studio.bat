@echo off
title PackVision AI - Local Application Launcher
color 0A
echo =======================================================================
echo         PackVision AI - Sketch-to-Photoreal Packaging Studio
echo =======================================================================
echo.
cd /d "%~dp0"

echo [1/2] Launching Backend Server on http://127.0.0.1:8000...
start "PackVision Backend" cmd /k "%~dp0start_backend.bat"

timeout /t 3 /nobreak >nul

echo [2/2] Launching React Frontend on http://localhost:5173...
start "PackVision Frontend" cmd /k "%~dp0start_frontend.bat"

echo.
echo =======================================================================
echo   PackVision AI Studio is now running!
echo   Frontend:  http://localhost:5173
echo   Backend:   http://127.0.0.1:8000
echo   API Docs:  http://127.0.0.1:8000/docs
echo =======================================================================
