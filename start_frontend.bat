@echo off
title PackVision AI - Frontend Studio
color 0D
cd /d "%~dp0frontend"

echo =======================================================================
echo          PackVision AI - React + Vite Frontend Studio
echo =======================================================================
echo.
echo Starting Vite Dev Server on http://localhost:5173 ...
echo.

npm run dev

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Frontend failed to start.
    pause
)
