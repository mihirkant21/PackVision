@echo off
title PackVision AI Studio (background)
cd /d "%~dp0"
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 >> storage\uvicorn_studio.log 2>&1
