@echo off
title PackVision AI - Automated Test Suite
color 0E
echo =======================================================================
echo          PackVision AI - Comprehensive Validation & Test Suite
echo =======================================================================
echo.
cd /d "%~dp0"

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
)

echo [*] Running Pytest test suite using %PYTHON_CMD%...
echo.
%PYTHON_CMD% -m pytest backend/tests -v
echo.
echo =======================================================================
echo All tests executed.
echo =======================================================================
pause
