@echo off
title PackVision AI - Neural Network Model Training
color 0D
echo =======================================================================
echo          PackVision AI - Custom PackVisionNet Model Trainer
echo =======================================================================
echo.
cd /d "%~dp0"

set PYTHON_CMD=python
if exist ".venv\Scripts\python.exe" (
    set PYTHON_CMD=.venv\Scripts\python.exe
)

echo [*] Using Environment Python: %PYTHON_CMD%
echo.

REM --- Dataset check -----------------------------------------------------------
REM Regenerating the dataset overwrites existing sketch/render/depth triplets,
REM so it is only offered when the split manifests are missing.
if exist "data\dataset\train_manifest.jsonl" if exist "data\dataset\val_manifest.jsonl" (
    echo [*] Found data\dataset\train_manifest.jsonl and val_manifest.jsonl - using existing dataset.
    goto :train
)

echo [!] Split manifests not found in data\dataset.
echo     Regenerating the dataset will (re)write triplet images into
echo     data\dataset\train and data\dataset\val.
echo.
set /p REGEN="Generate a fresh dataset now (210 triplets, ~5 min on CPU)? [y/N]: "
if /I not "%REGEN%"=="y" (
    echo [x] Aborting: training requires data\dataset\train_manifest.jsonl and val_manifest.jsonl.
    echo     Tip: if the triplet images exist but manifests are missing, run:
    echo         %PYTHON_CMD% data/repair_manifests.py --write
    pause
    exit /b 1
)

echo.
echo [*] Generating paired packaging dataset...
%PYTHON_CMD% data/dataset_generator.py
if errorlevel 1 (
    echo [x] Dataset generation failed.
    pause
    exit /b 1
)

:train
echo.
echo [*] Initiating model training on custom packaging dataset...
%PYTHON_CMD% scripts/train_custom_model.py --dataset-dir data/dataset --epochs 10 --batch-size 4 --lr 0.0002

echo.
echo =======================================================================
echo Model training routine completed.
echo Checkpoint stored at: models/checkpoints/packvision_custom_v1.pt
echo =======================================================================
pause
