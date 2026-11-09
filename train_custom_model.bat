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
echo [*] Verifying paired packaging dataset...
%PYTHON_CMD% data/dataset_generator.py --samples 80 --output-dir data/dataset

echo.
echo [*] Initiating model training on custom packaging dataset...
%PYTHON_CMD% scripts/train_custom_model.py --dataset-dir data/dataset --epochs 10 --batch-size 4 --lr 0.0002

echo.
echo =======================================================================
echo Model training routine completed.
echo Checkpoint stored at: models/checkpoints/packvision_custom_v1.pt
echo =======================================================================
pause
